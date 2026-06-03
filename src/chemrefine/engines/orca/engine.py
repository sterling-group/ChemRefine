"""``OrcaEngine`` — the standard DFT engine, driven from ORCA inputs.

Implements :class:`~chemrefine.engines.base.CalculationEngine` by
composing the smaller modules in this package:

* :mod:`engines.orca.input` writes the per-structure ``.inp`` files.
* :mod:`chemrefine.slurm` + :mod:`chemrefine.throttle` build and submit
  SLURM scripts under the PAL budget.
* :mod:`engines.orca.output` parses the resulting ``.out`` files into
  :class:`~chemrefine.state.Structure` instances.
* :mod:`engines.orca.nms` runs normal-mode sampling for steps that
  request it.

Submit and wait are intentionally collapsed: :meth:`submit` registers
every job with the throttler and blocks until every job finishes, so
:meth:`wait` is a no-op. The engine instance therefore carries no
state between steps.
"""

from __future__ import annotations

from pathlib import Path

from ase import Atoms

from chemrefine.engines.base import SlurmBatchEngine, register
from chemrefine.engines.orca import input as orca_input
from chemrefine.engines.orca import nms, output
from chemrefine.ids import allocate_child_ids, structure_artifact_path
from chemrefine.io import write_xyz
from chemrefine.state import StepContext, StepInputs, StepResults, Structure


@register("orca")
class OrcaEngine(SlurmBatchEngine):
    """Standard ORCA DFT engine."""

    name = "orca"
    supports_nms = True
    label = "ORCA"
    template_suffix = "inp"
    output_globs = ("*.out", "*.xyz", "*.gbw", "*.hess")

    # -- prepare -----------------------------------------------------------

    def prepare(self, ctx: StepContext) -> StepInputs:
        """Write one ``.xyz`` + ``.inp`` per seed structure."""
        ctx.step_dir.mkdir(parents=True, exist_ok=True)
        template = self._resolve_template(ctx)

        files: list[tuple[Path, Path, str]] = []
        for struct in ctx.prev_state.structures:
            xyz_paths = write_xyz(
                [struct.atoms],
                [struct.id],
                step_number=ctx.step_cfg.step,
                output_dir=ctx.step_dir,
            )
            xyz_path = xyz_paths[0]
            step = ctx.step_cfg.step
            inp_path = structure_artifact_path(ctx.step_dir, step, struct.id, "inp")
            out_path = structure_artifact_path(ctx.step_dir, step, struct.id, "out")
            orca_input.build_input(
                xyz_path=xyz_path,
                template_path=template,
                output_path=inp_path,
                charge=ctx.charge,
                multiplicity=ctx.multiplicity,
                extra_blocks=self._extra_blocks(ctx),
            )
            files.append((inp_path, out_path, struct.id))
        return StepInputs(files=tuple(files))

    def _extra_blocks(self, ctx: StepContext) -> str:
        """Subclass hook for engines that need extra ORCA blocks (e.g. MLFF ``%method``).

        The base ORCA engine has nothing extra to add; MLFF and PySCF
        override this to inject their ``%method ProgExt …`` block.
        """
        return ""

    # -- submit (PAL + run_block hooks; the loop lives on SlurmBatchEngine) -

    def _pal(self, ctx: StepContext) -> int:
        """Read PAL once from the step template.

        PAL is a property of the template, not of any individual structure:
        ORCA copies the same ``%pal`` block into every generated ``.inp``, so
        read it once from the template rather than the per-structure copies.
        """
        return orca_input.parse_pal(self._resolve_template(ctx))

    def _run_block(self, ctx: StepContext, inp_path: Path, out_path: Path) -> str:
        """Engine-specific bash that runs inside ``$WORK_DIR``."""
        orca = ctx.executables.get("orca", "orca")
        return (
            "export OMP_NUM_THREADS=1\n"
            f"{orca} {inp_path.name} > $OUTPUT_DIR/{out_path.name}"
        )

    def _extra_header_fields(self, ctx: StepContext) -> tuple[tuple[str, object], ...]:
        """Record which ORCA binary ran in the runlog header."""
        return (("orca_executable", ctx.executables.get("orca", "orca")),)

    # -- parse -------------------------------------------------------------

    def parse(self, inputs: StepInputs, ctx: StepContext) -> StepResults:
        """Read each output file and build :class:`Structure` instances."""
        operation = ctx.step_cfg.operation
        prev_by_id = {s.id: s for s in ctx.prev_state.structures}

        # Parse every output first so each input's fan-out is known, then mint
        # child IDs through the shared lineage convention in `ids` instead of
        # re-implementing the ``{parent}-{i}`` format here.
        parsed_per_input = [
            (sid, output.parse_output(out_path, operation))
            for _inp, out_path, sid in inputs.files
        ]
        parents = [sid for sid, _ in parsed_per_input]
        fanouts = [len(parsed) for _, parsed in parsed_per_input]
        child_ids = iter(allocate_child_ids(parents, fanouts))

        out_structures: list[Structure] = []
        for sid, parsed in parsed_per_input:
            input_struct = prev_by_id.get(sid)
            is_fanout = len(parsed) > 1
            for ps in parsed:
                # Fan-out: parent is the input that fanned out.
                # 1:1: child inherits the input's parent lineage unchanged.
                child_parent = (
                    sid
                    if is_fanout
                    else (input_struct.parent_id if input_struct is not None else None)
                )
                atoms = Atoms(symbols=list(ps.symbols), positions=ps.positions)
                out_structures.append(
                    Structure(
                        id=next(child_ids),
                        atoms=atoms,
                        parent_id=child_parent,
                        energy_hartree=ps.energy_hartree,
                        forces_ev_per_a=ps.forces_ev_per_a,
                        converged=ps.converged,
                        terminated=ps.terminated,
                    )
                )
        return StepResults(structures=tuple(out_structures))

    # -- nms ---------------------------------------------------------------

    def normal_mode_sample(self, results: StepResults, ctx: StepContext) -> StepResults:
        """Delegate to :mod:`engines.orca.nms`."""
        return nms.normal_mode_sample(results, ctx)

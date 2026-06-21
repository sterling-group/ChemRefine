"""``OrcaEngine`` — the standard DFT engine, driven from ORCA inputs.

Implements :class:`~chemrefine.engines.base.CalculationEngine` by
composing the smaller modules in this package:

* :mod:`engines.orca.input` writes the per-structure ``.inp`` files.
* :mod:`chemrefine.slurm` + :mod:`chemrefine.throttle` build and submit
  SLURM scripts under the PAL budget.
* :mod:`engines.orca.output` parses the resulting ``.out`` files into
  :class:`~chemrefine.state.Structure` instances.

Normal-mode sampling is engine-independent — the two-round algorithm lives in
:mod:`chemrefine.nms`, which drives this engine through the two NMS hooks it
supplies: :meth:`nms_input_info` (read the template's keywords) and
:meth:`read_frequencies` (parse a ``.out``'s imaginary modes + normal-mode tensor).

Submit and wait are intentionally collapsed: :meth:`submit` registers
every job with the throttler and blocks until every job finishes, so
:meth:`wait` is a no-op. The engine instance is stateless.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

from ase import Atoms

from chemrefine.engines.base import FrequencyData, NmsInputInfo, SlurmBatchEngine, register
from chemrefine.engines.orca import frequencies, inspect, output
from chemrefine.engines.orca import input as orca_input
from chemrefine.errors import OutputParseError
from chemrefine.ids import allocate_child_ids, input_geometry_path, structure_artifact_path
from chemrefine.io import write_single_xyz
from chemrefine.state import StepContext, StepInputs, StepResults, Structure

logger = logging.getLogger(__name__)


@register("orca")
class OrcaEngine(SlurmBatchEngine):
    """Standard ORCA DFT engine."""

    name: ClassVar[str] = "orca"
    supports_nms: ClassVar[bool] = True
    label: ClassVar[str] = "ORCA"
    template_suffix: ClassVar[str] = "inp"
    output_globs: ClassVar[tuple[str, ...]] = ("*.out", "*.xyz", "*.gbw", "*.hess")

    # -- prepare -----------------------------------------------------------

    def prepare(self, ctx: StepContext) -> StepInputs:
        """Write one ``.xyz`` + ``.inp`` per seed structure."""
        ctx.step_dir.mkdir(parents=True, exist_ok=True)
        template = self._resolve_template(ctx)

        files: list[tuple[Path, Path, str]] = []
        step = ctx.step_cfg.step
        for struct in ctx.prev_state.structures:
            xyz_path = write_single_xyz(
                struct.atoms,
                input_geometry_path(ctx.step_dir, step, struct.id),
                comment=f"step {step} {struct.id} input",
            )
            inp_path = structure_artifact_path(ctx.step_dir, step, struct.id, "inp")
            out_path = structure_artifact_path(ctx.step_dir, step, struct.id, "out")
            orca_input.build_input(
                xyz_path=xyz_path,
                template_path=template,
                output_path=inp_path,
                charge=ctx.charge,
                multiplicity=ctx.multiplicity,
                extra_blocks=self._extra_blocks(ctx),
                # The submit loop clamps the SLURM allocation to max_cores; the
                # .inp must declare the same PAL or ORCA over-spawns MPI ranks.
                max_pal=ctx.max_cores,
            )
            files.append((inp_path, out_path, struct.id))
        return StepInputs(files=tuple(files))

    def _extra_blocks(self, ctx: StepContext) -> str:
        """Subclass hook for engines that need extra ORCA blocks (e.g. MLIP ``%method``).

        The base ORCA engine has nothing extra to add; MLIP and PySCF
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
        return f"export OMP_NUM_THREADS=1\n{orca} {inp_path.name} > $OUTPUT_DIR/{out_path.name}"

    def _extra_header_fields(self, ctx: StepContext) -> tuple[tuple[str, object], ...]:
        """Record which ORCA binary ran in the runlog header."""
        return (("orca_executable", ctx.executables.get("orca", "orca")),)

    def _effective_operation(self, ctx: StepContext) -> str:
        """Operation to parse with: the explicit one if set, else inspect the template.

        An explicit ``operation`` always wins; otherwise the run type is inferred
        from the template's ORCA keywords (:mod:`chemrefine.engines.orca.inspect`).
        """
        if ctx.step_cfg.operation is not None:
            return ctx.step_cfg.operation
        return inspect.inspect_template(self._resolve_template(ctx)).operation

    # -- parse -------------------------------------------------------------

    def parse(self, inputs: StepInputs, ctx: StepContext) -> StepResults:
        """Parse each output once into :class:`Structure` instances.

        For ``.out``-based operations the file is read a single time and that
        text yields geometry/energy/forces + the run-status flags. Ensemble
        operations read their sidecar. ``parse`` is pure: NMS reads frequencies
        separately through :meth:`read_frequencies`.
        """
        operation = self._effective_operation(ctx)
        text_based = operation.lower().replace("+", "_") in output.TEXT_BASED_OPERATIONS
        prev_by_id = {s.id: s for s in ctx.prev_state.structures}

        # Parse every output first so each input's fan-out is known, then mint
        # child IDs through the shared lineage convention in `ids` instead of
        # re-implementing the ``{parent}-{i}`` format here.
        parsed_per_input: list[tuple[str, list[output.ParsedStructure]]] = []
        for _inp, out_path, sid in inputs.files:
            if text_based:
                parsed = output.parse_text(
                    out_path.read_text(encoding="utf-8", errors="replace"),
                    operation,
                    src=str(out_path),
                )
            else:
                parsed = output.parse_output(out_path, operation)
            parsed_per_input.append((sid, parsed))

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
                        gibbs_hartree=ps.gibbs_hartree,
                        enthalpy_hartree=ps.enthalpy_hartree,
                        energy_zpe_hartree=ps.energy_zpe_hartree,
                    )
                )
        return StepResults(structures=tuple(out_structures))

    # -- nms hooks (the engine-specific half of chemrefine.nms) ------------

    def nms_input_info(self, ctx: StepContext) -> NmsInputInfo:
        """Whether the step's template runs a TS search and computes frequencies.

        Both come from the ORCA template's keywords (:mod:`engines.orca.inspect`):
        ``OptTS`` ⇒ a TS target, ``…Freq`` ⇒ a frequency calc. The generic NMS
        coordinator uses these for the default target (F1) and the freq gate (B9).
        """
        run = inspect.inspect_template(self._resolve_template(ctx))
        return NmsInputInfo(is_transition_state=run.is_ts, computes_frequencies=run.has_freq)

    def read_frequencies(
        self, structure_id: str, step_dir: Path, ctx: StepContext
    ) -> FrequencyData:
        """Parse a structure's imaginary frequencies + normal-mode tensor from its ``.out``.

        The output is located via the shared per-id layout under ``step_dir``.
        ``imaginary`` is ``None`` when the output has **no** frequency table at all
        (distinct from ``{}`` = zero imaginary modes), so a run that never produced
        frequencies is never mistaken for a verified minimum; ``modes`` is ``None`` when
        the displacement tensor is absent / unparseable.
        """
        out_path = structure_artifact_path(step_dir, ctx.step_cfg.step, structure_id, "out")
        if not out_path.is_file():
            return FrequencyData(imaginary=None, modes=None)
        text = out_path.read_text(encoding="utf-8", errors="replace")
        if "VIBRATIONAL FREQUENCIES" not in text:
            logger.warning(
                "NMS: %s produced no frequency table — the template must request a "
                "frequency calc (e.g. opt+freq); the structure is left unresolved",
                structure_id,
            )
            return FrequencyData(imaginary=None, modes=None)
        try:
            parsed = output.parse_dft_from_text(text, src=str(out_path))
            n_atoms = len(parsed[0].symbols) if parsed else 0
            modes = (
                frequencies.parse_normal_modes_tensor_from_text(text, num_atoms=n_atoms)
                if n_atoms
                else None
            )
        except (OutputParseError, ValueError):
            modes = None
        return FrequencyData(
            imaginary=frequencies.parse_imaginary_frequencies_from_text(text), modes=modes
        )

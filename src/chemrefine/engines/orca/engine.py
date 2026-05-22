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

import logging
from pathlib import Path

from ase import Atoms

from chemrefine import slurm, throttle
from chemrefine.engines.base import register
from chemrefine.engines.orca import input as orca_input
from chemrefine.engines.orca import nms, output
from chemrefine.io import write_xyz
from chemrefine.state import (
    JobBatch,
    StepContext,
    StepInputs,
    StepResults,
    Structure,
)

logger = logging.getLogger(__name__)


@register("orca")
class OrcaEngine:
    """Standard ORCA DFT engine."""

    name = "orca"
    supports_nms = True

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
            inp_path = ctx.step_dir / f"step{ctx.step_cfg.step}_structure_{struct.id}.inp"
            out_path = ctx.step_dir / f"step{ctx.step_cfg.step}_structure_{struct.id}.out"
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

    def _resolve_template(self, ctx: StepContext) -> Path:
        """Pick the ORCA input template for this step."""
        name = ctx.step_cfg.template or f"step{ctx.step_cfg.step}.inp"
        template = ctx.template_dir / name
        if not template.is_file():
            raise FileNotFoundError(f"ORCA template not found: {template}")
        return template

    def _extra_blocks(self, ctx: StepContext) -> str:
        """Subclass hook for engines that need extra ORCA blocks (e.g. MLFF ``%method``).

        The base ORCA engine has nothing extra to add; MLFF and PySCF
        override this to inject their ``%method ProgExt …`` block.
        """
        return ""

    # -- submit / wait -----------------------------------------------------

    def submit(self, inputs: StepInputs, ctx: StepContext) -> JobBatch:
        """Generate SLURM scripts, submit under the PAL budget, block until done."""
        throttler = throttle.Throttler(max_cores=ctx.max_cores)
        header_path = ctx.template_dir / ctx.slurm_template
        if not header_path.is_file():
            raise FileNotFoundError(f"SLURM header template not found: {header_path}")

        step_label = ctx.step_cfg.dir_name()
        jobs: dict[Path, str] = {}
        for inp, out, sid in inputs.files:
            pal = min(orca_input.parse_pal(inp), ctx.max_cores)
            throttler.wait_for_room(pal, is_finished=slurm.is_finished)
            script_path = inp.with_suffix(".slurm")
            run_block = self._run_block(ctx, inp, out)
            slurm.build_script(
                job_name=inp.stem,
                pal=pal,
                template_path=header_path,
                script_path=script_path,
                input_path=inp,
                output_dir=out.parent,
                scratch_dir=ctx.scratch_dir,
                run_block=run_block,
                engine=ctx.step_cfg.engine,
                operation=ctx.step_cfg.operation,
                step=ctx.step_cfg.step,
                structure_id=sid,
                step_label=step_label,
                output_globs=("*.out", "*.xyz", "*.gbw", "*.hess"),
                extra_header_fields=(("orca_executable", ctx.orca_executable),),
            )
            job_id = slurm.submit(script_path)
            throttler.register(job_id, pal)
            jobs[inp] = job_id
            logger.info("submitted %s as job %s (pal=%d)", inp.name, job_id, pal)

        throttler.wait_all(is_finished=slurm.is_finished)
        return JobBatch(jobs=jobs)

    def _run_block(self, ctx: StepContext, inp_path: Path, out_path: Path) -> str:
        """Engine-specific bash that runs inside ``$SCRATCH_DIR``."""
        return (
            "export OMP_NUM_THREADS=1\n"
            f"{ctx.orca_executable} {inp_path.name} > $OUTPUT_DIR/{out_path.name}"
        )

    def wait(self, batch: JobBatch) -> None:
        """No-op: :meth:`submit` already blocked until every job finished.

        The parameter is kept for ``CalculationEngine`` Protocol parity.
        """
        return None

    # -- parse -------------------------------------------------------------

    def parse(self, inputs: StepInputs, ctx: StepContext) -> StepResults:
        """Read each output file and build :class:`Structure` instances."""
        operation = ctx.step_cfg.operation
        out_structures: list[Structure] = []
        for _inp, out_path, sid in inputs.files:
            parsed = output.parse_output(out_path, operation)
            for index, ps in enumerate(parsed):
                # If one input produces multiple structures (GOAT/PES), assign
                # hyphen-suffixed child IDs; single structures keep the parent ID.
                child_id = sid if len(parsed) == 1 else f"{sid}-{index}"
                atoms = Atoms(symbols=list(ps.symbols), positions=ps.positions)
                out_structures.append(
                    Structure(
                        id=child_id,
                        atoms=atoms,
                        energy_hartree=ps.energy_hartree,
                        forces_ev_per_a=ps.forces_ev_per_a,
                    )
                )
        return StepResults(structures=tuple(out_structures))

    # -- nms ---------------------------------------------------------------

    def normal_mode_sample(self, results: StepResults, ctx: StepContext) -> StepResults:
        """Delegate to :mod:`engines.orca.nms`."""
        return nms.normal_mode_sample(results, ctx)

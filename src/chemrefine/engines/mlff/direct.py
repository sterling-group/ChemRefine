"""Direct in-process MLFF engine — no ORCA, no SLURM.

This engine evaluates the MLFF calculator on each seed structure in
the same Python process and writes a per-structure ``.runlog`` plus a
tiny JSON output. Useful for fast pre-screening before the expensive
ORCA refinement stage.

The calculator is built once per engine instance and cached so the
GPU model load cost is paid only once per pipeline run.
"""

from __future__ import annotations

import logging

import numpy as np

from chemrefine import job_log
from chemrefine.constants import HARTREE_TO_EV
from chemrefine.engines.base import register
from chemrefine.engines.mlff.calculator import MlffCalculator
from chemrefine.state import (
    JobBatch,
    StepContext,
    StepInputs,
    StepResults,
    Structure,
)

logger = logging.getLogger(__name__)

# MLFF backends report energy in eV; ChemRefine stores Hartree internally.
_EV_TO_HARTREE: float = 1.0 / HARTREE_TO_EV


@register("mlff-direct")
class MlffDirectEngine:
    """In-process MLFF scorer satisfying :class:`CalculationEngine`."""

    name = "mlff-direct"
    supports_nms = False

    def __init__(self) -> None:
        self._calculator: MlffCalculator | None = None

    # -- lifecycle ---------------------------------------------------------

    def prepare(self, ctx: StepContext) -> StepInputs:
        """No input files needed — record paths so cache + manifest still work."""
        ctx.step_dir.mkdir(parents=True, exist_ok=True)
        files: list[tuple] = []
        for struct in ctx.prev_state.structures:
            stem = f"step{ctx.step_cfg.step}_structure_{struct.id}"
            placeholder_inp = ctx.step_dir / f"{stem}.json"
            placeholder_out = ctx.step_dir / f"{stem}.json.out"
            placeholder_inp.write_text(
                f'{{"id": "{struct.id}", "engine": "mlff-direct"}}\n',
                encoding="utf-8",
            )
            files.append((placeholder_inp, placeholder_out, struct.id))
        return StepInputs(files=tuple(files))

    def submit(self, inputs: StepInputs, ctx: StepContext) -> JobBatch:
        """Score every structure in-process; emit one ``.runlog`` per structure."""
        calc = self._get_calculator(ctx)
        step_label = ctx.step_cfg.dir_name()
        engine_name = ctx.step_cfg.engine
        for _inp, out, sid in inputs.files:
            log_path = ctx.step_dir / f"step{ctx.step_cfg.step}_structure_{sid}.runlog"
            job_log.python_header(
                engine=engine_name,
                operation=ctx.step_cfg.operation,
                step=ctx.step_cfg.step,
                structure_id=sid,
                step_label=step_label,
                step_dir=ctx.step_dir,
                log_path=log_path,
            )
            start = job_log.monotonic_seconds()
            exit_code = 0
            try:
                atoms = self._find_structure(ctx, sid).atoms.copy()
                energy_ev, _gradient = calc.single_point(atoms)
                energy_hartree = energy_ev * _EV_TO_HARTREE
                out.write_text(
                    f'{{"id": "{sid}", "energy_hartree": {energy_hartree}}}\n',
                    encoding="utf-8",
                )
            except Exception:
                exit_code = 1
                job_log.python_footer(
                    engine=engine_name,
                    step_label=step_label,
                    log_path=log_path,
                    exit_code=exit_code,
                    elapsed_seconds=job_log.monotonic_seconds() - start,
                )
                raise
            job_log.python_footer(
                engine=engine_name,
                step_label=step_label,
                log_path=log_path,
                exit_code=exit_code,
                elapsed_seconds=job_log.monotonic_seconds() - start,
                files_copied=1,
            )
        return JobBatch(
            jobs={inp: f"direct-{i}" for i, (inp, *_rest) in enumerate(inputs.files)}
        )

    def wait(self, batch: JobBatch) -> None:
        """No-op — :meth:`submit` ran inline."""
        return None

    def parse(self, inputs: StepInputs, ctx: StepContext) -> StepResults:
        """Re-read the per-structure JSON files into :class:`Structure` instances."""
        import json

        seeds = {s.id: s for s in ctx.prev_state.structures}
        results: list[Structure] = []
        for _inp, out, sid in inputs.files:
            data = json.loads(out.read_text(encoding="utf-8"))
            seed = seeds[sid]
            results.append(
                Structure(
                    id=sid,
                    atoms=seed.atoms.copy(),
                    energy_hartree=float(data["energy_hartree"]),
                    forces_eV_per_A=np.zeros((len(seed.atoms), 3)),
                )
            )
        return StepResults(structures=tuple(results))

    def normal_mode_sample(self, results: StepResults, ctx: StepContext) -> StepResults:
        """Not supported."""
        raise NotImplementedError("mlff-direct does not support NMS")

    # -- helpers -----------------------------------------------------------

    def _get_calculator(self, ctx: StepContext) -> MlffCalculator:
        """Build the calculator once and cache it on the engine instance."""
        if self._calculator is None:
            options = ctx.step_cfg.options or {}
            self._calculator = MlffCalculator(
                model_name=options.get("model_name") or options.get("model") or "",
                task_name=options.get("task_name") or options.get("task") or "mace_off",
                device=options.get("device", "cuda"),
                model_path=options.get("model_path"),
            )
        return self._calculator

    def _find_structure(self, ctx: StepContext, sid: str) -> Structure:
        """Locate a seed structure by its ID."""
        for struct in ctx.prev_state.structures:
            if struct.id == sid:
                return struct
        raise KeyError(f"unknown structure id {sid!r}")



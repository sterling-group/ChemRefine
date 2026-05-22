"""Direct in-process PySCF engine — no ORCA, no SLURM.

Mirrors :class:`~chemrefine.engines.mlff.direct.MlffDirectEngine`:
evaluates PySCF on each seed structure in the same Python process and
emits a per-structure ``.runlog``. The SCF + gradient call routes
through :mod:`chemrefine.engines.pyscf._runtime` so direct mode and
ExtOpt mode share one PySCF integration.

PySCF imports happen lazily inside the runtime helpers, so importing
this module is safe even when PySCF isn't installed — only an actual
``pyscf-direct`` step touches the heavy backend code.
"""

from __future__ import annotations

import json
import logging

import numpy as np
from ase import Atoms

from chemrefine import job_log
from chemrefine.engines.base import register
from chemrefine.engines.pyscf import _runtime
from chemrefine.engines.pyscf.options import PyscfOptions
from chemrefine.state import (
    JobBatch,
    StepContext,
    StepInputs,
    StepResults,
    Structure,
)

logger = logging.getLogger(__name__)


@register("pyscf-direct")
class PyscfDirectEngine:
    """In-process PySCF scorer satisfying :class:`CalculationEngine`."""

    name = "pyscf-direct"
    supports_nms = False

    # -- lifecycle ---------------------------------------------------------

    def prepare(self, ctx: StepContext) -> StepInputs:
        """Write a tiny per-structure settings JSON so cache + manifest work."""
        ctx.step_dir.mkdir(parents=True, exist_ok=True)
        files: list[tuple] = []
        for struct in ctx.prev_state.structures:
            inp = ctx.step_dir / f"step{ctx.step_cfg.step}_structure_{struct.id}.json"
            out = ctx.step_dir / f"step{ctx.step_cfg.step}_structure_{struct.id}.json.out"
            inp.write_text(
                json.dumps({"id": struct.id, "engine": "pyscf-direct"}) + "\n",
                encoding="utf-8",
            )
            files.append((inp, out, struct.id))
        return StepInputs(files=tuple(files))

    def submit(self, inputs: StepInputs, ctx: StepContext) -> JobBatch:
        """Score every structure in-process; emit one ``.runlog`` per structure."""
        options = PyscfOptions.from_raw(ctx.step_cfg.options)
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
                struct = self._find_structure(ctx, sid)
                energy_hartree, gradient = _score_one(
                    struct=struct,
                    options=options,
                    charge=ctx.charge,
                    multiplicity=ctx.multiplicity,
                )
                out.write_text(
                    json.dumps(
                        {
                            "id": sid,
                            "energy_hartree": energy_hartree,
                            "gradient_hartree_per_bohr": gradient,
                        }
                    )
                    + "\n",
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
        seeds = {s.id: s for s in ctx.prev_state.structures}
        results: list[Structure] = []
        for _inp, out, sid in inputs.files:
            data = json.loads(out.read_text(encoding="utf-8"))
            seed = seeds[sid]
            forces = _gradient_to_forces(data.get("gradient_hartree_per_bohr"))
            results.append(
                Structure(
                    id=sid,
                    atoms=seed.atoms.copy(),
                    energy_hartree=float(data["energy_hartree"]),
                    forces_ev_per_a=forces,
                )
            )
        return StepResults(structures=tuple(results))

    def normal_mode_sample(self, results: StepResults, ctx: StepContext) -> StepResults:
        """Not supported."""
        raise NotImplementedError("pyscf-direct does not support NMS")

    # -- helpers -----------------------------------------------------------

    def _find_structure(self, ctx: StepContext, sid: str) -> Structure:
        """Locate a seed structure by its ID."""
        for struct in ctx.prev_state.structures:
            if struct.id == sid:
                return struct
        raise KeyError(f"unknown structure id {sid!r}")


# ---------------------------------------------------------------------------
# Free helpers (also used by tests)
# ---------------------------------------------------------------------------


def _score_one(
    *,
    struct: Structure,
    options: PyscfOptions,
    charge: int,
    multiplicity: int,
) -> tuple[float, list[list[float]]]:
    """Run one SCF + gradient via :mod:`._runtime` and return atomic-unit results."""
    atoms: Atoms = struct.atoms
    mol = _runtime.build_mol(
        symbols=tuple(atoms.get_chemical_symbols()),
        positions_angstrom=np.asarray(atoms.get_positions(), dtype=float),
        charge=charge,
        multiplicity=multiplicity,
        basis=options.basis,
    )
    energy, gradient, _meta, _mf = _runtime.run_dft(
        mol,
        method=options.method,
        xc=options.xc,
        use_df=options.df,
        want_gpu=options.gpu,
        nthreads=1,
        dograd=True,
    )
    return energy, gradient


def _gradient_to_forces(
    gradient: list[list[float]] | None,
) -> np.ndarray | None:
    """Convert ``∂E/∂x`` Hartree/Bohr → ``F`` eV/Å for the v4 ``Structure`` field."""
    from chemrefine.constants import HARTREE_PER_BOHR_TO_EV_PER_A

    if not gradient:
        return None
    arr = np.asarray(gradient, dtype=float) * (-HARTREE_PER_BOHR_TO_EV_PER_A)
    return arr

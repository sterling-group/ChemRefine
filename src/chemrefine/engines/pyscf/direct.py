"""Template-driven PySCF engine — runs the user's PySCF script per structure.

The user provides a ``step{N}.py`` PySCF script template (mirror of an
ORCA ``step{N}.inp`` template) with ``string.Template`` placeholders
for the geometry; ChemRefine substitutes them and runs each rendered
script through the same SLURM-or-local submission machinery the ORCA
engine uses.

Lifecycle:

* :meth:`prepare` writes one ``.xyz`` per structure plus a rendered
  per-structure ``.py`` (via :mod:`engines.pyscf.input`).
* :meth:`submit` builds a SLURM script whose ``run_block`` is
  ``python <rendered.py>``, submits via :func:`chemrefine.slurm.submit`
  (which auto-falls-back to local bash execution when ``sbatch``
  isn't installed), and waits for every job via the shared throttler.
* :meth:`parse` reads the JSON the user's script wrote at the
  contracted ``$OUTPUT_JSON`` path and builds a :class:`Structure`.

The ExtOpt-driven PySCF engine (``engine: pyscf``) is unrelated and
unchanged — ORCA owns that flow.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from ase import Atoms

from chemrefine import slurm, throttle
from chemrefine.engines.base import register
from chemrefine.engines.pyscf import input as pyscf_input
from chemrefine.errors import OutputParseError
from chemrefine.ids import structure_artifact_path
from chemrefine.io import write_xyz
from chemrefine.quantities import HARTREE_PER_BOHR_TO_EV_PER_A
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
    """Template-driven direct PySCF engine satisfying :class:`CalculationEngine`."""

    name = "pyscf-direct"
    supports_nms = False

    # -- prepare -----------------------------------------------------------

    def prepare(self, ctx: StepContext) -> StepInputs:
        """Render one ``.py`` + ``.xyz`` per seed structure."""
        ctx.step_dir.mkdir(parents=True, exist_ok=True)
        template = self._resolve_template(ctx)
        step = ctx.step_cfg.step

        files: list[tuple[Path, Path, str]] = []
        for struct in ctx.prev_state.structures:
            xyz_paths = write_xyz(
                [struct.atoms],
                [struct.id],
                step_number=step,
                output_dir=ctx.step_dir,
            )
            xyz_path = xyz_paths[0]
            script_path = structure_artifact_path(ctx.step_dir, step, struct.id, "py")
            output_json = structure_artifact_path(ctx.step_dir, step, struct.id, "json")
            pyscf_input.build_input(
                xyz_path=xyz_path,
                template_path=template,
                output_path=script_path,
                output_json_path=output_json,
                charge=ctx.charge,
                multiplicity=ctx.multiplicity,
            )
            files.append((script_path, output_json, struct.id))
        return StepInputs(files=tuple(files))

    def _resolve_template(self, ctx: StepContext) -> Path:
        """Pick the PySCF Python-script template for this step."""
        name = ctx.step_cfg.template or f"step{ctx.step_cfg.step}.py"
        template = ctx.template_dir / name
        if not template.is_file():
            raise FileNotFoundError(f"PySCF template not found: {template}")
        return template

    # -- submit / wait -----------------------------------------------------

    def submit(self, inputs: StepInputs, ctx: StepContext) -> JobBatch:
        """Wrap each rendered ``.py`` in a SLURM script and submit it."""
        throttler = throttle.Throttler(max_cores=ctx.max_cores)
        header_path = ctx.template_dir / ctx.slurm_template
        if not header_path.is_file():
            raise FileNotFoundError(f"SLURM header template not found: {header_path}")

        options = ctx.step_cfg.options or {}
        pal = min(int(options.get("cores", 1)), ctx.max_cores)

        step_label = ctx.step_cfg.dir_name()
        jobs: dict[Path, str] = {}
        for inp, out, sid in inputs.files:
            throttler.wait_for_room(pal, is_finished=slurm.is_finished)
            script_path = inp.with_suffix(".slurm")
            slurm.build_script(
                job_name=inp.stem,
                pal=pal,
                template_path=header_path,
                script_path=script_path,
                input_path=inp,
                output_dir=out.parent,
                scratch_dir=ctx.scratch_dir,
                run_block=f"python {inp.name}",
                engine=ctx.step_cfg.engine,
                operation=ctx.step_cfg.operation,
                step=ctx.step_cfg.step,
                structure_id=sid,
                step_label=step_label,
                output_globs=("*.json", "*.xyz"),
            )
            job_id = slurm.submit(script_path)
            throttler.register(job_id, pal)
            jobs[inp] = job_id
            logger.info("submitted %s as job %s (pal=%d)", inp.name, job_id, pal)

        throttler.wait_all(is_finished=slurm.is_finished)
        return JobBatch(jobs=jobs)

    def wait(self, batch: JobBatch) -> None:
        """No-op: :meth:`submit` already blocked until every job finished."""
        return None

    # -- parse -------------------------------------------------------------

    def parse(self, inputs: StepInputs, ctx: StepContext) -> StepResults:
        """Read each output JSON into a :class:`Structure`."""
        prev_by_id = {s.id: s for s in ctx.prev_state.structures}
        out_structures: list[Structure] = []
        for _inp, out_path, sid in inputs.files:
            data = _load_output_json(out_path)
            seed = prev_by_id.get(sid)
            atoms = _atoms_from_output(data, fallback=seed.atoms if seed else None)
            forces = _forces_from_gradient(data.get("gradient_hartree_per_bohr"))
            out_structures.append(
                Structure(
                    id=sid,
                    atoms=atoms,
                    parent_id=seed.parent_id if seed is not None else None,
                    energy_hartree=float(data["energy_hartree"]),
                    forces_ev_per_a=forces,
                )
            )
        return StepResults(structures=tuple(out_structures))

    def normal_mode_sample(self, results: StepResults, ctx: StepContext) -> StepResults:
        """Not supported — the orchestrator gates this on ``supports_nms``."""
        raise NotImplementedError("pyscf-direct does not support normal-mode sampling")


# ---------------------------------------------------------------------------
# Output parsing helpers (top-level for testability)
# ---------------------------------------------------------------------------


def _load_output_json(out_path: Path) -> dict:
    """Read the user's script output JSON; raise :class:`OutputParseError` if malformed."""
    if not out_path.is_file():
        raise OutputParseError(f"PySCF output not found: {out_path}")
    try:
        data = json.loads(out_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise OutputParseError(f"PySCF output {out_path} is not valid JSON: {e}") from e
    if "energy_hartree" not in data:
        raise OutputParseError(
            f"PySCF output {out_path} missing required 'energy_hartree' field"
        )
    return data


def _atoms_from_output(data: dict, *, fallback: Atoms | None) -> Atoms:
    """Return ASE ``Atoms`` from the output JSON, falling back to the seed geometry.

    If the user's script wrote a ``positions_angstrom`` block (an
    optimised geometry, say), the returned ``Atoms`` carries those
    positions and the seed's symbols. Otherwise the seed atoms come
    back unchanged.
    """
    positions = data.get("positions_angstrom")
    if positions is None or fallback is None:
        if fallback is None:
            raise OutputParseError(
                "PySCF output lacks positions_angstrom and no seed atoms are available"
            )
        return fallback.copy()
    updated = fallback.copy()
    updated.set_positions(np.asarray(positions, dtype=float))
    return updated


def _forces_from_gradient(
    gradient: list[list[float]] | None,
) -> np.ndarray | None:
    """Convert PySCF gradient (Hartree/Bohr) to ASE forces (eV/Å)."""
    if not gradient:
        return None
    return np.asarray(gradient, dtype=float) * (-HARTREE_PER_BOHR_TO_EV_PER_A)

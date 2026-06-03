"""High-level pipeline orchestrator — iterate steps, thread immutable state.

Two functions are enough:

* :func:`bootstrap` turns ``config.input`` (an ``.xyz`` file, a ``.csv``
  of SMILES, a directory of ``.xyz`` files, or — for backwards
  compatibility — ``templates/step1.xyz``) into the initial
  :class:`~chemrefine.state.PipelineState`.
* :func:`run` iterates :attr:`Config.steps` in order, threading state
  through :func:`chemrefine.step.run_step` and stopping early if a step
  yields no survivors.

Side-effect import of :mod:`chemrefine.engines` at module top
populates the :data:`~chemrefine.engines.base.ENGINES` registry so the
orchestrator never imports a concrete engine directly.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ase.io import read as ase_read

from chemrefine import io
from chemrefine.config import Config, StepConfig
from chemrefine.errors import ConfigError
from chemrefine.quantities import DEFAULT_TEMPERATURE_K
from chemrefine.state import PipelineState, Structure

# Importing :mod:`chemrefine.step` pulls in :mod:`chemrefine.engines.base`,
# which runs :mod:`chemrefine.engines`'s ``__init__`` and self-registers every
# bundled engine. No explicit ``import chemrefine.engines`` needed.
from chemrefine.step import StepOutcome, run_step

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def bootstrap(config: Config) -> PipelineState:
    """Construct the initial :class:`PipelineState` from ``config.input``.

    Resolution order:

    1. ``config.input`` is set → infer format from suffix / type:
       - directory: read every ``*.xyz`` inside, in natural sort order.
       - ``.csv``: convert SMILES to per-row XYZ files under ``output_dir/_seed``.
       - ``.xyz``: read one structure (assigned ID ``"0"``).
    2. ``config.input`` is None → fall back to
       ``templates/step1.xyz`` (the historic default).

    Raises :class:`ConfigError` if no seed source can be located.
    """
    path = config.input
    if path is None:
        default = config.template_dir / "step1.xyz"
        if not default.is_file():
            raise ConfigError(
                f"no 'input' declared and default {default} does not exist"
            )
        return _seed_from_xyz(default)

    if path.is_dir():
        return _seed_from_directory(path)
    if path.suffix.lower() == ".csv":
        return _seed_from_smiles_csv(path, config.output_dir / "_seed")
    if path.suffix.lower() == ".xyz":
        return _seed_from_xyz(path)
    raise ConfigError(f"unsupported input format: {path}")


def _seed_from_xyz(path: Path) -> PipelineState:
    """Seed the pipeline from a single XYZ file (one structure, ID ``"0"``)."""
    atoms = ase_read(str(path), format="xyz")
    return PipelineState(structures=(Structure(id="0", atoms=atoms),))


def _seed_from_directory(directory: Path) -> PipelineState:
    """Seed the pipeline from every ``*.xyz`` under ``directory``, sorted naturally."""
    xyz_files = io.gather_output_files(directory, "*.xyz")
    if not xyz_files:
        raise ConfigError(f"no .xyz files found under {directory}")
    structures = tuple(
        Structure(id=str(i), atoms=ase_read(str(f), format="xyz"))
        for i, f in enumerate(xyz_files)
    )
    return PipelineState(structures=structures)


def _seed_from_smiles_csv(csv_path: Path, out_dir: Path) -> PipelineState:
    """Seed the pipeline by converting each SMILES row in a CSV to a 3D XYZ structure."""
    xyz_files = io.smiles_to_xyz(csv_path, out_dir)
    if not xyz_files:
        raise ConfigError(f"no SMILES in {csv_path} converted to 3D structures")
    structures = tuple(
        Structure(id=str(i), atoms=ase_read(str(f), format="xyz"))
        for i, f in enumerate(xyz_files)
    )
    return PipelineState(structures=structures)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _write_step_csv(config: Config, step_cfg: StepConfig, state: PipelineState) -> None:
    """Append this step's survivor energies to the cumulative ``steps.csv``.

    Uses the step's own ``sample.temperature_k`` when a sample filter is set
    so the report's Boltzmann weights match the temperature the step filtered
    at; otherwise the standard reference temperature. A step with no survivors
    has nothing to summarise, so no row is written.
    """
    if not state.structures:
        return
    temperature_k = (
        step_cfg.sample.temperature_k
        if step_cfg.sample is not None
        else DEFAULT_TEMPERATURE_K
    )
    io.save_step_csv(
        energies_hartree=[s.energy_hartree for s in state.structures],
        structure_ids=[s.id for s in state.structures],
        step_number=step_cfg.step,
        output_dir=config.output_dir,
        temperature_k=temperature_k,
    )


def run(
    config: Config, *, use_cache: bool = True, rerun_step: int | None = None
) -> list[StepOutcome]:
    """Run every step in order; return the per-step outcomes.

    If a step produces no survivors the pipeline stops early — there is
    nothing to feed the next step. When ``rerun_step`` is set, that step
    resubmits only its failed jobs (recorded in its ``failed_jobs.json``)
    instead of taking the normal cache hit.
    """
    state = bootstrap(config)
    logger.info(
        "bootstrapped pipeline with %d seed structure(s)", len(state.structures)
    )

    outcomes: list[StepOutcome] = []
    for step_cfg in config.steps:
        logger.info(
            "=== step %d (%s, engine=%s) ===",
            step_cfg.step,
            step_cfg.dir_name(),
            step_cfg.engine,
        )
        outcome = run_step(
            config, step_cfg, state, use_cache=use_cache,
            rerun=(rerun_step == step_cfg.step),
        )
        outcomes.append(outcome)
        _write_step_csv(config, step_cfg, outcome.state)
        state = outcome.state
        if not state:
            logger.warning(
                "step %d produced no survivors; stopping pipeline early",
                step_cfg.step,
            )
            break
    logger.info("pipeline finished after %d step(s)", len(outcomes))
    return outcomes

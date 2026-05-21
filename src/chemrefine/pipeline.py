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

from chemrefine import io
from chemrefine.config import Config
from chemrefine.errors import ConfigError
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
    atoms = io.read_xyz(path)
    return PipelineState(structures=(Structure(id="0", atoms=atoms),))


def _seed_from_directory(directory: Path) -> PipelineState:
    """Seed the pipeline from every ``*.xyz`` under ``directory``, sorted naturally."""
    xyz_files = io.gather_output_files(directory, "*.xyz")
    if not xyz_files:
        raise ConfigError(f"no .xyz files found under {directory}")
    structures = tuple(
        Structure(id=str(i), atoms=io.read_xyz(f)) for i, f in enumerate(xyz_files)
    )
    return PipelineState(structures=structures)


def _seed_from_smiles_csv(csv_path: Path, out_dir: Path) -> PipelineState:
    """Seed the pipeline by converting each SMILES row in a CSV to a 3D XYZ structure."""
    xyz_files = io.smiles_to_xyz(csv_path, out_dir)
    if not xyz_files:
        raise ConfigError(f"no SMILES in {csv_path} converted to 3D structures")
    structures = tuple(
        Structure(id=str(i), atoms=io.read_xyz(f)) for i, f in enumerate(xyz_files)
    )
    return PipelineState(structures=structures)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(config: Config, *, use_cache: bool = True) -> list[StepOutcome]:
    """Run every step in order; return the per-step outcomes.

    If a step produces no survivors the pipeline stops early — there is
    nothing to feed the next step.
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
        outcome = run_step(config, step_cfg, state, use_cache=use_cache)
        outcomes.append(outcome)
        state = outcome.state
        if not state:
            logger.warning(
                "step %d produced no survivors; stopping pipeline early",
                step_cfg.step,
            )
            break
    logger.info("pipeline finished after %d step(s)", len(outcomes))
    return outcomes

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
populates the :data:`~chemrefine.engines.api.ENGINES` registry so the
orchestrator never imports a concrete engine directly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

from ase import Atoms
from ase.io import read as ase_read

from chemrefine import io
from chemrefine.config import Config, StepConfig
from chemrefine.engines import preflight_backends
from chemrefine.errors import ConfigError
from chemrefine.quantities import DEFAULT_TEMPERATURE_K
from chemrefine.state import PipelineState, Structure

# Importing :mod:`chemrefine.step` pulls in :mod:`chemrefine.engines.api`,
# which runs :mod:`chemrefine.engines`'s ``__init__`` and self-registers every
# bundled engine. No explicit ``import chemrefine.engines`` needed.
from chemrefine.step import (
    StepOutcome,
    halt_if_pending,
    rebuild_cache_step,
    run_step,
)

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
       - ``.xyz``: read every frame (IDs ``"0"``, ``"1"``, … in file order).
    2. ``config.input`` is None → fall back to
       ``templates/step1.xyz`` (the historic default).

    Raises :class:`ConfigError` if no seed source can be located.
    """
    path = config.input
    if path is None:
        default = config.template_dir / "step1.xyz"
        if not default.is_file():
            raise ConfigError(f"no 'input' declared and default {default} does not exist")
        return _seed_from_xyz(default)

    if path.is_dir():
        return _seed_from_directory(path)
    if path.suffix.lower() == ".csv":
        return _seed_from_smiles_csv(path, config.output_dir / "_seed")
    if path.suffix.lower() == ".xyz":
        return _seed_from_xyz(path)
    raise ConfigError(f"unsupported input format: {path}")


def _seed_from_xyz(path: Path) -> PipelineState:
    """Seed the pipeline from an XYZ file — one structure **per frame**, in file order.

    ``index=":"`` reads every frame; ASE's default would silently keep only
    the *last* one, losing the rest of a multi-conformer seed file. A
    single-frame file still yields exactly one structure with ID ``"0"``.
    """
    frames = ase_read(str(path), index=":", format="xyz")
    return PipelineState(
        structures=tuple(Structure(id=str(i), atoms=atoms) for i, atoms in enumerate(frames))
    )


def _seed_from_directory(directory: Path) -> PipelineState:
    """Seed the pipeline from every ``*.xyz`` under ``directory``, sorted naturally.

    Every **frame** of every file becomes a structure (``index=":"`` —
    like :func:`_seed_from_xyz`, ASE's default would silently keep only
    the last frame of a multi-conformer file). IDs number continuously
    across files in natural-sort order.
    """
    xyz_files = io.gather_output_files(directory, "*.xyz")
    if not xyz_files:
        raise ConfigError(f"no .xyz files found under {directory}")
    frames = [atoms for f in xyz_files for atoms in ase_read(str(f), index=":", format="xyz")]
    return PipelineState(
        structures=tuple(Structure(id=str(i), atoms=atoms) for i, atoms in enumerate(frames))
    )


def _seed_from_smiles_csv(csv_path: Path, out_dir: Path) -> PipelineState:
    """Seed the pipeline by converting each SMILES row in a CSV to a 3D XYZ structure."""
    xyz_files = io.smiles_to_xyz(csv_path, out_dir)
    if not xyz_files:
        raise ConfigError(f"no SMILES in {csv_path} converted to 3D structures")
    structures = tuple(
        # Each file holds exactly one embedded conformer, so the read is a
        # single Atoms — ase types it as a frame-list union.
        Structure(id=str(i), atoms=cast(Atoms, ase_read(str(f), format="xyz")))
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
        step_cfg.sample.temperature_k if step_cfg.sample is not None else DEFAULT_TEMPERATURE_K
    )
    io.save_step_csv(
        # filtering.apply drops None-energy structures, so survivors all carry one.
        energies_hartree=cast("list[float]", [s.energy_hartree for s in state.structures]),
        structure_ids=[s.id for s in state.structures],
        step_number=step_cfg.step,
        output_dir=config.output_dir,
        temperature_k=temperature_k,
    )


def run(
    config: Config,
    *,
    use_cache: bool = True,
    rebuild_step: int | None = None,
    resubmit_step: int | None = None,
) -> list[StepOutcome]:
    """Run every step in order; return the per-step outcomes.

    If a step produces no survivors the pipeline stops early — there is
    nothing to feed the next step. ``resume`` (``use_cache=True``) re-attempts
    the failed jobs of any pending ``on_failure: stop`` step; ``resubmit_step``
    (set by ``rerun-errors``) scopes that re-attempt to one step. When
    ``rebuild_step`` is set, that one step is rebuilt **from outputs already on
    disk** (parse only, no submission) instead of executing.
    """
    logger.info(
        "config: max_cores=%d, max_gpus=%s, output_dir=%s",
        config.max_cores,
        config.max_gpus if config.max_gpus is not None else "auto",
        config.output_dir,
    )
    # Fail fast: every step's backend env must be resolvable before ANY job submits.
    preflight_backends(config.steps)
    state = bootstrap(config)
    logger.info("bootstrapped pipeline with %d seed structure(s)", len(state.structures))

    outcomes: list[StepOutcome] = []
    for step_cfg in config.steps:
        logger.info(
            "=== step %d (%s, engine=%s) ===",
            step_cfg.step,
            step_cfg.dir_name(),
            step_cfg.engine,
        )
        if rebuild_step == step_cfg.step:
            outcome = rebuild_cache_step(config, step_cfg, state)
        else:
            outcome = run_step(
                config, step_cfg, state, use_cache=use_cache, resubmit_step=resubmit_step
            )
            # Single halt point: an on_failure=stop step that still has pending
            # failures stops the run here (after its successes were cached).
            halt_if_pending(config, step_cfg, resubmit_step)
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

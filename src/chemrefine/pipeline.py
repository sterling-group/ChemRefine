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
from collections.abc import Iterable
from pathlib import Path

from ase import Atoms

from chemrefine import filtering, io, slurm
from chemrefine.config import Config, StepConfig
from chemrefine.engines import preflight_backends
from chemrefine.errors import ConfigError
from chemrefine.quantities import DEFAULT_TEMPERATURE_K
from chemrefine.state import PipelineState, Structure

# Importing :mod:`chemrefine.step` pulls in :mod:`chemrefine.engines.api`,
# which runs :mod:`chemrefine.engines`'s ``__init__`` and self-registers every
# bundled engine. No explicit ``import chemrefine.engines`` needed.
from chemrefine.step import (
    RunPlan,
    StepMode,
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


def _state_from_frames(frames: Iterable[Atoms]) -> PipelineState:
    """Number a sequence of frames into a :class:`PipelineState` — IDs ``"0"``, ``"1"``, ….

    The shared tail of every seeder: continuous IDs in iteration order, so a
    directory's frames number straight on from the previous file's.
    """
    return PipelineState(
        structures=tuple(Structure(id=str(i), atoms=atoms) for i, atoms in enumerate(frames))
    )


def _seed_from_xyz(path: Path) -> PipelineState:
    """Seed the pipeline from an XYZ file — one structure **per frame**, in file order.

    :func:`chemrefine.io.read_xyz_frames` reads every frame; ASE's default
    would silently keep only the *last* one, losing the rest of a
    multi-conformer seed file. A single-frame file still yields exactly one
    structure with ID ``"0"``.
    """
    return _state_from_frames(io.read_xyz_frames(path))


def _seed_from_directory(directory: Path) -> PipelineState:
    """Seed the pipeline from every ``*.xyz`` under ``directory``, sorted naturally.

    Every **frame** of every file becomes a structure (via
    :func:`chemrefine.io.read_xyz_frames` — like :func:`_seed_from_xyz`,
    ASE's default would silently keep only the last frame of a multi-conformer
    file). IDs number continuously across files in natural-sort order.
    """
    xyz_files = io.gather_output_files(directory, "*.xyz")
    if not xyz_files:
        raise ConfigError(f"no .xyz files found under {directory}")
    return _state_from_frames(atoms for f in xyz_files for atoms in io.read_xyz_frames(f))


def _seed_from_smiles_csv(csv_path: Path, out_dir: Path) -> PipelineState:
    """Seed the pipeline by converting each SMILES row in a CSV to a 3D XYZ structure."""
    xyz_files = io.smiles_to_xyz(csv_path, out_dir)
    if not xyz_files:
        raise ConfigError(f"no SMILES in {csv_path} converted to 3D structures")
    # Each converted file holds exactly one embedded conformer — take its sole frame.
    return _state_from_frames(io.read_xyz_frames(f)[0] for f in xyz_files)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _write_step_csv(config: Config, step_cfg: StepConfig, state: PipelineState) -> None:
    """Append this step's survivor energies to the cumulative ``steps.csv``.

    The report summarises **the same energy the step filtered on**: a step sampling
    on ``gibbs`` gets Gibbs energies and Gibbs-derived Boltzmann weights, at that
    step's own ``sample.temperature_k``. Reporting electronic energies for a
    thermochemistry-filtered step produced a table that silently contradicted the
    survivor set it was describing. With no sample filter, the electronic energy
    at the standard reference temperature.

    A step with no survivors has nothing to summarise, so no row is written.
    """
    if not state.structures:
        return
    sample = step_cfg.sample
    temperature_k = sample.temperature_k if sample is not None else DEFAULT_TEMPERATURE_K
    energy_type = sample.energy_type if sample is not None else "electronic"
    energy_attr = filtering.ENERGY_ATTR[energy_type]
    io.save_step_csv(
        energies_hartree=[getattr(s, energy_attr) for s in state.structures],
        structure_ids=[s.id for s in state.structures],
        step_number=step_cfg.step,
        output_dir=config.output_dir,
        temperature_k=temperature_k,
        energy_type=energy_type,
    )


def run(config: Config, plan: RunPlan | None = None) -> list[StepOutcome]:
    """Run every step in order; return the per-step outcomes.

    ``plan`` says which :class:`~chemrefine.recovery.StepMode` each step runs in —
    already resolved by :mod:`chemrefine.recovery` from the requested action, so
    nothing here has to re-derive it. The default plan is a plain cache-honouring
    pass, which is what a caller with no recovery intent wants.

    If a step produces no survivors the pipeline stops early — there is nothing to
    feed the next step.
    """
    plan = plan if plan is not None else RunPlan()
    logger.info(
        "config: max_cores=%d, max_gpus=%s, output_dir=%s",
        config.max_cores,
        config.max_gpus if config.max_gpus is not None else "auto",
        config.output_dir,
    )
    # Fail fast: every step's backend env must be resolvable before ANY job submits,
    # and `dispatch: slurm` must actually have sbatch available.
    preflight_backends(config.steps)
    slurm.dispatch_locally(config.dispatch)
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
        mode = plan.for_step(step_cfg.step)
        if mode is StepMode.REBUILD:
            outcome = rebuild_cache_step(config, step_cfg, state)
        else:
            outcome = run_step(config, step_cfg, state, mode=mode)
        outcomes.append(outcome)
        # Summarise before halting, so a run that stops still reports the work it
        # actually completed — otherwise the halted step's cached successes are
        # missing from steps.csv.
        _write_step_csv(config, step_cfg, outcome.state)
        state = outcome.state
        # Single halt point, reached by every mode: an on_failure=stop step with
        # pending failures stops the run here, after its successes are cached and
        # summarised. `rebuild` is included on purpose — re-parsing from disk does
        # not make a failed structure succeed, and continuing would run the next
        # step against a partial survivor set the user asked to stop on.
        halt_if_pending(config, step_cfg, mode)
        if not state:
            logger.warning(
                "step %d produced no survivors; stopping pipeline early",
                step_cfg.step,
            )
            break
    logger.info("pipeline finished after %d step(s)", len(outcomes))
    return outcomes

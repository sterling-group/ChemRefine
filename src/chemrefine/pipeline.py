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

import contextlib
import json
import logging
import os
import socket
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path

from ase import Atoms

from chemrefine import filtering, io, slurm
from chemrefine.config import Config, StepConfig
from chemrefine.engines import preflight_backends
from chemrefine.errors import ConfigError, RunLockError
from chemrefine.quantities import DEFAULT_TEMPERATURE_K
from chemrefine.state import PipelineState, Structure

# Importing :mod:`chemrefine.step` pulls in :mod:`chemrefine.engines.api`,
# which runs :mod:`chemrefine.engines`'s ``__init__`` and self-registers every
# bundled engine. No explicit ``import chemrefine.engines`` needed.
# No ``StepMode`` import: with the three questions about it answered by the enum's own
# predicates, this module composes modes without naming a single member.
from chemrefine.step import (
    RunPlan,
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
       ``templates/step1.xyz`` (the conventional default).

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
# Run lock — one driver per output tree
# ---------------------------------------------------------------------------

RUN_LOCK_NAME = ".chemrefine.lock"
"""The advisory lock file :func:`run_lock` holds at the root of ``output_dir``."""


def _lock_holder(lock: Path) -> tuple[str, int, str] | None:
    """The ``(host, pid, started)`` recorded in ``lock``, or ``None`` if unreadable.

    ``None`` covers both a lock vacated between the failed claim and this read, and one
    whose writer died between creating the file and writing it — the caller treats the
    two alike, because neither names a holder whose liveness can be checked.
    """
    try:
        record = json.loads(lock.read_text(encoding="utf-8"))
        return str(record["host"]), int(record["pid"]), str(record["started"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _pid_alive(pid: int) -> bool:
    """Whether ``pid`` names a live process on *this* host.

    ``PermissionError`` means the process exists and belongs to someone else — alive.
    Only :class:`ProcessLookupError` proves it gone.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _reclaim_stale(lock: Path, holder: tuple[str, int, str]) -> bool:
    """Atomically take a dead holder's lock off its path; ``True`` if this process did.

    Reclaiming by ``unlink()`` was a race: two drivers that both probed the same dead pid
    could interleave — A unlinks, A creates, B unlinks *A's fresh lock*, B creates — and
    both acquire, which is the one state the lock exists to prevent. ``os.replace`` moves
    the inode to a per-pid claim name instead: exactly one process wins the rename (atomic
    on POSIX, and on NFS where this tree lives), and the loser sees
    :class:`FileNotFoundError` and re-reads whatever now holds the path.

    The claim is then verified against ``holder`` — the record that justified it. Between
    the caller's read and the rename, another process can have completed its *own* reclaim
    **and** created a fresh live lock at the same path, and the rename cannot tell those
    apart. A claim whose record no longer matches is therefore put back with ``os.link``,
    the atomic fail-if-exists primitive: if a third process created a lock meanwhile, the
    link fails and that lock stands, which leaves the swept-up holder no worse off than
    before this function ran. Either way the reclaim is reported as not-ours, and the
    caller's next pass answers to whatever the path now holds.
    """
    claim = lock.with_name(f"{RUN_LOCK_NAME}.reclaim.{os.getpid()}")
    try:
        lock.replace(claim)
    except FileNotFoundError:
        return False
    if _lock_holder(claim) != holder:
        with contextlib.suppress(OSError):
            os.link(claim, lock)
        claim.unlink(missing_ok=True)
        return False
    claim.unlink(missing_ok=True)
    return True


@contextlib.contextmanager
def run_lock(output_dir: Path) -> Iterator[None]:
    """Hold ``output_dir`` for one driver; raise :class:`RunLockError` if another has it.

    **Why a lock at all.** The resume machinery cannot tell a live concurrent driver from
    a dead one: :func:`chemrefine.step._partial_step_outcome` treats a manifest whose
    fingerprint matches as proof it is safe to continue, and a *running* driver leaves
    exactly that state on disk. A second driver would then parse outputs the first one's
    jobs are still writing, archive their directories out from under those jobs, and
    resubmit duplicates — silently, since nothing in that sequence is an error.

    **Why a pidfile and not ``flock``.** The output tree lives on a shared filesystem on
    HPC, where ``flock`` semantics are the least reliable part of NFS; an ``O_EXCL``
    create is atomic everywhere. The cost is that a lock can outlive a driver killed with
    SIGKILL, so a holder on *this* host is probed with ``os.kill(pid, 0)`` and reclaimed
    when dead. A holder on another host cannot be probed from here — that lock is treated
    as live, and the error says to delete it once its run is known dead.

    **Reclaim is an atomic rename, and release is ownership-checked.** Deleting a stale
    lock with ``unlink()`` let two drivers that both probed the same dead pid interleave —
    one deleted the other's freshly created lock — and both acquire; :func:`_reclaim_stale`
    renames the stale lock to a per-pid claim instead, so exactly one process wins, and
    verifies the claim against the record that justified it in case a fresh lock was swept
    up in the window. The release mirrors it: unlinking whatever sits at the lock path
    would let a driver whose lock was deleted out from under it (the error message's own
    advice, followed against a run that was in fact alive) remove the *current* holder's
    lock on exit, reopening the tree to a third driver — so the ``finally`` deletes the
    file only while it still names this process.

    **Reentrant by pid**: :func:`chemrefine.recovery.execute` takes the lock around a
    whole action — ``run``'s cache invalidation mutates the tree *before*
    :func:`run` starts — and :func:`run` takes it again for callers that drive the
    pipeline directly. The inner acquisition sees its own pid in the record and yields
    without ownership, so the one release still happens at the outermost exit.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = output_dir / RUN_LOCK_NAME
    reclaimed = False
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            holder = _lock_holder(lock)
            if holder is not None and holder[0] == socket.gethostname():
                host, pid, started = holder
                if pid == os.getpid():
                    # Ours, taken further out — see "reentrant by pid" above.
                    yield
                    return
                if not reclaimed and not _pid_alive(pid):
                    # One attempt per process, won or lost: a lost claim means another
                    # driver got there first, and the next pass answers to its lock.
                    reclaimed = True
                    if _reclaim_stale(lock, holder):
                        logger.warning(
                            "reclaiming stale run lock %s (pid %d on %s, started %s, now dead)",
                            lock,
                            pid,
                            host,
                            started,
                        )
                    continue
            # `from None`: the FileExistsError is this branch's condition, not a cause —
            # everything it could say is already in the message.
            raise RunLockError(
                f"another ChemRefine run holds this output tree: {lock} "
                + (
                    f"names pid {holder[1]} on {holder[0]}, started {holder[2]}"
                    if holder is not None
                    else "exists but is unreadable"
                )
                + ". Two drivers on one tree archive and resubmit each other's work, so "
                "this run stops here. Wait for that run to finish — or, if it is known "
                "dead (e.g. killed on another node), delete the lock file and retry."
            ) from None
    # The record in `_lock_holder`'s field order, so the release below can compare whole.
    me = (socket.gethostname(), os.getpid(), datetime.now(UTC).isoformat(timespec="seconds"))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"pid": me[1], "host": me[0], "started": me[2]}, fh)
        yield
    finally:
        # On success and on failure alike: a raise must not leave the tree locked, and
        # back-to-back runs in one process (tests, notebooks) must each acquire cleanly.
        # Only while the file is still ours, though — see "release is ownership-checked".
        if _lock_holder(lock) == me:
            lock.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _write_step_csv(config: Config, step_cfg: StepConfig, state: PipelineState) -> None:
    """Append this step's survivor energies to the cumulative ``steps.csv``.

    The report summarises **the same energy the step filtered on**: a step sampling
    on ``gibbs`` gets Gibbs energies and Gibbs-derived Boltzmann weights, at that
    step's own ``sample.temperature_k``. Reporting electronic energies for a
    thermochemistry-filtered step gives a table that silently contradicts the survivor set
    it describes. With no sample filter, the electronic energy at the standard reference
    temperature.

    A step with no survivors has nothing to summarise, so no row is written — that
    decision lives in :func:`~chemrefine.io.save_step_csv` along with the step-1
    truncation, rather than being half here and half there.
    """
    sample = step_cfg.sample
    temperature_k = sample.temperature_k if sample is not None else DEFAULT_TEMPERATURE_K
    ranking = filtering.ranking_energy(sample)
    io.save_step_csv(
        energies_hartree=[getattr(s, ranking.attr) for s in state.structures],
        structure_ids=[s.id for s in state.structures],
        step_number=step_cfg.step,
        output_dir=config.output_dir,
        temperature_k=temperature_k,
        energy_type=ranking.energy_type,
    )


def run(config: Config, plan: RunPlan | None = None) -> list[StepOutcome]:
    """Run every step in order; return the per-step outcomes.

    ``plan`` says which :class:`~chemrefine.recovery.StepMode` each step runs in —
    already resolved by :mod:`chemrefine.recovery` from the requested action, so
    nothing here has to re-derive it. The default plan is a plain cache-honouring
    pass, which is what a caller with no recovery intent wants.

    If a step produces no survivors the pipeline stops early — there is nothing to
    feed the next step.

    The whole run holds the output tree's :func:`run_lock`: a second driver pointed at
    the same ``output_dir`` raises :class:`~chemrefine.errors.RunLockError` instead of
    archiving and resubmitting this one's in-flight work.
    """
    plan = plan if plan is not None else RunPlan()
    with run_lock(config.output_dir):
        logger.info(
            "config: max_cores=%d, max_gpus=%s, output_dir=%s",
            config.max_cores,
            config.max_gpus if config.max_gpus is not None else "auto",
            config.output_dir,
        )
        # Fail fast: every step's backend env must be resolvable before ANY job submits,
        # and `dispatch: slurm` must actually have sbatch available.
        #
        # The steps checked are the ones that *can* submit, which is `StepMode.may_submit` and
        # nothing else. A guard for something that will not happen is just a wall: it would
        # make `chemrefine rebuild-cache` refuse to run wherever the backend is not installed,
        # which is exactly where you want to rebuild — a login node, or any machine holding
        # the output tree but not the MLIP/PySCF stack that produced it.
        #
        # Asking `may_submit` rather than excluding `REBUILD` by name is what makes that hold
        # for the whole command. `rebuild-cache N` puts N in `REBUILD` and every *other* step
        # in `CACHE_ONLY`, which equally cannot submit — so excluding only the named step
        # leaves the wall standing on all the others, and a two-step MLIP config still cannot
        # be rebuilt off-cluster. Nothing is weakened by the wider exemption: a step that
        # cannot submit reaches `ChemRefineError` from `run_step` if its cache is unusable,
        # never the engine.
        preflight_backends([cfg for cfg in config.steps if plan.for_step(cfg.step).may_submit()])
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
            outcome = (
                run_step(config, step_cfg, state, mode=mode)
                if mode.runs_through_run_step()
                else rebuild_cache_step(config, step_cfg, state)
            )
            outcomes.append(outcome)
            # Summarise before halting, so a run that stops still reports the work it
            # actually completed — otherwise the halted step's cached successes are
            # missing from steps.csv.
            _write_step_csv(config, step_cfg, outcome.state)
            state = outcome.state
            # Single halt point, reached by every mode: an on_failure=stop step with
            # pending failures stops the run here, after its successes are cached and
            # summarised. Which modes may halt is `StepMode.can_halt`.
            halt_if_pending(config, step_cfg, mode)
            if not state:
                logger.warning(
                    "step %d produced no survivors; stopping pipeline early",
                    step_cfg.step,
                )
                break
            # A scoped plan can end before the last step — see `RunPlan.stop_after`.
            if not plan.covers(step_cfg.step):
                logger.info("step %d is the last this action covers; stopping here", step_cfg.step)
                break
        logger.info("pipeline finished after %d step(s)", len(outcomes))
        return outcomes

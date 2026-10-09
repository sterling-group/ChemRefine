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
import errno
import json
import logging
import os
import signal
import socket
import threading
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import NoReturn

import numpy as np
from ase import Atoms

from chemrefine import filtering, ids, io, slurm
from chemrefine.config import Config, StepConfig
from chemrefine.engines import preflight_backends, preflight_steps
from chemrefine.errors import ConfigError, NoUsableCacheError, OutputTreeError, RunLockError
from chemrefine.quantities import DEFAULT_TEMPERATURE_K
from chemrefine.state import PipelineState, Structure

# Importing :mod:`chemrefine.step` pulls in :mod:`chemrefine.engines.api`,
# which runs :mod:`chemrefine.engines`'s ``__init__`` and self-registers every
# bundled engine. No explicit ``import chemrefine.engines`` needed.
# No ``StepMode`` import: with every question about it answered by the enum's own
# predicates, this module composes modes without naming a single member.
from chemrefine.step import (
    RunPlan,
    StepOutcome,
    halt_if_pending,
    preflight_batches,
    rebuild_cache_step,
    run_step,
)
from chemrefine.validate import undeclared_options

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

    Raises :class:`ConfigError` if no seed source can be located — or can be located but
    not used. The readers raise their own exceptions for a file that is missing, malformed
    or not text (ASE's ``XYZError`` and ``FileNotFoundError``, a ``ValueError`` for a
    coordinate that is not a number, a ``UnicodeDecodeError``, pandas' own for a CSV), and
    left to escape they reached the user as a traceback with exit 1 — outside the exit-code
    contract the template and header checks honour — while ``chemrefine validate`` said OK.
    The net is the one :func:`chemrefine.agent_tools.build_structures` puts around the same
    read. A seed that parses to no structure at all is refused for the same reason: a
    zero-byte ``.xyz`` reads as zero frames, and a run over zero seeds exited 0 having
    computed nothing.
    """
    path = config.input
    if path is None:
        default = config.template_dir / "step1.xyz"
        if not default.is_file():
            raise ConfigError(f"no 'input' declared and default {default} does not exist")
        path = default
    try:
        state = _seed(path, config.output_dir / "_seed")
    except (OSError, ValueError, IndexError, KeyError) as e:
        raise ConfigError(f"cannot seed from {path}: {e}") from e
    if not state.structures:
        raise ConfigError(f"{path} holds no structures")
    return state


def _seed(path: Path, smiles_out_dir: Path) -> PipelineState:
    """Dispatch on what ``path`` is — a directory, a SMILES CSV or an XYZ file."""
    if path.is_dir():
        return _seed_from_directory(path)
    if path.suffix.lower() == ".csv":
        return _seed_from_smiles_csv(path, smiles_out_dir)
    if path.suffix.lower() == ".xyz":
        return _seed_from_xyz(path)
    raise ConfigError(f"unsupported input format: {path}")


def _state_from_frames(frames: Iterable[Atoms]) -> PipelineState:
    """Number a sequence of frames into a :class:`PipelineState` — IDs ``"0"``, ``"1"``, ….

    The shared tail of every seeder: continuous IDs in iteration order, so a
    directory's frames number straight on from the previous file's.

    It is also where a seed geometry is checked for finiteness, and being the shared tail is
    why: one guard covers the file, the directory and the SMILES paths alike. ASE's reader
    accepts a ``nan`` coordinate, and a seed is the one geometry no parse boundary ever sees
    — so left here it would reach :func:`chemrefine.cache.save`, whose coordinates go to the
    ``arrays.npz`` sidecar rather than through ``write_json``'s ``allow_nan=False``. Nothing
    downstream would object: it round-trips the cache and :func:`~chemrefine.cache.structure_digest`
    hashes it to a perfectly stable key, so every later step would be computed from
    coordinates that are not numbers, silently. The engine's own parse guard cannot catch
    this one, because on the ``on_failure: best`` path the seed is carried forward *instead*
    of a parse.

    :class:`~chemrefine.errors.ConfigError` rather than a parse error: the offending file is
    the user's input, and it is named so they can find the row.
    """
    structures = []
    for i, atoms in enumerate(frames):
        if not np.isfinite(atoms.get_positions()).all():
            raise ConfigError(
                f"seed structure {i} has a non-finite coordinate (nan/inf). A geometry that "
                f"is not numbers cannot be computed on, and nothing downstream would reject "
                f"it — check the input file this seed was read from."
            )
        structures.append(Structure(id=str(i), atoms=atoms))
    return PipelineState(structures=tuple(structures))


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
    damaged from outside — a partial copy, a truncating editor. This program never creates
    the path without its record (:func:`run_lock` links a complete claim into place), so an
    unreadable lock is never one of its own half-written. The caller treats the two alike,
    because neither names a holder whose liveness can be checked.
    """
    try:
        record = json.loads(lock.read_text(encoding="utf-8"))
        if type(record["pid"]) is not int:
            return None
        return str(record["host"]), record["pid"], str(record["started"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _pid_alive(pid: int) -> bool | None:
    """Whether ``pid`` names a live process on *this* host; ``None`` where that cannot be asked.

    ``PermissionError`` means the process exists and belongs to someone else — alive.
    Only :class:`ProcessLookupError` proves it gone.

    ``None`` is a number the kernel cannot be asked about as one process's: one not above
    zero, which ``os.kill`` reads as a process group or as every process, and one past its
    pid type (``OverflowError``). This program records neither, so such a record is
    damaged from outside, and it proves nothing about the driver or job that wrote it.
    Three answers rather than two, so no caller takes "cannot be asked" for "gone": a
    holder is reclaimed, and a lease swept, on ``False`` alone.
    """
    if pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OverflowError:
        return None
    return True


@dataclass(frozen=True)
class LockStatus:
    """One read of the run lock, for a *reporting* caller (status tools, the GUI).

    ``held`` is what a would-be driver cares about: a lock file this read cannot prove
    safe to claim — one naming a holder not provably dead, or one that exists but is
    unreadable (every holder field then ``None``: the claim path refuses that file too,
    so held is the answer the two readers must share). ``alive`` is three-valued —
    ``True`` / ``False`` only for a same-host holder, ``None`` when the holder is on
    another host, or its record names a number no process can have, and liveness cannot
    be probed from here (still ``held``: the reclaim machinery, not a status read, is the
    only thing entitled to call a foreign lock stale). Reporting only — claiming the lock
    remains :func:`run_lock`'s job, with its atomic-rename reclaim dance.
    """

    held: bool
    host: str | None
    pid: int | None
    started: str | None
    alive: bool | None


def lock_status(output_dir: Path) -> LockStatus:
    """Read ``output_dir``'s run lock without touching it.

    The status view :func:`run_lock`'s internals already imply, made public for the
    agent tools and the GUI: *is a driver running this tree, and which one?* A lock
    naming a same-host dead pid reports ``held=False`` — that is exactly the stale case
    :func:`run_lock` would reclaim on the next claim.

    A lock file that exists but names no holder reports ``held=True`` with every
    holder field ``None`` — the same side of "held" :func:`run_lock` puts it on. That
    file was damaged from outside (a partial copy, a truncating editor — this program
    never creates the lock without its record), and the claim path refuses it until
    someone deletes a lock they know is dead; reporting it "not held" here would send
    :func:`~chemrefine.agent_tools.start_run` past its own gate into a child that exits
    into a log nobody is watching yet.
    """
    lock = output_dir / RUN_LOCK_NAME
    holder = _lock_holder(lock)
    if holder is None:
        return LockStatus(held=not io.absent(lock), host=None, pid=None, started=None, alive=None)
    host, pid, started = holder
    alive = _pid_alive(pid) if host == socket.gethostname() else None
    return LockStatus(held=alive is not False, host=host, pid=pid, started=started, alive=alive)


def _reclaim_stale(lock: Path, holder: tuple[str, int, str]) -> bool:
    """Atomically take a dead holder's lock off its path; ``True`` if this process did.

    Reclaiming by ``unlink()`` would be a race: two drivers that both probed the same dead
    pid could interleave — A unlinks, A creates, B unlinks *A's fresh lock*, B creates — and
    both acquire, which is the one state the lock exists to prevent. ``os.replace`` moves
    the inode to a per-pid claim name instead: exactly one process wins the rename (atomic
    on POSIX, and on NFS where this tree lives), and the loser sees
    :class:`FileNotFoundError` and re-reads whatever now holds the path. Any other refusal
    of the rename — a sticky directory guarding another account's lock, a read-only
    mount — is the output tree's :class:`~chemrefine.errors.OutputTreeError`.

    The claim is then verified against ``holder`` — the record that justified it. Between
    the caller's read and the rename, another process can have completed its *own* reclaim
    **and** created a fresh live lock at the same path, and the rename cannot tell those
    apart. A claim whose record no longer matches is therefore put back with ``os.link``,
    the atomic fail-if-exists primitive. When the restore fails, the swept-up driver is
    already running with no lock on the path, and nothing here can un-overlap it. That
    residue is said rather than swallowed (:func:`_unrestored`): which run it is, and that
    the claim file is kept as the only surviving copy of its record. A third driver's lock
    on the path is the ordinary failure: the residue is logged, the reclaim is reported as
    not-ours, and the caller's next pass answers to that lock. Any other refusal of the
    restore stops the run as the output tree's, its refusal carrying the residue and what
    to do about it, where a next pass would link this driver's own lock into the path it
    emptied, beside the driver it swept up.
    """
    claim = lock.with_name(f"{RUN_LOCK_NAME}.reclaim.{os.getpid()}")
    with io.tree_write(lock):
        try:
            lock.replace(claim)
        except FileNotFoundError:
            return False
    swept = _lock_holder(claim)
    if swept != holder:
        try:
            with io.tree_write(lock):
                try:
                    os.link(claim, lock)
                except FileExistsError:
                    current = _lock_holder(lock)
                    logger.error(
                        "run lock %s: a stale-lock reclaim swept up a live lock and could not "
                        "restore it, the path now being held by %s, so two drivers may share "
                        "this tree; %s",
                        lock,
                        f"pid {current[1]} on {current[0]}" if current else "an unreadable lock",
                        _unrestored(claim, swept),
                    )
                    return False
        except OutputTreeError as e:
            raise OutputTreeError(
                f"{e}. That link was putting back a live lock a stale-lock reclaim had swept "
                f"up: {_unrestored(claim, swept)}. Wait for that run to end, or stop it, "
                f"before running the command again."
            ) from e.__cause__
        io.remove_tree_file(claim)
        return False
    io.remove_tree_file(claim)
    return True


def _unrestored(claim: Path, swept: tuple[str, int, str] | None) -> str:
    """What a swept-up live lock that could not be put back leaves: its run and its record."""
    who = f"pid {swept[1]} on {swept[0]}, started {swept[2]}" if swept else "unreadable"
    return f"the run it belongs to ({who}) holds no lock, and its record is kept at {claim}"


@contextlib.contextmanager
def _sigterm_unwinds() -> Generator[None]:
    """Turn SIGTERM into an orderly unwind while a run holds resources.

    ``scancel`` and a walltime kill deliver SIGTERM, and Python's default disposition
    terminates without unwinding — no ``finally``, no ``atexit`` — so the run lock would
    stay behind (stranding a cross-host resume on a lock nobody can probe) and the local
    jobs would keep burning their cores. Raising :class:`SystemExit` from the handler makes
    the signal an ordinary exception at whatever bytecode is executing: :func:`run_lock`'s
    ``finally`` releases the lock (a release the filesystem refuses is logged beside the
    ``143``, never raised over it), :func:`chemrefine.engines._execution.run_batch`'s
    unwind reaps the local jobs, and ``atexit`` still fires. ``143`` is the
    conventional ``128+SIGTERM`` status, so the driver's exit code still says what stopped
    it. A genuine SIGKILL remains unhandleable — the same-host dead-pid reclaim stays as the
    net for that.

    The previous handler is restored on exit, and off the main thread this is a no-op:
    ``signal.signal`` raises there, and a host application driving the pipeline from a
    worker thread must not fail over a nicety it cannot have.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def _raise(signum: int, frame: FrameType | None) -> NoReturn:
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, _raise)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


_LINKS_REFUSED = frozenset({errno.EPERM, errno.EOPNOTSUPP, errno.ENOSYS, errno.EXDEV})
"""What ``os.link`` answers on a filesystem that cannot link the run lock into place at all.

vfat and exfat answer ``EPERM``; some FUSE and SMB mounts ``EOPNOTSUPP`` or ``ENOSYS``; a
union mount that would put the link on another branch than its source, ``EXDEV``. Such a
filesystem can never hold the lock, which is the lock's own refusal,
:class:`~chemrefine.errors.RunLockError`. Any other refusal of the link — a full disk or
quota, a read-only mount — is the output tree refusing a write, as for every write the run
makes.
"""


def _link_claim(claim: Path, lock: Path, output_dir: Path) -> None:
    """Link the complete record ``claim`` into place as ``lock``, or refuse while one stands.

    ``FileExistsError`` is the lock being held, the ordinary answer, decided past the link:
    a same-host holder that is dead is reclaimed (:func:`_reclaim_stale`) — one attempt per
    process, won or lost, since a lost claim means another driver got there first and the
    next pass answers to its lock — and any other holder is refused by name as
    :class:`~chemrefine.errors.RunLockError`. So is a filesystem that cannot hard-link at
    all (:data:`_LINKS_REFUSED`); any other refusal of the link is the output tree's
    (:func:`chemrefine.io.tree_write`).
    """
    reclaimed = False
    while True:
        with io.tree_write(lock):
            try:
                os.link(claim, lock)
                return
            except FileExistsError:
                pass
            except OSError as e:
                if e.errno not in _LINKS_REFUSED:
                    raise
                raise RunLockError(
                    f"cannot lock {output_dir}: the filesystem refuses hard links "
                    f"({e}), which the run lock is built on. Put output_dir on a "
                    f"filesystem that supports them."
                ) from e
        # Decided past the handler, so the refusal below carries no FileExistsError in its
        # traceback.
        holder = _lock_holder(lock)
        if holder is not None and holder[0] == socket.gethostname():
            host, pid, started = holder
            if not reclaimed and _pid_alive(pid) is False:
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
        )


@contextlib.contextmanager
def run_lock(output_dir: Path) -> Generator[None]:
    """Hold ``output_dir`` for one driver; raise :class:`RunLockError` if another has it.

    **Why a lock at all.** The resume machinery cannot tell a live concurrent driver from
    a dead one: :func:`chemrefine.step._incremental_step_outcome` treats a manifest whose
    fingerprint matches as proof it is safe to continue, and a *running* driver leaves
    exactly that state on disk. A second driver would then parse outputs the first one's
    jobs are still writing, archive their directories out from under those jobs, and
    resubmit duplicates — silently, since nothing in that sequence is an error.

    **Why a pidfile and not ``flock``.** The output tree lives on a shared filesystem on
    HPC, where ``flock`` semantics are the least reliable part of NFS; a fail-if-exists
    ``os.link`` is atomic on every filesystem that offers it. One that does not (a vfat or
    exfat drive, some FUSE and SMB mounts) is refused by name, by the errno such a mount
    answers (:data:`_LINKS_REFUSED`); any other refusal of the claim or the link — a full
    disk or a read-only mount — is the output tree's
    :class:`~chemrefine.errors.OutputTreeError`, as for every write the run makes. The cost
    of a pidfile is that a lock can outlive a killed driver.
    :func:`_sigterm_unwinds` narrows that to SIGKILL alone — a ``scancel`` or walltime
    SIGTERM unwinds and releases — and a SIGKILLed holder on *this* host is probed with
    ``os.kill(pid, 0)`` and reclaimed when dead. A holder on another host cannot be probed
    from here — that lock is treated as live, and the error says to delete it once its run
    is known dead.

    **The record and the lock are one operation.** The record is written to a per-pid
    claim beside the lock and *linked* into place, so the lock never exists without a
    complete record from this program. Created empty and written afterwards, it could:
    a full disk at that instant would leave a 0-byte lock the ownership-checked release
    below cannot recognise as its own, and every later driver would refuse the tree as
    "unreadable" until someone deleted it by hand. So an unreadable lock can only have been
    damaged from outside.

    **Reclaim is an atomic rename, and release is ownership-checked.** Deleting a stale lock
    with ``unlink()`` would let two drivers that both probed the same dead pid interleave —
    one deleting the other's freshly created lock — and both acquire; :func:`_reclaim_stale`
    renames the stale lock to a per-pid claim instead, so exactly one process wins, and
    verifies the claim against the record that justified it in case a fresh lock was swept
    up in the window. The release mirrors it: unlinking whatever sits at the lock path would
    let a driver whose lock was deleted out from under it (the error message's own advice,
    followed against a run that was in fact alive) remove the *current* holder's lock on
    exit, reopening the tree to a third driver — so the ``finally`` deletes the file only
    while it still names this process. A release the filesystem refuses after a run that
    returned is that run's :class:`~chemrefine.errors.OutputTreeError`; while an exception
    is on its way out it is logged beside it instead (:func:`chemrefine.io.cleanup`), so a
    SIGTERM's ``143`` stays ``143``.

    **Reentrant by pid**: :func:`chemrefine.recovery.execute` takes the lock around a
    whole action — ``run``'s cache invalidation mutates the tree *before*
    :func:`run` starts — and :func:`run` takes it again for callers that drive the
    pipeline directly. The inner acquisition reads its own pid in the lock before it writes
    anything and yields without ownership: it writes no claim of its own, and the one
    release still happens at the outermost exit.

    **The lock fences drivers, not their jobs.** A dead holder's SLURM jobs run on
    after its lock is reclaimed (and a SIGKILLed holder's local ones too); the leases
    every batch and every submission record are the fence for that window — see
    :func:`require_no_live_jobs`, which both takers of this lock call before touching
    the tree.
    """
    with _sigterm_unwinds():
        lock = output_dir / RUN_LOCK_NAME
        held = _lock_holder(lock)
        if held is not None and held[:2] == (socket.gethostname(), os.getpid()):
            # Ours, taken further out — see "reentrant by pid" above.
            yield
            return
        with io.tree_write(output_dir):
            output_dir.mkdir(parents=True, exist_ok=True)
        # The record in `_lock_holder`'s field order, so the release below compares whole.
        me = (socket.gethostname(), os.getpid(), datetime.now(UTC).isoformat(timespec="seconds"))
        # Complete on disk before the lock exists — see "the record and the lock are one
        # operation" above. `os.link` is the fail-if-exists primitive `_reclaim_stale`
        # already relies on; exactly one claimant's link lands.
        claim = lock.with_name(f"{RUN_LOCK_NAME}.claim.{me[1]}")
        try:
            io.write_tree_file(claim, json.dumps({"pid": me[1], "host": me[0], "started": me[2]}))
            _link_claim(claim, lock, output_dir)
        except BaseException:
            with io.cleanup():
                io.remove_tree_file(claim)
            raise
        try:
            # Linked, the claim is a second name for the lock's record. Removed inside the
            # release's guard, so a refusal here still lets go of the lock just taken.
            io.remove_tree_file(claim)
            yield
        finally:
            # On success and on failure alike: a raise must not leave the tree locked, and
            # back-to-back runs in one process (tests, notebooks) must each acquire cleanly.
            # Only while the file is still ours — see "release is ownership-checked".
            with io.cleanup():
                if _lock_holder(lock) == me:
                    io.remove_tree_file(lock)


def require_no_live_jobs(config: Config) -> None:
    """Refuse to drive a tree while a dead driver's recorded jobs are still running.

    The lock above fences *drivers*, and its release does not outlive their **jobs**: a
    SIGTERM unwind deliberately leaves SLURM jobs running, a SIGKILL leaves the local
    ones too, and the reclaim machinery then hands the lock to the next driver — which
    would archive and re-parse directories the old jobs' exit traps are still copying
    into, the very overlap the lock exists to prevent, reopened through its own release.
    Every submission therefore records the jobs in flight
    (:func:`chemrefine.slurm.dispatch.record_leases`, released when its batch drains),
    and this fence — called with the lock already held, before anything mutates the
    tree — walks every step's ledger and answers each lease by kind:

    * a **SLURM lease** is put to the queue, all of them in one probe: an id the
      scheduler returned, and the tag a batch recorded before it asked for its first job
      (:func:`chemrefine.slurm.dispatch.lease_batch`), which the queue answers by job
      name. The tag is what finds a job whose id never reached the ledger — the driver
      was killed, or ``sbatch`` failed, once the scheduler had it — and the refusal names
      the jobs the queue listed, not the record. A host with no ``squeue`` cannot verify
      them, and a tree copied to a laptop must not be fenced harder than the original —
      so that case warns and proceeds instead of refusing on evidence no one there can
      produce, keeping those leases on record for the next run that can ask.
    * a **local job** is probed by pid on its own host, and refused unless the probe
      proves it gone (:func:`_pid_alive`); recorded on another host it cannot be probed
      from here, and the fence refuses until someone who knows that run is dead deletes
      the record — the lock's own escape, worded the same way.

    Leases every probe proves dead are garbage from a run that never unwound; the fence
    deletes them (it holds the lock, so they are its to clean), which is also what keeps
    a long-lived tree from re-probing long-drained ids on every resume — and, once the
    kernel recycles a dead pid, from refusing a run over a process that was never ours.
    """
    hostname = socket.gethostname()
    visited: list[Path] = []
    scheduled: dict[str, Path] = {}
    for step_cfg in config.steps:
        step_dir = config.step_dir(step_cfg)
        leases = slurm.load_leases(step_dir)
        if not leases:
            continue
        ledger = slurm.lease_path(step_dir)
        for lease in leases:
            if not lease.local:
                scheduled[lease.id] = ledger
            elif lease.host != hostname:
                raise RunLockError(
                    f"a previous run left local job {lease.id} recorded on {lease.host} "
                    f"({ledger}), and its liveness cannot be probed from {hostname}. Two "
                    f"drivers on one tree archive and resubmit each other's work, so this "
                    f"run stops here. If that run is known dead, delete the record and retry."
                )
            else:
                alive = None if lease.pid is None else _pid_alive(lease.pid)
                if alive is not False:
                    state = (
                        f"(pid {lease.pid}) is still running" if alive else "is not provably dead"
                    )
                    raise RunLockError(
                        f"a previous run's local job {lease.id} {state} on this host "
                        f"({ledger} records it), and its exit trap will still write into "
                        f"this tree, so this run stops here. Wait for it to finish — or "
                        f"kill it, delete the record, and retry."
                    )
        visited.append(step_dir)
    if scheduled:
        ledgers = {str(ledger) for ledger in scheduled.values()}
        if not slurm.scheduler_reachable():
            logger.warning(
                "cannot verify the SLURM jobs recorded in %d ledger(s) from this host (no "
                "squeue); proceeding — if the run that submitted them is still live on the "
                "cluster, stop this one now",
                len(ledgers),
            )
            # Every local lease above was proven dead (any other kind raised) and only
            # the SLURM ones are unverified — so shed exactly the former and keep exactly
            # the latter, whether or not they share a ledger. A dead pid left beside a
            # SLURM lease would be re-probed by every later resume and, once recycled,
            # refuse a run whose only escape deletes the SLURM evidence with it.
            for step_dir in visited:
                slurm.release_leases(step_dir, keep_slurm=True)
            return
        queue = slurm.poll_jobs(tuple(scheduled))
        if queue.rows is None:
            raise RunLockError(
                f"squeue could not say whether a previous run's SLURM jobs are still "
                f"queued or running (recorded in {', '.join(sorted(ledgers))}), and their "
                f"exit traps would still write into this tree, so this run stops here. Run "
                f"the command again once the scheduler answers — or, if the record is known "
                f"stale, delete it and retry."
            )
        if queue.rows:
            # A row is a job, or one array's task: `12345_7`, `12345_[8-99%4]`.
            jobs = sorted({row.partition("_")[0] for row in queue.rows})
            live = {str(scheduled[key]) for key in set(scheduled) - queue.finished}
            raise RunLockError(
                f"a previous run's SLURM job(s) {', '.join(jobs)} are still queued or "
                f"running (recorded in {', '.join(sorted(live))}), and their exit traps "
                f"still write into this tree, so this run stops here. Wait for them to "
                f"drain or scancel them — or, if the record is known stale, delete it and "
                f"retry."
            )
    for step_dir in visited:
        slurm.release_leases(step_dir)


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
        path=ids.steps_report_path(config.output_dir),
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
        # Lock first, fence second: the lock answers for live *drivers*, the fence for a
        # dead driver's live *jobs* — and it must run before anything mutates the tree.
        require_no_live_jobs(config)
        logger.info(
            "config: max_cores=%d, max_gpus=%s, output_dir=%s",
            config.max_cores,
            config.max_gpus if config.max_gpus is not None else "auto",
            config.output_dir,
        )
        # Fail fast: every step's backend env must be resolvable before ANY job submits,
        # every step's own preflight refusals (a training step without a device, an
        # open-shell save_tensors) must fire now rather than at that step's turn, and so
        # must the scheduler's (a layout over max_cores, a CUDA step with no GPU to run
        # on) — and `dispatch: slurm` must actually have sbatch available.
        #
        # The steps checked are the ones that *can* submit, which is `StepMode.may_submit` and
        # nothing else. A guard for something that will not happen is just a wall: it would
        # make `chemrefine rebuild-cache` refuse to run wherever the backend is not installed,
        # which is exactly where you want to rebuild — a login node, or any machine holding
        # the output tree but not the backend stack that produced it.
        #
        # Asking `may_submit` rather than excluding `REBUILD` by name is what makes that hold
        # for the whole command. `rebuild-cache N` puts N in `REBUILD` and every *other* step
        # in `CACHE_ONLY`, which equally cannot submit — so excluding only the named step
        # leaves the wall standing on all the others, and a two-step config of provisioned
        # backends still cannot be rebuilt off-cluster. Nothing is weakened by the wider
        # exemption: a step that cannot submit reaches `ChemRefineError` from `run_step` if
        # its cache is unusable, never the engine, so it never reaches the `prepare` this
        # walk stands before.
        submittable = [cfg for cfg in config.steps if plan.for_step(cfg.step).may_submit()]
        preflight_backends(submittable)
        preflight_steps(submittable, charge=config.charge, multiplicity=config.multiplicity)
        # A key no reader declares changes nothing, and the walk says so as `chemrefine
        # validate` does, because a `resume` after an edit, the GUI's Run button and an
        # agent's `start_run` never pass through validate. A warning rather than a refusal,
        # for the report's own reason: the lenient script-engine read is the documented
        # design, and with `nms: true` the NMS knobs share this dict with the engine's.
        for step_cfg in submittable:
            if (silent := undeclared_options(step_cfg)) is not None:
                logger.warning("step %d: %s", step_cfg.step, silent)
        preflight_batches(config, submittable, local=slurm.dispatch_locally(config.dispatch))
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
            if plan.best_effort(step_cfg.step):
                # Past a scoped rebuild's target the walk is reporting, not computing:
                # serve what the current configuration can vouch for — a cache *load*,
                # never a re-parse and never a submission — and let the first cache it
                # cannot serve end the run instead of failing it. The exception's own
                # message names the cheapest command that repairs the tail (see
                # `run_step`'s diagnosis), so it is logged verbatim.
                try:
                    outcome = run_step(config, step_cfg, state, mode=mode)
                except NoUsableCacheError as e:
                    logger.info("stopping the report at step %d: %s", step_cfg.step, e)
                    break
            else:
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
        logger.info("pipeline finished after %d step(s)", len(outcomes))
        return outcomes

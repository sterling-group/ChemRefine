"""Run the scripts: sbatch, squeue, and the local-process fallback.

The other half of :mod:`chemrefine.slurm` — :mod:`chemrefine.slurm.script` writes the
bash, this runs it. Everything here shells out to a real binary or to the OS, which is
why the tests patch ``subprocess.run`` rather than standing up a cluster.

It also owns the **local dispatch** path, which is what lets the same pipeline run on a
laptop: with no ``sbatch``, scripts are launched as background processes under the same
core budget a SLURM run would get, and :data:`_LOCAL_PROCS` tracks them so none is
orphaned when the interpreter exits.
"""

from __future__ import annotations

import atexit
import contextlib
import functools
import getpass
import itertools
import logging
import os
import re
import shutil
import signal
import subprocess
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from chemrefine.errors import ConfigError, JobSubmissionError
from chemrefine.throttle import GpuBudget, StallDeadline

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _current_user() -> str:
    """The username for ``squeue -u``, resolved once (cached for the polling loop).

    Resolved lazily, not at import: ``getpass.getuser()`` raises in
    passwd-less environments (containers running under an arbitrary UID),
    and an import-time call would make the whole package unimportable
    there. The numeric UID is an equally valid ``squeue -u`` argument.
    """
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        return str(os.getuid())


_PARSABLE_ID_RE = re.compile(r"^(\d+)(?:;.*)?$")
"""One line of ``sbatch --parsable`` output: the job id alone, or ``id;cluster`` on a
federation."""

_SUBMITTED_LINE_RE = re.compile(r"Submitted batch job (\d+)")
"""sbatch's human-facing line — the fallback for a site wrapper that swallows
``--parsable``."""


def _parse_job_id(stdout: str) -> str:
    """The job id in sbatch's output — anchored, never "the first integer anywhere".

    Both submitters pass ``--parsable``, so the id is normally a line of its own. The scan
    is per line rather than one search over the whole output because site wrappers prepend
    banners, and an unanchored search took the first number in one: a "90% of quota used"
    warning became job 90, the throttler polled a job that did not exist, ``squeue``
    reported it absent, and the whole batch was ledgered ``MISSING_OUTPUT`` while the real
    jobs ran. The human ``Submitted batch job N`` line stays as an explicit fallback for a
    wrapper that swallows the flag entirely.
    """
    for line in stdout.splitlines():
        if m := _PARSABLE_ID_RE.match(line.strip()):
            return m.group(1)
    if m := _SUBMITTED_LINE_RE.search(stdout):
        return m.group(1)
    raise JobSubmissionError(f"could not parse job ID from sbatch output: {stdout!r}")

_LOCAL_JOB_PREFIX = "local-"
"""Synthetic job-ID prefix used by :func:`_submit_local`.

:func:`is_finished` polls the matching background process for any ID
starting with this prefix.
"""
_LOCAL_JOB_COUNTER = itertools.count(1)

_LOCAL_TERMINATE_GRACE_SECONDS = 5.0
"""How long a local job gets to exit on SIGTERM before :func:`terminate_local_jobs`
escalates to SIGKILL."""

_LOCAL_PROCS: dict[str, tuple[subprocess.Popen[bytes], TextIO, TextIO]] = {}
"""Background local jobs, keyed by ``local-N`` id → ``(proc, out_fh, err_fh)``.

:func:`_submit_local` launches each script with :class:`subprocess.Popen` and
records it here; :func:`is_finished` polls the process, closes its log handles,
and drops the entry on completion. Running local jobs in the background (rather
than blocking) is what lets a laptop run the same throttled parallelism a SLURM
run gets — many scripts run at once under the PAL budget instead of one-at-a-time.

**Module-level on purpose, and this is where that decision is recorded.** Module-global
state should have an owner, and this module is it: the registry models a process-wide fact
(the child processes *this interpreter* started), and the :func:`atexit`-registered sweep
that stops them orphaning cores is a process-wide hook. Hanging it off
:class:`~chemrefine.throttle.Throttler` would not remove the global, only move it to a
registry of live throttlers for ``atexit`` to walk, while coupling a core-budget
abstraction to subprocess handle lifecycles. The scope of the state and the scope of its
owner already agree.
"""


def sbatch_available(*, sbatch_cmd: str = "sbatch") -> bool:
    """Return True when ``sbatch`` is on ``PATH``.

    The raw PATH probe that :func:`dispatch_locally` — the actual SLURM-vs-local
    decision — builds on.
    """
    return shutil.which(sbatch_cmd) is not None


def dispatch_locally(dispatch: str = "auto", *, sbatch_cmd: str = "sbatch") -> bool:
    """Resolve SLURM-vs-local for this run — the single source of truth.

    ``local`` always uses the background bash runner (even when an ``sbatch``
    binary is on PATH); ``slurm`` requires ``sbatch`` and raises
    :class:`~chemrefine.errors.ConfigError` when it is missing — never silently
    local; ``auto`` picks ``sbatch`` when available, the local runner when not.
    """
    if dispatch == "local":
        return True
    available = sbatch_available(sbatch_cmd=sbatch_cmd)
    if dispatch == "slurm" and not available:
        raise ConfigError(
            "`dispatch: slurm` is set but `sbatch` is not on PATH; "
            "use `dispatch: auto`/`local` or run on a SLURM host"
        )
    return not available


# ---------------------------------------------------------------------------
# GPU budget + device-aware header selection
# ---------------------------------------------------------------------------

_UNLIMITED_GPUS = 1_000_000
"""Effectively-unlimited GPU budget used under SLURM, where the scheduler — not
chemrefine — places GPUs (one ``--gres=gpu`` allocation per job)."""


def header_name_for_device(device: str) -> str:
    """Map a compute device to its SLURM header basename.

    ``"cuda"`` → ``cuda.slurm.header`` (requests a GPU node via ``--gres=gpu``),
    anything else → ``cpu.slurm.header``. The trainer and the batch engine both
    call this so the cuda/cpu choice lives in one place.
    """
    return "cuda.slurm.header" if str(device).lower() == "cuda" else "cpu.slurm.header"


def _index_tokens(count: int) -> tuple[str, ...]:
    """``("0", "1", …)`` — the device tokens of a host addressed by plain index."""
    return tuple(str(i) for i in range(count))


def _inherited_devices() -> tuple[str, ...] | None:
    """The device tokens ``CUDA_VISIBLE_DEVICES`` grants this process, or ``None`` if unset.

    Deliberately **not** cached, unlike :func:`_detected_devices`: that probes the host,
    which cannot change under a running process, while this reads an allocation the caller
    controls and a test can vary.

    Tokens are kept verbatim — the variable takes indices, GPU UUIDs (``GPU-8a2f…``) and MIG
    handles interchangeably, and a MIG instance can be named no other way. An empty value is
    a real answer (no devices at all), which is why "unset" is ``None`` rather than an empty
    tuple.
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    return tuple(token for token in (part.strip() for part in raw.split(",")) if token)


@functools.lru_cache(maxsize=1)
def _detected_devices() -> tuple[str, ...]:
    """Best-effort device tokens for the local host, via ``nvidia-smi -L``.

    Counts MIG instances when the card is MIG-partitioned (each is its own CUDA
    device) and whole cards otherwise. Falls back to one device when ``nvidia-smi`` is
    absent or errors — a single-device budget serialises GPU jobs, and the
    engine's availability guard reports a genuinely missing GPU separately.

    The tokens are plain indices, which is what an unpartitioned host's
    ``CUDA_VISIBLE_DEVICES`` takes. A MIG instance can only be selected by its handle, so on
    a MIG host set ``CUDA_VISIBLE_DEVICES`` (SLURM does) and :func:`_inherited_devices`
    supplies the real names instead — extracting MIG UUIDs from ``nvidia-smi -L`` would be a
    genuine improvement and is deliberately out of scope here, since handing out indices is
    what this already did.

    Cached for the life of the process, like :func:`_current_user` above: the device set
    is a property of the host, and this is asked once per batch — per step *and* per retry
    batch — so an uncached probe forks ``nvidia-smi`` throughout a run to re-learn a
    constant. Tests that vary it call ``_detected_devices.cache_clear()``.
    """
    try:
        # Resolved via PATH on purpose: nvidia-smi lives in different places per driver
        # packaging, so a hardcoded path would be more brittle, not less.
        result = subprocess.run(
            ["nvidia-smi", "-L"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return _index_tokens(1)
    lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    mig = [ln for ln in lines if ln.startswith("MIG")]
    gpus = [ln for ln in lines if ln.startswith("GPU")]
    return _index_tokens(max(1, len(mig) or len(gpus)))


def resolve_gpu_budget(configured: int | None, *, local: bool) -> GpuBudget:
    """Resolve the concurrent-GPU budget, and the devices it may hand out.

    **Unlimited under SLURM** and with no device list: the scheduler arbitrates GPUs via
    ``--gres`` and pins each job itself, so chemrefine must not second-guess it. An explicit
    ``Config.max_gpus`` still caps the count there, for a user who wants one.

    Locally the devices come from an inherited ``CUDA_VISIBLE_DEVICES`` when there is one,
    and from the ``nvidia-smi`` probe otherwise. The distinction is between an *allocation*
    and a *guess*, and it decides what ``max_gpus`` is allowed to do:

    * No inherited variable — the probe is a guess about the host, so an explicit
      ``max_gpus`` overrides it outright. That is what the knob is for where ``nvidia-smi``
      is absent or wrong.
    * An inherited variable — this run owns those devices and no others. ``max_gpus`` may
      narrow the allocation and must **not** widen it: the throttler pins one job per token
      and :func:`_submit_local` merges its value *over* the inherited one, so a token this
      run was never granted would reach the job with nothing downstream to catch it.

    Takes the resolved ``local`` rather than the raw ``dispatch`` mode: every caller has
    already asked :func:`dispatch_locally` — it is what decides the whole submission path —
    so re-deriving it here would be a second ``PATH`` probe for an answer already in hand,
    and one more place that could disagree with the path actually taken.
    """
    if not local:
        return GpuBudget(count=_UNLIMITED_GPUS if configured is None else configured)
    granted = _inherited_devices()
    if granted is None:
        devices = _detected_devices() if configured is None else _index_tokens(configured)
        return GpuBudget(count=len(devices), devices=devices)
    if configured is not None and configured > len(granted):
        logger.warning(
            "max_gpus=%d exceeds the %d device(s) in CUDA_VISIBLE_DEVICES (%s); using those",
            configured,
            len(granted),
            ",".join(granted),
        )
    devices = granted if configured is None else granted[:configured]
    return GpuBudget(count=len(devices), devices=devices)


# ---------------------------------------------------------------------------
# sbatch / squeue
# ---------------------------------------------------------------------------


def _submit_local(script_path: str | Path, *, env: dict[str, str] | None = None) -> str:
    """Launch a generated SLURM script via ``bash`` in the background; return a ``local-N`` id.

    The script's ``#SBATCH`` directives are no-ops to bash, so the
    ``--output`` / ``--error`` redirection SLURM normally provides
    doesn't fire. We redirect both streams to the same ``script.runlog``
    / ``script.err`` paths the SBATCH directives point at, so users see
    the same on-disk artifacts in local mode as they do under SLURM.

    Unlike a foreground run this returns immediately — the throttler
    polls :func:`is_finished` to reap it — so several local jobs run
    concurrently under the PAL budget. A non-zero exit is **not** raised
    here; it surfaces through the engine's output parsing (the same path
    SLURM failures take), so it lands in the ``on_failure`` ledger.

    ``start_new_session=True`` gives the job a process group of its own, which is what
    makes it *stoppable*: the thing that has to die is the calculation, and that is a
    grandchild — ``bash`` runs it as a foreground child. Without a group of its own the
    job sits in the driver's, and :func:`terminate_local_jobs` can only reach the shell.
    See there for what that cost.
    """
    script_path = Path(script_path)
    out_handle = script_path.with_suffix(".runlog").open("w", encoding="utf-8")
    err_handle = script_path.with_suffix(".err").open("w", encoding="utf-8")
    try:
        # `bash` from PATH is the point — this is the no-SLURM fallback — and
        # script_path is a script this process generated moments ago.
        proc = subprocess.Popen(  # noqa: S603
            ["bash", str(script_path)],  # noqa: S607
            stdout=out_handle,
            stderr=err_handle,
            env={**os.environ, **env} if env else None,
            start_new_session=True,
        )
    except Exception:
        out_handle.close()
        err_handle.close()
        raise
    job_id = f"{_LOCAL_JOB_PREFIX}{next(_LOCAL_JOB_COUNTER)}"
    _LOCAL_PROCS[job_id] = (proc, out_handle, err_handle)
    logger.info("launched %s locally as job %s (pid %s)", script_path, job_id, proc.pid)
    return job_id


def _local_is_finished(job_id: str) -> bool:
    """Poll a background local job; close its log handles and reap it when done."""
    entry = _LOCAL_PROCS.get(job_id)
    if entry is None:
        return True  # never registered here, or already reaped
    proc, out_handle, err_handle = entry
    if proc.poll() is None:
        return False
    out_handle.close()
    err_handle.close()
    _LOCAL_PROCS.pop(job_id, None)
    if proc.returncode != 0:
        logger.warning("local job %s exited %d (see %s)", job_id, proc.returncode, out_handle.name)
    return True


def terminate_local_jobs(job_ids: Collection[str] | None = None) -> None:
    """Kill and reap background local jobs, closing their log handles.

    Called from the scheduler's ``finally`` so an abnormal exit — a throttle timeout,
    a mid-batch submission failure, Ctrl-C — doesn't orphan the ``bash`` children this
    interpreter launched. Left running they keep burning the cores the user's next
    attempt needs, and their ``.runlog`` / ``.err`` handles stay open.

    ``job_ids`` limits the sweep to one batch; ``None`` — no argument at all — means *every*
    registered local job, which is what the interpreter-exit hook wants. Already-finished
    jobs are simply absent from the registry, so this is safe to call unconditionally.

    **An empty collection means empty, not "everything".** The scheduler passes its live job
    ids from a ``finally``, and on every clean exit that collection is empty; overloading
    falsiness made the ordinary end of a batch mean "kill every local job this process
    started", which is the opposite of the caller's intent and would reach another batch's
    jobs the moment anything ran two.

    **The signal goes to the process group, not to ``bash``.** What has to stop is the
    calculation, and that is a grandchild — the shell runs it as a foreground child and
    then waits. ``proc.terminate()`` reaches only the shell, which defers its ``TERM``
    trap until that child returns, so the grace period expires, the shell is SIGKILLed,
    and the calculation goes on running, reparented to init — silently, because the EXIT
    trap never fires either: no copy-back, no scratch teardown, no runlog footer, while
    ORCA's ``.out``, redirected straight to ``$OUTPUT_DIR``, still parses. Signalling the
    group reaches both, which is why :func:`_submit_local` gives each job a session of its
    own.

    ``ProcessLookupError`` is suppressed because the group can drain between the poll and
    the signal; ``PermissionError`` because a job that re-execs under another uid is the
    scheduler's business, not ours, and neither is a reason to abandon the rest of the sweep.
    """
    targets = list(_LOCAL_PROCS) if job_ids is None else list(job_ids)
    for job_id in targets:
        entry = _LOCAL_PROCS.pop(job_id, None)
        if entry is None:
            continue
        proc, out_handle, err_handle = entry
        if proc.poll() is None:
            logger.warning("terminating local job %s (pgid %s)", job_id, proc.pid)
            _signal_group(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=_LOCAL_TERMINATE_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                _signal_group(proc.pid, signal.SIGKILL)
                proc.wait()
        out_handle.close()
        err_handle.close()


def _signal_group(pgid: int, sig: signal.Signals) -> None:
    """Signal a local job's whole process group, tolerating one that has already gone.

    ``pgid`` is the leader's pid: :func:`_submit_local` starts each job in a new session,
    so the two are the same number.
    """
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, sig)


atexit.register(terminate_local_jobs)


def submit(
    script_path: str | Path,
    *,
    sbatch_cmd: str = "sbatch",
    env: dict[str, str] | None = None,
    dispatch: str = "auto",
) -> str:
    """Submit a SLURM script and return the assigned job ID.

    Falls back to running the generated script directly via ``bash``
    when :func:`dispatch_locally` says so (``dispatch: local``, or
    ``auto`` with no ``sbatch_cmd`` on ``PATH``), so a user can run
    ChemRefine on a laptop without SLURM the same way it runs on an
    HPC node. The local fallback executes synchronously and returns a
    synthetic ``"local-N"`` job ID; :func:`is_finished` treats that
    prefix as already-complete.

    Raises :class:`~chemrefine.errors.JobSubmissionError` if ``sbatch``
    exits non-zero or its output lacks a numeric job ID. The local
    fallback launches in the background and never raises on a non-zero
    exit — that surfaces through output parsing instead.

    ``env`` (e.g. ``{"CUDA_VISIBLE_DEVICES": "1"}``) is applied only on the
    local fallback; under SLURM the scheduler sets the per-job GPU environment.
    """
    if dispatch_locally(dispatch, sbatch_cmd=sbatch_cmd):
        return _submit_local(script_path, env=env)
    try:
        result = subprocess.run(  # noqa: S603
            [sbatch_cmd, "--parsable", str(script_path)],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        detail = (getattr(e, "stderr", None) or "").strip()  # FileNotFoundError has no stderr
        raise JobSubmissionError(
            f"sbatch failed for {script_path}: {e}"
            + (f"\nsbatch said: {detail}" if detail else "")
            + "\n(to run without SLURM set `dispatch: local` in the YAML)"
        ) from e
    job_id = _parse_job_id(result.stdout)
    logger.info("submitted %s as job %s", script_path, job_id)
    return job_id


@dataclass(frozen=True)
class QueueState:
    """One poll of the scheduler: which ids are gone, and what is still queued for the rest.

    Two facts from one query, because they come from one query — splitting them into
    ``finished_jobs`` plus a second "how much is left" call would double the ``squeue`` rate
    of every polling loop to learn two things about the same output.

    ``rows`` is what makes an *array's* progress visible. A parent id stays in
    :attr:`finished`'s complement until its very last task exits, so a caller watching ids
    alone sees a 1000-task array as one motionless job for hours; the rows underneath it
    (``12345_7`` running, ``12345_[9-999]`` pending) change every time a task starts or ends.
    A caller compares them between polls — *changed* means the scheduler is doing something.
    Deliberately not a count: tasks starting split a pending range into more rows and tasks
    finishing remove them, so the number moves in both directions while the set moves
    whenever anything happens.
    """

    finished: frozenset[str]
    """The polled ids with nothing left in the queue (local jobs: the process has exited)."""

    rows: frozenset[str] | None
    """Scheduler rows still queued for the polled ids — array tasks individually.

    ``None`` when the poll never reached the scheduler. That is a third state, not the same
    as an empty set: no rows says the queue drained, ``None`` says nothing was learned.
    Reporting the polled *ids* back as though they were rows collapsed the two, and the
    fabricated set differs from the real task rows every time — so a ``squeue`` alternating
    between working and failing read as a queue moving on every single poll, and the stall
    bound behind :attr:`~chemrefine.config.Config.job_timeout_seconds` never fired."""


def poll_jobs(job_ids: Collection[str], *, squeue_cmd: str = "squeue") -> QueueState:
    """Poll the whole batch's state in **one** ``squeue``.

    The whole set is answered by a single scheduler query. Asking per job means a step
    with N concurrent jobs runs N ``squeue`` subprocesses every poll interval: at
    ``max_cores: 512`` with ``pal: 1`` that is ~50 invocations a second against the
    controller, sustained for the length of the step. Sites rate-limit or ban for exactly
    that, and it is pure waste — one full job list fetched N times to answer N questions
    about it.

    ``"local-N"`` ids are polled via their background process
    (:func:`_local_is_finished`, which also reaps them, so each is polled exactly once
    per call); real SLURM ids are matched against the current user's queue. An array's
    parent id matches its task rows by prefix — ``squeue`` prints running tasks as
    ``12345_0`` and pending ones as ``12345_[5-999]``, never the bare parent id.

    ``--noheader`` is requested explicitly rather than slicing the first line off the
    output: a site that injects ``--noheader`` (via a ``squeue`` wrapper or
    ``SQUEUE_FORMAT``) would otherwise have its first *real* job id discarded as if it
    were the header, reporting a still-running job as finished and misclassifying it as
    a failure.

    A failing ``squeue`` (transient on busy clusters) yields no scheduler ids this
    tick — "not finished", to be retried — rather than falsely reporting the batch done.
    A **missing** ``squeue`` is treated the same way rather than raising: a host with
    ``sbatch`` but no ``squeue`` (a partially-installed client) would otherwise raise on
    every poll of a batch that is already running, which is the worst moment to fail.
    :func:`submit` handles both the same way. Neither case can be told from a genuinely
    idle queue, so neither adds to :attr:`~QueueState.finished` — and
    :attr:`~QueueState.rows` is ``None``, saying the poll learned nothing rather than
    inventing rows for it. Reporting the polled *ids* there instead is what made an
    intermittent ``squeue`` restart the stall clock forever: ids made up from the parents
    differ from the real task rows on every alternation, so a caller watching for movement
    saw it on every tick.
    """
    ids = set(job_ids)
    local = {jid for jid in ids if jid.startswith(_LOCAL_JOB_PREFIX)}
    done = {jid for jid in local if _local_is_finished(jid)}
    scheduled = ids - local
    rows = local - done
    if not scheduled:
        return QueueState(frozenset(done), frozenset(rows))
    try:
        result = subprocess.run(  # noqa: S603
            [squeue_cmd, "--noheader", "-u", _current_user(), "-o", "%i"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return QueueState(frozenset(done), None)
    running = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    for jid in scheduled:
        mine = {line for line in running if line == jid or line.startswith(f"{jid}_")}
        if mine:
            rows |= mine
        else:
            done.add(jid)
    return QueueState(frozenset(done), frozenset(rows))


def finished_jobs(job_ids: Collection[str], *, squeue_cmd: str = "squeue") -> set[str]:
    """Return the subset of ``job_ids`` that is no longer running — **one** ``squeue``.

    The :data:`~chemrefine.throttle.FinishedFn` half of :func:`poll_jobs`, for the caller
    that only needs to know what finished: :class:`chemrefine.throttle.Throttler` frees a
    budget per completion and has no use for what is still queued behind it.
    """
    return set(poll_jobs(job_ids, squeue_cmd=squeue_cmd).finished)


def is_finished(job_id: str, *, squeue_cmd: str = "squeue") -> bool:
    """Return True if ``job_id`` is no longer running.

    The single-job form of :func:`finished_jobs`, kept for callers that genuinely have
    one job to wait on (the MLIP trainer submits exactly one). Batch callers must use
    :func:`finished_jobs` — calling this in a loop is the pathology it exists to avoid.
    """
    return job_id in finished_jobs([job_id], squeue_cmd=squeue_cmd)


def wait_for_jobs(
    job_ids: Collection[str],
    *,
    poll_interval: float,
    poll: Callable[[Collection[str]], QueueState],
    max_wait_seconds: float | None = None,
) -> None:
    """Block until every id in ``job_ids`` reports finished, polling at ``poll_interval``.

    The single canonical "wait for these SLURM jobs to drain" loop — used by the job-array
    path in :mod:`chemrefine.engines._execution` and by the MLIP trainer, which submits one
    job (the per-structure path uses the budget-aware
    :class:`chemrefine.throttle.Throttler` instead, which reaps as it waits). ``poll`` is
    injected (the caller passes :func:`poll_jobs`) so it stays mockable, mirroring the
    throttler.

    ``max_wait_seconds`` mirrors :meth:`chemrefine.throttle.Throttler.wait_for_completion` so
    ``Config.job_timeout_seconds`` means the same thing on every path — otherwise setting it
    would silently do nothing for a ``slurm_array: true`` step. ``None`` waits indefinitely.
    Both share one :class:`~chemrefine.throttle.StallDeadline`, so "the same thing" is one
    implementation rather than two that agree today.

    It bounds each stretch **without progress**, not the whole drain, and *progress is
    measured in tasks* — the reason this takes a :class:`QueueState` rather than a set of
    finished ids. A ≤1000-structure step is a single array, so ``pending`` holds exactly one
    parent id that does not disappear until the last task exits: re-anchoring on ids alone
    turned ``job_timeout_seconds`` into a total-runtime bound there, and a healthy long array
    tripped a timeout meant to catch a stuck one. The rows underneath the parent move
    whenever a task does, and that is what restarts the clock.
    """
    pending = set(job_ids)
    deadline = StallDeadline(max_wait_seconds)
    rows: frozenset[str] | None = None
    while pending:
        state = poll(pending)
        remaining = pending - state.finished
        if not remaining:
            return
        # A poll that never reached the scheduler carries the last known rows forward, so it
        # compares equal and cannot be mistaken for the queue moving.
        seen = rows if state.rows is None else state.rows
        if remaining != pending or seen != rows:
            deadline.progress()
        pending, rows = remaining, seen
        deadline.check(f"{len(pending)} job(s) to finish: {sorted(pending)}")
        time.sleep(poll_interval)


def submit_array(
    script_path: str | Path,
    *,
    n_tasks: int,
    max_concurrent: int,
    manifest: Path,
    sbatch_cmd: str = "sbatch",
) -> str:
    """Submit one array chunk; return the parent job ID.

    ``--export=ALL,CR_MANIFEST=...`` points the shared script at this chunk's
    manifest; ``%max_concurrent`` is the scheduler-enforced concurrency cap
    (the caller computes this chunk's *share* of ``max_cores // PAL``, so all
    of a step's arrays together respect the same core budget the per-job
    throttler enforces — a per-array limit alone would grant it once per
    chunk). Raises
    :class:`~chemrefine.errors.JobSubmissionError` like :func:`submit`.
    There is no local fallback — the engine only takes this path under SLURM.
    """
    try:
        # remaining argv entries are paths we generated.
        result = subprocess.run(  # noqa: S603
            [
                sbatch_cmd,
                "--parsable",
                f"--export=ALL,CR_MANIFEST={manifest}",
                f"--array=0-{n_tasks - 1}%{max_concurrent}",
                str(script_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        detail = (getattr(e, "stderr", None) or "").strip()  # FileNotFoundError has no stderr
        raise JobSubmissionError(
            f"sbatch --array failed for {script_path}: {e}"
            + (f"\nsbatch said: {detail}" if detail else "")
            + "\n(to run without SLURM set `dispatch: local` in the YAML)"
        ) from e
    job_id = _parse_job_id(result.stdout)
    logger.info(
        "submitted %s as array job %s (%d tasks, max %d concurrent)",
        script_path,
        job_id,
        n_tasks,
        max_concurrent,
    )
    return job_id

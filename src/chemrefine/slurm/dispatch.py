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
import functools
import getpass
import itertools
import logging
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Collection
from pathlib import Path
from typing import TextIO

from chemrefine.errors import ConfigError, JobSubmissionError, ThrottleTimeoutError

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


_SBATCH_OVERRIDE_RE = re.compile(r"--(?:ntasks|cpus-per-task|job-name|output|error)(?=[=\s]|$)")
_JOB_ID_RE = re.compile(r"\b(\d+)\b")

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

**Module-level on purpose, and this is where that decision is recorded.** It was
considered for a move onto :class:`~chemrefine.throttle.Throttler`, on the grounds that
module-global state should have an owner. It should — and this module is it: the registry
models a process-wide fact (the child processes *this interpreter* started), and the
:func:`atexit`-registered sweep that stops them orphaning cores is a process-wide hook.
Hanging it off the throttler would not remove the global, only move it to a registry of
live throttlers for ``atexit`` to walk, while coupling a core-budget abstraction to
subprocess handle lifecycles. The scope of the state and the scope of its owner already
agree.

Note the audit's original symptom was different: ``run_batch`` had no ``try/finally``, so
an exception mid-batch orphaned the children. That is fixed, and it was never about the
state being global.
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


def _detect_local_gpus() -> int:
    """Best-effort count of locally visible CUDA devices via ``nvidia-smi -L``.

    Counts MIG instances when the card is MIG-partitioned (each is its own CUDA
    device) and whole cards otherwise. Falls back to ``1`` when ``nvidia-smi`` is
    absent or errors — a single-device budget serialises GPU jobs, and the
    engine's availability guard reports a genuinely missing GPU separately.
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
        return 1
    lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    mig = [ln for ln in lines if ln.startswith("MIG")]
    gpus = [ln for ln in lines if ln.startswith("GPU")]
    return max(1, len(mig) or len(gpus))


def resolve_gpu_budget(configured: int | None, *, dispatch: str = "auto") -> int:
    """Resolve the concurrent-GPU budget for the throttler.

    An explicit ``Config.max_gpus`` wins. Otherwise: **unlimited under SLURM**
    (the scheduler arbitrates GPUs, so chemrefine must not second-guess it) and
    the **detected local device count** off-cluster.
    """
    if configured is not None:
        return configured
    return _detect_local_gpus() if dispatch_locally(dispatch) else _UNLIMITED_GPUS


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


def terminate_local_jobs(job_ids: Collection[str] = ()) -> None:
    """Kill and reap background local jobs, closing their log handles.

    Called from the scheduler's ``finally`` so an abnormal exit — a throttle timeout,
    a mid-batch submission failure, Ctrl-C — doesn't orphan the ``bash`` children this
    interpreter launched. Left running they keep burning the cores the user's next
    attempt needs, and their ``.runlog`` / ``.err`` handles stay open.

    ``job_ids`` limits the sweep to one batch; the default empty tuple means *every*
    registered local job, which is what the interpreter-exit hook wants. Already-finished
    jobs are simply absent from the registry, so this is safe to call unconditionally.
    """
    targets = list(job_ids) if job_ids else list(_LOCAL_PROCS)
    for job_id in targets:
        entry = _LOCAL_PROCS.pop(job_id, None)
        if entry is None:
            continue
        proc, out_handle, err_handle = entry
        if proc.poll() is None:
            logger.warning("terminating local job %s (pid %s)", job_id, proc.pid)
            proc.terminate()
            try:
                proc.wait(timeout=_LOCAL_TERMINATE_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        out_handle.close()
        err_handle.close()


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
            [sbatch_cmd, str(script_path)],
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
    m = _JOB_ID_RE.search(result.stdout)
    if not m:
        raise JobSubmissionError(f"could not parse job ID from sbatch output: {result.stdout!r}")
    job_id = m.group(1)
    logger.info("submitted %s as job %s", script_path, job_id)
    return job_id


def finished_jobs(job_ids: Collection[str], *, squeue_cmd: str = "squeue") -> set[str]:
    """Return the subset of ``job_ids`` that is no longer running — **one** ``squeue``.

    The whole set is answered by a single scheduler query. Asking per job instead meant
    a step with N concurrent jobs ran N ``squeue`` subprocesses every poll interval: at
    ``max_cores: 512`` with ``pal: 1`` that is ~50 invocations a second against the
    controller, sustained for the length of the step. Sites rate-limit or ban for
    exactly that, and it is pure waste — the same full job list was being fetched N
    times to answer N questions about it.

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
    :func:`submit` handles both the same way.
    """
    ids = set(job_ids)
    local = {jid for jid in ids if jid.startswith(_LOCAL_JOB_PREFIX)}
    done = {jid for jid in local if _local_is_finished(jid)}
    scheduled = ids - local
    if not scheduled:
        return done
    try:
        result = subprocess.run(  # noqa: S603
            [squeue_cmd, "--noheader", "-u", _current_user(), "-o", "%i"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return done
    running = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    done |= {
        jid
        for jid in scheduled
        if not any(line == jid or line.startswith(f"{jid}_") for line in running)
    }
    return done


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
    finished: Callable[[Collection[str]], set[str]],
    max_wait_seconds: float | None = None,
) -> None:
    """Block until every id in ``job_ids`` reports finished, polling at ``poll_interval``.

    The single canonical "wait for these SLURM jobs to drain" loop — used by the job-array
    path in :mod:`chemrefine.engines._execution` (the per-job path uses the budget-aware
    :class:`chemrefine.throttle.Throttler` instead, which reaps as it waits). ``finished``
    is injected (the caller passes :func:`finished_jobs`) so it stays mockable, mirroring
    the throttler.

    ``max_wait_seconds`` mirrors :meth:`chemrefine.throttle.Throttler.wait_all` so
    ``Config.job_timeout_seconds`` means the same thing on both paths — otherwise setting it
    would silently do nothing for a ``slurm_array: true`` step. ``None`` waits indefinitely.
    """
    deadline = time.monotonic() + max_wait_seconds if max_wait_seconds is not None else None
    pending = set(job_ids)
    while pending:
        pending -= finished(pending)
        if not pending:
            return
        if deadline is not None and time.monotonic() >= deadline:
            raise ThrottleTimeoutError(
                f"timed out after {max_wait_seconds}s waiting for {len(pending)} array job(s)"
            )
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
    (the caller computes ``max_cores // PAL``, so the array natively respects
    the same core budget the per-job throttler enforces). Raises
    :class:`~chemrefine.errors.JobSubmissionError` like :func:`submit`.
    There is no local fallback — the engine only takes this path under SLURM.
    """
    try:
        # remaining argv entries are paths we generated.
        result = subprocess.run(  # noqa: S603
            [
                sbatch_cmd,
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
    m = _JOB_ID_RE.search(result.stdout)
    if not m:
        raise JobSubmissionError(f"could not parse job ID from sbatch output: {result.stdout!r}")
    job_id = m.group(1)
    logger.info(
        "submitted %s as array job %s (%d tasks, max %d concurrent)",
        script_path,
        job_id,
        n_tasks,
        max_concurrent,
    )
    return job_id

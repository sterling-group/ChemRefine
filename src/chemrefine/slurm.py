"""SLURM mechanics: script generation, ``sbatch`` submission, ``squeue`` polling.

This module is engine-agnostic. It knows how to assemble a SLURM script
from a cluster-specific header template plus an engine-provided
``run_block`` (the bash that actually invokes the calculation), how to
submit that script with ``sbatch``, and how to poll job completion with
``squeue``. PAL-budget bookkeeping lives in :mod:`chemrefine.throttle`;
this module only deals with the SLURM commands themselves.

The :func:`submit` and :func:`is_finished` functions shell out to real
binaries. Tests patch ``subprocess.run`` to avoid needing a live SLURM
cluster.
"""

from __future__ import annotations

import functools
import getpass
import itertools
import logging
import os
import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from chemrefine import job_log
from chemrefine.errors import JobSubmissionError

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


# The lookahead requires `=`, whitespace, or end-of-line after the flag name so
# only the exact directives we re-add are dropped — `--ntasks` must not swallow
# a cluster header's `--ntasks-per-node` / `--ntasks-per-core`.
_SBATCH_OVERRIDE_RE = re.compile(r"--(?:ntasks|cpus-per-task|job-name|output|error)(?=[=\s]|$)")
_JOB_ID_RE = re.compile(r"\b(\d+)\b")

_LOCAL_JOB_PREFIX = "local-"
"""Synthetic job-ID prefix used by :func:`_submit_local`.

:func:`is_finished` polls the matching background process for any ID
starting with this prefix.
"""
_LOCAL_JOB_COUNTER = itertools.count(1)

_LOCAL_PROCS: dict[str, tuple[subprocess.Popen, object, object]] = {}
"""Background local jobs, keyed by ``local-N`` id → ``(proc, out_fh, err_fh)``.

:func:`_submit_local` launches each script with :class:`subprocess.Popen` and
records it here; :func:`is_finished` polls the process, closes its log handles,
and drops the entry on completion. Running local jobs in the background (rather
than blocking) is what lets a laptop run the same throttled parallelism a SLURM
run gets — many scripts run at once under the PAL budget instead of one-at-a-time.
"""


def sbatch_available(*, sbatch_cmd: str = "sbatch") -> bool:
    """Return True when ``sbatch`` is on ``PATH`` (i.e. we're on a real SLURM host).

    The single source of truth for "SLURM vs local": :func:`submit` uses it to
    pick ``sbatch`` over the local-bash fallback, and the batch engine uses it to
    pick the poll cadence and GPU budget.
    """
    return shutil.which(sbatch_cmd) is not None


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
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return 1
    lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    mig = [ln for ln in lines if ln.startswith("MIG")]
    gpus = [ln for ln in lines if ln.startswith("GPU")]
    return max(1, len(mig) or len(gpus))


def resolve_gpu_budget(configured: int | None) -> int:
    """Resolve the concurrent-GPU budget for the throttler.

    An explicit ``Config.max_gpus`` wins. Otherwise: **unlimited under SLURM**
    (the scheduler arbitrates GPUs, so chemrefine must not second-guess it) and
    the **detected local device count** off-cluster.
    """
    if configured is not None:
        return configured
    return _UNLIMITED_GPUS if sbatch_available() else _detect_local_gpus()


def _compute_work_dir_expr(output_dir: Path, scratch_dir: Path | None) -> str:
    """Return the bash expression for ``$WORK_DIR``.

    With no ``scratch_dir``, the per-calc work dir is a sibling under
    ``output_dir``; otherwise it lives under the shared scratch root.
    Both forms append a SLURM-job + timestamp + random suffix so
    concurrent jobs on the same node never collide.
    """
    suffix = "${SLURM_JOB_ID:-$$}_${ts}_${rand}"
    if scratch_dir is None:
        return f"{output_dir}/_work_{suffix}"
    return f"{scratch_dir}/ChemRefine_{suffix}"


# ---------------------------------------------------------------------------
# Script assembly
# ---------------------------------------------------------------------------


def _read_header(template_path: Path) -> tuple[list[str], list[str]]:
    """Split a SLURM header template into ``(#SBATCH lines we keep, body lines)``.

    Drops any ``#SBATCH`` directive we override later (``--ntasks`` /
    ``--cpus-per-task`` / ``--job-name`` / ``--output`` / ``--error``) so PAL and
    log paths stay consistent regardless of what the cluster header declares.
    Longer flags that merely share a prefix (``--ntasks-per-node``) are kept.
    """
    if not template_path.is_file():
        raise FileNotFoundError(f"SLURM header template {template_path} not found")
    sbatch_lines: list[str] = []
    body_lines: list[str] = []
    for raw in template_path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped.startswith("#SBATCH"):
            if not _SBATCH_OVERRIDE_RE.search(stripped):
                sbatch_lines.append(raw.rstrip())
        else:
            body_lines.append(raw.rstrip())
    return sbatch_lines, body_lines


def _run_body_lines(
    *,
    work_dir_expr: str,
    output_dir: Path,
    input_path: Path,
    header: str,
    footer: str,
    globs_expr: str,
    cleanup: str,
    run_block: str,
) -> list[str]:
    """The generated bash after the header: scratch setup, on-exit trap, run block.

    Sets up ``$WORK_DIR`` (a fresh ``ts``/``rand``-suffixed dir), copies the input
    in, and installs an ``EXIT`` trap that always copies ``globs_expr`` back to
    ``$OUTPUT_DIR`` and emits the runlog footer — success or failure — before
    running the engine's ``run_block``.
    """
    return [
        "# Scratch + run block (generated by ChemRefine)",
        "set -euo pipefail",
        "ts=$(date +%Y%m%d%H%M%S)",
        # bash-native random suffix; avoids ``tr ... | head -c`` which fires
        # SIGPIPE on ``head`` close and trips ``pipefail``.
        'rand=$(printf "%04x%04x" "$RANDOM" "$RANDOM")',
        f'export WORK_DIR="{work_dir_expr}"',
        f'export OUTPUT_DIR="{output_dir}"',
        'mkdir -p "$WORK_DIR"',
        f'cp "{input_path}" "$WORK_DIR/"',
        'cd "$WORK_DIR"',
        "",
        header,
        "",
        # Always emit the footer on exit, success or failure. The trap
        # captures $? immediately so it survives the cp/cleanup steps.
        "exit_code=0",
        "files_copied=0",
        "scratch_kept=false",
        "_on_exit() {",
        "  exit_code=$?",
        "  set +e",
        f"  files_copied=$(ls {globs_expr} 2>/dev/null | wc -l)",
        f'  cp {globs_expr} "$OUTPUT_DIR/" 2>/dev/null || true',
        f"  {cleanup}",
        footer,
        "}",
        "trap _on_exit EXIT",
        "",
        run_block,
        "",
    ]


def build_script(
    *,
    job_name: str,
    pal: int,
    template_path: Path,
    script_path: Path,
    input_path: Path,
    output_dir: Path,
    scratch_dir: Path | None,
    run_block: str,
    engine: str,
    operation: str,
    step: int,
    structure_id: str,
    step_label: str,
    output_globs: Sequence[str],
    extra_header_fields: Sequence[tuple[str, object]] = (),
    save_scratch: bool = False,
) -> Path:
    """Assemble a SLURM script at ``script_path`` from a header template + a run block.

    Reads ``template_path`` (a cluster header), strips any ``#SBATCH``
    directives we own (``--ntasks``/``--cpus-per-task``/``--job-name``/
    ``--output``/``--error``) and re-adds them so PAL + log paths stay
    consistent, then appends a scratch-setup + on-exit trap that runs the
    engine's ``run_block`` in a fresh ``$WORK_DIR`` and copies ``output_globs``
    back to ``output_dir``. Notable args:

    * ``pal`` → ``#SBATCH --ntasks`` (``--cpus-per-task`` pinned to 1).
    * ``scratch_dir`` → base for the per-calc ``$WORK_DIR``; ``None`` auto-derives
      ``_work_<jobid>_<ts>_<rand>`` under ``output_dir`` (see :class:`Config`).
    * ``run_block`` → engine bash run after ``cd $WORK_DIR`` (may use
      ``$WORK_DIR``/``$OUTPUT_DIR`` + the input basename).
    * ``output_globs`` → result-file globs copied back on exit (engine-declared).
    * ``engine``/``operation``/``step``/``structure_id``/``step_label`` /
      ``extra_header_fields`` → forwarded to :mod:`chemrefine.job_log` for the
      runlog header/footer.
    """
    sbatch_lines, body_lines = _read_header(template_path)
    runlog_path = output_dir / f"{job_name}.runlog"
    err_path = output_dir / f"{job_name}.err"
    sbatch_lines += [
        f"#SBATCH --job-name={job_name}",
        f'#SBATCH --output="{runlog_path}"',
        f'#SBATCH --error="{err_path}"',
        f"#SBATCH --ntasks={pal}",
        "#SBATCH --cpus-per-task=1",
    ]

    cleanup = (
        'scratch_kept=true; echo "scratch kept at $WORK_DIR"'
        if save_scratch
        else 'scratch_kept=false; cd "$OUTPUT_DIR" && rm -rf "$WORK_DIR"'
    )
    header = job_log.bash_header(
        engine=engine,
        operation=operation,
        step=step,
        structure_id=structure_id,
        step_label=step_label,
        step_dir=output_dir,
        cores=pal,
        extra_fields=extra_header_fields,
    )
    footer = job_log.bash_footer(engine=engine, step_label=step_label)

    script_lines = [
        "#!/bin/bash",
        "",
        *sbatch_lines,
        "",
        *body_lines,
        "",
        *_run_body_lines(
            work_dir_expr=_compute_work_dir_expr(output_dir, scratch_dir),
            output_dir=output_dir,
            input_path=input_path,
            header=header,
            footer=footer,
            globs_expr=" ".join(output_globs),
            cleanup=cleanup,
            run_block=run_block,
        ),
    ]

    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("\n".join(script_lines), encoding="utf-8")
    return script_path


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
        proc = subprocess.Popen(
            ["bash", str(script_path)],
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


def submit(
    script_path: str | Path,
    *,
    sbatch_cmd: str = "sbatch",
    env: dict[str, str] | None = None,
) -> str:
    """Submit a SLURM script and return the assigned job ID.

    Falls back to running the generated script directly via ``bash``
    when ``sbatch_cmd`` is not on ``PATH``, so a user can run
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
    if not sbatch_available(sbatch_cmd=sbatch_cmd):
        return _submit_local(script_path, env=env)
    try:
        result = subprocess.run(
            [sbatch_cmd, str(script_path)],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise JobSubmissionError(f"sbatch failed for {script_path}: {e}") from e
    m = _JOB_ID_RE.search(result.stdout)
    if not m:
        raise JobSubmissionError(f"could not parse job ID from sbatch output: {result.stdout!r}")
    job_id = m.group(1)
    logger.info("submitted %s as job %s", script_path, job_id)
    return job_id


def is_finished(job_id: str, *, squeue_cmd: str = "squeue") -> bool:
    """Return True if ``job_id`` is no longer running.

    ``"local-N"`` IDs are polled via their background process
    (:func:`_local_is_finished`); real SLURM IDs are checked against the
    current user's ``squeue``.
    """
    if job_id.startswith(_LOCAL_JOB_PREFIX):
        return _local_is_finished(job_id)
    try:
        result = subprocess.run(
            [squeue_cmd, "-u", _current_user(), "-o", "%i"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        # squeue is transient on busy clusters; treat as "not finished" and try again later.
        return False
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    # First line is the header ("JOBID"); drop it before checking membership.
    running = lines[1:] if lines else []
    return job_id not in running

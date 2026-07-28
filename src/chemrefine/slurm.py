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
from collections.abc import Callable, Collection, Sequence
from pathlib import Path
from typing import TextIO

from chemrefine import job_log
from chemrefine.errors import ConfigError, JobSubmissionError

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


def _compute_work_dir_expr(output_dir: Path | str, scratch_dir: Path | None) -> str:
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
        raise ConfigError(f"SLURM header template {template_path} not found")
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


_SCRATCH_CLEANUP = 'scratch_kept=false; cd "$OUTPUT_DIR" && rm -rf "$WORK_DIR"'
"""On-exit scratch removal shared by the per-job and array scripts."""


def _run_body_lines(
    *,
    work_dir_expr: str,
    output_dir: Path | str,
    input_path: Path | str,
    header: str,
    footer: str,
    globs_expr: str,
    cleanup: str,
    run_block: str,
    output_dirs: Sequence[str] = (),
) -> list[str]:
    """The generated bash after the header: scratch setup, on-exit trap, run block.

    Sets up ``$WORK_DIR`` (a fresh ``ts``/``rand``-suffixed dir), copies the input
    in, and installs an ``EXIT`` trap that always copies ``globs_expr`` (files) and
    each ``output_dirs`` entry (a whole directory, e.g. pyscf's ``tensors/``) back
    to ``$OUTPUT_DIR`` and emits the runlog footer — success or failure — before
    running the engine's ``run_block``.

    **An engine's run block must not install its own ``trap … EXIT``**: bash keeps one
    handler per signal, so a second one silently *replaces* this handler and takes the
    copy-back, the ``output_dirs`` copy, the scratch teardown and the footer with it. An
    engine that needs teardown of its own defines ``_chemrefine_engine_cleanup`` instead;
    this handler calls it first, before copying anything back.

    ``TERM``/``INT`` are trapped explicitly so a cancelled job records its real exit code:
    the ``EXIT`` trap does fire on a fatal signal, but with ``$?`` already reset to 0, which
    made a ``scancel``-ed run look successful in its runlog. ``_cr_done`` keeps the handler
    single-shot, since ``TERM`` then ``EXIT`` would otherwise run it twice.
    """
    dir_copies = [f'  cp -r "{d}" "$OUTPUT_DIR/" 2>/dev/null || true' for d in output_dirs]
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
        'mkdir -p "$OUTPUT_DIR"',
        f'cp "{input_path}" "$WORK_DIR/"',
        'cd "$WORK_DIR"',
        "",
        header,
        "",
        # Always emit the footer on exit, success or failure.
        "exit_code=0",
        "files_copied=0",
        "scratch_kept=false",
        "_cr_done=false",
        "_on_exit() {",
        # `local rc=$?` must be the FIRST statement: the `$_cr_done` test below is itself a
        # command, so checking the latch first would overwrite the status we came here to record.
        "  local rc=$?",
        "  $_cr_done && return 0",
        "  _cr_done=true",
        # An explicit code from the signal traps wins; otherwise the status we were entered with.
        "  exit_code=${1:-$rc}",
        # `set +e` before the hook, so an engine whose cleanup fails cannot abort the copy-back.
        "  set +e",
        "  declare -F _chemrefine_engine_cleanup >/dev/null && _chemrefine_engine_cleanup",
        f"  files_copied=$(ls {globs_expr} 2>/dev/null | wc -l)",
        f'  cp {globs_expr} "$OUTPUT_DIR/" 2>/dev/null || true',
        *dir_copies,
        f"  {cleanup}",
        footer,
        "}",
        "trap _on_exit EXIT",
        "trap '_on_exit 143' TERM",
        "trap '_on_exit 130' INT",
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
    output_dirs: Sequence[str] = (),
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
        'scratch_kept=true; echo "scratch kept at $WORK_DIR"' if save_scratch else _SCRATCH_CLEANUP
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
            output_dirs=output_dirs,
        ),
    ]

    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("\n".join(script_lines), encoding="utf-8")
    return script_path


# ---------------------------------------------------------------------------
# Job arrays — one script + manifest(s) per step (Config.slurm_array)
# ---------------------------------------------------------------------------

_MAX_ARRAY_SIZE = 1000
"""Tasks per array chunk. Clusters commonly cap ``MaxArraySize`` at 1001
(highest index 1000), so a larger step submits several arrays — indices
restart at 0 per chunk and each chunk gets its own manifest, with the one
shared script pointed at it via ``--export=ALL,CR_MANIFEST=...``."""


def write_array_manifests(
    files: Sequence[tuple[Path, Path, str]],
    step_dir: Path,
    *,
    step_label: str,
) -> list[tuple[Path, tuple[tuple[Path, Path, str], ...]]]:
    """Write the per-chunk task manifests; return ``[(manifest_path, chunk), ...]``.

    Line ``i`` of a manifest is the tab-separated ``input<TAB>output<TAB>id``
    of array task ``i``; the generated array script looks its own line up via
    ``$SLURM_ARRAY_TASK_ID``. Returning the file chunks alongside the paths
    lets the caller map every input to its chunk's parent job id.
    """
    manifests: list[tuple[Path, tuple[tuple[Path, Path, str], ...]]] = []
    for chunk_no, start in enumerate(range(0, len(files), _MAX_ARRAY_SIZE)):
        chunk = tuple(files[start : start + _MAX_ARRAY_SIZE])
        path = step_dir / f"{step_label}_array.manifest.{chunk_no}"
        path.write_text(
            "".join(f"{inp}\t{out}\t{sid}\n" for inp, out, sid in chunk),
            encoding="utf-8",
        )
        manifests.append((path, chunk))
    return manifests


def build_array_script(
    *,
    step_label: str,
    pal: int,
    template_path: Path,
    script_path: Path,
    output_dir: Path,
    scratch_dir: Path | None,
    run_block: str,
    engine: str,
    operation: str,
    step: int,
    output_globs: Sequence[str],
    output_dirs: Sequence[str] = (),
    extra_header_fields: Sequence[tuple[str, object]] = (),
) -> Path:
    """Assemble the one-per-step SLURM array script at ``script_path``.

    Same header handling and run body as :func:`build_script`, but the
    per-structure values resolve at **runtime**: the task reads line
    ``$SLURM_ARRAY_TASK_ID`` of ``$CR_MANIFEST`` (input, output, structure id),
    derives ``$INP_NAME`` / ``$OUT_NAME`` / ``$OUT_DIR`` (the structure's own
    directory = ``dirname $OUT``), and ``exec``-redirects itself to the canonical
    per-structure ``.runlog`` / ``.err`` *inside that directory* so on-disk
    artifacts match the per-job path exactly. ``run_block`` must therefore
    reference the bash variables rather than literal filenames — the engine
    renders it against sentinel paths named ``$INP_NAME`` / ``$OUT_NAME``.
    Failures before the redirect land in the SBATCH fallback log
    ``array_%A_%a.log`` (under the step dir).
    """
    sbatch_lines, body_lines = _read_header(template_path)
    fallback_log = output_dir / "array_%A_%a.log"
    sbatch_lines += [
        f"#SBATCH --job-name={step_label}_array",
        f'#SBATCH --output="{fallback_log}"',
        f'#SBATCH --error="{fallback_log}"',
        f"#SBATCH --ntasks={pal}",
        "#SBATCH --cpus-per-task=1",
    ]
    # ``$SID`` expands inside the runlog heredoc at runtime, like $(hostname).
    header = job_log.bash_header(
        engine=engine,
        operation=operation,
        step=step,
        structure_id="$SID",
        step_label=step_label,
        step_dir=output_dir,
        cores=pal,
        extra_fields=extra_header_fields,
    )
    footer = job_log.bash_footer(engine=engine, step_label=step_label)

    resolution = [
        "# Array task resolution (generated by ChemRefine)",
        "set -euo pipefail",
        'line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CR_MANIFEST")',
        "IFS=$'\\t' read -r INP OUT SID <<< \"$line\"",
        'INP_NAME=$(basename "$INP")',
        'OUT_NAME=$(basename "$OUT")',
        # Each structure has its own directory (= dirname of its output path).
        'OUT_DIR=$(dirname "$OUT")',
        'mkdir -p "$OUT_DIR"',
        "# Per-structure logs — the same artifacts the per-job path writes.",
        'exec >"$OUT_DIR/${INP_NAME%.*}.runlog" 2>"$OUT_DIR/${INP_NAME%.*}.err"',
        "",
    ]

    script_lines = [
        "#!/bin/bash",
        "",
        *sbatch_lines,
        "",
        *body_lines,
        "",
        *resolution,
        *_run_body_lines(
            work_dir_expr=_compute_work_dir_expr("$OUT_DIR", scratch_dir),
            output_dir="$OUT_DIR",
            input_path="$INP",
            header=header,
            footer=footer,
            globs_expr=" ".join(output_globs),
            cleanup=_SCRATCH_CLEANUP,
            run_block=run_block,
            output_dirs=output_dirs,
        ),
    ]

    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("\n".join(script_lines), encoding="utf-8")
    return script_path


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
) -> None:
    """Block until every id in ``job_ids`` reports finished, polling at ``poll_interval``.

    The single canonical "wait for these SLURM jobs to drain" loop — used by the job-array
    path in :mod:`chemrefine.engines._execution` (the per-job path uses the budget-aware
    :class:`chemrefine.throttle.Throttler` instead, which reaps as it waits). ``finished``
    is injected (the caller passes :func:`finished_jobs`) so it stays mockable, mirroring
    the throttler.
    """
    pending = set(job_ids)
    while pending:
        pending -= finished(pending)
        if pending:
            time.sleep(poll_interval)

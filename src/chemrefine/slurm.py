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

import getpass
import itertools
import logging
import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from chemrefine import job_log
from chemrefine.errors import JobSubmissionError

logger = logging.getLogger(__name__)

# Resolved once at import time so the polling loop in :class:`Throttler`
# doesn't re-query ``getpass`` per ``is_finished`` call.
_CURRENT_USER: str = getpass.getuser()

_SBATCH_OVERRIDES = ("--ntasks", "--cpus-per-task", "--job-name", "--output", "--error")
_JOB_ID_RE = re.compile(r"\b(\d+)\b")

_LOCAL_JOB_PREFIX = "local-"
"""Synthetic job-ID prefix used by :func:`_submit_local`.

:func:`is_finished` treats any ID starting with this prefix as
already-completed because the local fallback runs synchronously
inside :func:`submit`.
"""
_LOCAL_JOB_COUNTER = itertools.count(1)


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
    """Assemble a SLURM script at ``script_path`` and return its path.

    Parameters
    ----------
    job_name:
        Becomes ``#SBATCH --job-name`` and the stem of the ``.runlog`` /
        ``.err`` files.
    pal:
        Becomes ``#SBATCH --ntasks``. ``--cpus-per-task`` is pinned to 1.
    template_path:
        Cluster-specific SLURM header (e.g. ``cpu.slurm.header``). Any
        ``--ntasks`` / ``--cpus-per-task`` / ``--job-name`` /
        ``--output`` / ``--error`` directives the user wrote there are
        dropped and replaced with our overrides so PAL stays consistent.
    script_path:
        Output path for the generated ``.slurm`` script (parents will be
        created).
    input_path:
        The engine input file. Copied into the per-calculation
        ``$WORK_DIR`` so the calculation runs on local fast storage.
    output_dir:
        Where to copy results back to once the calculation finishes.
        Used as the runlog destination via absolute ``#SBATCH --output``.
    scratch_dir:
        Fast-storage base for the per-calculation work dir. When
        ``None``, the work dir is auto-derived as a sibling under
        ``output_dir`` (``_work_<jobid>_<ts>_<rand>``). When set, the
        work dir lives under ``scratch_dir`` instead.
    run_block:
        Engine-specific bash that actually invokes the calculation. It
        runs after ``cd $WORK_DIR``; it can reference ``$OUTPUT_DIR``,
        ``$WORK_DIR``, and the basename of the input file.
    engine, operation, step, structure_id, step_label:
        Forwarded to :mod:`chemrefine.job_log` so the runlog header /
        footer carry the same fields direct-mode engines emit.
    output_globs:
        Shell globs of result files to copy back to ``output_dir`` once
        the calculation finishes. Engines declare what they produce
        (e.g. ORCA: ``("*.out", "*.xyz", "*.gbw", "*.hess")``); the
        SLURM layer never assumes a specific engine's file set.
    extra_header_fields:
        Engine-specific ``(key, value)`` rows appended after the
        generic runlog header fields. ORCA uses this for
        ``orca_executable``; other engines pass whatever identifies the
        binary they shelled out to.
    """
    if not template_path.is_file():
        raise FileNotFoundError(f"SLURM header template {template_path} not found")

    sbatch_lines: list[str] = []
    body_lines: list[str] = []
    for raw in template_path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped.startswith("#SBATCH"):
            if any(flag in stripped for flag in _SBATCH_OVERRIDES):
                continue
            sbatch_lines.append(raw.rstrip())
        else:
            body_lines.append(raw.rstrip())

    runlog_path = output_dir / f"{job_name}.runlog"
    err_path = output_dir / f"{job_name}.err"

    sbatch_lines.append(f"#SBATCH --job-name={job_name}")
    sbatch_lines.append(f'#SBATCH --output="{runlog_path}"')
    sbatch_lines.append(f'#SBATCH --error="{err_path}"')
    sbatch_lines.append(f"#SBATCH --ntasks={pal}")
    sbatch_lines.append("#SBATCH --cpus-per-task=1")

    work_dir_expr = _compute_work_dir_expr(output_dir, scratch_dir)

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

    globs_expr = " ".join(output_globs)

    script_lines = [
        "#!/bin/bash",
        "",
        *sbatch_lines,
        "",
        *body_lines,
        "",
        "# Scratch + run block (generated by ChemRefine)",
        "set -euo pipefail",
        'ts=$(date +%Y%m%d%H%M%S)',
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

    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("\n".join(script_lines), encoding="utf-8")
    return script_path


# ---------------------------------------------------------------------------
# sbatch / squeue
# ---------------------------------------------------------------------------


def _submit_local(script_path: str | Path) -> str:
    """Run a generated SLURM script directly via ``bash``; return a synthetic job ID.

    The script's ``#SBATCH`` directives are no-ops to bash, so the
    ``--output`` / ``--error`` redirection SLURM normally provides
    doesn't fire. We capture both streams instead and write them to
    the same ``script.runlog`` / ``script.err`` paths the SBATCH
    directives point at, so users see the same on-disk artifacts in
    local mode as they do under SLURM.
    """
    script_path = Path(script_path)
    result = subprocess.run(
        ["bash", str(script_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    script_path.with_suffix(".runlog").write_text(result.stdout, encoding="utf-8")
    script_path.with_suffix(".err").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise JobSubmissionError(
            f"local execution of {script_path} failed "
            f"(exit {result.returncode}): {result.stderr.strip()[:500]}"
        )
    job_id = f"{_LOCAL_JOB_PREFIX}{next(_LOCAL_JOB_COUNTER)}"
    logger.info("ran %s locally as job %s", script_path, job_id)
    return job_id


def submit(script_path: str | Path, *, sbatch_cmd: str = "sbatch") -> str:
    """Submit a SLURM script and return the assigned job ID.

    Falls back to running the generated script directly via ``bash``
    when ``sbatch_cmd`` is not on ``PATH``, so a user can run
    ChemRefine on a laptop without SLURM the same way it runs on an
    HPC node. The local fallback executes synchronously and returns a
    synthetic ``"local-N"`` job ID; :func:`is_finished` treats that
    prefix as already-complete.

    Raises :class:`~chemrefine.errors.JobSubmissionError` if ``sbatch``
    exits non-zero, its output lacks a numeric job ID, or the local
    fallback script exits non-zero.
    """
    if shutil.which(sbatch_cmd) is None:
        return _submit_local(script_path)
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
        raise JobSubmissionError(
            f"could not parse job ID from sbatch output: {result.stdout!r}"
        )
    job_id = m.group(1)
    logger.info("submitted %s as job %s", script_path, job_id)
    return job_id


def is_finished(job_id: str, *, squeue_cmd: str = "squeue") -> bool:
    """Return True if ``job_id`` is no longer in the current user's ``squeue``.

    Synthetic ``"local-N"`` IDs are reported finished immediately —
    those scripts already ran to completion inside :func:`submit`.
    """
    if job_id.startswith(_LOCAL_JOB_PREFIX):
        return True
    try:
        result = subprocess.run(
            [squeue_cmd, "-u", _CURRENT_USER, "-o", "%i"],
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

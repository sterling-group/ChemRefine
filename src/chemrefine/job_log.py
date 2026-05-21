"""Per-job operational logs (one file per structure per step).

A *runlog* is one file per structure per step at
``<step_dir>/step{N}_structure_{ID}.runlog``. Both bash (SLURM) and
direct (in-process) engines emit the same skeleton:

* A start header with host, job_id, mode, engine, operation, step,
  structure_id, scratch path, output path, cores, and orca_executable.
* The execution phase content (engine-specific stdout interleaved).
* A finish footer with exit_code, elapsed_seconds, files_copied,
  scratch_kept.

The same fields appear in both modes so a maintainer grepping
``outputs/step*/step*_structure_*.runlog`` sees a uniform corpus.
"""

from __future__ import annotations

import os
import socket
import time
from datetime import datetime
from pathlib import Path

_HEADER_KEYS = (
    "host",
    "job_id",
    "mode",
    "engine",
    "operation",
    "step",
    "structure_id",
    "scratch",
    "output",
    "cores",
    "orca_executable",
)
_FOOTER_KEYS = (
    "exit_code",
    "elapsed_seconds",
    "files_copied",
    "scratch_kept",
)


def _format_field(key: str, value: object) -> str:
    """Render one ``key=value`` field line, indented two spaces."""
    return f"  {key}={value}"


def _now_iso() -> str:
    """Return a timezone-aware ISO-8601 timestamp at second precision."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Bash side (embedded in SLURM scripts)
# ---------------------------------------------------------------------------


def bash_header(
    *,
    engine: str,
    operation: str,
    step: int,
    structure_id: str,
    step_label: str,
    step_dir: Path,
    cores: int,
    orca_executable: str,
) -> str:
    """Return a bash snippet that prints the job-start header to stdout.

    The snippet relies on ``$WORK_DIR`` being set in the surrounding bash
    scope (the generated SLURM script does this immediately above the
    snippet). It also leaves ``start_time`` set so :func:`bash_footer`
    can compute the elapsed seconds.
    """
    fields = [
        ("host", "$(hostname)"),
        ("job_id", "${SLURM_JOB_ID:-$$}"),
        ("mode", "$__cr_mode"),
        ("engine", engine),
        ("operation", operation),
        ("step", step),
        ("structure_id", structure_id),
        ("scratch", "$WORK_DIR"),
        ("output", step_dir),
        ("cores", cores),
        ("orca_executable", orca_executable),
    ]
    field_lines = "\n".join(_format_field(k, v) for k, v in fields)
    return (
        'if [ -n "${SLURM_JOB_ID:-}" ]; then __cr_mode=slurm; else __cr_mode=bash; fi\n'
        "cat <<EOF\n"
        f"[$(date -Iseconds)] ChemRefine {engine} {step_label} starting\n"
        f"{field_lines}\n"
        "EOF\n"
        "start_time=$(date +%s)"
    )


def bash_footer(*, engine: str, step_label: str) -> str:
    """Return a bash snippet that prints the job-end footer.

    Reads ``$?``, ``start_time``, ``files_copied``, and ``scratch_kept``
    from the surrounding scope. ``files_copied`` and ``scratch_kept``
    should be set by the surrounding script before this footer fires;
    fallbacks of ``0`` and ``false`` are used when they are not.
    """
    fields = [
        ("exit_code", "$exit_code"),
        ("elapsed_seconds", "$elapsed_seconds"),
        ("files_copied", "${files_copied:-0}"),
        ("scratch_kept", "${scratch_kept:-false}"),
    ]
    field_lines = "\n".join(_format_field(k, v) for k, v in fields)
    return (
        "end_time=$(date +%s)\n"
        "elapsed_seconds=$((end_time - start_time))\n"
        "cat <<EOF\n"
        f"[$(date -Iseconds)] ChemRefine {engine} {step_label} finished\n"
        f"{field_lines}\n"
        "EOF"
    )


# ---------------------------------------------------------------------------
# Python side (direct in-process engines)
# ---------------------------------------------------------------------------


def python_header(
    *,
    engine: str,
    operation: str,
    step: int,
    structure_id: str,
    step_label: str,
    step_dir: Path,
    log_path: Path,
    cores: int = 1,
    orca_executable: str = "—",
) -> Path:
    """Write the job-start header to ``log_path`` and return that path.

    ``orca_executable`` defaults to an em-dash because direct engines do
    not shell out to ORCA. ``cores`` defaults to ``1`` for the same
    reason — in-process scoring is single-threaded by convention.
    """
    fields = [
        ("host", socket.gethostname()),
        ("job_id", os.getpid()),
        ("mode", "direct"),
        ("engine", engine),
        ("operation", operation),
        ("step", step),
        ("structure_id", structure_id),
        ("scratch", step_dir),
        ("output", step_dir),
        ("cores", cores),
        ("orca_executable", orca_executable),
    ]
    lines = [
        f"[{_now_iso()}] ChemRefine {engine} {step_label} starting",
        *(_format_field(k, v) for k, v in fields),
        "",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines), encoding="utf-8")
    return log_path


def python_footer(
    *,
    engine: str,
    step_label: str,
    log_path: Path,
    exit_code: int,
    elapsed_seconds: int,
    files_copied: int = 0,
    scratch_kept: bool = False,
) -> None:
    """Append the job-end footer to ``log_path``."""
    fields = [
        ("exit_code", exit_code),
        ("elapsed_seconds", elapsed_seconds),
        ("files_copied", files_copied),
        ("scratch_kept", "true" if scratch_kept else "false"),
    ]
    lines = [
        f"[{_now_iso()}] ChemRefine {engine} {step_label} finished",
        *(_format_field(k, v) for k, v in fields),
        "",
    ]
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Convenience: time the direct-mode payload
# ---------------------------------------------------------------------------


def monotonic_seconds() -> int:
    """Return an integer monotonic second counter for elapsed timing."""
    return int(time.monotonic())

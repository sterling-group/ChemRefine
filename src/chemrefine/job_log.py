"""Per-job operational logs (one file per structure per step).

A *runlog* is one file per job, written beside the calculation it
describes. A structure's own runs sit at its canonical path,
``<step_dir>/<structure_id>/step{N}_{structure_id}.runlog``; the
attempt directories hold the rest, in two shapes because two things
put them there:

* ``<structure_id>/attempt{K}/step{N}_{structure_id}.runlog`` — a
  *superseded* run, moved there wholesale when the next attempt began
  (:func:`chemrefine.attempts.archive_previous`). A convergence retry
  re-runs at the canonical path, so this is where its first try went.
* ``<structure_id>/attempt{K}/<child_id>/step{N}_{child_id}.runlog`` —
  a job that *ran* inside the attempt: an NMS round-2 child, or that
  child's own retry one level deeper.

(The ``step{N}_structure_{ID}`` basename this module once documented is
the **v1.3.1** spelling; ``structure_`` was dropped from every artifact
name in 2.0 — see the v1→v2 migration guide.)

The bash header/footer snippets here are embedded in every generated
SLURM script (which also runs via the local bash fallback), so every
engine emits the same skeleton:

* A start header with host, job_id, mode, engine, operation, step,
  structure_id, scratch path, output path, and cores. Engines append
  their own ``(key, value)`` rows via ``extra_fields`` (e.g. ORCA adds
  ``orca_executable``).
* The execution phase content (engine-specific stdout interleaved).
* A finish footer with exit_code, elapsed_seconds, files_copied,
  scratch_kept.

The fixed fields (``_HEADER_KEYS`` / ``_FOOTER_KEYS``) drive the field
order, so a maintainer grepping ``outputs/step*/**/step*.runlog`` sees a
uniform corpus — every attempt included; engine-specific rows extend the
header without disturbing that shape. (In bash that pattern needs
``shopt -s globstar``; without it ``**`` collapses to one level and finds
only the canonical runlogs. zsh, ``Path.glob`` and ripgrep need nothing.)
"""

from __future__ import annotations

from collections.abc import Sequence
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
    extra_fields: Sequence[tuple[str, object]] = (),
) -> str:
    """Return a bash snippet that prints the job-start header to stdout.

    The snippet relies on ``$WORK_DIR`` being set in the surrounding bash
    scope (the generated SLURM script does this immediately above the
    snippet). It also leaves ``start_time`` set so :func:`bash_footer`
    can compute the elapsed seconds. Engines can append their own
    ``(key, value)`` rows via ``extra_fields`` — they render after the
    fixed runlog skeleton.
    """
    values = {
        "host": "$(hostname)",
        "job_id": "${SLURM_JOB_ID:-$$}",
        "mode": "$__cr_mode",
        "engine": engine,
        "operation": operation,
        "step": step,
        "structure_id": structure_id,
        "scratch": "$WORK_DIR",
        "output": step_dir,
        "cores": cores,
    }
    fields = [(k, values[k]) for k in _HEADER_KEYS] + list(extra_fields)
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
    values = {
        "exit_code": "$exit_code",
        "elapsed_seconds": "$elapsed_seconds",
        "files_copied": "${files_copied:-0}",
        "scratch_kept": "${scratch_kept:-false}",
    }
    fields = [(k, values[k]) for k in _FOOTER_KEYS]
    field_lines = "\n".join(_format_field(k, v) for k, v in fields)
    return (
        "end_time=$(date +%s)\n"
        "elapsed_seconds=$((end_time - start_time))\n"
        "cat <<EOF\n"
        f"[$(date -Iseconds)] ChemRefine {engine} {step_label} finished\n"
        f"{field_lines}\n"
        "EOF"
    )

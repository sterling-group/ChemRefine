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

The bash header/footer snippets here are embedded in every generated
SLURM script (which also runs via the local bash fallback), so every
engine emits the same skeleton:

* A start header with host, job_id, mode, engine, operation, step,
  structure_id, scratch path, output path, and cores. Engines append
  their own ``(key, value)`` rows via ``extra_fields`` (e.g. the binary
  an engine ran).
* The execution phase content (engine-specific stdout interleaved).
* A finish footer with exit_code, elapsed_seconds, files_copied,
  scratch_kept.

The fixed fields (``_HEADER_KEYS`` / ``_FOOTER_KEYS``) drive the field
order, so a maintainer grepping ``outputs/step*/**/step*.runlog`` sees a
uniform corpus — every attempt included; engine-specific rows extend the
header without disturbing that shape. (In bash that pattern needs
``shopt -s globstar``; without it ``**`` collapses to one level and finds
only the canonical runlogs. zsh, ``Path.glob`` and ripgrep need nothing.)

The footer is the job's own verdict
-----------------------------------

It is the one record of how a job ended that outlives the driver which
submitted it, and :func:`read_footer` is its one reader. Two of its
fields are the verdict, one fact each:

* ``exit_code`` — how the run block ended: its own status, ``0`` for one
  that ran to its end, or ``143`` / ``130`` for a job stopped by ``TERM``
  / ``INT`` (a ``scancel``, a walltime, the driver's own unwind),
  whatever the program inside made of the signal.
* ``scratch_kept`` — ``true`` when a result the engine declared could not
  be copied home. The scratch directory, the only whole copy, is then
  left at the header's ``scratch`` path on its ``host``, and the script
  exits :data:`COPY_BACK_FAILED` if its run block had succeeded.

``files_copied`` counts the loose result files that did arrive. A job killed
outright — ``SIGKILL``, a node that went down — writes no footer at all,
which is why no footer is never read as success.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

from chemrefine.config import reject_shell_unsafe
from chemrefine.errors import ConfigError

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

_STARTING = "starting"
_FINISHED = "finished"
"""The word that ends a run's first line, and the one that ends its footer's."""

COPY_BACK_FAILED = 74
"""The status a job exits with when its run block succeeded and a result did not come home.

``EX_IOERR`` of ``sysexits.h``. The generated script's exit handler exits with it
(:func:`chemrefine.slurm.script._run_body_lines`), so a scheduler's accounting and the
local runner's warning report a job whose footer says ``exit_code=0`` and
``scratch_kept=true`` as the failure it is."""


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

    ``operation`` is the one the step's outputs are read as
    (:attr:`~chemrefine.state.StepContext.operation`) — declared, or inferred from the
    template — so the runlog says what the step ran as, not only what the config spelled.

    **The operation and every ``extra_fields`` value are held to**
    :func:`~chemrefine.config.reject_shell_unsafe` **here, at emission.** The header is an
    unquoted heredoc, so a value is interpolated as bash — a ``$(...)`` in one executes when
    the job runs, and a bare ``$NAME`` is a nounset abort under the script's ``set -u`` when
    nothing has exported it yet (the header runs *before* the engine's run block). Checked at
    the point the value becomes bash rather than per engine, so a row added by any engine,
    and an operation any engine infers, is covered by existing — the same property-keyed
    placement as the rule itself. The rest of the skeleton is exempt on purpose:
    ``$(hostname)`` and ``${SLURM_JOB_ID:-$$}`` are this module's own deliberate
    expansions, and the other values are safe by construction.

    A refused value is a :class:`~chemrefine.errors.ConfigError` naming the field: the
    text originates in the user's YAML (an engine's option, an ``executables`` entry),
    and a bare ``ValueError`` from inside script assembly would leave the exit-code
    contract as a traceback.
    """
    for key, value in (("operation", operation), *extra_fields):
        try:
            reject_shell_unsafe(
                str(value), what=f"runlog field {key!r}", fix="remove the character"
            )
        except ValueError as e:
            raise ConfigError(str(e)) from e
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
        f"[$(date -Iseconds)] ChemRefine {engine} {step_label} {_STARTING}\n"
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
        f"[$(date -Iseconds)] ChemRefine {engine} {step_label} {_FINISHED}\n"
        f"{field_lines}\n"
        "EOF"
    )


# ---------------------------------------------------------------------------
# Python side (the driver reads what a job left)
# ---------------------------------------------------------------------------


def runlog_path(output_dir: Path, job_name: str) -> Path:
    """Where the job named ``job_name`` logs its run: in the directory it writes into.

    The canonical path of this module's docstring, for the writers that are Python and for
    the reader: the generated script's ``#SBATCH --output``
    (:func:`chemrefine.slurm.script.build_script`), the local runner's redirect
    (:func:`chemrefine.slurm.dispatch.submit`, which names the job by its script, written
    beside its output) and :func:`read_footer`'s callers. An array task redirects to the
    same path in bash, from its own input and output
    (:func:`chemrefine.slurm.script.build_array_script`). A job is named after its input,
    so for a prepared ``(input, output, id)`` triple this is
    ``runlog_path(output.parent, input.stem)``.
    """
    return output_dir / f"{job_name}.runlog"


# The stamp is `date -Iseconds`, 25 characters; bounded, so a line full of brackets (a
# progress bar, colour codes) is not rescanned from each of them.
_RUN_LINE = r"\[[^\]\n]{0,40}\] ChemRefine [^\n]* "
_STARTED_RE = re.compile(_RUN_LINE + _STARTING + r"\n")
_FOOTER_RE = re.compile(
    _RUN_LINE + _FINISHED + r"\n" + "".join(rf"  {key}=(?P<{key}>[^\n]*)\n" for key in _FOOTER_KEYS)
)
"""A footer as :func:`bash_footer` writes it, whole: its line and every field, in order.

Built from ``_FOOTER_KEYS``, so the reader follows the format. Not anchored to a line
start: a program whose last output carries no newline leaves the footer's line joined to
it."""

_TAIL_BYTES = 65536
"""How much of a runlog's end :func:`read_footer` reads. A footer is five short lines and
the last thing its script writes; a training's log above it can run to gigabytes."""

_BOOLEANS = {"true": True, "false": False}


class Footer(NamedTuple):
    """What a job said about how it ended — the two verdict fields of its footer."""

    exit_code: int
    """How the run block ended; ``0`` for one that ran to its end."""

    scratch_kept: bool
    """Whether a declared result could not be copied home, so scratch was left holding it."""


def read_footer(runlog: Path) -> Footer | None:
    """The footer of a job's last run, as it wrote it; ``None`` when it wrote none.

    ``None`` is no verdict at all, and a caller must never read it as success: there is
    no runlog, it cannot be read, the footer is cut short, or the job was killed before
    it could write one.

    The verdict is the **last run's**. A scheduler that appends to a job's output file
    leaves an earlier run's footer above a later run's header, and a later run that was
    killed leaves no footer of its own — so a footer followed by another start says
    nothing about the run that produced what is on disk.

    What is read is text in the job's own output, so a program that printed a whole
    footer of its own and was then killed outright, before the script's handler could
    write the real one, would be believed. No engine's program prints one.
    """
    try:
        with runlog.open("rb") as handle:
            handle.seek(max(0, handle.seek(0, os.SEEK_END) - _TAIL_BYTES))
            tail = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    footers = list(_FOOTER_RE.finditer(tail))
    if not footers or _STARTED_RE.search(tail, footers[-1].end()):
        return None
    try:
        return Footer(int(footers[-1]["exit_code"]), _BOOLEANS[footers[-1]["scratch_kept"]])
    except (ValueError, KeyError):
        return None

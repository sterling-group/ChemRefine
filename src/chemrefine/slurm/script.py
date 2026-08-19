"""Assemble the bash a job runs: the SLURM script, and the job-array variants.

Everything here turns values into text and returns it — nothing submits, polls or
touches the scheduler; that is :mod:`chemrefine.slurm.dispatch`. The split is what
lets the shell-safety rule be checked: every value that ends up inside a generated
script passes through one of the builders below, so
``tests/test_engines_invariants.py`` can enumerate them from these signatures rather
than from memory.

A script is a cluster-specific header template plus an engine-provided
:class:`~chemrefine.state.RunBlock` (the bash that invokes the calculation). This
module is engine-agnostic: it never imports an engine, and the only thing it knows
about a calculation is the block of bash it was handed.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from chemrefine import job_log
from chemrefine.errors import ConfigError
from chemrefine.state import JobTriple, RunBlock

logger = logging.getLogger(__name__)

# The lookahead requires `=`, whitespace, or end-of-line after the flag name so
# only the exact directives we re-add are dropped — `--ntasks` must not swallow
# a cluster header's `--ntasks-per-node` / `--ntasks-per-core`.
_SBATCH_OVERRIDE_RE = re.compile(r"--(?:ntasks|cpus-per-task|job-name|output|error)(?=[=\s]|$)")


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
"""On-exit scratch removal shared by the per-job and array scripts.

Unconditional on both paths. :func:`build_script` once took a ``save_scratch`` flag that
swapped this for a keep-and-announce line, but it was a v1 concept carried into the
rewrite's signature and never wired to anything — no config knob, no caller — so it was
removed rather than left as a knob nobody could reach. The engine-owned equivalents are the
ones that work, and there are two of them: an engine copies a directory home by naming it in
``output_dirs`` (``pyscf-extopt``'s ``tensors/``), or runs its own teardown by returning it as
:attr:`~chemrefine.state.RunBlock.cleanup` (``qchem``'s ``options.save``, which copies
``$QCSCRATCH/$QCSAVE``; the ExtOpt server stop). Both are honoured on the array path as well
— where a builder flag would have had to be added twice.

``scratch_kept`` stays in the runlog: the footer is a published artifact, the field still
reports truthfully, and changing that format is a decision of its own."""

# ``--mem`` and ``--mem-per-cpu`` in either ``=`` or space form. ``--mem-per-gpu`` cannot
# match: after ``--mem`` the optional group rejects ``-per-gpu`` and the mandatory ``[=\s]``
# rejects the ``-`` that follows, so a GPU memory directive is neither read nor stripped.
_MEM_DIRECTIVE_RE = re.compile(r"--mem(?:-per-cpu)?[=\s]+(\d+)([KkMmGgTt]?)")


def _directive_mb(value: int, unit: str) -> int:
    """One SLURM memory value in MB. Bare numbers are MB (SLURM's default unit).

    ``K`` floors to MB — understating the header's grant errs toward extending it, which
    only ever raises an allocation, never starves one.
    """
    unit = unit.upper()
    if unit == "K":
        return value // 1024
    return value * {"": 1, "M": 1, "G": 1024, "T": 1024 * 1024}[unit]


def _header_memory_mb(sbatch_lines: list[str], *, ntasks: int, cpus_per_task: int) -> int | None:
    """Per-job MB the header's own memory directives grant; ``None`` when it has none.

    ``--mem-per-cpu`` multiplies by the layout; ``--mem`` is the per-node total, which for
    the single-node jobs this builder emits is the job's. ``--mem=0`` is SLURM's "all the
    node's memory" — granted as unbounded rather than nothing. With several directives the
    most generous wins, mirroring how sbatch resolves duplicates (last wins) closely enough
    for a sufficiency test that only decides whether to extend.
    """
    granted: int | None = None
    for line in sbatch_lines:
        m = _MEM_DIRECTIVE_RE.search(line)
        if not m:
            continue
        value = _directive_mb(int(m.group(1)), m.group(2))
        if "per-cpu" in m.group(0):
            total = value * ntasks * cpus_per_task
        elif int(m.group(1)) == 0:
            total = sys.maxsize
        else:
            total = value
        granted = max(granted or 0, total)
    return granted


def _apply_memory(
    sbatch_lines: list[str],
    memory_mb: int | None,
    *,
    ntasks: int,
    cpus_per_task: int,
    job_name: str,
) -> list[str]:
    """Honour a header allocation that covers ``memory_mb``; extend one that falls short.

    ``None`` — the engine declares nothing — leaves the header byte-untouched, which is
    what every job got before engines could declare memory. With a requirement, a header
    whose own ``--mem``/``--mem-per-cpu`` already covers it stands as written: the cluster's
    policy wins whenever it is adequate. Only a short or absent allocation is replaced, with
    the derived ``--mem-per-cpu`` and one log line naming both numbers, so an override never
    happens silently.
    """
    if memory_mb is None:
        return sbatch_lines
    granted = _header_memory_mb(sbatch_lines, ntasks=ntasks, cpus_per_task=cpus_per_task)
    if granted is not None and granted >= memory_mb:
        return sbatch_lines
    per_cpu = -(-memory_mb // (ntasks * cpus_per_task))  # ceil
    logger.info(
        "%s: the input declares %d MB; the header grants %s — requesting --mem-per-cpu=%d",
        job_name,
        memory_mb,
        f"{granted} MB" if granted is not None else "no memory",
        per_cpu,
    )
    kept = [line for line in sbatch_lines if not _MEM_DIRECTIVE_RE.search(line)]
    return [*kept, f"#SBATCH --mem-per-cpu={per_cpu}"]


def _run_body_lines(
    *,
    work_dir_expr: str,
    output_dir: Path | str,
    input_path: Path | str,
    header: str,
    footer: str,
    globs_expr: str,
    cleanup: str,
    run_block: RunBlock,
    output_dirs: Sequence[str] = (),
) -> list[str]:
    """The generated bash after the header: scratch setup, on-exit trap, run block.

    Sets up ``$WORK_DIR`` (a fresh ``ts``/``rand``-suffixed dir), copies the input
    in, and installs an ``EXIT`` trap that always copies ``globs_expr`` (files) and
    each ``output_dirs`` entry (a whole directory, e.g. pyscf's ``tensors/``) back
    to ``$OUTPUT_DIR`` and emits the runlog footer — success or failure — before
    running ``run_block.body``.

    **This must remain the only ``trap`` in the generated script.** bash keeps one handler per
    signal, so a second one silently *replaces* this handler and takes the copy-back, the
    ``output_dirs`` copy, the scratch teardown and the footer with it. An engine that needs
    teardown returns it as :attr:`~chemrefine.state.RunBlock.cleanup`, which is interpolated
    *inside* this handler, so there is no reason for an engine to trap at all.

    The handler is armed before ``run_block.body`` runs — it has to be, or a failure inside the
    body would clean up nothing — which means a body that trapped ``EXIT`` anyway would still
    displace it. That rule is held by ``test_no_engine_emits_a_trap_of_its_own``.

    ``TERM``/``INT`` are trapped explicitly so a cancelled job records its real exit code:
    the ``EXIT`` trap does fire on a fatal signal, but with ``$?`` already reset to 0, which
    makes a ``scancel``-ed run look successful in its runlog. ``_cr_done`` keeps the handler
    single-shot, since ``TERM`` then ``EXIT`` would otherwise run it twice.
    """
    dir_copies = [f'  cp -r "{d}" "$OUTPUT_DIR/" 2>/dev/null || true' for d in output_dirs]
    # The engine's teardown, indented into this handler's body. Blank when it has none.
    engine_cleanup = [f"  {line}" for line in run_block.cleanup.splitlines()]
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
        # `set +e` before the engine's teardown, so a cleanup that fails cannot abort the
        # copy-back that follows it.
        "  set +e",
        *engine_cleanup,
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
        run_block.body,
        "",
    ]


def build_script(
    *,
    job_name: str,
    ntasks: int,
    cpus_per_task: int = 1,
    memory_mb: int | None = None,
    template_path: Path,
    script_path: Path,
    input_path: Path,
    output_dir: Path,
    scratch_dir: Path | None,
    run_block: RunBlock,
    engine: str,
    operation: str,
    step: int,
    structure_id: str,
    step_label: str,
    output_globs: Sequence[str],
    output_dirs: Sequence[str] = (),
    extra_header_fields: Sequence[tuple[str, object]] = (),
) -> Path:
    """Assemble a SLURM script at ``script_path`` from a header template + a run block.

    Reads ``template_path`` (a cluster header), strips any ``#SBATCH``
    directives we own (``--ntasks``/``--cpus-per-task``/``--job-name``/
    ``--output``/``--error``) and re-adds them so the core layout + log paths stay
    consistent, then appends a scratch-setup + on-exit trap that runs the
    engine's ``run_block`` in a fresh ``$WORK_DIR`` and copies ``output_globs``
    back to ``output_dir``. Notable args:

    * ``ntasks`` / ``cpus_per_task`` → the SBATCH pair, straight through. The pair is the
      engine's :meth:`~chemrefine.engines.api.JobExecutable.slurm_layout`: MPI ranks are
      ``(pal, 1)``, one threaded process is ``(1, threads)`` — the same core count, spelled
      the way the program will actually use it.
    * ``memory_mb`` → the engine's declared requirement
      (:meth:`~chemrefine.engines.api.JobExecutable.memory_mb`). A header allocation that
      covers it stands untouched; a short or absent one is replaced by the derived
      ``--mem-per-cpu`` — see :func:`_apply_memory`. ``None`` never touches the header.
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
    sbatch_lines = _apply_memory(
        sbatch_lines, memory_mb, ntasks=ntasks, cpus_per_task=cpus_per_task, job_name=job_name
    )
    runlog_path = output_dir / f"{job_name}.runlog"
    err_path = output_dir / f"{job_name}.err"
    sbatch_lines += [
        f"#SBATCH --job-name={job_name}",
        f'#SBATCH --output="{runlog_path}"',
        f'#SBATCH --error="{err_path}"',
        f"#SBATCH --ntasks={ntasks}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
    ]

    header = job_log.bash_header(
        engine=engine,
        operation=operation,
        step=step,
        structure_id=structure_id,
        step_label=step_label,
        step_dir=output_dir,
        cores=ntasks * cpus_per_task,
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
            cleanup=_SCRATCH_CLEANUP,
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
    files: Sequence[JobTriple],
    step_dir: Path,
    *,
    step_label: str,
) -> list[tuple[Path, tuple[JobTriple, ...]]]:
    """Write the per-chunk task manifests; return ``[(manifest_path, chunk), ...]``.

    Line ``i`` of a manifest is the tab-separated ``input<TAB>output<TAB>id``
    of array task ``i``; the generated array script looks its own line up via
    ``$SLURM_ARRAY_TASK_ID``. Returning the file chunks alongside the paths
    lets the caller map every input to its chunk's parent job id.
    """
    manifests: list[tuple[Path, tuple[JobTriple, ...]]] = []
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
    ntasks: int,
    cpus_per_task: int = 1,
    memory_mb: int | None = None,
    template_path: Path,
    script_path: Path,
    output_dir: Path,
    scratch_dir: Path | None,
    run_block: RunBlock,
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
    sbatch_lines = _apply_memory(
        sbatch_lines,
        memory_mb,
        ntasks=ntasks,
        cpus_per_task=cpus_per_task,
        job_name=f"{step_label}_array",
    )
    fallback_log = output_dir / "array_%A_%a.log"
    sbatch_lines += [
        f"#SBATCH --job-name={step_label}_array",
        f'#SBATCH --output="{fallback_log}"',
        f'#SBATCH --error="{fallback_log}"',
        f"#SBATCH --ntasks={ntasks}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
    ]
    # ``$SID`` expands inside the runlog heredoc at runtime, like $(hostname).
    header = job_log.bash_header(
        engine=engine,
        operation=operation,
        step=step,
        structure_id="$SID",
        step_label=step_label,
        step_dir=output_dir,
        cores=ntasks * cpus_per_task,
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

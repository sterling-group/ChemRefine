"""ORCA input file generation from a template + XYZ geometry (the writer half).

The template is a user-provided ORCA ``.inp`` file (e.g. ``step1.inp``)
that declares the method, basis, and any ``%pal`` / ``%scf`` blocks.
This module appends an ``* xyzfile`` directive (pointing at the per-structure
``_inp.xyz``) onto the end of the template, stripping any pre-existing
``* xyzfile`` line first so the template can be reused across steps with
different seed geometries. No ``%base`` is emitted — ORCA defaults the base to
the input filename's stem, keeping its outputs distinct from the input geometry.

Engines that drive ORCA from an external program (MLIP, PySCF) pass
their own ``extra_blocks`` argument — typically a ``%method ... end``
block that points to the external server wrapper.

*Reading* facts from a template (run type, PAL count) is the reader half,
:mod:`chemrefine.engines.orca.inspect`. ``build_input`` clamps any template PAL
declaration to the caller's ``max_pal`` (via :func:`clamp_pal`, reusing the inspect
module's PAL grammar) so the generated input never requests more MPI ranks than its
SLURM allocation grants.
"""

from __future__ import annotations

import re
from pathlib import Path

from chemrefine.engines.orca.inspect import PAL_PATTERNS
from chemrefine.errors import ConfigError

_XYZFILE_DIRECTIVE_RE = re.compile(r"^\s*\*\s+xyzfile.*$", re.MULTILINE)

_QUOTED_PATH_RE = re.compile(r'"([^"\n]+)"')

_JSON_PROP_BLOCK = "%output\n  JSONPropFile True\nend"
"""Ask ORCA (≥ 6) to drop its native ``basename.property.json`` next to the run.

A convenience artifact for users and tooling — chemrefine itself parses the
``.out`` (and writes its own canonical ``*.result.json``). Omitted when the
template already sets ``JSONPropFile`` so a user override wins."""


def _referenced_file(raw: str, template_dir: Path) -> Path | None:
    """The existing file a quoted template string names, or ``None`` if it names none.

    The single rule behind both consumers of a quoted reference — the rewriter that pins
    it into the rendered input and the enumeration the cache key digests it through — so
    a file cannot be pinned into a job yet missing from the step's identity, or the
    reverse. Relative references resolve against the template's own directory (the
    rewriter's contract); absolute ones stand as written; anything that is not an
    existing file (a non-path string, a file created at run time) is not a reference.
    """
    candidate = Path(raw)
    resolved = candidate if candidate.is_absolute() else (template_dir / candidate).resolve()
    return resolved if resolved.is_file() else None


def _absolutize_template_paths(text: str, template_dir: Path) -> str:
    """Rewrite quoted relative file references to absolute paths.

    Templates name auxiliary files relative to their own directory (e.g.
    ``%DOCKER GUEST "../templates/cl.xyz"``), but ORCA runs inside a scratch
    work dir where those paths cannot resolve. Any quoted path that exists
    relative to the template's directory is pinned to its absolute location;
    everything else (absolute paths, non-path strings, files created at run
    time) passes through untouched.
    """

    def _sub(m: re.Match[str]) -> str:
        if Path(m.group(1)).is_absolute():
            return m.group(0)
        resolved = _referenced_file(m.group(1), template_dir)
        return m.group(0) if resolved is None else f'"{resolved}"'

    return _QUOTED_PATH_RE.sub(_sub, text)


def referenced_aux_files(template_path: Path) -> dict[str, Path]:
    """The files ``template_path`` names by quoted reference — as written → as read.

    The cache-key half of :func:`_referenced_file`'s rule: every file this returns has
    its bytes rewritten into the step's identity
    (:meth:`chemrefine.cache.StepKey.of`), because a job's result depends on a guest
    geometry or point-charge file exactly as it depends on the template text — the
    template digests only the *path string*, and editing the referenced file in place
    otherwise changed every job's answer while every fingerprint stood still. Absolute
    references are included on the same grounds: the rewriter leaves them alone, but
    ORCA reads them all the same. A quoted string that names no existing file
    contributes nothing — like :func:`chemrefine.cache.option_file_digests`, "not a
    path" and "missing" both mean no entry, and the key moves when the file appears.

    Keyed by the reference **as the template writes it**, not by the resolved path:
    the resolved path is absolute, and an absolute string in a key breaks the one
    guarantee the cache must keep under a moved tree — a relocated project's
    ``rebuild-cache`` re-derives the same keys from the same bytes. The written
    reference travels with the template (it is already inside the template digest),
    and it also tells two same-content files apart, which a bare set of digests
    cannot.

    An unreadable template answers ``{}`` rather than raising: the missing-template
    story belongs to :func:`chemrefine.cache.template_digest` and
    :func:`chemrefine.ids.require_template`, and this enumeration must not front-run
    their error with its own.
    """
    try:
        text = template_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {}
    references: dict[str, Path] = {}
    for match in _QUOTED_PATH_RE.finditer(text):
        resolved = _referenced_file(match.group(1), template_path.parent)
        if resolved is not None:
            references[match.group(1)] = resolved
    return references


WHITESPACE_PATH_REASON = (
    "ORCA reads each geometry through `* xyzfile <path>`, a whitespace-delimited field it "
    "does not treat as quotable, and execs an ExtOpt wrapper through `sh`, which splits on "
    "whitespace too"
)
"""Why an ORCA-family step cannot be handed a path with whitespace in it.

Quoted verbatim by :func:`require_whitespace_free` and by
:mod:`chemrefine.validate`'s early warning, which reaches it through
:class:`~chemrefine.engines.api.WhitespacePathIntolerant` rather than by importing this
module — so the reason a user reads is written once, here, beside the directive that
imposes it."""


def require_whitespace_free(path: Path, *, what: str) -> None:
    """Raise :class:`~chemrefine.errors.ConfigError` if ``path`` contains whitespace.

    **Verified against ORCA 6.1.1 rather than assumed**, in both directions:

    * ``* xyzfile 0 1 /…/my outputs/step1_0_inp.xyz`` makes ORCA report
      ``CANNOT OPEN FILE`` naming the truncated prefix ``/…/my``. Quoting the value fails
      identically — the field is not quotable.
    * ``%method ProgExt "/…/my ext/wrapper.sh"`` *is* read as a quoted string, and then
      handed to ``sh``: ``sh: 1: /…/my: not found``.
    * By contrast a quoted ``%``-block *filename* is read correctly
      (``%pointcharges "/…/my templates/x.pc"`` → ``Reading point charge file ... ok``),
      which is why this refuses only the paths ChemRefine itself composes and not the
      auxiliary files a template names.

    Checked here, at the moment the path is written into an input, rather than at config
    load. That is what makes it the *resolved* path: :func:`chemrefine.step.step_dir_for`
    calls ``Path.resolve()``, so a directory symlinked through a spaced parent produces a
    spaced path from a config in which no space appears anywhere — invisible to any check
    on the configured value.

    The known gap, stated because it is narrow rather than absent: a template that names
    its *own* ExtOpt wrapper relatively (``ProgExt "extprog.sh"``) has that reference
    absolutised against ``template_dir`` by :func:`_absolutize_template_paths`, so a spaced
    ``template_dir`` can still manufacture a spaced ProgExt. That path is the user's to
    name and ORCA reads every other quoted ``%``-block path with a space correctly, so it
    is left to the ORCA error rather than guessed at here.
    """
    if re.search(r"\s", str(path)):
        raise ConfigError(
            f"{what} contains whitespace: {str(path)!r}. {WHITESPACE_PATH_REASON} — so this "
            f"step cannot run from here. Point `output_dir` at a path without whitespace "
            f"(a relative one inherits the config file's own directory), or move the "
            f"project. Engines that do not write paths into an ORCA input — mlip, pyscf, "
            f"qchem — are unaffected and need no change."
        )


def clamp_pal(text: str, max_pal: int) -> str:
    """Rewrite every PAL / ``nprocs`` declaration above ``max_pal`` down to ``max_pal``.

    The SLURM allocation is clamped to ``max_cores`` at submit time; the
    generated ``.inp`` must declare the same count or ORCA launches more MPI
    ranks than the job owns (``mpirun`` "not enough slots" under SLURM, silent
    oversubscription locally). Declarations at or below the budget pass
    through unchanged.
    """

    def _sub(m: re.Match[str]) -> str:
        return m.group(1) + str(min(int(m.group(2)), max_pal))

    for pattern in PAL_PATTERNS:
        text = pattern.sub(_sub, text)
    return text


def build_input(
    *,
    xyz_path: Path,
    template_path: Path,
    output_path: Path,
    charge: int,
    multiplicity: int,
    extra_blocks: str = "",
    max_pal: int | None = None,
) -> Path:
    """Write an ORCA ``.inp`` to ``output_path``; return that path.

    No explicit ``%base`` is emitted: ORCA defaults the base to the input
    filename's stem, so all derived files (``.out``, ``.engrad``, optimized
    ``.xyz``, ``.gbw`` …) share that stem while staying distinct from the
    ``_inp.xyz`` input geometry (``xyz_path``). ``max_pal`` (the engine passes
    ``Config.max_cores``) clamps any PAL declaration in the template via
    :func:`clamp_pal` so the input never asks for more ranks than the SLURM
    allocation grants.
    """
    if not template_path.is_file():
        raise ConfigError(f"ORCA template not found: {template_path}")
    require_whitespace_free(xyz_path, what="the ORCA geometry path")

    template = template_path.read_text(encoding="utf-8")
    cleaned = _XYZFILE_DIRECTIVE_RE.sub("", template).rstrip()
    cleaned = _absolutize_template_paths(cleaned, template_path.parent)
    if max_pal is not None:
        cleaned = clamp_pal(cleaned, max_pal)

    lines = [cleaned, ""]
    if "jsonpropfile" not in cleaned.lower():
        lines.extend([_JSON_PROP_BLOCK, ""])
    extra = extra_blocks.strip()
    if extra:
        lines.extend([extra, ""])
    # No explicit %base: ORCA defaults the base to the input filename's stem
    # (``step{N}_{id}``), so its outputs (optimized ``.xyz``, ``.gbw``, ``.hess``)
    # are named distinctly from the ``_inp.xyz`` input and both survive copy-back.
    lines.append(f"* xyzfile {charge} {multiplicity} {xyz_path}")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path

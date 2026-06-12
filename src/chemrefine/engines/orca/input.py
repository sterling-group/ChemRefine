"""ORCA input file generation from a template + XYZ geometry.

The template is a user-provided ORCA ``.inp`` file (e.g. ``step1.inp``)
that declares the method, basis, and any ``%pal`` / ``%scf`` blocks.
This module pastes a fresh ``%base`` line and an ``* xyzfile`` directive
onto the end of the template, stripping any pre-existing ``* xyzfile``
line first so the template can be reused across steps with different
seed geometries.

Engines that drive ORCA from an external program (MLIP, PySCF) pass
their own ``extra_blocks`` argument — typically a ``%method ... end``
block that points to the external server wrapper.

The :func:`parse_pal` / :func:`clamp_pal` helpers also live here so
PAL-budget extraction and enforcement are co-located with the rest of
the ORCA input vocabulary — generic SLURM machinery is engine-agnostic
and must not parse ORCA's ``%pal`` directive itself. ``build_input``
clamps any template PAL declaration to the caller's ``max_pal`` so the
generated input never requests more MPI ranks than its SLURM
allocation grants.
"""

from __future__ import annotations

import re
from pathlib import Path

_XYZFILE_DIRECTIVE_RE = re.compile(r"^\s*\*\s+xyzfile.*$", re.MULTILINE)

# Every spelling of an ORCA PAL declaration, as ``(prefix)(count)`` pairs so
# :func:`parse_pal` reads the count and :func:`clamp_pal` rewrites it in place.
_PAL_PATTERNS = (
    re.compile(r"(nprocs\s+)(\d+)", re.IGNORECASE),
    re.compile(r"\b(PAL)(\d+)\b", re.IGNORECASE),
    re.compile(r"(^\s*PAL\s+)(\d+)\b", re.IGNORECASE | re.MULTILINE),
)


def parse_pal(input_file: str | Path) -> int:
    """Return the PAL / ``nprocs`` value declared in an ORCA input or template.

    Per-structure ``.inp`` files inherit their ``%pal`` block from the
    step template, so callers typically pass the template path once
    per step rather than re-reading every generated copy. Falls back
    to ``1`` when no PAL directive is found, matching ORCA's own
    default for serial runs.
    """
    text = Path(input_file).read_text(encoding="utf-8")
    for pattern in _PAL_PATTERNS:
        m = pattern.search(text)
        if m:
            return int(m.group(2))
    return 1


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

    for pattern in _PAL_PATTERNS:
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

    ``output_path``'s stem becomes the ORCA ``%base`` so all derived
    files (``.out``, ``.engrad``, ``.xyz``, ``.gbw`` …) share a prefix.
    ``max_pal`` (the engine passes ``Config.max_cores``) clamps any PAL
    declaration in the template via :func:`clamp_pal` so the input never
    asks for more ranks than the SLURM allocation grants.
    """
    if not template_path.is_file():
        raise FileNotFoundError(f"ORCA template not found: {template_path}")

    template = template_path.read_text(encoding="utf-8")
    cleaned = _XYZFILE_DIRECTIVE_RE.sub("", template).rstrip()
    if max_pal is not None:
        cleaned = clamp_pal(cleaned, max_pal)

    lines = [cleaned, ""]
    extra = extra_blocks.strip()
    if extra:
        lines.extend([extra, ""])
    lines.append(f'%base "{output_path.stem}"')
    lines.append(f"* xyzfile {charge} {multiplicity} {xyz_path}")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path

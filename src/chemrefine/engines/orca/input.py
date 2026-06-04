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

The :func:`parse_pal` helper also lives here so PAL-budget extraction is
co-located with the rest of the ORCA input vocabulary — generic SLURM
machinery is engine-agnostic and must not parse ORCA's ``%pal``
directive itself.
"""

from __future__ import annotations

import re
from pathlib import Path

_XYZFILE_DIRECTIVE_RE = re.compile(r"^\s*\*\s+xyzfile.*$", re.MULTILINE)

_PAL_PATTERNS = (
    re.compile(r"nprocs\s+(\d+)", re.IGNORECASE),
    re.compile(r"\bPAL(\d+)\b", re.IGNORECASE),
    re.compile(r"^\s*PAL\s+(\d+)\b", re.IGNORECASE | re.MULTILINE),
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
            return int(m.group(1))
    return 1


def build_input(
    *,
    xyz_path: Path,
    template_path: Path,
    output_path: Path,
    charge: int,
    multiplicity: int,
    extra_blocks: str = "",
) -> Path:
    """Write an ORCA ``.inp`` to ``output_path``; return that path.

    ``output_path``'s stem becomes the ORCA ``%base`` so all derived
    files (``.out``, ``.engrad``, ``.xyz``, ``.gbw`` …) share a prefix.
    """
    if not template_path.is_file():
        raise FileNotFoundError(f"ORCA template not found: {template_path}")

    template = template_path.read_text(encoding="utf-8")
    cleaned = _XYZFILE_DIRECTIVE_RE.sub("", template).rstrip()

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

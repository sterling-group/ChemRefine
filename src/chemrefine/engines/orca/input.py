"""ORCA input file generation from a template + XYZ geometry.

The template is a user-provided ORCA ``.inp`` file (e.g. ``step1.inp``)
that declares the method, basis, and any ``%pal`` / ``%scf`` blocks.
This module pastes a fresh ``%base`` line and an ``* xyzfile`` directive
onto the end of the template, stripping any pre-existing ``* xyzfile``
line first so the template can be reused across steps with different
seed geometries.

Engines that drive ORCA from an external program (MLFF, PySCF) pass
their own ``extra_blocks`` argument — typically a ``%method ... end``
block that points to the external server wrapper.
"""

from __future__ import annotations

import re
from pathlib import Path

_XYZFILE_DIRECTIVE_RE = re.compile(r"^\s*\*\s+xyzfile.*$", re.MULTILINE)


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

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

from chemrefine.engines.orca.inspect import _PAL_PATTERNS

_XYZFILE_DIRECTIVE_RE = re.compile(r"^\s*\*\s+xyzfile.*$", re.MULTILINE)


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

    No explicit ``%base`` is emitted: ORCA defaults the base to the input
    filename's stem, so all derived files (``.out``, ``.engrad``, optimized
    ``.xyz``, ``.gbw`` …) share that stem while staying distinct from the
    ``_inp.xyz`` input geometry (``xyz_path``). ``max_pal`` (the engine passes
    ``Config.max_cores``) clamps any PAL declaration in the template via
    :func:`clamp_pal` so the input never asks for more ranks than the SLURM
    allocation grants.
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
    # No explicit %base: ORCA defaults the base to the input filename's stem
    # (``step{N}_{id}``), so its outputs (optimized ``.xyz``, ``.gbw``, ``.hess``)
    # are named distinctly from the ``_inp.xyz`` input and both survive copy-back.
    lines.append(f"* xyzfile {charge} {multiplicity} {xyz_path}")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path

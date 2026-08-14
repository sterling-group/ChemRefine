"""Q-Chem input generation from a template + XYZ geometry (the writer half).

The template is a user-provided Q-Chem input (e.g. ``step1.in``) declaring the ``$rem``
settings; this module puts the per-structure geometry into it. Unlike ORCA — whose
``* xyzfile`` directive is appended at the end, because ORCA reads geometry from a named
file wherever the directive sits — Q-Chem takes its geometry **inline**, in the first job's
``$molecule`` block, and a multi-job ``@@@`` template's later jobs read it back with
``$molecule read $end``. So the generated block must land in job 1 and nowhere else:

* a template with a ``$molecule`` block gets its **first** one replaced in place — first
  whatever its body, because a job-1 ``read`` has nothing to read from in the fresh scratch
  each per-structure job runs in, while every *later* block (the ``read`` idiom) survives
  untouched;
* a template with none gets the generated block prepended, which is job 1 by construction.

Q-Chem reads its sections in any order within a job, so the replacement never has to move a
block — position is preserved, and the diff between template and rendered input is exactly
the geometry.
"""

from __future__ import annotations

import re
from pathlib import Path

from chemrefine.errors import ConfigError
from chemrefine.io import read_xyz_frames

_MOLECULE_BLOCK_RE = re.compile(r"\$molecule\b.*?\$end", re.IGNORECASE | re.DOTALL)


def _molecule_block(xyz_path: Path, charge: int, multiplicity: int) -> str:
    """Render the ``$molecule`` block for one structure's ``_inp.xyz`` geometry.

    Rows are formatted like every geometry this package writes
    (:func:`chemrefine.io.write_single_xyz`'s six decimals) — the ``_inp.xyz`` this reads is
    already written at that precision, so nothing is lost restating it.
    """
    atoms = read_xyz_frames(xyz_path)[0]
    rows = [
        f"{symbol:2s} {x:.6f} {y:.6f} {z:.6f}"
        for symbol, (x, y, z) in zip(
            atoms.get_chemical_symbols(), atoms.get_positions(), strict=True
        )
    ]
    return "\n".join([r"$molecule", f"{charge} {multiplicity}", *rows, r"$end"])


def build_input(
    *,
    xyz_path: Path,
    template_path: Path,
    output_path: Path,
    charge: int,
    multiplicity: int,
) -> Path:
    """Write a Q-Chem input to ``output_path``; return that path.

    The geometry replaces the template's first ``$molecule … $end`` block in place, or is
    prepended when the template declares none — see the module docstring for why job 1 and
    only job 1. Everything else in the template, including any later ``$molecule read $end``
    of a ``@@@`` chain, passes through byte-for-byte.
    """
    if not template_path.is_file():
        raise ConfigError(f"Q-Chem template not found: {template_path}")
    template = template_path.read_text(encoding="utf-8")
    block = _molecule_block(xyz_path, charge, multiplicity)
    rendered, replaced = _MOLECULE_BLOCK_RE.subn(lambda _m: block, template, count=1)
    if not replaced:
        rendered = f"{block}\n\n{template.lstrip()}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered.rstrip() + "\n", encoding="utf-8")
    return output_path

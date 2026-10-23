"""Read an xtb XYZ geometry: the optimised ``xtbopt.xyz``, or a structure's input geometry.

xtb writes the optimised geometry in its input's format. ChemRefine hands it an XYZ file,
so ``xtbopt.xyz`` is XYZ in Å: an atom count, an extended-XYZ comment line (xtb's
``Lattice=`` / ``Properties=`` keys and an ``energy:`` note), and one ``symbol x y z`` row
per atom. The same reader reads the structure's ``_inp.xyz``, which is the geometry of
every run that does not move the atoms.

Read by hand rather than through ASE's extended-XYZ reader: that reader would make a cell
of xtb's all-zero ``Lattice`` and judge the comment line's keys, where this needs only the
rows, held to the atom count and to finite values.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def parse_xyz_text(text: str) -> tuple[tuple[str, ...], NDArray[np.float64]]:
    """``(symbols, positions in Å)`` from the first frame of an XYZ file's text.

    Raises :class:`ValueError` when the atom count does not read, a row does not read or
    is not finite, or there are fewer rows than the count: a file cut off mid-write would
    otherwise be a smaller molecule.
    """
    lines = text.splitlines()
    try:
        count = int(lines[0].split()[0])
    except (IndexError, ValueError):
        raise ValueError("no atom count on the first line") from None
    rows = lines[2 : 2 + count]
    if count < 1 or len(rows) < count:
        raise ValueError(f"{len(rows)} coordinate rows for {count} atoms")
    symbols: list[str] = []
    positions: list[list[float]] = []
    for line in rows:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"unreadable row {line.strip()!r}")
        xyz = [float(token) for token in parts[1:4]]
        if not np.isfinite(xyz).all():
            raise ValueError(f"non-finite coordinate in {line.strip()!r}")
        symbols.append(parts[0])
        positions.append(xyz)
    return tuple(symbols), np.array(positions, dtype=np.float64)

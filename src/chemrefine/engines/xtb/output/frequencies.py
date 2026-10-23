"""Read xtb's ``g98.out``: the vibrational frequencies and normal modes, the NMS feed.

A Hessian run writes a Gaussian-98-style frequency file beside its output. It lists the
vibrational modes only, three to a block::

                          1                      2                      3
                            a                      a                      a
     Frequencies --  1443.8562              3594.8048              3596.8747
     ...
     Atom AN      X      Y      Z        X      Y      Z        X      Y      Z
       1   8     0.00   0.00   0.27    -0.00  -0.27  -0.00    -0.00   0.00  -0.19
       ...

**The mode-index contract.** The table and the tensor share one 0-based index over all
3N Cartesian modes, as ORCA's do. The modes ``g98.out`` leaves out are the translations
and rotations: 3N - n_vib of them (six, or five for a linear molecule), which take the
first indices. So vibrational mode ``k`` (``g98.out`` counts from 1) is index
``(3N - n_vib) + k - 1``. Their tensor columns stay zero, and no table entry names them.
An imaginary mode is a **negative** frequency.

The displacements are unit-length Cartesian vectors printed to two decimals. Each column
is scaled back to unit length, so a displacement along it is the length NMS asked for.
"""

from __future__ import annotations

import re

import numpy as np
from numpy.typing import NDArray

_FREQUENCIES_RE = re.compile(r"^\s*Frequencies --(.*)$")
_ATOM_HEADER_RE = re.compile(r"^\s*Atom\s+AN(?:\s+X\s+Y\s+Z)+\s*$")


def parse_g98_text(
    text: str, *, n_atoms: int
) -> tuple[dict[int, float], dict[int, float], NDArray[np.float64]]:
    """``(imaginary, table, tensor)`` from a ``g98.out``'s text.

    ``table`` maps each vibrational mode's index (see the module docstring) to cm⁻¹;
    ``imaginary`` is its subset with negative frequencies (``{}`` for a minimum);
    ``tensor`` is ``(n_atoms, 3, 3·n_atoms)``. Raises :class:`ValueError` when no mode
    block reads, a block's displacement rows do not number ``n_atoms``, a value is not
    finite, or there are more modes than 3N - 5 — each a file that cannot be the
    molecule's.
    """
    lines = text.splitlines()
    frequencies: list[float] = []
    columns: list[NDArray[np.float64]] = []
    for i, line in enumerate(lines):
        found = _FREQUENCIES_RE.match(line)
        if found is None:
            continue
        # Every token is read, and one that does not read refuses the file: a block skipped
        # would shift each later mode onto another mode's index.
        values = [float(token) for token in found.group(1).split()]
        if not values:
            raise ValueError(f"an empty frequency row {line.strip()!r}")
        frequencies += values
        columns += _block_columns(lines, i + 1, n_atoms=n_atoms, n_modes=len(values))
    if not frequencies:
        raise ValueError("no `Frequencies --` row")
    if not np.isfinite(frequencies).all():
        raise ValueError("a frequency is not finite")
    n_modes = 3 * n_atoms
    trivial = n_modes - len(frequencies)
    if trivial < 5:
        raise ValueError(f"{len(frequencies)} vibrational modes for {n_atoms} atoms")
    table = {trivial + k: value for k, value in enumerate(frequencies)}
    imaginary = {index: value for index, value in table.items() if value < 0.0}
    tensor = np.zeros((n_atoms, 3, n_modes), dtype=np.float64)
    for k, column in enumerate(columns):
        norm = np.linalg.norm(column)
        if norm == 0.0:
            raise ValueError(f"vibrational mode {k + 1} has no displacement")
        tensor[:, :, trivial + k] = column / norm
    return imaginary, table, tensor


def _block_columns(
    lines: list[str], start: int, *, n_atoms: int, n_modes: int
) -> list[NDArray[np.float64]]:
    """The ``(n_atoms, 3)`` displacement of each of one block's modes, read from ``start``."""
    header = next((j for j in range(start, len(lines)) if _ATOM_HEADER_RE.match(lines[j])), None)
    if header is None:
        raise ValueError("a frequency block has no `Atom AN` displacement table")
    rows: list[list[float]] = []
    for line in lines[header + 1 : header + 1 + n_atoms]:
        parts = line.split()
        if len(parts) != 2 + 3 * n_modes:
            raise ValueError(f"unreadable displacement row {line.strip()!r}")
        rows.append([float(token) for token in parts[2:]])
    if len(rows) != n_atoms:
        raise ValueError(f"{len(rows)} displacement rows for {n_atoms} atoms")
    block = np.array(rows, dtype=np.float64)
    if not np.isfinite(block).all():
        raise ValueError("a displacement is not finite")
    return [block[:, 3 * m : 3 * m + 3] for m in range(n_modes)]

"""Parse the molecular geometry (``CARTESIAN COORDINATES`` block) from an ORCA ``.out``.

One section, one module: this owns the coordinate-block grammar. The
:mod:`chemrefine.engines.orca.output` coordinator reads the file once and calls this over the
shared text.
"""

from __future__ import annotations

import re

import numpy as np
from numpy.typing import NDArray

# Last block, since geometry optimisation re-prints the table as it iterates.
_COORD_BLOCK_RE = re.compile(
    r"CARTESIAN COORDINATES\s+\(ANGSTROEM\)\s*\n-+\n((?:.*?\n)+?)-+\n",
    re.DOTALL,
)


def parse_coordinates_from_text(
    text: str,
) -> tuple[tuple[str, ...], NDArray[np.float64]] | None:
    """Return ``(symbols, positions)`` from the **last** coordinates block, or ``None``.

    ``None`` means the block is absent. Raises :class:`ValueError` on a malformed coordinate
    row (e.g. a ``*****`` overflow token) — the coordinator turns that into an
    :class:`~chemrefine.errors.OutputParseError` with file context.
    """
    blocks = _COORD_BLOCK_RE.findall(text)
    if not blocks:
        return None
    return _parse_coord_block(blocks[-1])


def _parse_coord_block(block: str) -> tuple[tuple[str, ...], NDArray[np.float64]]:
    """Parse the body of a ``CARTESIAN COORDINATES (ANGSTROEM)`` block."""
    symbols: list[str] = []
    positions: list[list[float]] = []
    for line in block.strip().splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        symbols.append(parts[0])
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return tuple(symbols), np.array(positions, dtype=np.float64)

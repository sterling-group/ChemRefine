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
    row — an unparseable token (``*****`` overflow) or a non-finite one (``nan`` / ``inf``,
    which ``float()`` accepts) — and the coordinator turns that into an
    :class:`~chemrefine.errors.OutputParseError` with file context.
    """
    blocks = _COORD_BLOCK_RE.findall(text)
    if not blocks:
        return None
    return _parse_coord_block(blocks[-1])


def _parse_coord_block(block: str) -> tuple[tuple[str, ...], NDArray[np.float64]]:
    """Parse the body of a ``CARTESIAN COORDINATES (ANGSTROEM)`` block.

    Raises :class:`ValueError` on a row that is not three finite numbers. ``float()``
    already does that for an overflow token (``*****``), but it *accepts* ``nan`` and
    ``inf`` — and a non-finite coordinate is the same unusable geometry by a quieter
    route. Raising the same exception for both means the caller's existing handler turns
    them into the same :class:`~chemrefine.errors.OutputParseError`.
    """
    symbols: list[str] = []
    positions: list[list[float]] = []
    for line in block.strip().splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        row = [float(parts[1]), float(parts[2]), float(parts[3])]
        if not np.isfinite(row).all():
            raise ValueError(f"non-finite coordinate in {line.strip()!r}")
        symbols.append(parts[0])
        positions.append(row)
    return tuple(symbols), np.array(positions, dtype=np.float64)

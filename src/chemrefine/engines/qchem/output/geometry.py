"""Parse the molecular geometry from a Q-Chem ``.out``.

One section, one module: this owns the ``Standard Nuclear Orientation`` grammar, and the
coordinator reads the file once and calls this over the shared text.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

ORIENTATION_MARKER = "Standard Nuclear Orientation (Angstroms)"
"""The banner every geometry block opens with, **unit suffix included** — public because
the coordinator names it in the error a block-less output raises.

The suffix is the family rule (stated in this package's docstring): a geometry banner
pins its units, or the parser refuses. Q-Chem prints ``(Bohr)`` here under
``$rem input_bohr true``, and matched without the suffix those values landed unconverted
in :class:`~chemrefine.engines.api.ParsedResult.positions`, whose contract is Å — a
geometry silently wrong by a factor of 0.529 everywhere downstream. Unmatched, a Bohr
output is an ordinary "no block" refusal instead (and the input writer refuses the rem
up front — :func:`chemrefine.engines.qchem.input.build_input`). The ORCA counterpart
pins ``(ANGSTROEM)`` the same way."""


def parse_orientation_from_text(
    text: str,
) -> tuple[tuple[str, ...], NDArray[np.float64]] | None:
    """``(symbols, positions)`` from the **last** orientation block, or ``None``.

    A geometry optimisation re-prints the block per cycle; the last describes the
    geometry the final energy belongs to. ``None`` means no block at all — the
    coordinator decides what that missing section means for the run. Rows are
    ``index symbol x y z``; the column-header and dashed-separator lines around them
    never parse as rows, and the closing separator (or the next section) ends the block.

    Raises :class:`ValueError` on a malformed coordinate row — an unparseable token
    (``*****`` overflow) or a non-finite one (``nan`` / ``inf``, which ``float()``
    accepts) — and the coordinator turns that into an
    :class:`~chemrefine.errors.OutputParseError` with file context, so the reader stays
    pure text-to-values. An empty block (the marker with no atom rows)
    returns an empty tuple for the coordinator to refuse: the marker alone proves
    nothing.
    """
    start = text.rfind(ORIENTATION_MARKER)
    if start == -1:
        return None
    symbols: list[str] = []
    rows: list[list[float]] = []
    for line in text[start:].splitlines()[1:]:
        parts = line.split()
        if len(parts) == 5 and parts[0].isdecimal():
            row = [float(parts[2]), float(parts[3]), float(parts[4])]
            # `float()` rejects a `*****` overflow but accepts `nan` / `inf`, and a
            # non-finite coordinate is the same unusable geometry by a quieter route.
            # Raising the same exception for both means the coordinator's one handler
            # turns them into the same OutputParseError.
            if not np.isfinite(row).all():
                raise ValueError(f"non-finite coordinate in {line.strip()!r}")
            rows.append(row)
            symbols.append(parts[1])
            continue
        if symbols:
            break  # the closing separator (or the next section) ends the block
    return tuple(symbols), np.array(rows, dtype=np.float64)

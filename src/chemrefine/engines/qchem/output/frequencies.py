"""Q-Chem frequency-block parsing: the vibrational table + the normal-mode tensor.

One section, one module — the NMS feed: ``imaginary_freqs`` and ``normal_modes`` are
what :mod:`chemrefine.nms` displaces along, parsed in the same pass as everything else.

**The mode-index contract.** Q-Chem's ``VIBRATIONAL ANALYSIS`` prints only the 3N-6
vibrational modes, numbered from 1 — translations and rotations are already projected
out. ChemRefine's NMS contract counts the opposite way: the tensor carries the six
trivial modes *first* (:class:`~chemrefine.engines.api.NmsCapableEngine`,
``_TRIVIAL_MODES`` in :mod:`chemrefine.nms`), and ``random`` sampling skips the leading
six by index. So this parser pads six zero columns in front and maps Q-Chem's 1-based
mode ``k`` to tensor index ``k + 5`` — a zero column can never be selected
(``minimum``/``ts`` displace only along imaginary indices, ``random`` draws real
vibrations, and a zero displacement would move nothing), so the padding is inert by
construction. An imaginary mode is a **negative** ``Frequency:`` value.
"""

from __future__ import annotations

import re

import numpy as np
from numpy.typing import NDArray

VIB_MARKER = "VIBRATIONAL ANALYSIS"
"""The banner the frequency scan anchors on — public because the coordinator's docstring
names the section it dispatches to this module."""

_MODE_LINE_RE = re.compile(r"^\s*Mode:\s+(\d+(?:\s+\d+)*)\s*$")
_FREQ_LINE_RE = re.compile(r"^\s*Frequency:\s+(-?\d+\.\d+(?:\s+-?\d+\.\d+)*)\s*$")
_DISPLACEMENT_HEADER_RE = re.compile(r"^\s*(?:X\s+Y\s+Z\s*)+$")

_TRIVIAL_MODES = 6
"""Leading zero-padded columns of the tensor — the NMS ordering contract's trivial modes.

The same constant as :mod:`chemrefine.nms`'s, restated rather than imported: the engine
subsystem sits below the NMS coordinator and must not import it.
"""


def parse_frequency_block(
    text: str, *, n_atoms: int
) -> tuple[dict[int, float] | None, dict[int, float] | None, NDArray[np.float64] | None]:
    """Imaginary modes, the whole table, and the padded displacement tensor.

    ``None`` for all three when the output has no ``VIBRATIONAL ANALYSIS`` at all **or**
    the section is there but no mode block parses — a job truncated or died mid-print.
    Both are "no data", distinct from ``{}`` = a parsed table with zero imaginary modes —
    a verified minimum; conflating the truncated case with ``{}`` called a killed freq
    job a minimum, silently, through ``get_frequencies``' ``imaginary_count: 0``. The
    tensor alone is ``None`` when only the displacement rows do not parse.
    ``rpartition`` takes the **last** analysis, matching the energy and geometry readers
    — in an ``@@@`` chain that is the freq job's.
    """
    _head, sep, tail = text.rpartition(VIB_MARKER)
    if not sep:
        return None, None, None
    freqs: dict[int, float] = {}
    columns: dict[int, list[list[float]]] = {}
    lines = tail.splitlines()
    i = 0
    while i < len(lines):
        mode_match = _MODE_LINE_RE.match(lines[i])
        if not mode_match:
            i += 1
            continue
        indices = [int(token) for token in mode_match.group(1).split()]
        i += 1
        values = _frequency_values(lines, i, expected=len(indices))
        if values is None:
            continue  # a Mode: line without its Frequency: row — skip the block
        for index, value in zip(indices, values, strict=True):
            freqs[index] = value
        i, block_rows = _displacement_rows(lines, i, n_atoms=n_atoms, n_modes=len(indices))
        if block_rows is not None:
            for column, index in enumerate(indices):
                columns[index] = [row[3 * column : 3 * column + 3] for row in block_rows]
    if not freqs:
        # marker present, nothing parsed: no data, not a verified minimum
        return None, None, None
    # Q-Chem numbers only the non-trivial modes, from 1; the NMS index space counts the
    # six translations and rotations first. The whole table is shifted into that space
    # once, and the imaginary subset taken from it, so the two cannot come to disagree.
    table = {index + _TRIVIAL_MODES - 1: v for index, v in freqs.items()}
    imaginary = {index: v for index, v in table.items() if v < 0.0}
    if not columns:
        return imaginary, table, None
    tensor = np.zeros((n_atoms, 3, _TRIVIAL_MODES + max(freqs)), dtype=np.float64)
    for index, block in columns.items():
        tensor[:, :, index + _TRIVIAL_MODES - 1] = np.array(block, dtype=np.float64)
    return imaginary, table, tensor


def _frequency_values(lines: list[str], start: int, *, expected: int) -> list[float] | None:
    """The ``Frequency:`` row's values for one mode block, or ``None`` if it is not there."""
    for j in range(start, min(start + 3, len(lines))):
        m = _FREQ_LINE_RE.match(lines[j])
        if m:
            values = [float(token) for token in m.group(1).split()]
            return values if len(values) == expected else None
    return None


def _displacement_rows(
    lines: list[str], start: int, *, n_atoms: int, n_modes: int
) -> tuple[int, list[list[float]] | None]:
    """One block's per-atom displacement rows; ``(next_line, None)`` when they don't parse.

    The block is the ``X Y Z`` column header followed by one row per atom (symbol + three
    components per mode); the ``TransDip`` row after them never matches the width test.
    """
    j = start
    while j < len(lines) and not _DISPLACEMENT_HEADER_RE.match(lines[j]):
        if _MODE_LINE_RE.match(lines[j]):
            return j, None  # next block began — this one had no displacement table
        j += 1
    rows: list[list[float]] = []
    j += 1
    while j < len(lines) and len(rows) < n_atoms:
        parts = lines[j].split()
        if len(parts) == 1 + 3 * n_modes and not parts[0].startswith("TransDip"):
            try:
                row = [float(token) for token in parts[1:]]
                # `float()` accepts ``nan`` / ``inf``. A non-finite displacement would ride
                # the tensor into `nms.displace_along_mode` and put NaN into every child
                # geometry it builds, so it withholds the tensor like a corrupt token does.
                if not np.isfinite(row).all():
                    return j, None
            except ValueError:
                return j, None
            rows.append(row)
        j += 1
    return j, rows if len(rows) == n_atoms else None

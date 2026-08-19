"""Parse a Q-Chem ``.out``: energy + final geometry, and the frequency section for NMS.

Two halves with two different standings:

* **Energy, geometry, status — a placeholder.** The last ``Total energy in the final basis
  set`` and the last ``Standard Nuclear Orientation`` block (both repeat per optimisation
  cycle — last match wins, the discipline every ORCA reader follows), with
  ``converged``/``terminated_normally`` left ``None`` ("no signal is not a failure
  signal"). The full parsers — status banners, thermochemistry, correlated energies,
  per-operation dispatch — are being written separately and replace/extend this half; the
  submission path only needs a structure to come back.
* **Frequencies — the real NMS feed.** ``imaginary_freqs`` and ``normal_modes`` are what
  :mod:`chemrefine.nms` displaces along, parsed here in the same pass.

**The mode-index contract.** Q-Chem's ``VIBRATIONAL ANALYSIS`` prints only the 3N-6
vibrational modes, numbered from 1 — translations and rotations are already projected out.
ChemRefine's NMS contract is the opposite convention: the tensor carries the six trivial
modes *first* (:class:`~chemrefine.engines.api.NmsCapableEngine`, ``_TRIVIAL_MODES`` in
:mod:`chemrefine.nms`), because ORCA prints all 3N and ``random`` sampling skips the leading
six by index. So this parser pads six zero columns in front and maps Q-Chem's 1-based mode
``k`` to tensor index ``k + 5`` — a zero column can never be selected (``minimum``/``ts``
displace only along imaginary indices, ``random`` draws real vibrations, and a zero
displacement would move nothing), so the padding is inert by construction. An imaginary
mode is a **negative** ``Frequency:`` value.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from chemrefine.engines.api import ParsedResult
from chemrefine.errors import OutputParseError

_FINAL_ENERGY_RE = re.compile(r"Total energy in the final basis set =\s*(-?\d+\.\d+)")
_ORIENTATION_MARKER = "Standard Nuclear Orientation"
_VIB_MARKER = "VIBRATIONAL ANALYSIS"
_MODE_LINE_RE = re.compile(r"^\s*Mode:\s+(\d+(?:\s+\d+)*)\s*$")
_FREQ_LINE_RE = re.compile(r"^\s*Frequency:\s+(-?\d+\.\d+(?:\s+-?\d+\.\d+)*)\s*$")
_DISPLACEMENT_HEADER_RE = re.compile(r"^\s*(?:X\s+Y\s+Z\s*)+$")

_TRIVIAL_MODES = 6
"""Leading zero-padded columns of the tensor — the NMS ordering contract's trivial modes.

The same constant as :mod:`chemrefine.nms`'s, restated rather than imported: the engine
subsystem sits below the NMS coordinator and must not import it.
"""


def parse_qchem(path: str | Path) -> list[ParsedResult]:
    """Parse one Q-Chem ``.out`` into a single-element ``[ParsedResult]``."""
    return parse_qchem_text(Path(path).read_text(encoding="utf-8", errors="replace"), src=str(path))


def parse_qchem_text(text: str, *, src: str = "<text>") -> list[ParsedResult]:
    """Assemble one :class:`ParsedResult` from a single read of a Q-Chem ``.out``.

    Raises :class:`OutputParseError` when the energy or the geometry is missing or
    malformed. ``src`` only labels error messages.
    """
    energies = _FINAL_ENERGY_RE.findall(text)
    if not energies:
        raise OutputParseError(f"no 'Total energy in the final basis set' in {src}")
    symbols, positions = _last_orientation(text, src)
    imaginary, modes = _parse_frequencies(text, n_atoms=len(symbols))
    return [
        ParsedResult(
            symbols=symbols,
            positions=positions,
            energy_hartree=float(energies[-1]),
            forces_ev_per_a=None,
            imaginary_freqs=imaginary,
            normal_modes=modes,
        )
    ]


def _last_orientation(text: str, src: str) -> tuple[tuple[str, ...], NDArray[np.float64]]:
    """``(symbols, positions)`` from the **last** ``Standard Nuclear Orientation`` block.

    A geometry optimisation re-prints the block per cycle; the last describes the geometry
    the final energy belongs to. Rows are ``index symbol x y z``; the column-header and
    dashed-separator lines around them never parse as rows, and the closing separator ends
    the block.
    """
    start = text.rfind(_ORIENTATION_MARKER)
    if start == -1:
        raise OutputParseError(f"no '{_ORIENTATION_MARKER}' block in {src}")
    symbols: list[str] = []
    rows: list[list[float]] = []
    for line in text[start:].splitlines()[1:]:
        parts = line.split()
        if len(parts) == 5 and parts[0].isdigit():
            try:
                row = [float(parts[2]), float(parts[3]), float(parts[4])]
                # `float()` rejects a `*****` overflow but accepts `nan` / `inf`, and a
                # non-finite coordinate is the same unusable geometry by a quieter route.
                if not np.isfinite(row).all():
                    raise ValueError(f"non-finite coordinate in {line.strip()!r}")
            except ValueError as e:
                raise OutputParseError(f"malformed coordinate row in {src}: {e}") from e
            rows.append(row)
            symbols.append(parts[1])
            continue
        if symbols:
            break  # the closing separator (or the next section) ends the block
    if not symbols:
        raise OutputParseError(f"'{_ORIENTATION_MARKER}' block in {src} has no atoms")
    return tuple(symbols), np.array(rows, dtype=np.float64)


def _parse_frequencies(
    text: str, *, n_atoms: int
) -> tuple[dict[int, float] | None, NDArray[np.float64] | None]:
    """Imaginary modes + padded displacement tensor, or ``(None, None)`` without the section.

    ``None`` for both when the output has no ``VIBRATIONAL ANALYSIS`` at all (distinct from
    ``{}`` = a frequency section with zero imaginary modes — a verified minimum); the tensor
    alone is ``None`` when its rows do not parse. ``rpartition`` takes the **last** analysis,
    matching the energy and geometry readers — in an ``@@@`` chain that is the freq job's.
    """
    _head, sep, tail = text.rpartition(_VIB_MARKER)
    if not sep:
        return None, None
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
    imaginary = {index + _TRIVIAL_MODES - 1: v for index, v in freqs.items() if v < 0.0}
    if not freqs or not columns:
        return imaginary, None
    tensor = np.zeros((n_atoms, 3, _TRIVIAL_MODES + max(freqs)), dtype=np.float64)
    for index, block in columns.items():
        tensor[:, :, index + _TRIVIAL_MODES - 1] = np.array(block, dtype=np.float64)
    return imaginary, tensor


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

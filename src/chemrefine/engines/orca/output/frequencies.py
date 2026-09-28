"""ORCA frequency-block parsing: the vibrational table + the normal-mode tensor.

One section, one module. Thermochemistry — Gibbs / enthalpy / ZPE — lives in
:mod:`chemrefine.engines.orca.output.energy`, since those are energies. The
:mod:`chemrefine.engines.orca.output.coordinator` calls these over the text it already read.

The ``VIBRATIONAL FREQUENCIES`` table looks like::

    -----------------------
    VIBRATIONAL FREQUENCIES
    -----------------------

    Scaling factor for frequencies =  1.000000000  (already applied!)

         0:       0.00 cm**-1
         ...
        37:   -118.27 cm**-1  ***imaginary mode***

The mode-index → frequency mapping is returned whole; the caller picks the subset it
needs (imaginary modes for NMS, all modes for a spectrum, …).
"""

from __future__ import annotations

import re
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

_FREQ_BLOCK_MARKER = "VIBRATIONAL FREQUENCIES"
"""Banner each Hessian's frequency table opens with; a TS search prints one per recompute."""

_LINEAR_STATEMENT = "The molecule is recognized as being linear"
"""ORCA's own word, in the thermochemistry after a frequency table, that the molecule is linear.

A linear molecule has five translations and rotations (indices 0-4) where any other has six
(0-5); ORCA prints them at 0.00 cm⁻¹ before the real vibrations. Read from ORCA rather than
decided from the geometry, which would need a collinearity tolerance of its own."""


class _Table(NamedTuple):
    """The last frequency block: every row, and the highest index that is not a vibration."""

    rows: list[tuple[int, float, bool]]
    last_trivial: int


_NORMAL_MODES_MARKER = "NORMAL MODES"
"""Banner the displacement-tensor scan anchors on (see
:func:`parse_normal_modes_tensor_from_text`)."""

_FREQ_LINE_RE = re.compile(r"^\s*(?P<index>\d+):\s+(?P<value>-?\d+\.\d+)\s*cm\*\*-1(?P<rest>.*)$")
_IMAG_TAG_RE = re.compile(r"imaginary mode", re.IGNORECASE)
_MODE_COL_HEADER_RE = re.compile(r"^\s*(\d+\s+)+\d+\s*$")
_MODE_ROW_RE = re.compile(r"^\s*\d+\s+[-\d.Ee\s]+$")


def _frequency_rows(text: str) -> _Table:
    """``(mode_index, cm⁻¹, is_imaginary)`` for every row of the last frequency block.

    The single scan the three public selectors share, so "which block, where does it
    start, where does it end" is answered once. Each of them then filters this list by its
    own rule; splitting the scan per rule is how two of them would come to disagree about
    which Hessian's table they are reading.

    Reads the **last** ``VIBRATIONAL FREQUENCIES`` group, matching every sibling parser
    (:func:`~chemrefine.engines.orca.output.energy.parse_final_energy_from_text`,
    the thermochemistry, the coordinates — all take the last match). A TS search recomputes
    the Hessian as it goes and prints one group per recompute, so reading the *first* reports
    a converged transition state with the imaginary modes it had on the way there: energy and
    geometry describe the final structure while the frequencies describe an earlier one, and
    NMS re-optimises along modes that no longer exist. Same discipline as
    :func:`~chemrefine.engines.orca.output.status.parse_converged`: last verdict wins.

    Returned with the highest trivial index for that block, from ORCA's own statement that
    the molecule is linear (:data:`_LINEAR_STATEMENT`), which follows the table.
    """
    # `sep` empty means the banner is absent — leave `text` alone so a frequency-less
    # output still yields `[]` rather than being scanned from the top.
    _head, sep, tail = text.rpartition(_FREQ_BLOCK_MARKER)
    # The marker itself has to survive: the scan below only enters the block when it *sees*
    # that line, so slicing it off would return nothing at all.
    text = sep + tail if sep else text

    in_block = False
    after_scaling = False
    rows: list[tuple[int, float, bool]] = []

    for line in text.splitlines():
        if _FREQ_BLOCK_MARKER in line:
            in_block = True
            continue
        if not in_block:
            continue
        if "Scaling factor for frequencies" in line:
            after_scaling = True
            continue
        if not after_scaling:
            continue

        m = _FREQ_LINE_RE.match(line)
        if m is None:
            # End of the block: a blank line or new section header. `rows` guards against
            # the blank line that separates the scaling factor from the table itself.
            if line.strip() == "" and rows:
                break
            continue
        rows.append(
            (
                int(m.group("index")),
                float(m.group("value")),
                bool(_IMAG_TAG_RE.search(m.group("rest"))),
            )
        )
    return _Table(rows, 4 if _LINEAR_STATEMENT in text else 5)


def parse_frequencies_from_text(text: str, *, only_imaginary: bool = False) -> dict[int, float]:
    """Return ``{mode_index: frequency_cm_inverse}`` from already-read ORCA output text.

    ``only_imaginary`` keeps only the modes ORCA flagged ``***imaginary mode***`` (the
    NMS-targeting subset); otherwise the real vibrations, every mode past the translations
    and rotations — six of them, or five for a molecule ORCA recognizes as linear.

    Neither subset is the whole table — see :func:`parse_mode_table_from_text`.
    """
    table = _frequency_rows(text)
    return {
        index: value
        for index, value, is_imaginary in table.rows
        if (is_imaginary if only_imaginary else index > table.last_trivial)
    }


def parse_imaginary_frequencies_from_text(text: str) -> dict[int, float]:
    """Imaginary-modes subset, operating on already-read output text."""
    return parse_frequencies_from_text(text, only_imaginary=True)


def parse_mode_table_from_text(text: str) -> dict[int, float]:
    """Every mode worth naming: the real vibrations *and* the imaginary ones.

    ORCA prints the translations and rotations first, at 0.00 cm⁻¹, and the vibrations after
    them in ascending order, so a transition state's imaginary mode leads the vibrations.
    The table is the union of the two subsets above rather than the index window alone so
    that :attr:`~chemrefine.state.Structure.imaginary_freqs` is a subset of it by
    construction, not by that layout.

    This is the table a viewer offers, :func:`chemrefine.agent_tools.analyze_mode` names a
    frequency from and ``random`` normal-mode sampling draws from, so ``imaginary_freqs``
    stays a subset of it by construction.
    """
    table = _frequency_rows(text)
    return {
        index: value
        for index, value, is_imaginary in table.rows
        if is_imaginary or index > table.last_trivial
    }


def parse_normal_modes_tensor_from_text(text: str, *, num_atoms: int) -> NDArray[np.float64]:
    """Return the per-mode displacement tensor ``(num_atoms, 3, n_modes)`` from output text.

    Each ``[atom, axis, mode]`` slice is one Cartesian-displacement component for one normal
    mode. ORCA prints the tensor in column-major blocks (a header line with the mode indices,
    then ``3 N`` rows), so the blocks are collected and ``hstack``-ed to recover the full
    ``(3N, n_modes)`` matrix before reshaping. Raises :class:`ValueError` if no mode blocks are
    present or the shape doesn't match ``3·num_atoms``.

    The scan is anchored on the ``NORMAL MODES`` banner. Without that anchor it started at
    the top of the file, and the column-header pattern — a line of two or more integers —
    matches plenty of earlier ORCA tables (symmetry, basis-set summaries, internal
    coordinates). Latching onto one of those and stopping at its separator produced a
    wrongly-shaped array, which the coordinator swallows into ``modes=None``, at which point
    every structure fails NMS with "no normal-mode tensor" — a message pointing at the
    frequency job rather than at the parser.
    """
    # `rpartition` for the same reason as the frequency table: the last Hessian is the one
    # describing the converged geometry. Guard on `sep`, not `tail` — `partition` yields an
    # empty tail when the marker is absent, but `rpartition` yields the *whole text*, so
    # testing `tail` here would stop catching a non-frequency output.
    _head, sep, tail = text.rpartition(_NORMAL_MODES_MARKER)
    if not sep:
        raise ValueError(f"no {_NORMAL_MODES_MARKER!r} section; is this a frequency output?")
    collecting = False
    block_rows: list[list[float]] = []
    blocks: list[NDArray[np.float64]] = []
    for line in tail.splitlines():
        if _MODE_COL_HEADER_RE.match(line):
            collecting = True
            if block_rows:
                blocks.append(np.asarray(block_rows, dtype=float))
                block_rows = []
            continue
        if not collecting:
            continue
        if _MODE_ROW_RE.match(line):
            parts = line.split()
            # First column is the row index; the rest are mode components.
            block_rows.append([float(x) for x in parts[1:]])
            continue
        # Block ends on a separator / next section header.
        if "IR SPECTRUM" in line or line.startswith("-"):
            if block_rows:
                blocks.append(np.asarray(block_rows, dtype=float))
                block_rows = []
            break

    if not blocks:
        raise ValueError("no normal-mode blocks found; is this a frequency-calculation output?")

    full = np.hstack(blocks)
    if not np.isfinite(full).all():
        # `float()` accepts ``nan`` / ``inf``. A non-finite displacement would ride the
        # tensor into `nms.displace_along_mode` and put NaN into every child geometry it
        # builds, so it withholds the tensor exactly as a wrong row count does.
        raise ValueError("non-finite component in the normal-mode block")
    if full.shape[0] != 3 * num_atoms:
        raise ValueError(
            f"malformed normal-mode block: got {full.shape[0]} rows, "
            f"expected {3 * num_atoms} (3 axes by {num_atoms} atoms)"
        )
    return full.reshape(num_atoms, 3, -1)

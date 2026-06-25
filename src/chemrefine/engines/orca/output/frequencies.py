"""ORCA frequency-block parsing: the vibrational table + the normal-mode tensor.

One section, one module. (Thermochemistry — Gibbs / enthalpy / ZPE — moved to
:mod:`chemrefine.engines.orca.output.energy`, since those are energies.) The
:mod:`chemrefine.engines.orca.output.coordinator` calls these over the text it already read.

The ``VIBRATIONAL FREQUENCIES`` table looks like::

    -----------------------
    VIBRATIONAL FREQUENCIES
    -----------------------

    Scaling factor for frequencies =  1.000000000  (already applied!)

         0:       0.00 cm**-1
         ...
        37:   -118.27 cm**-1  ***imaginary mode***

We parse the mode-index → frequency mapping and let the caller pick the subset (imaginary
modes for NMS, all modes for a spectrum, …).
"""

from __future__ import annotations

import re

import numpy as np
from numpy.typing import NDArray

_FREQ_LINE_RE = re.compile(r"^\s*(?P<index>\d+):\s+(?P<value>-?\d+\.\d+)\s*cm\*\*-1(?P<rest>.*)$")
_IMAG_TAG_RE = re.compile(r"imaginary mode", re.IGNORECASE)
_MODE_COL_HEADER_RE = re.compile(r"^\s*(\d+\s+)+\d+\s*$")
_MODE_ROW_RE = re.compile(r"^\s*\d+\s+[-\d.Ee\s]+$")


def parse_frequencies_from_text(
    text: str,
    *,
    only_imaginary: bool = False,
    skip_first_real: int = 5,
) -> dict[int, float]:
    """Return ``{mode_index: frequency_cm_inverse}`` from already-read ORCA output text.

    ``only_imaginary`` keeps only the modes ORCA flagged ``***imaginary mode***`` (the
    NMS-targeting subset); otherwise ``skip_first_real`` low-index translation/rotation modes
    are dropped (ORCA prints six zero modes, five for linear molecules; default drops five).
    """
    in_block = False
    after_scaling = False
    out: dict[int, float] = {}

    for line in text.splitlines():
        if "VIBRATIONAL FREQUENCIES" in line:
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
            # End of the block: a blank line or new section header.
            if line.strip() == "" and out:
                break
            continue

        index = int(m.group("index"))
        value = float(m.group("value"))
        is_imaginary = bool(_IMAG_TAG_RE.search(m.group("rest")))

        if only_imaginary:
            if is_imaginary:
                out[index] = value
        elif index > skip_first_real:
            out[index] = value
    return out


def parse_imaginary_frequencies_from_text(text: str) -> dict[int, float]:
    """Imaginary-modes subset, operating on already-read output text."""
    return parse_frequencies_from_text(text, only_imaginary=True)


def parse_normal_modes_tensor_from_text(text: str, *, num_atoms: int) -> NDArray[np.float64]:
    """Return the per-mode displacement tensor ``(num_atoms, 3, n_modes)`` from output text.

    Each ``[atom, axis, mode]`` slice is one Cartesian-displacement component for one normal
    mode. ORCA prints the tensor in column-major blocks (a header line with the mode indices,
    then ``3 N`` rows); we collect each block and ``hstack`` them to recover the full
    ``(3N, n_modes)`` matrix before reshaping. Raises :class:`ValueError` if no mode blocks are
    present or the shape doesn't match ``3·num_atoms``.
    """
    collecting = False
    block_rows: list[list[float]] = []
    blocks: list[NDArray[np.float64]] = []
    for line in text.splitlines():
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
    if full.shape[0] != 3 * num_atoms:
        raise ValueError(
            f"malformed normal-mode block: got {full.shape[0]} rows, "
            f"expected {3 * num_atoms} (3 axes by {num_atoms} atoms)"
        )
    return full.reshape(num_atoms, 3, -1)

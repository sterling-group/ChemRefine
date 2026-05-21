"""ORCA frequency-block parsing.

The ``VIBRATIONAL FREQUENCIES`` table in an ORCA output looks like::

    -----------------------
    VIBRATIONAL FREQUENCIES
    -----------------------

    Scaling factor for frequencies =  1.000000000  (already applied!)

         0:       0.00 cm**-1
         1:       0.00 cm**-1
         ...
         6:      15.11 cm**-1
       ...
        37:   -118.27 cm**-1  ***imaginary mode***

We parse the mode-index → frequency mapping and let the caller decide
which subset they want (imaginary modes for NMS, all modes for
spectrum extraction, etc.).

Verified against the v3 :func:`OrcaInterface.parse_imaginary_frequency`
behaviour from ``orca_interface.py`` on ``main`` plus the real
frequency tables in ``Conformational-Sampling/outputs/step4/*.out``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_FREQ_LINE_RE = re.compile(
    r"^\s*(?P<index>\d+):\s+(?P<value>-?\d+\.\d+)\s*cm\*\*-1(?P<rest>.*)$"
)
_IMAG_TAG_RE = re.compile(r"imaginary mode", re.IGNORECASE)
_MODE_COL_HEADER_RE = re.compile(r"^\s*(\d+\s+)+\d+\s*$")
_MODE_ROW_RE = re.compile(r"^\s*\d+\s+[-\d.Ee\s]+$")


def parse_frequencies(
    path: str | Path,
    *,
    only_imaginary: bool = False,
    skip_first_real: int = 5,
) -> dict[int, float]:
    """Return ``{mode_index: frequency_cm_inverse}`` from an ORCA output.

    Parameters
    ----------
    path:
        Path to an ORCA ``.out`` from a frequency calculation
        (``! ... FREQ``).
    only_imaginary:
        When ``True`` return only modes ORCA flagged with
        ``***imaginary mode***`` — this is the NMS-targeting subset.
    skip_first_real:
        Number of low-index translational / rotational modes to drop
        when ``only_imaginary`` is ``False``. ORCA always prints six
        zero modes (five for linear molecules); the default drops the
        first 5 to match v3 behaviour.
    """
    in_block = False
    after_scaling = False
    out: dict[int, float] = {}

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
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


def parse_imaginary_frequencies(path: str | Path) -> dict[int, float]:
    """Convenience wrapper — return only the imaginary modes."""
    return parse_frequencies(path, only_imaginary=True)


# ---------------------------------------------------------------------------
# Normal-mode displacement tensor
# ---------------------------------------------------------------------------


def parse_normal_modes_tensor(
    path: str | Path, *, num_atoms: int
) -> NDArray[np.float64]:
    """Return the per-mode displacement tensor for an ORCA frequency output.

    The returned array has shape ``(num_atoms, 3, n_modes)`` — each
    ``[atom, axis, mode]`` slice gives one Cartesian-displacement
    component for one normal mode.

    ORCA prints the tensor in column-major blocks (header line with
    the mode indices, then ``3 N`` rows). We collect each block as a
    matrix and ``hstack`` them to recover the full ``(3N, n_modes)``
    matrix before reshaping.

    Ported verbatim from v3's ``OrcaInterface.parse_normal_modes_tensor``.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")

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
        raise ValueError(
            f"no normal-mode blocks found in {path!s}; "
            "is this a frequency-calculation output?"
        )

    full = np.hstack(blocks)
    if full.shape[0] != 3 * num_atoms:
        raise ValueError(
            f"malformed normal-mode block: got {full.shape[0]} rows, "
            f"expected {3 * num_atoms} (3 axes by {num_atoms} atoms)"
        )
    return full.reshape(num_atoms, 3, -1)

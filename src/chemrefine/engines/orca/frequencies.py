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
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_FREQ_LINE_RE = re.compile(r"^\s*(?P<index>\d+):\s+(?P<value>-?\d+\.\d+)\s*cm\*\*-1(?P<rest>.*)$")
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
        first five.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return parse_frequencies_from_text(
        text, only_imaginary=only_imaginary, skip_first_real=skip_first_real
    )


def parse_frequencies_from_text(
    text: str,
    *,
    only_imaginary: bool = False,
    skip_first_real: int = 5,
) -> dict[int, float]:
    """Parse a frequency table from already-read ORCA output text.

    Same contract as :func:`parse_frequencies`, but accepts the file
    contents directly so callers (e.g. NMS) that need to parse
    multiple sections of the same output avoid re-reading.
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


def parse_imaginary_frequencies(path: str | Path) -> dict[int, float]:
    """Convenience wrapper — return only the imaginary modes."""
    return parse_imaginary_frequencies_from_text(
        Path(path).read_text(encoding="utf-8", errors="replace")
    )


def parse_imaginary_frequencies_from_text(text: str) -> dict[int, float]:
    """Imaginary-modes subset, operating on already-read output text."""
    return parse_frequencies_from_text(text, only_imaginary=True)


# ---------------------------------------------------------------------------
# Normal-mode displacement tensor
# ---------------------------------------------------------------------------


def parse_normal_modes_tensor(path: str | Path, *, num_atoms: int) -> NDArray[np.float64]:
    """Return the per-mode displacement tensor for an ORCA frequency output.

    The returned array has shape ``(num_atoms, 3, n_modes)`` — each
    ``[atom, axis, mode]`` slice gives one Cartesian-displacement
    component for one normal mode.

    ORCA prints the tensor in column-major blocks (header line with
    the mode indices, then ``3 N`` rows). We collect each block as a
    matrix and ``hstack`` them to recover the full ``(3N, n_modes)``
    matrix before reshaping.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return parse_normal_modes_tensor_from_text(text, num_atoms=num_atoms)


def parse_normal_modes_tensor_from_text(text: str, *, num_atoms: int) -> NDArray[np.float64]:
    """Same contract as :func:`parse_normal_modes_tensor` but on already-read text."""
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


# ---------------------------------------------------------------------------
# Thermochemistry (Gibbs / enthalpy / electronic+ZPE) from a freq output
# ---------------------------------------------------------------------------

_THERMO_MARKER = "THERMOCHEMISTRY"
# ORCA prints "<label>   ...   <value> Eh"; tolerate the dotted padding.
_GIBBS_RE = re.compile(r"Final Gibbs free energy\s*\.*\s*(-?\d+\.\d+)")
_ENTHALPY_RE = re.compile(r"Total Enthalpy\s*\.*\s*(-?\d+\.\d+)")
_ZPE_RE = re.compile(r"Zero point energy\s*\.*\s*(-?\d+\.\d+)")


@dataclass(frozen=True)
class Thermochemistry:
    """Absolute thermochemistry (Hartree) extracted from an ORCA freq output.

    Any quantity whose line is absent from the block is ``None``.
    """

    gibbs_hartree: float | None
    enthalpy_hartree: float | None
    energy_zpe_hartree: float | None


def parse_thermochemistry_from_text(
    text: str, *, electronic_hartree: float
) -> Thermochemistry | None:
    """Return Gibbs / enthalpy / electronic+ZPE (Hartree), or ``None`` if no block.

    ORCA's ``THERMOCHEMISTRY`` section prints an absolute ``Final Gibbs free
    energy`` and ``Total Enthalpy``; ``Zero point energy`` is the (positive) ZPE
    *correction*, so electronic+ZPE = ``electronic_hartree`` + that correction.
    Returns ``None`` when the output has no thermochemistry block at all (e.g. a
    plain ``opt_sp`` with no frequencies).
    """
    if _THERMO_MARKER not in text:
        return None
    # Take the last of each (a compound job may print thermochemistry more than
    # once; the final block is the one we want), matching the energy parser.
    gibbs = _GIBBS_RE.findall(text)
    enthalpy = _ENTHALPY_RE.findall(text)
    zpe = _ZPE_RE.findall(text)
    return Thermochemistry(
        gibbs_hartree=float(gibbs[-1]) if gibbs else None,
        enthalpy_hartree=float(enthalpy[-1]) if enthalpy else None,
        energy_zpe_hartree=(electronic_hartree + float(zpe[-1])) if zpe else None,
    )

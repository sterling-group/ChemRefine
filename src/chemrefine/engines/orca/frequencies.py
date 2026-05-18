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

_FREQ_LINE_RE = re.compile(
    r"^\s*(?P<index>\d+):\s+(?P<value>-?\d+\.\d+)\s*cm\*\*-1(?P<rest>.*)$"
)
_IMAG_TAG_RE = re.compile(r"imaginary mode", re.IGNORECASE)


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

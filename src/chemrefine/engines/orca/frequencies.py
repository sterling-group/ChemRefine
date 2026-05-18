"""Frequency-block parsing for ORCA ``opt_freq`` outputs.

TODO: port ``parse_imaginary_frequency`` and the surrounding helpers
from ``orca_interface.py`` on ``main``. They live next to the normal-
mode tensor parsers — both are exercised by the NMS pipeline.

This module is split out so the ``opt_sp`` happy path in
:mod:`engines.orca.output` stays small.
"""

from __future__ import annotations

from pathlib import Path


def parse_imaginary_frequencies(path: str | Path) -> list[float]:
    """Return imaginary-mode frequencies (cm⁻¹) from an ORCA frequency output.

    Placeholder — needs a real opt+freq output to verify regex.
    """
    raise NotImplementedError(
        "ORCA frequency parser not yet ported — see TODO in engines/orca/frequencies.py"
    )

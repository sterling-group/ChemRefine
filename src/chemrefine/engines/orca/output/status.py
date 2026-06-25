"""ORCA run-status markers: did the job terminate normally / converge.

A tiny shared concern read by both the single-structure coordinator
(:mod:`chemrefine.engines.orca.output`) and the PES parser
(:mod:`chemrefine.engines.orca.output.ensembles`), so it lives in its own module rather than inline
in either. ORCA prints ``****ORCA TERMINATED NORMALLY****`` only on a clean exit; any
``... NOT CONVERGED ...`` (SCF or geometry/MaxIter) marks a failed stationary point.
"""

from __future__ import annotations

import re

_TERMINATED_RE = re.compile(r"ORCA TERMINATED NORMALLY")
_NOT_CONVERGED_RE = re.compile(r"NOT CONVERGED", re.IGNORECASE)


def parse_terminated(text: str) -> bool:
    """``True`` if ORCA printed its normal-termination banner."""
    return bool(_TERMINATED_RE.search(text))


def parse_converged(text: str) -> bool:
    """``True`` unless ORCA reported a ``NOT CONVERGED`` (SCF or geometry/MaxIter)."""
    return not bool(_NOT_CONVERGED_RE.search(text))

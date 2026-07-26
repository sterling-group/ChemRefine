"""ORCA run-status markers: did the job terminate normally / converge.

A tiny shared concern read by both the single-structure coordinator
(:mod:`chemrefine.engines.orca.output.coordinator`) and the PES parser
(:mod:`chemrefine.engines.orca.output.ensembles`), so it lives in its own module rather than inline
in either. ORCA prints ``****ORCA TERMINATED NORMALLY****`` only on a clean exit.

Convergence is read as the **last verdict wins**, not "did the file ever say
``NOT CONVERGED``". ORCA prints a verdict per SCF and per geometry optimisation, and a run
routinely recovers: an early SCF fails, ORCA retries with a different guess, and the
optimisation then converges. Scanning the whole file for a negative marker calls that run
failed — which, on an ``on_failure: stop`` step, halts a pipeline that actually succeeded and
sends a converged structure back through the retry path. Taking the last verdict matches the
discipline the energy readers already use: ``parse_final_energy_from_text`` keeps the *last*
``FINAL SINGLE POINT ENERGY``.
"""

from __future__ import annotations

import re

_TERMINATED_RE = re.compile(r"ORCA TERMINATED NORMALLY")

#: Every convergence verdict ORCA prints, positive or negative, in one pass so the
#: **last** one decides. The bare ``NOT CONVERGED`` alternative is last among the
#: negatives so the specific spellings win the match when both could apply; it stays
#: as the catch-all for verdict banners this list doesn't enumerate.
_VERDICT_RE = re.compile(
    r"(?P<neg>SCF NOT CONVERGED|THE OPTIMIZATION HAS NOT CONVERGED|NOT CONVERGED)"
    r"|(?P<pos>SCF CONVERGED AFTER|THE OPTIMIZATION HAS CONVERGED)",
    re.IGNORECASE,
)


def parse_terminated(text: str) -> bool:
    """``True`` if ORCA printed its normal-termination banner."""
    return bool(_TERMINATED_RE.search(text))


def parse_converged(text: str) -> bool:
    """``True`` unless ORCA's **last** convergence verdict was a failure.

    An output with no verdict at all (a plain single point, a trimmed fixture) returns
    ``True`` — "no signal is not a failure signal", matching
    :func:`chemrefine.step_failures.succeeded`, which treats only an explicit ``False``
    as a failure.
    """
    last = None
    for match in _VERDICT_RE.finditer(text):
        last = match
    if last is None:
        return True
    return last.lastgroup == "pos"

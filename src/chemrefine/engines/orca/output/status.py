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

"Last verdict wins" is also what makes ``The optimization has not yet converged`` usable as a
negative: ORCA prints it after every non-final geometry cycle, so 72 of the 108 recorded
outputs contain it — but in none of them is it the *last* verdict, because the success banner
always follows. When it genuinely is last, the optimisation ran out of cycles.
"""

from __future__ import annotations

import re

_TERMINATED_RE = re.compile(r"ORCA TERMINATED NORMALLY")

#: Every convergence verdict ORCA prints, positive or negative, in one pass so the
#: **last** one decides. Each alternative is a string ORCA 6.1.1 actually emits — verified
#: against its binaries and against 108 recorded outputs — because guessing at the wording
#: here fails silently in both directions.
#:
#: Two deliberate omissions:
#:
#: * There is **no** ``THE OPTIMIZATION HAS NOT CONVERGED``. ORCA has no such banner; a
#:   geometry optimisation that runs out of cycles simply leaves ``The optimization has not
#:   yet converged`` as its last word, with no success banner after it. That is the
#:   spelling to match, and matching it is what makes a cycle-exhausted optimisation
#:   detectable at all.
#: * No bare ``NOT CONVERGED`` catch-all. Its only real-world matches are
#:   ``LOCALIZATION HAS NOT CONVERGED`` and ``MAXIMUM NO OF ITERATIONS EXCEEDED -
#:   LOCALIZATION NOT CONVERGED`` — orbital localisation is post-processing for printing,
#:   and it failing says nothing about the energy or the geometry. Catching it would fail
#:   a perfectly good run.
_VERDICT_RE = re.compile(
    r"(?P<neg>SCF NOT CONVERGED|The optimization has not yet converged)"
    r"|(?P<pos>SCF CONVERGED AFTER|THE OPTIMIZATION HAS CONVERGED)",
    re.IGNORECASE,
)


def parse_terminated_normally(text: str) -> bool:
    """``True`` if ORCA printed its normal-termination banner (i.e. it exited cleanly).

    Named to match :attr:`chemrefine.state.Structure.terminated_normally`, whose value
    this becomes: ``True`` is success, not "the job was terminated".
    """
    return bool(_TERMINATED_RE.search(text))


def parse_converged(text: str) -> bool:
    """``True`` unless ORCA's **last** convergence verdict was a failure.

    An output with no verdict at all (a plain single point, a trimmed fixture) returns
    ``True`` — "no signal is not a failure signal", matching
    :func:`chemrefine.lifecycle.succeeded`, which treats only an explicit ``False``
    as a failure.
    """
    last = None
    for match in _VERDICT_RE.finditer(text):
        last = match
    if last is None:
        return True
    return last.lastgroup == "pos"

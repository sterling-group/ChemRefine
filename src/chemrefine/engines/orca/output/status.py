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

The last verdict is taken **of each kind**, SCF and optimiser apart, and a run is converged
unless either is a failure. The two judge different things — one evaluation's SCF, and
whether the geometry has settled — and each geometry's SCF verdict comes after the
optimiser's verdict on the one before it. Read as one stream, a run killed after the SCF of a
geometry the optimiser had not yet judged reads as converged.

"Last verdict wins" is also what makes ``The optimization has not yet converged`` usable as a
negative: ORCA prints it after every non-final geometry cycle, so most recorded optimisation
outputs contain it — but in none of them is it the *last* verdict, because the success banner
always follows. When it genuinely is last, the optimisation ran out of cycles.
"""

from __future__ import annotations

import re

_TERMINATED_RE = re.compile(r"ORCA TERMINATED NORMALLY")

#: The convergence verdicts ORCA prints, positive and negative, one pattern per kind so the
#: **last** of each decides. Each alternative is a string ORCA 6.1.1 actually emits —
#: verified against its binaries and against the recorded outputs — because guessing at the
#: wording here fails silently in both directions. Matched case and all: other text that
#: says "SCF not converged" reaches an output too — an ExtOpt external program's own log
#: (PySCF prints ``SCF not converged.``), ORCA's NEB warning about images that are not the
#: structure read — and none of it is a verdict on this one, which a verdict of its kind
#: printed later would not always override.
#:
#: Two deliberate omissions:
#:
#: * There is **no** ``THE OPTIMIZATION HAS NOT CONVERGED``. ORCA has no such banner; a
#:   geometry optimisation that runs out of cycles leaves ``The optimization has not yet
#:   converged`` as its last verdict, followed only by a notice that it reached its cycle
#:   limit and never by a success banner. That is the spelling to match, and matching it
#:   is what makes a cycle-exhausted optimisation detectable at all.
#: * No bare ``NOT CONVERGED`` catch-all. Its only real-world matches are
#:   ``LOCALIZATION HAS NOT CONVERGED`` and ``MAXIMUM NO OF ITERATIONS EXCEEDED -
#:   LOCALIZATION NOT CONVERGED`` — orbital localisation is post-processing for printing,
#:   and it failing says nothing about the energy or the geometry. Catching it would fail
#:   a perfectly good run.
_VERDICT_RES = (
    re.compile(r"(?P<neg>SCF NOT CONVERGED)|(?P<pos>SCF CONVERGED AFTER)"),
    re.compile(
        r"(?P<neg>The optimization has not yet converged)|(?P<pos>THE OPTIMIZATION HAS CONVERGED)"
    ),
)


def parse_terminated_normally(text: str) -> bool:
    """``True`` if ORCA printed its normal-termination banner (i.e. it exited cleanly).

    Named to match :attr:`chemrefine.state.Structure.terminated_normally`, whose value
    this becomes: ``True`` is success, not "the job was terminated".
    """
    return bool(_TERMINATED_RE.search(text))


def parse_converged(text: str) -> bool:
    """``True`` unless ORCA's **last** SCF or **last** optimiser verdict was a failure.

    A kind with no verdict at all (a plain single point prints no optimiser verdict, a
    trimmed fixture may print none) counts as passed — "no signal is not a failure signal",
    matching :func:`chemrefine.lifecycle.succeeded`, which treats only an explicit ``False``
    as a failure.
    """
    return not any(_last_failed(pattern, text) for pattern in _VERDICT_RES)


def _last_failed(pattern: re.Pattern[str], text: str) -> bool:
    """Whether the last verdict ``pattern`` finds in ``text`` is a failure."""
    last = None
    for match in pattern.finditer(text):
        last = match
    return last is not None and last.lastgroup == "neg"

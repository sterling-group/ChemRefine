"""Q-Chem run-status markers: did the job terminate normally / did it converge.

A small shared concern the coordinator reads in the same pass as everything else and
threads onto every parsed structure. The discipline throughout is **last verdict wins**:
a run that recovers — an early SCF failure, then convergence — must not be condemned by
a scan for any negative marker anywhere in the file. Every matched spelling must be one
Q-Chem actually emits, pinned against full captured output; a guessed banner fails
silently in both directions, and the shipped fixtures are trimmed and carry no banner.
"""

from __future__ import annotations


def parse_terminated_normally(text: str) -> bool | None:
    """Whether the program exited cleanly; ``None`` when the output reports no verdict.

    ``True`` for an output carrying Q-Chem's clean-exit ending ("Thank you very much
    for using Q-Chem" / the ``Total job time:`` line); ``False`` for one carrying the
    fatal banner ("Q-Chem fatal error") or cut off before any ending — a walltime kill
    leaves exactly that file. In an ``@@@`` multi-job chain the verdict belongs to the
    *run*, not to one job of it. A wrongly-strict match ledgers good jobs, which is why
    the strings are held to captured output rather than guessed.
    """
    return None


def parse_converged(text: str) -> bool | None:
    """Whether the SCF/geometry converged; ``None`` when the output reports no verdict.

    Every convergence verdict the output prints is collected, positive and negative,
    and the **last** one decides. An output with no verdict at all stays ``None`` —
    no signal is not a failure signal.
    """
    return None

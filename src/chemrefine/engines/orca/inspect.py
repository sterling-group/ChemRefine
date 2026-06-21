"""Infer what an ORCA run does by reading its input template's keywords.

When a step omits ``operation``, ChemRefine inspects the resolved ORCA template
to decide which parser to use and (for NMS) whether the run optimises a TS and
whether it computes frequencies. An explicit ``operation`` always wins over this
— inspection is the convenience default, not an override.

Only the ``!`` simple-input lines are read for keywords, and ORCA ``#`` comments
(whole-line *and* trailing) are stripped first, so a keyword that only appears in
a comment (``# add Opt later``) is never mistaken for a real directive. A relaxed
surface scan is detected from a ``%geom … Scan … end`` block (also comment-stripped).

The returned :attr:`OrcaRunType.operation` is one of the strings
:func:`chemrefine.engines.orca.output.parse_output` already understands
(``opt_sp`` / ``sp`` / ``pes`` / ``goat`` / ``docker`` / ``solvator``), so the
existing parser dispatch is reused unchanged. With no run-type keyword at all it
defaults to ``sp`` — ORCA's own fallback (a single point).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# A ``%geom … Scan … end`` block (a relaxed surface scan) → the ``pes`` parser.
_GEOM_SCAN_RE = re.compile(r"%geom\b.*?\bscan\b.*?\bend\b", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class OrcaRunType:
    """What an ORCA template will do, inferred from its keywords.

    ``operation`` is the parser key (see :mod:`chemrefine.engines.orca.output`);
    ``is_ts`` marks an ``OptTS`` (drives the NMS ``ts`` target); ``has_freq`` marks
    a frequency calc (gates NMS).
    """

    operation: str
    is_ts: bool
    has_freq: bool


def _strip_orca_comments(text: str) -> str:
    """Drop ORCA ``#`` comments (everything from the first ``#`` on each line)."""
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def inspect_template(template_path: str | Path) -> OrcaRunType:
    """Infer the run type from an ORCA template's ``!`` keyword lines (+ ``%geom`` scan).

    Ensemble runs are simple keywords (``! GOAT`` / ``! DOCKER`` / ``! SOLVATOR``);
    a relaxed scan is a ``%geom … Scan … end`` block; an ``Opt`` (or ``OptTS``) is an
    optimization (``opt_sp``); anything else — including a bare single point or a
    frequency-only job — parses from the ``.out`` like ``sp``. ``OptTS`` and any
    ``…Freq`` are flagged regardless. Comments are ignored and matching is
    case-insensitive.
    """
    decommented = _strip_orca_comments(
        Path(template_path).read_text(encoding="utf-8", errors="replace")
    )
    # The keyword surface is the set of ``!`` simple-input lines (comments removed).
    keywords = " ".join(
        ln for ln in decommented.lower().splitlines() if ln.lstrip().startswith("!")
    )
    if "goat" in keywords:
        operation = "goat"
    elif "docker" in keywords:
        operation = "docker"
    elif "solvator" in keywords:
        operation = "solvator"
    elif _GEOM_SCAN_RE.search(decommented):
        operation = "pes"
    elif "opt" in keywords:  # Opt or OptTS — an optimization
        operation = "opt_sp"
    else:
        operation = "sp"  # ORCA's own fallback: a single point
    return OrcaRunType(
        operation=operation,
        is_ts="optts" in keywords,
        has_freq="freq" in keywords,  # matches freq / numfreq / anfreq
    )

"""Read facts from an ORCA input template (the reader half of the ORCA input file).

The writer half — generating the per-structure ``.inp`` — lives in
:mod:`chemrefine.engines.orca.input`. This module only *reads* a template: it infers the
run type (for parser dispatch + NMS) and the PAL count, in a single pass.

When a step omits ``operation``, ChemRefine inspects the resolved ORCA template to decide
which parser to use and (for NMS) whether the run optimises a TS and whether it computes
frequencies. An explicit ``operation`` always wins — inspection is the convenience default.

Only the ``!`` simple-input lines are read for keywords, and ORCA ``#`` comments (whole-line
*and* trailing) are stripped first, so a keyword that only appears in a comment
(``# add Opt later``) is never mistaken for a real directive. A relaxed surface scan is
detected from a ``%geom … Scan … end`` block (also comment-stripped). The PAL count is read
from the ``%pal``/``nprocs``/``PALn`` declaration anywhere in the template.

The returned :attr:`OrcaInputInfo.operation` is one of the strings
:func:`chemrefine.engines.orca.output.parse_output` already understands
(``opt_sp`` / ``sp`` / ``pes`` / ``goat`` / ``docker`` / ``solvator``). With no run-type
keyword at all it defaults to ``sp`` — ORCA's own fallback (a single point).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# A ``%geom … Scan … end`` block (a relaxed surface scan) → the ``pes`` parser.
_GEOM_SCAN_RE = re.compile(r"%geom\b.*?\bscan\b.*?\bend\b", re.IGNORECASE | re.DOTALL)

# Whole-token spellings of the two run-type keywords this keys on, as ORCA 6.1.1 accepts
# them. Matched against whole tokens, so a keyword that merely contains one of these
# substrings cannot trip them.
#
# The optimisation prefixes are the convergence levels (``SloppyOpt`` … ``VeryTightOpt``)
# and the Cartesian / external-optimiser variants (``COpt``, ``ExtOpt``). ``ExtOpt`` counts
# as an optimisation because it is one — ORCA runs its own optimiser over gradients from an
# external program, which is what the ``mlip-extopt`` / ``pyscf-extopt`` engines do.
#
# ``TS`` is a separate alternative rather than a suffix on the prefixed forms, because it
# only attaches to the bare keyword: ORCA takes ``OptTS`` and rejects ``TightOptTS``,
# ``COptTS`` and ``ExtOptTS`` outright ("UNRECOGNIZED OR DUPLICATED KEYWORD"). A tightness
# level for a saddle-point search is written as its own keyword — ``! OptTS TightOpt`` —
# which this reads as the two tokens it is. That is also why :attr:`OrcaInputInfo.is_ts`
# can test for the exact token ``optts`` and not a family of spellings.
_OPT_TOKEN_RE = re.compile(
    r"(?:sloppy|loose|normal|verytight|tight|c|ext)?opt|optts", re.IGNORECASE
)
_FREQ_TOKEN_RE = re.compile(r"(?:num|an)?freq", re.IGNORECASE)

# Every spelling of an ORCA PAL declaration, as ``(prefix)(count)`` pairs so :func:`_read_pal`
# reads the count and :func:`chemrefine.engines.orca.input.clamp_pal` rewrites it in place.
_PAL_PATTERNS = (
    re.compile(r"(nprocs\s+)(\d+)", re.IGNORECASE),
    re.compile(r"\b(PAL)(\d+)\b", re.IGNORECASE),
    re.compile(r"(^\s*PAL\s+)(\d+)\b", re.IGNORECASE | re.MULTILINE),
)


@dataclass(frozen=True)
class OrcaInputInfo:
    """Facts read from an ORCA input template in one pass.

    ``operation`` is the parser key (see :mod:`chemrefine.engines.orca.output`); ``is_ts``
    marks an ``OptTS`` (drives the NMS ``ts`` target); ``has_freq`` marks a frequency calc
    (gates NMS); ``pal`` is the declared core count (``1`` if none).
    """

    operation: str
    is_ts: bool
    has_freq: bool
    pal: int


def _strip_orca_comments(text: str) -> str:
    """Drop ORCA ``#`` comments — everything from the first *unquoted* ``#`` on each line.

    The quote tracking is not pedantry: templates name auxiliary files in double quotes
    (``%DOCKER GUEST "lig#3.xyz"``), and cutting the line at that ``#`` silently truncated
    it. Whatever followed — an ``Opt`` or ``Freq`` keyword on the same line, the ``end`` of a
    ``%geom … Scan`` block — then vanished from the keyword surface, so the step was
    classified with the wrong parser or refused NMS for a reason that was not true. A
    failure that reads as a chemistry problem, caused by a character in a filename.
    """
    lines = []
    for raw in text.splitlines():
        in_quotes = False
        cut = len(raw)
        for index, char in enumerate(raw):
            if char == '"':
                in_quotes = not in_quotes
            elif char == "#" and not in_quotes:
                cut = index
                break
        lines.append(raw[:cut])
    return "\n".join(lines)


def _read_pal(text: str) -> int:
    """Return the PAL / ``nprocs`` count declared in ORCA input text (``1`` if none)."""
    for pattern in _PAL_PATTERNS:
        m = pattern.search(text)
        if m:
            return int(m.group(2))
    return 1


def inspect_template(template_path: str | Path) -> OrcaInputInfo:
    """Infer run type + PAL from an ORCA template in a single read.

    Ensemble runs are simple keywords (``! GOAT`` / ``! DOCKER`` / ``! SOLVATOR``); a relaxed
    scan is a ``%geom … Scan … end`` block; an ``Opt`` (or ``OptTS``) is an optimization
    (``opt_sp``); anything else — including a bare single point or a frequency-only job —
    parses from the ``.out`` like ``sp``. ``OptTS`` and any ``…Freq`` are flagged regardless.
    Comments are ignored and matching is case-insensitive.
    """
    text = Path(template_path).read_text(encoding="utf-8", errors="replace")
    decommented = _strip_orca_comments(text)
    # The keyword surface is the set of ``!`` simple-input lines (comments removed),
    # split into whole tokens. Matching substrings against the joined line instead
    # would let any future ORCA keyword that merely *contains* "opt" or "freq" — or a
    # basis-set or functional name that does — silently pick the wrong parser, and via
    # the NMS frequency gate, reject or admit a step for the wrong reason.
    keywords = {
        token
        for line in decommented.lower().splitlines()
        if line.lstrip().startswith("!")
        for token in line.lstrip().lstrip("!").split()
    }
    if "goat" in keywords:
        operation = "goat"
    elif "docker" in keywords:
        operation = "docker"
    elif "solvator" in keywords:
        operation = "solvator"
    elif _GEOM_SCAN_RE.search(decommented):
        operation = "pes"
    elif any(_OPT_TOKEN_RE.fullmatch(token) for token in keywords):
        operation = "opt_sp"
    else:
        operation = "sp"  # ORCA's own fallback: a single point
    return OrcaInputInfo(
        operation=operation,
        is_ts="optts" in keywords,
        has_freq=any(_FREQ_TOKEN_RE.fullmatch(token) for token in keywords),
        pal=_read_pal(decommented),
    )

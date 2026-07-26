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

# Whole-token spellings of the two run-type keywords we key on. ORCA prefixes
# convergence tightness (``TightOpt``) and Cartesian/TS variants (``COpt``, ``OptTS``),
# and frequencies come as ``Freq`` / ``NumFreq`` / ``AnFreq``. Matched against whole
# tokens, so a keyword that merely contains one of these substrings cannot trip them.
_OPT_TOKEN_RE = re.compile(r"(?:very|tight|normal|loose|c)?opt(?:ts)?", re.IGNORECASE)
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
    """Drop ORCA ``#`` comments (everything from the first ``#`` on each line)."""
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def _read_pal(text: str) -> int:
    """Return the PAL / ``nprocs`` count declared in ORCA input text (``1`` if none)."""
    for pattern in _PAL_PATTERNS:
        m = pattern.search(text)
        if m:
            return int(m.group(2))
    return 1


def parse_pal(input_file: str | Path) -> int:
    """Return the PAL / ``nprocs`` value declared in an ORCA input or template file.

    Per-structure ``.inp`` files inherit their ``%pal`` block from the step template, so
    callers typically pass the template path once per step. Falls back to ``1`` when no PAL
    directive is found, matching ORCA's own default for serial runs.
    """
    return _read_pal(Path(input_file).read_text(encoding="utf-8"))


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

"""xtb output parsing: one file per module, assembled by a coordinator that reads stdout once.

The per-file readers live in :mod:`.stdout` (energy, verdicts, thermochemistry, and which
files xtb wrote), :mod:`.gradient` (the Turbomole ``gradient`` file), :mod:`.geometry`
(``xtbopt.xyz`` and the input XYZ) and :mod:`.frequencies` (``g98.out``). The
:mod:`.coordinator` assembles them into one :class:`~chemrefine.engines.api.ParsedResult`,
and this package re-exports its entry points.

The readers take text and paths, never a step context, so code that runs xtb itself — with
the command line :mod:`chemrefine.engines.xtb.command` builds — reads xtb's results through
the same readers the engine does.
"""

from chemrefine.engines.xtb.output.coordinator import parse_xtb, parse_xtb_text

__all__ = ["parse_xtb", "parse_xtb_text"]

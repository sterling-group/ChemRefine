"""ORCA output parsing: one section per module, assembled by a read-once coordinator.

The per-section extractors live in sibling modules — :mod:`.geometry`, :mod:`.energy`,
:mod:`.forces`, :mod:`.frequencies`, :mod:`.status` — and the multi-structure parsers in
:mod:`.ensembles`. The :mod:`.coordinator` reads a single-structure ``.out`` **once** and runs
every extractor over the shared text to assemble one
:class:`~chemrefine.engines.api.ParsedResult`. This package re-exports the coordinator's entry
points, so ``from chemrefine.engines.orca import output`` then ``output.parse_output(...)`` works.
"""

from chemrefine.engines.orca.output.coordinator import (
    TEXT_BASED_OPERATIONS,
    parse_dft,
    parse_dft_from_text,
    parse_output,
    parse_text,
)

__all__ = [
    "TEXT_BASED_OPERATIONS",
    "parse_dft",
    "parse_dft_from_text",
    "parse_output",
    "parse_text",
]

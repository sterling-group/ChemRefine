"""ORCA output parsing: one section per module, assembled by a read-once coordinator.

The per-section extractors live in sibling modules — :mod:`.geometry`, :mod:`.energy`,
:mod:`.forces`, :mod:`.frequencies`, :mod:`.status` — and the multi-structure parsers in
:mod:`.ensembles`. The :mod:`.coordinator` reads a single-structure ``.out`` **once** and runs
every extractor over the shared text to assemble one
:class:`~chemrefine.engines.api.ParsedResult`. This package re-exports the coordinator's entry
points, so ``from chemrefine.engines.orca import output`` then ``output.parse_output(...)`` works.

ORCA prints with CODATA 2018, the digits :mod:`chemrefine.quantities` keeps, so a value
converted from this output round-trips with it.
"""

from chemrefine.engines.orca.output.coordinator import (
    known_operations,
    multi_structure_operations,
    parse_dft,
    parse_dft_from_text,
    parse_output,
)

__all__ = [
    "known_operations",
    "multi_structure_operations",
    "parse_dft",
    "parse_dft_from_text",
    "parse_output",
]

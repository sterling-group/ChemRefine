"""Q-Chem output parsing: one section per module, assembled by a read-once coordinator.

The per-section readers live in :mod:`.geometry`, :mod:`.energy`, :mod:`.forces`,
:mod:`.frequencies` and :mod:`.status`; the :mod:`.coordinator` reads a ``.out``
**once**, runs every reader over the shared text, and threads each answer onto one
:class:`~chemrefine.engines.api.ParsedResult`. This package re-exports the
coordinator's entry points, so ``from chemrefine.engines.qchem import output`` then
``output.parse_qchem(...)`` works exactly as it did when this was one module.
"""

from chemrefine.engines.qchem.output.coordinator import (
    known_operations,
    parse_output,
    parse_qchem,
    parse_qchem_text,
)

__all__ = [
    "known_operations",
    "parse_output",
    "parse_qchem",
    "parse_qchem_text",
]

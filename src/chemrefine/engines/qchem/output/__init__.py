"""Q-Chem output parsing: one section per module, assembled by a read-once coordinator.

The per-section readers live in :mod:`.geometry`, :mod:`.energy`, :mod:`.forces`,
:mod:`.frequencies` and :mod:`.status`; the :mod:`.coordinator` reads a ``.out``
**once**, runs every reader over the shared text, and threads each answer onto one
:class:`~chemrefine.engines.api.ParsedResult`. This package re-exports the
coordinator's entry points, so ``from chemrefine.engines.qchem import output`` then
``output.parse_qchem(...)`` works exactly as it did when this was one module.

**Family rule, for every reader here — the stubs included when they are implemented: a
banner that can be printed in more than one unit pins the unit inside its match, or the
reader refuses.** Q-Chem prints its orientation in Bohr under ``input_bohr``, and a
match that accepts both spellings consumes Bohr as Å — a geometry silently wrong by
0.529 everywhere downstream, which no later check can see
(:class:`~chemrefine.engines.api.ParsedResult` declares no unit, only a contract). The
ORCA family pins ``(ANGSTROEM)`` for the same reason; :data:`.geometry.ORIENTATION_MARKER`
is this package's instance of the rule.
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

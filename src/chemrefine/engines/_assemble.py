"""Turn per-input parser output into lineage-correct :class:`Structure` results.

Every engine's job is to parse one of its output files into a list of
:class:`ParsedResult` (geometry + energy + optional forces / status flags /
thermochemistry). Assembling those into :class:`~chemrefine.state.Structure`
objects — minting child IDs and threading the parent lineage through a step's
fan-out — is **engine-independent**, so it lives here in one place rather than in
each engine's ``parse``.

``parsed_per_input`` is ``[(input_id, [ParsedResult, ...]), ...]``: one entry per
submitted structure, each holding however many structures that input produced
(1 for a normal calc, many for a GOAT/Docker/PES ensemble). The fan-out drives ID
allocation via :func:`chemrefine.ids.allocate_child_ids`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from numpy.typing import NDArray

from chemrefine.ids import allocate_child_ids
from chemrefine.state import PipelineState, StepResults, Structure


@dataclass(frozen=True)
class ParsedResult:
    """One structure extracted from an engine's output, before lineage is assigned.

    ``terminated`` / ``converged`` are run-status flags (``None`` when the engine
    doesn't report them, e.g. sidecar ensemble frames); a structure is a *failure*
    only when one is explicitly ``False``. The thermochemistry fields are populated
    only by a frequency run.
    """

    symbols: tuple[str, ...]
    positions: NDArray[np.float64]
    energy_hartree: float
    forces_ev_per_a: NDArray[np.float64] | None
    converged: bool | None = None
    terminated: bool | None = None
    gibbs_hartree: float | None = None
    enthalpy_hartree: float | None = None
    energy_zpe_hartree: float | None = None


def build_structures(
    parsed_per_input: Sequence[tuple[str, list[ParsedResult]]],
    prev_state: PipelineState,
) -> StepResults:
    """Assemble parsed results into :class:`Structure` objects with correct lineage.

    Child IDs come from :func:`chemrefine.ids.allocate_child_ids` (1:1 inherits the
    input's ID; a fan-out gets ``{parent}-{i}``). A fan-out child's ``parent_id`` is
    the input that produced it; a 1:1 child inherits the input's own ``parent_id``.
    """
    prev_by_id = {s.id: s for s in prev_state.structures}
    parents = [sid for sid, _ in parsed_per_input]
    fanouts = [len(parsed) for _, parsed in parsed_per_input]
    child_ids = iter(allocate_child_ids(parents, fanouts))

    out: list[Structure] = []
    for sid, parsed in parsed_per_input:
        input_struct = prev_by_id.get(sid)
        is_fanout = len(parsed) > 1
        for ps in parsed:
            child_parent = (
                sid if is_fanout else (input_struct.parent_id if input_struct is not None else None)
            )
            out.append(
                Structure(
                    id=next(child_ids),
                    atoms=Atoms(symbols=list(ps.symbols), positions=ps.positions),
                    parent_id=child_parent,
                    energy_hartree=ps.energy_hartree,
                    forces_ev_per_a=ps.forces_ev_per_a,
                    converged=ps.converged,
                    terminated=ps.terminated,
                    gibbs_hartree=ps.gibbs_hartree,
                    enthalpy_hartree=ps.enthalpy_hartree,
                    energy_zpe_hartree=ps.energy_zpe_hartree,
                )
            )
    return StepResults(structures=tuple(out))

"""Reduce a step's parsed results to the structures that move on to the next step.

The orchestrator calls :func:`apply` once per step with the engine's
parsed :class:`~chemrefine.state.StepResults` and the step's
:class:`~chemrefine.config.SampleConfig`. ``apply`` returns a
:class:`~chemrefine.state.PipelineState` containing only the survivors.

Three filter methods are supported (one Pydantic variant each):

* ``boltzmann`` — keep structures whose cumulative Boltzmann weight (at
  the given temperature) reaches a target percentage.
* ``min`` — keep the lowest-energy structures: the ``count`` lowest
  (``count == 0`` keeps everything) or all within ``window_kcalmol`` of
  the minimum (exactly one selector).
* ``max`` — keep the *highest*-energy structures (PES-style sampling):
  the ``count`` highest or all within ``window_kcalmol`` of the maximum
  (exactly one selector).

Setting ``by_parent: true`` on the sample config applies the same
method independently within each parent-ID group instead of globally.
"""

from __future__ import annotations

import logging
import operator
from collections import defaultdict
from collections.abc import Callable
from typing import Any, cast

import numpy as np

from chemrefine.config import (
    BoltzmannSample,
    MaxSample,
    MinSample,
    SampleConfig,
)
from chemrefine.quantities import HARTREE_TO_KCALMOL, boltzmann_weights
from chemrefine.state import PipelineState, StepResults, Structure

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def apply(results: StepResults, sample: SampleConfig | None) -> PipelineState:
    """Return the survivors of ``results`` under ``sample``.

    ``sample is None`` is the identity filter: every structure (with a
    computed energy) passes through.
    """
    structures = [s for s in results.structures if s.energy_hartree is not None]
    if sample is None or not structures:
        return PipelineState(structures=tuple(structures))

    if sample.by_parent:
        survivors = _filter_by_parent(structures, sample)
    else:
        survivors = _dispatch(sorted(structures, key=operator.attrgetter("energy_hartree")), sample)
    return PipelineState(structures=tuple(survivors))


# ---------------------------------------------------------------------------
# By-parent grouping
# ---------------------------------------------------------------------------


def _filter_by_parent(structures: list[Structure], sample: SampleConfig) -> list[Structure]:
    """Group by parent ID, filter each group, return concatenated survivors."""
    groups: dict[str, list[Structure]] = defaultdict(list)
    for struct in structures:
        # Seed structures (parent_id=None) form their own singleton groups
        # by falling back to their own id, matching the historical behaviour
        # of ``parent_of`` on flat IDs.
        groups[struct.parent_id or struct.id].append(struct)
    sort_key = operator.attrgetter("energy_hartree")
    survivors: list[Structure] = []
    for parent, group in groups.items():
        kept = _dispatch(sorted(group, key=sort_key), sample)
        logger.debug("parent %s: %d structures -> %d survivors", parent, len(group), len(kept))
        survivors.extend(kept)
    return survivors


# ---------------------------------------------------------------------------
# Method dispatch
# ---------------------------------------------------------------------------


# The second lambda argument is the *matching* variant (keyed by type), a
# per-key correlation a dict value type can't express — hence ``Any``.
_DISPATCHERS: dict[type, Callable[[list[Structure], Any], list[Structure]]] = {
    MinSample: lambda s, c: _filter_min(s, c),
    MaxSample: lambda s, c: _filter_max(s, c),
    BoltzmannSample: lambda s, c: _filter_boltzmann(s, c.percent_cumulative, c.temperature_k),
}


def _dispatch(sorted_structures: list[Structure], sample: SampleConfig) -> list[Structure]:
    """Pick the per-method filter implementation for ``sample``."""
    handler = _DISPATCHERS.get(type(sample))
    if handler is None:
        raise TypeError(f"unsupported sample config: {type(sample).__name__}")
    return handler(sorted_structures, sample)


def _filter_min(sorted_structures: list[Structure], sample: MinSample) -> list[Structure]:
    """Keep the lowest-energy structures by ``count`` or ``window_kcalmol``.

    Exactly one selector is set (the config validator guarantees it). ``count``
    keeps the N lowest (``0`` = keep all); ``window_kcalmol`` keeps everything
    within that window of the minimum. Precondition: ascending by ``energy_hartree``.
    """
    if sample.count is not None:
        return list(sorted_structures) if sample.count <= 0 else sorted_structures[: sample.count]
    min_e = cast(float, sorted_structures[0].energy_hartree)
    window_h = cast(float, sample.window_kcalmol) / HARTREE_TO_KCALMOL
    return [s for s in sorted_structures if cast(float, s.energy_hartree) <= min_e + window_h]


def _filter_max(sorted_structures: list[Structure], sample: MaxSample) -> list[Structure]:
    """Keep the highest-energy structures by ``count`` or ``window_kcalmol``.

    Exactly one selector is set (the config validator guarantees it). ``count``
    keeps the N highest; ``window_kcalmol`` keeps everything within that window of
    the maximum. Returned highest-first. Precondition: ascending by ``energy_hartree``.
    """
    high_first = list(reversed(sorted_structures))
    if sample.count is not None:
        return high_first[: sample.count]
    max_e = cast(float, high_first[0].energy_hartree)
    window_h = cast(float, sample.window_kcalmol) / HARTREE_TO_KCALMOL
    return [s for s in high_first if cast(float, s.energy_hartree) >= max_e - window_h]


def _filter_boltzmann(
    sorted_structures: list[Structure],
    percent_cumulative: float,
    temperature_k: float,
) -> list[Structure]:
    """Keep structures until cumulative Boltzmann weight reaches ``percent_cumulative``.

    Precondition: ``sorted_structures`` must be sorted ascending by
    ``energy_hartree`` — callers are responsible for sorting before dispatch.
    """
    if len(sorted_structures) <= 1:
        return list(sorted_structures)
    energies_kcal = np.array([s.energy_hartree for s in sorted_structures]) * HARTREE_TO_KCALMOL
    weights = boltzmann_weights(energies_kcal - energies_kcal.min(), temperature_k)
    cumulative = np.cumsum(weights * 100.0)
    # Keep every structure whose cumulative weight is still below the
    # threshold, plus the one that crosses it.
    n_below = int(np.sum(cumulative < percent_cumulative))
    return list(sorted_structures[: n_below + 1])

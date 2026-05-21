"""Reduce a step's parsed results to the structures that move on to the next step.

The orchestrator calls :func:`apply` once per step with the engine's
parsed :class:`~chemrefine.state.StepResults` and the step's
:class:`~chemrefine.config.SampleConfig`. ``apply`` returns a
:class:`~chemrefine.state.PipelineState` containing only the survivors.

Four filter methods are supported (one Pydantic variant each):

* ``boltzmann`` — keep structures whose cumulative Boltzmann weight (at
  the given temperature) reaches a target percentage.
* ``energy_window`` — keep structures within ``window_kcal`` of the
  lowest-energy one.
* ``integer`` — keep the ``count`` lowest-energy structures
  (``count == 0`` keeps everything).
* ``high_energy`` — keep the ``count`` *highest*-energy structures
  (used for PES-style sampling).

Setting ``by_parent: true`` on the sample config applies the same
method independently within each parent-ID group instead of globally.
"""

from __future__ import annotations

import logging
import operator
from collections import defaultdict
from collections.abc import Callable

import numpy as np

from chemrefine.config import (
    BoltzmannSample,
    EnergyWindowSample,
    HighEnergySample,
    IntegerSample,
    SampleConfig,
)
from chemrefine.constants import HARTREE_TO_KCALMOL, R_KCALMOL_K
from chemrefine.ids import parent_of
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
        survivors = _dispatch(
            sorted(structures, key=operator.attrgetter("energy_hartree")), sample
        )
    return PipelineState(structures=tuple(survivors))


# ---------------------------------------------------------------------------
# By-parent grouping
# ---------------------------------------------------------------------------


def _filter_by_parent(structures: list[Structure], sample: SampleConfig) -> list[Structure]:
    """Group by parent ID, filter each group, return concatenated survivors."""
    groups: dict[str, list[Structure]] = defaultdict(list)
    for struct in structures:
        groups[parent_of(struct.id)].append(struct)
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


_DISPATCHERS: dict[type, Callable[[list[Structure], SampleConfig], list[Structure]]] = {
    IntegerSample: lambda s, c: _filter_integer(s, c.count),
    EnergyWindowSample: lambda s, c: _filter_energy_window(s, c.window_kcal),
    BoltzmannSample: lambda s, c: _filter_boltzmann(
        s, c.percent_cumulative, c.temperature_k
    ),
    HighEnergySample: lambda s, c: _filter_high_energy(s, c.count),
}


def _dispatch(sorted_structures: list[Structure], sample: SampleConfig) -> list[Structure]:
    """Pick the per-method filter implementation for ``sample``."""
    handler = _DISPATCHERS.get(type(sample))
    if handler is None:
        raise TypeError(f"unsupported sample config: {type(sample).__name__}")
    return handler(sorted_structures, sample)


def _filter_integer(sorted_structures: list[Structure], count: int) -> list[Structure]:
    """Keep the ``count`` lowest-energy structures (``count == 0`` keeps all)."""
    if count <= 0:
        return list(sorted_structures)
    return sorted_structures[:count]


def _filter_energy_window(
    sorted_structures: list[Structure], window_kcal: float
) -> list[Structure]:
    """Keep structures within ``window_kcal`` of the lowest-energy structure."""
    min_e = sorted_structures[0].energy_hartree
    window_h = window_kcal / HARTREE_TO_KCALMOL
    return [s for s in sorted_structures if s.energy_hartree <= min_e + window_h]


def _filter_boltzmann(
    sorted_structures: list[Structure],
    percent_cumulative: float,
    temperature_k: float,
) -> list[Structure]:
    """Keep structures until cumulative Boltzmann weight reaches ``percent_cumulative``."""
    if len(sorted_structures) <= 1:
        return list(sorted_structures)
    energies_kcal = (
        np.array([s.energy_hartree for s in sorted_structures]) * HARTREE_TO_KCALMOL
    )
    delta = energies_kcal - energies_kcal.min()
    weights = np.exp(-delta / (R_KCALMOL_K * temperature_k))
    weights /= weights.sum()
    cumulative = np.cumsum(weights * 100.0)
    survivors: list[Structure] = []
    for struct, cum in zip(sorted_structures, cumulative, strict=True):
        survivors.append(struct)
        if cum >= percent_cumulative:
            break
    return survivors  # pragma: no cover - unreachable: Pydantic constrains percent_cumulative <= 100 so the loop always breaks


def _filter_high_energy(
    sorted_structures: list[Structure], count: int
) -> list[Structure]:
    """Keep the ``count`` highest-energy structures (PES-style sampling)."""
    if not sorted_structures:
        return []
    return list(reversed(sorted_structures))[:count]



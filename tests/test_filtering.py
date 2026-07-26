"""Tests for the per-step filter dispatch."""

from __future__ import annotations

import pytest
from ase import Atoms

from chemrefine.config import (
    BoltzmannSample,
    MaxSample,
    MinSample,
)
from chemrefine.filtering import apply
from chemrefine.quantities import HARTREE_TO_KCALMOL
from chemrefine.state import StepResults, Structure


def _results(*pairs: tuple) -> StepResults:
    """Build a ``StepResults`` from ``(id, energy)`` or ``(id, energy, parent_id)`` tuples."""
    structures = tuple(
        Structure(
            id=pair[0],
            atoms=Atoms("H"),
            parent_id=pair[2] if len(pair) > 2 else None,
            energy_hartree=pair[1],
        )
        for pair in pairs
    )
    return StepResults(structures=structures)


# ---------------------------------------------------------------------------
# Identity (sample=None)
# ---------------------------------------------------------------------------


def test_apply_with_none_sample_keeps_everything():
    r = _results(("0", -1.0), ("1", -0.5))
    state = apply(r, None)
    assert [s.id for s in state.structures] == ["0", "1"]


def test_apply_with_none_sample_keeps_energyless_structures():
    """The identity filter is the identity — energy-less structures survive it.

    ``on_failure: best`` backfills a failed structure from its submitted input. On
    step 1 those are bootstrap seeds, which have no energy yet; dropping them here
    would silently turn ``best`` into ``skip`` (B5).
    """
    r = StepResults(
        structures=(
            Structure(id="0", atoms=Atoms("H"), energy_hartree=-1.0),
            Structure(id="1", atoms=Atoms("H"), energy_hartree=None),
        )
    )
    state = apply(r, None)
    assert [s.id for s in state.structures] == ["0", "1"]


def test_apply_drops_structures_with_no_energy():
    r = StepResults(
        structures=(
            Structure(id="0", atoms=Atoms("H"), energy_hartree=-1.0),
            Structure(id="1", atoms=Atoms("H"), energy_hartree=None),
        )
    )
    state = apply(r, MinSample(method="min", count=10))
    assert [s.id for s in state.structures] == ["0"]


def test_apply_empty_returns_empty():
    state = apply(StepResults(structures=()), MinSample(method="min", count=5))
    assert state.structures == ()


# ---------------------------------------------------------------------------
# min — count
# ---------------------------------------------------------------------------


def test_min_keeps_lowest_n():
    r = _results(("a", -1.0), ("b", -2.0), ("c", -0.5), ("d", -3.0))
    state = apply(r, MinSample(method="min", count=2))
    assert [s.id for s in state.structures] == ["d", "b"]


def test_min_count_zero_keeps_all_sorted():
    r = _results(("a", -1.0), ("b", -2.0))
    state = apply(r, MinSample(method="min", count=0))
    assert [s.id for s in state.structures] == ["b", "a"]


# ---------------------------------------------------------------------------
# min — window_kcalmol (replaces the old energy_window method)
# ---------------------------------------------------------------------------


def test_min_window_keeps_structures_within_window():
    # window = 1.0 kcal/mol -> in Hartree ~= 1/627.5 ~= 0.00159
    # min energy -1.0; threshold -1.0 + 1.0 kcal/mol ≈ -0.9984
    r = _results(("a", -1.0), ("b", -0.9985), ("c", -0.5))
    state = apply(r, MinSample(method="min", window_kcalmol=1.0))
    ids = {s.id for s in state.structures}
    assert "a" in ids
    assert "c" not in ids


def test_min_window_negative_results_are_sorted():
    r = _results(("a", -1.0), ("b", -1.0 + 1e-6))
    state = apply(r, MinSample(method="min", window_kcalmol=10.0))
    assert state.structures[0].id == "a"


# ---------------------------------------------------------------------------
# boltzmann
# ---------------------------------------------------------------------------


def test_boltzmann_single_structure_kept():
    r = _results(("a", -1.0))
    state = apply(
        r, BoltzmannSample(method="boltzmann", percent_cumulative=50.0, temperature_k=300)
    )
    assert [s.id for s in state.structures] == ["a"]


def test_boltzmann_drops_high_energy_tail():
    # Two structures, one dominant (lowest energy)
    r = _results(("low", -1.0), ("high", -1.0 + 10.0 / HARTREE_TO_KCALMOL))
    state = apply(
        r, BoltzmannSample(method="boltzmann", percent_cumulative=99.0, temperature_k=298.15)
    )
    # 10 kcal/mol gap at room T → very high penalty, low gets almost all weight
    assert state.structures[0].id == "low"
    # 99% cumulative should already be reached by the lowest structure alone
    assert len(state.structures) == 1


def test_boltzmann_keeps_more_when_threshold_high():
    # Boltzmann with very small spread → need many structures to reach 99%
    energies = [-1.0 - i * 1e-6 for i in range(5)]
    r = _results(*[(str(i), e) for i, e in enumerate(energies)])
    state = apply(
        r, BoltzmannSample(method="boltzmann", percent_cumulative=99.9, temperature_k=298.15)
    )
    assert len(state.structures) >= 4


# ---------------------------------------------------------------------------
# max — count
# ---------------------------------------------------------------------------


def test_max_keeps_top_n():
    r = _results(("a", -1.0), ("b", -2.0), ("c", -0.5), ("d", -1.5))
    state = apply(r, MaxSample(method="max", count=2))
    # Sorted asc by energy: b(-2.0), d(-1.5), a(-1.0), c(-0.5)
    # Reversed asc gives desc; top 2 = c, a
    assert [s.id for s in state.structures] == ["c", "a"]


def test_max_count_one_keeps_max():
    r = _results(("a", -1.0), ("b", -2.0), ("c", -0.5))
    state = apply(r, MaxSample(method="max", count=1))
    assert [s.id for s in state.structures] == ["c"]


def test_max_empty_input_returns_empty():
    """apply with MaxSample short-circuits on an empty StepResults."""
    state = apply(StepResults(structures=()), MaxSample(method="max", count=5))
    assert state.structures == ()


# ---------------------------------------------------------------------------
# max — window_kcalmol (highest-energy side)
# ---------------------------------------------------------------------------


def test_max_window_keeps_structures_within_window_of_max():
    # max energy -0.5; threshold -0.5 - 1.0 kcal/mol ≈ -0.5016
    r = _results(("a", -1.0), ("b", -0.5015), ("c", -0.5))
    state = apply(r, MaxSample(method="max", window_kcalmol=1.0))
    ids = [s.id for s in state.structures]
    assert ids[0] == "c"  # highest first
    assert "b" in ids
    assert "a" not in ids


# ---------------------------------------------------------------------------
# energy_type (gibbs / enthalpy / electronic_zero_point)
# ---------------------------------------------------------------------------


def test_energy_type_gibbs_sorts_on_gibbs():
    """With energy_type=gibbs the filter ranks by Gibbs, not electronic energy."""
    r = StepResults(
        structures=(
            # electronic order: a < b, but gibbs order: b < a
            Structure(id="a", atoms=Atoms("H"), energy_hartree=-2.0, gibbs_hartree=-1.0),
            Structure(id="b", atoms=Atoms("H"), energy_hartree=-1.0, gibbs_hartree=-2.0),
        )
    )
    state = apply(r, MinSample(method="min", count=1, energy_type="gibbs"))
    assert [s.id for s in state.structures] == ["b"]


def test_energy_type_missing_thermochem_raises():
    from chemrefine.errors import ConfigError

    r = StepResults(
        structures=(Structure(id="a", atoms=Atoms("H"), energy_hartree=-1.0, gibbs_hartree=None),)
    )
    with pytest.raises(ConfigError, match="energy_type='gibbs' needs thermochemistry"):
        apply(r, MinSample(method="min", count=1, energy_type="gibbs"))


# ---------------------------------------------------------------------------
# by_parent
# ---------------------------------------------------------------------------


def test_by_parent_groups_by_lineage():
    r = _results(
        ("0-0", -1.0, "0"),
        ("0-1", -2.0, "0"),
        ("0-2", -0.5, "0"),
        ("1-0", -1.0, "1"),
        ("1-1", -0.5, "1"),
    )
    sample = MinSample(method="min", count=1, by_parent=True)
    state = apply(r, sample)
    ids = {s.id for s in state.structures}
    # One survivor per parent group
    assert ids == {"0-1", "1-0"}


def test_by_parent_handles_flat_ids():
    r = _results(("0", -1.0), ("1", -2.0))
    sample = MinSample(method="min", count=1, by_parent=True)
    state = apply(r, sample)
    # Seed IDs (parent_id=None) form their own singleton groups
    assert {s.id for s in state.structures} == {"0", "1"}


# ---------------------------------------------------------------------------
# Unsupported sample type
# ---------------------------------------------------------------------------


def test_unknown_sample_type_raises():
    class BogusSample:
        method = "nope"
        by_parent = False
        temperature_k = 298.15
        energy_type = "electronic"

    r = _results(("a", -1.0))
    with pytest.raises(TypeError):
        apply(r, BogusSample())  # type: ignore[arg-type]


def test_all_sample_variants_have_a_dispatcher():
    """Catches the "added a new SampleConfig variant but forgot to register it" bug."""
    from chemrefine.filtering import _DISPATCHERS

    assert MinSample in _DISPATCHERS
    assert MaxSample in _DISPATCHERS
    assert BoltzmannSample in _DISPATCHERS

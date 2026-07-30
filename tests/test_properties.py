"""Property tests for the pure, numerical core.

The example-based tests here pin down the cases someone thought of. These pin down the
*invariants* — the things that have to hold for every input, which is where a numerical
routine tends to go wrong: an empty sequence, one element, a huge energy spread that
overflows an exponential, a fanout of zero.

Scope is deliberately the side-effect-free functions: ID allocation, unit conversion,
Boltzmann statistics, the filters, and the NMS displacement maths. Anything that touches
a file or an engine belongs in the module tests, where a fixture can say what it did.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from chemrefine import quantities
from chemrefine.config import BoltzmannSample, MaxSample, MinSample
from chemrefine.filtering import apply
from chemrefine.ids import allocate_child_ids
from chemrefine.nms import displace_along_mode
from chemrefine.state import StepResults, Structure

# Energies a quantum-chemistry run could plausibly produce, in Hartree.
energies = st.floats(min_value=-5000.0, max_value=0.0, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# ids.allocate_child_ids
# ---------------------------------------------------------------------------


@given(
    st.lists(st.integers(min_value=0, max_value=5), min_size=0, max_size=20),
)
def test_allocate_child_ids_returns_exactly_the_requested_fanout(fanouts: list[int]) -> None:
    """One id per child, no matter how the fanout is distributed."""
    parents = [str(i) for i in range(len(fanouts))]
    assert len(allocate_child_ids(parents, fanouts)) == sum(fanouts)


@given(st.lists(st.integers(min_value=0, max_value=5), min_size=1, max_size=20))
def test_allocate_child_ids_are_unique(fanouts: list[int]) -> None:
    """Two structures must never end up sharing a directory."""
    parents = [str(i) for i in range(len(fanouts))]
    children = allocate_child_ids(parents, fanouts)
    assert len(set(children)) == len(children)


@given(st.lists(st.integers(min_value=1, max_value=4), min_size=1, max_size=10))
def test_allocate_child_ids_keeps_the_parent_id_for_a_single_child(fanouts: list[int]) -> None:
    """A 1:1 step must not rename its structures — that is what keeps a chain of
    refinements addressable by the same id from start to finish."""
    parents = [f"p{i}" for i in range(len(fanouts))]
    children = allocate_child_ids(parents, fanouts)
    cursor = 0
    for parent, fanout in zip(parents, fanouts, strict=True):
        if fanout == 1:
            assert children[cursor] == parent
        cursor += fanout


# ---------------------------------------------------------------------------
# quantities
# ---------------------------------------------------------------------------


@given(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
def test_convert_round_trips(value: float) -> None:
    """Every conversion pair has an inverse that gets you back.

    Worth asserting because the table stores each direction as its own constant rather
    than deriving one from the other, so a typo in either would go unnoticed.
    """
    for a, b in (
        ("hartree", "kcal/mol"),
        ("hartree", "kj/mol"),
        ("hartree", "ev"),
        ("bohr", "angstrom"),
        ("hartree/bohr", "ev/angstrom"),
    ):
        there_and_back = quantities.convert(quantities.convert(value, a, b), b, a)
        assert np.isclose(there_and_back, value, rtol=1e-9, atol=1e-12)


@given(st.lists(energies, min_size=1, max_size=50), st.floats(min_value=1.0, max_value=2000.0))
def test_boltzmann_weights_sum_to_one(values: list[float], temperature: float) -> None:
    """The defining property, including for absolute energies in the thousands.

    Passing raw absolute energies is the case that overflows a naive implementation:
    exp(+4000/RT) is inf, and inf/inf is nan. The minimum is subtracted internally.
    """
    kcal = np.asarray(values) * quantities.HARTREE_TO_KCALMOL
    weights = quantities.boltzmann_weights(kcal, temperature)
    assert np.all(np.isfinite(weights))
    assert np.all(weights >= 0.0)
    assert np.isclose(weights.sum(), 1.0)


@given(st.lists(energies, min_size=2, max_size=30), st.floats(min_value=1.0, max_value=2000.0))
def test_boltzmann_weights_favour_lower_energies(values: list[float], temperature: float) -> None:
    """A lower energy never gets less weight than a higher one."""
    kcal = np.asarray(sorted(values)) * quantities.HARTREE_TO_KCALMOL
    weights = quantities.boltzmann_weights(kcal, temperature)
    assert np.all(np.diff(weights) <= 1e-12)


# ---------------------------------------------------------------------------
# filtering
# ---------------------------------------------------------------------------


def _results(values: list[float]) -> StepResults:
    return StepResults(
        structures=tuple(
            Structure(id=str(i), atoms=Atoms("H"), energy_hartree=e) for i, e in enumerate(values)
        )
    )


@given(st.lists(energies, min_size=1, max_size=40), st.integers(min_value=1, max_value=40))
def test_min_keeps_at_most_count_and_always_the_lowest(values: list[float], count: int) -> None:
    survivors = apply(_results(values), MinSample(method="min", count=count)).structures
    assert len(survivors) == min(count, len(values))
    assert survivors[0].energy_hartree == min(values)


@given(st.lists(energies, min_size=1, max_size=40), st.integers(min_value=1, max_value=40))
def test_max_keeps_at_most_count_and_always_the_highest(values: list[float], count: int) -> None:
    survivors = apply(_results(values), MaxSample(method="max", count=count)).structures
    assert len(survivors) == min(count, len(values))
    assert survivors[0].energy_hartree == max(values)


@given(st.lists(energies, min_size=1, max_size=40))
def test_every_filter_returns_a_subset_of_its_input(values: list[float]) -> None:
    """A filter selects; it never invents a structure or duplicates one."""
    results = _results(values)
    original = {s.id for s in results.structures}
    for sample in (
        MinSample(method="min", count=3),
        MaxSample(method="max", count=3),
        BoltzmannSample(method="boltzmann", percent_cumulative=50.0),
        MinSample(method="min", window_kcalmol=5.0),
    ):
        survivors = apply(results, sample).structures
        ids = [s.id for s in survivors]
        assert set(ids) <= original
        assert len(set(ids)) == len(ids)
        assert survivors, "a filter over a non-empty input must keep at least one structure"


# ---------------------------------------------------------------------------
# nms displacement maths
# ---------------------------------------------------------------------------


@given(
    st.integers(min_value=1, max_value=12),
    st.floats(min_value=0.01, max_value=5.0, allow_nan=False),
)
@settings(max_examples=50)
def test_displacement_is_symmetric_about_the_input_geometry(n_atoms: int, amount: float) -> None:
    """The ± pair straddles the original geometry — their midpoint is where you started.

    NMS relies on this: it displaces along an imaginary mode in both directions and
    re-optimises, expecting the pair to fall into the two basins either side.
    """
    rng = np.random.default_rng(n_atoms)
    positions = rng.random((n_atoms, 3))
    mode = rng.standard_normal((n_atoms, 3))
    plus, minus = displace_along_mode(positions, mode, displacement=amount)
    assert np.allclose((plus + minus) / 2.0, positions)
    assert np.allclose(plus - positions, -(minus - positions))


@given(st.integers(min_value=1, max_value=8))
def test_zero_displacement_leaves_the_geometry_alone(n_atoms: int) -> None:
    rng = np.random.default_rng(n_atoms)
    positions = rng.random((n_atoms, 3))
    plus, minus = displace_along_mode(
        positions, rng.standard_normal((n_atoms, 3)), displacement=0.0
    )
    assert np.array_equal(plus, positions)
    assert np.array_equal(minus, positions)


@given(st.integers(min_value=1, max_value=6), st.integers(min_value=1, max_value=6))
def test_displacement_rejects_a_mode_of_the_wrong_shape(n_atoms: int, n_modes: int) -> None:
    """A mode tensor from a different molecule must fail loudly, not broadcast."""
    assume(n_atoms != n_modes)
    import pytest

    with pytest.raises(ValueError, match="shape mismatch"):
        displace_along_mode(np.zeros((n_atoms, 3)), np.zeros((n_modes, 3)), displacement=1.0)


# ---------------------------------------------------------------------------
# The parse boundary — what makes `energies` above a safe strategy
# ---------------------------------------------------------------------------


@given(st.floats())
@settings(max_examples=200)
def test_only_a_finite_energy_survives_the_parse_boundary(tmp_path_factory, value: float) -> None:
    """No float reaches a ``Structure`` unless it is finite — or parsing raises.

    Every filter property above draws from ``energies``, which excludes NaN and infinity.
    That exclusion is only legitimate if something *enforces* it, and this is the
    enforcement point: a diverged calculation writes ``nan``/``inf`` and the boundary must
    turn that into an ``OutputParseError``, never a rankable energy. Asserted over the whole
    float domain rather than the well-behaved subset, because the well-behaved subset is
    exactly what let a NaN through before.

    ``tmp_path_factory`` is session-scoped, so hypothesis may reuse it across examples;
    one file, rewritten per example, is what that scope allows.
    """
    import json

    import pytest

    from chemrefine.engines._script.output import _load_output_json
    from chemrefine.errors import OutputParseError

    out = tmp_path_factory.getbasetemp() / "parse_boundary.json"
    out.write_text(json.dumps({"energy_hartree": value}), encoding="utf-8")
    if np.isfinite(value):
        assert _load_output_json(out, label="MLIP")["energy_hartree"] == value
    else:
        with pytest.raises(OutputParseError):
            _load_output_json(out, label="MLIP")

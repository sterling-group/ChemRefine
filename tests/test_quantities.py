"""Tests for ``chemrefine.quantities`` — physical constants + unit helpers."""

from __future__ import annotations

import numpy as np
import pytest

from chemrefine.quantities import (
    DEFAULT_TEMPERATURE_K,
    HARTREE_TO_KCALMOL,
    R_KCALMOL_K,
    boltzmann_weights,
    convert,
)


def test_constants_use_codata_values():
    # 1 Ha ≈ 627.509 kcal/mol; allow loose float tolerance for derivation chain.
    assert abs(HARTREE_TO_KCALMOL - 627.5094740631) < 1e-3
    # R ~= 1.9872041e-3 kcal/(mol*K)
    assert abs(R_KCALMOL_K - 1.98720425e-3) < 1e-7


def test_convert_scalar_hartree_to_kcalmol():
    assert abs(convert(1.0, "hartree", "kcal/mol") - HARTREE_TO_KCALMOL) < 1e-9


def test_convert_round_trips_through_kcalmol():
    rt = convert(convert(0.5, "hartree", "kcal/mol"), "kcal/mol", "hartree")
    assert abs(rt - 0.5) < 1e-12


def test_convert_array_input_returns_ndarray():
    out = convert(np.array([1.0, 2.0]), "hartree", "kcal/mol")
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, [HARTREE_TO_KCALMOL, 2.0 * HARTREE_TO_KCALMOL])


def test_convert_same_unit_is_identity():
    assert convert(3.14, "hartree", "hartree") == 3.14


def test_convert_recognises_aliases():
    assert convert(1.0, "Ha", "kcal") == HARTREE_TO_KCALMOL
    assert abs(convert(1.0, "Å", "bohr") - 1.0 / 0.529177210903) < 1e-9


def test_convert_unknown_pair_raises():
    with pytest.raises(ValueError):
        convert(1.0, "ergs", "joules")


def test_convert_length_pair():
    from chemrefine.quantities import BOHR_TO_ANGSTROM

    assert abs(convert(1.0, "bohr", "angstrom") - BOHR_TO_ANGSTROM) < 1e-12


def test_convert_gradient_pair():
    from chemrefine.quantities import HARTREE_PER_BOHR_TO_EV_PER_A

    assert abs(convert(1.0, "hartree/bohr", "ev/angstrom") - HARTREE_PER_BOHR_TO_EV_PER_A) < 1e-9


# ---------------------------------------------------------------------------
# Boltzmann
# ---------------------------------------------------------------------------


def test_boltzmann_weights_sum_to_one():
    weights = boltzmann_weights([0.0, 1.0, 2.0])
    assert abs(weights.sum() - 1.0) < 1e-12


def test_boltzmann_weights_lowest_energy_dominates():
    weights = boltzmann_weights([0.0, 100.0, 100.0])
    assert weights[0] > 0.99


def test_boltzmann_weights_equal_energies_give_uniform():
    weights = boltzmann_weights([5.0, 5.0, 5.0, 5.0])
    np.testing.assert_allclose(weights, [0.25, 0.25, 0.25, 0.25])


def test_boltzmann_weights_uses_default_temperature():
    """The default temperature should be 298.15 K."""
    w1 = boltzmann_weights([0.0, 1.0])
    w2 = boltzmann_weights([0.0, 1.0], temperature_k=DEFAULT_TEMPERATURE_K)
    np.testing.assert_allclose(w1, w2)


def test_boltzmann_weights_returns_zeros_when_all_underflow():
    """At absurdly high relative energies all exponentials underflow to 0."""
    weights = boltzmann_weights([1e9, 1e9, 1e9])
    np.testing.assert_array_equal(weights, [0.0, 0.0, 0.0])

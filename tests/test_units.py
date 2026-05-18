"""Tests for energy-unit conversions and Boltzmann weights."""

from __future__ import annotations

import numpy as np

from chemrefine.units import (
    DEFAULT_TEMPERATURE_K,
    HARTREE_TO_KCAL_MOL,
    R_KCAL_MOL_K,
    boltzmann_weights,
    hartree_to_kcal,
    kcal_to_hartree,
)


def test_hartree_kcal_constants_match_reference():
    # 1 Hartree = 627.5095 kcal/mol to 4 sig fig
    assert abs(HARTREE_TO_KCAL_MOL - 627.5094740631) < 1e-3
    # R in kcal/(mol K) = 1.9872e-3
    assert abs(R_KCAL_MOL_K - 1.9872041e-3) < 1e-7


def test_hartree_to_kcal_roundtrip():
    energies = np.array([0.0, -1.5, 0.25])
    rt = kcal_to_hartree(hartree_to_kcal(energies))
    np.testing.assert_allclose(rt, energies, rtol=1e-12)


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

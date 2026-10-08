"""Dependency-free rejection of corrupted chemistry at the fermionic boundary."""

from types import SimpleNamespace

import numpy as np
import pytest

from chemrefine.engines.qiskit.fermionic import _real_dense, fermionic_integrals
from chemrefine.errors import ConfigError


@pytest.mark.parametrize("tensor,message", [([[1j]], "real"), ([[float("nan")]], "finite")])
def test_real_integral_adapter_never_drops_invalid_values(tensor, message):
    """Integral conversion rejects imaginary terms and nonfinite matrix entries."""
    with pytest.raises(ConfigError, match=message):
        _real_dense(tensor, name="one-body")


@pytest.mark.parametrize("failure", [None, "dimensions", "symmetry", "occupations"])
def test_one_body_only_adapter_validates_shape_symmetry_and_particles(failure):
    """A one-body Hamiltonian needs no optional provider until actual simulation."""
    h1 = np.diag([-1.0, 0.5])
    if failure == "dimensions":
        h1 = np.zeros((3, 3))
    elif failure == "symmetry":
        h1[0, 1] = 1
    integrals = SimpleNamespace(alpha={"+-": h1}, beta={}, beta_alpha={})
    problem = SimpleNamespace(
        hamiltonian=SimpleNamespace(electronic_integrals=integrals),
        properties=SimpleNamespace(angular_momentum=None),
        orbital_occupations=[1, 0],
        orbital_occupations_b=[1, 0],
    )
    if failure == "occupations":
        problem.orbital_occupations = [1, 1]
    prepared = SimpleNamespace(
        problem=problem, num_spatial_orbitals=2, num_particles=(1, 1), multiplicity=1
    )
    if failure:
        with pytest.raises(ConfigError):
            fermionic_integrals(prepared)
    else:
        data = fermionic_integrals(prepared)
        assert np.count_nonzero(data.h2) == 0
        assert data.occupations == ((0,), (0,))

"""Independent molecular integral contracts and active-space preparation tests."""

import numpy as np
import pytest
from pydantic import ValidationError

from chemrefine.engines.qiskit.data import ElectronicStructureData, MolecularMetadata
from chemrefine.engines.qiskit.options import ActiveSpaceOptions, QiskitOptions
from chemrefine.errors import ConfigError


def minimal_data(**kwargs):
    """Return a two-orbital input requiring no optional chemistry packages."""
    values = {
        "num_alpha": 1,
        "num_beta": 1,
        "num_spatial_orbitals": 2,
        "one_body_integrals": np.diag([-1.0, -0.5]),
        "two_body_integrals": np.zeros((2, 2, 2, 2)),
    }
    values.update(kwargs)
    return ElectronicStructureData(**values)


def test_integral_data_copies_arrays_and_resolves_occupations():
    source = np.eye(2)
    data = minimal_data(one_body_integrals=source)
    source[0, 0] = 12
    assert data.one_body_integrals[0, 0] == 1
    assert not data.one_body_integrals.flags.writeable
    assert data.num_particles == (1, 1)
    assert data.multiplicity == 1
    np.testing.assert_array_equal(data.orbital_occupations, [1, 0])


@pytest.mark.parametrize("name", ["one_body_integrals", "two_body_integrals"])
def test_primary_integral_blocks_cannot_be_none(name):
    """Only optional beta blocks may be omitted from an electronic Hamiltonian."""
    with pytest.raises(ConfigError, match=f"{name} is required"):
        minimal_data(**{name: None})


@pytest.mark.parametrize(
    ("alpha", "beta", "multiplicity"),
    [(2, 2, 3), (2, 1, 4), (1, 2, 4)],
)
def test_multiplicity_cannot_exceed_spin_orbital_capacity(alpha, beta, multiplicity):
    """Fully paired orbitals cannot supply unpaired spins even at a sufficient electron count."""
    with pytest.raises(ConfigError, match="multiplicity"):
        minimal_data(num_alpha=alpha, num_beta=beta, multiplicity=multiplicity)


@pytest.mark.parametrize(
    ("alpha", "beta", "multiplicity"),
    [(2, 2, 1), (2, 1, 2), (1, 2, 2), (2, 0, 3), (1, 1, 3)],
)
def test_allowed_spin_capacity_boundaries_remain_valid(alpha, beta, multiplicity):
    """Both electron-rich and hole-rich populations admit their physical spin limits."""
    data = minimal_data(num_alpha=alpha, num_beta=beta, multiplicity=multiplicity)
    assert data.multiplicity == multiplicity
    assert data.num_particles == (alpha, beta)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"num_spatial_orbitals": 0}, "counts"),
        ({"num_alpha": True}, "counts"),
        ({"num_beta": -1}, "alpha/beta"),
        ({"num_alpha": 3}, "alpha/beta"),
        ({"two_body_order": "automatic"}, "two_body_order"),
        ({"multiplicity": 2}, "multiplicity"),
        ({"multiplicity": 0}, "multiplicity"),
        ({"multiplicity": True}, "multiplicity"),
        ({"multiplicity": 5}, "multiplicity"),
        ({"one_body_integrals": [[1]]}, "shape"),
        ({"one_body_integrals": [[1, 2], [0, 1]]}, "Hermitian"),
        ({"one_body_integrals": np.full((2, 2), np.nan)}, "finite"),
        ({"one_body_integrals": np.eye(2, dtype=complex) * 1j}, "Hermitian"),
        ({"one_body_integrals": "bad"}, "valid numbers"),
        ({"two_body_integrals": np.ones((2, 2))}, "shape"),
        ({"one_body_integrals_beta": np.eye(2)}, "three beta"),
        ({"orbital_energies": [0]}, "shape"),
        ({"orbital_occupations": [0.5, 0.5]}, "0/1"),
        ({"orbital_occupations_beta": [1, 1]}, "summing"),
        ({"nuclear_repulsion_energy": float("inf")}, "nuclear"),
        ({"nuclear_repulsion_energy": -1}, "nuclear"),
        ({"nuclear_repulsion_energy": "bad"}, "nuclear"),
        ({"nuclear_repulsion_energy": True}, "nuclear"),
        ({"provenance": {"bad": object()}}, "JSON-compatible"),
        ({"metadata": {"bad": float("nan")}}, "JSON-compatible"),
        ({"metadata": []}, "mapping"),
    ],
)
def test_integral_data_rejects_inconsistent_science(options, message):
    with pytest.raises(ConfigError, match=message):
        minimal_data(**options)


def test_unrestricted_requires_complete_blocks_and_overlap():
    fields = {
        "one_body_integrals_beta": np.eye(2),
        "two_body_integrals_beta_beta": np.zeros((2,) * 4),
        "two_body_integrals_beta_alpha": np.zeros((2,) * 4),
    }
    with pytest.raises(ConfigError, match="overlap"):
        minimal_data(**fields)
    data = minimal_data(**fields, overlap_alpha_beta=np.eye(2), two_body_order="physicist")
    np.testing.assert_array_equal(data.overlap_alpha_beta, np.eye(2))
    assert data.multiplicity == 1


def test_metadata_and_explicit_active_space_validation():
    molecule = MolecularMetadata(("H", "H"), ((0, 0, 0), (0, 0, 0.735)))
    assert molecule.charge == 0
    assert QiskitOptions(freeze_core=True).freeze_core
    assert ActiveSpaceOptions(orbitals=2, active_orbitals=[0, 3]).active_orbitals == [0, 3]
    for indices in ([0], [0, 0], [-1, 0], [0, True]):
        with pytest.raises(ValidationError, match="active_orbitals"):
            ActiveSpaceOptions(orbitals=2, active_orbitals=indices)
    with pytest.raises(ConfigError, match="symbols"):
        MolecularMetadata(("Bad",), ((0, 0, 0),))
    with pytest.raises(ConfigError, match="charge"):
        MolecularMetadata(("H",), ((0, 0, 0),), charge=True)
    with pytest.raises(ConfigError, match="coordinates"):
        MolecularMetadata(("H",), ((0, 0),))

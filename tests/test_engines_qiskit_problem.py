"""Small deterministic integral/active-space tests using the optional Qiskit stack."""

import builtins
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from chemrefine.engines.qiskit.data import ElectronicStructureData, MolecularMetadata
from chemrefine.engines.qiskit.options import ActiveSpaceOptions, QiskitOptions
from chemrefine.engines.qiskit.problem import prepare_problem, prepare_pyscf_problem
from chemrefine.errors import ConfigError

pytest.importorskip("qiskit_nature")


@pytest.fixture
def h2_data():
    """Load fixed RHF/STO-3G integrals for H2 at 0.735 angstrom."""
    path = Path(__file__).parent / "data/engines/qiskit/h2_integrals.json"
    return ElectronicStructureData(**json.loads(path.read_text()))


def exact_energy(prepared):
    """Solve a prepared problem in its particle sector using the real Qiskit stack."""
    from qiskit_algorithms import NumPyMinimumEigensolver
    from qiskit_nature.second_q.algorithms import GroundStateEigensolver
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    solver = NumPyMinimumEigensolver(
        filter_criterion=prepared.problem.get_default_filter_criterion()
    )
    result = GroundStateEigensolver(JordanWignerMapper(), solver).solve(prepared.problem)
    return result.total_energies[0]


def test_h2_problem_constructs_without_pyscf_and_exact_energy_is_stable(h2_data, monkeypatch):
    original_import = builtins.__import__

    def no_pyscf(name, *args, **kwargs):
        if name == "pyscf" or name.startswith("pyscf."):
            raise AssertionError("integral preparation must not import PySCF")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pyscf)
    prepared = prepare_problem(h2_data)
    assert prepared.num_particles == (1, 1)
    assert prepared.num_spatial_orbitals == 2
    assert prepared.num_spin_orbitals == 4
    assert prepared.original_num_spatial_orbitals == 2
    assert prepared.active_orbitals == [0, 1]
    assert prepared.fermionic_hamiltonian.num_spin_orbitals == 4
    assert prepared.provenance["source"] == "stored_pyscf_integrals"
    assert prepared.energy_offsets == {"nuclear_repulsion_energy": h2_data.nuclear_repulsion_energy}
    assert exact_energy(prepared) == pytest.approx(-1.1373060357534, abs=1e-10)


def test_declared_integral_order_and_unrestricted_blocks_agree(h2_data):
    restricted = prepare_problem(h2_data)
    physicist = prepare_problem(
        replace(
            h2_data,
            two_body_integrals=h2_data.two_body_integrals.transpose(0, 2, 3, 1),
            two_body_order="physicist",
        )
    )
    unrestricted = prepare_problem(
        replace(
            h2_data,
            one_body_integrals_beta=h2_data.one_body_integrals,
            two_body_integrals_beta_beta=h2_data.two_body_integrals,
            two_body_integrals_beta_alpha=h2_data.two_body_integrals,
            overlap_alpha_beta=np.eye(2),
            orbital_energies_beta=h2_data.orbital_energies,
        )
    )
    assert restricted.fermionic_hamiltonian.equiv(physicist.fermionic_hamiltonian)
    assert restricted.fermionic_hamiltonian.equiv(unrestricted.fermionic_hamiltonian)
    assert exact_energy(unrestricted) == pytest.approx(exact_energy(restricted), abs=1e-12)


def test_explicit_orbital_order_is_preserved_and_constants_match_hf(h2_data):
    reordered = prepare_problem(h2_data, active_space=ActiveSpaceOptions(active_orbitals=[1, 0]))
    assert reordered.active_orbitals == [1, 0]
    np.testing.assert_array_equal(reordered.problem.orbital_occupations, [0, 1])
    assert exact_energy(reordered) == pytest.approx(exact_energy(prepare_problem(h2_data)))
    reduced = prepare_problem(
        h2_data, active_space=ActiveSpaceOptions(electrons=2, orbitals=1, active_orbitals=[0])
    )
    assert reduced.num_spatial_orbitals == 1
    assert exact_energy(reduced) == pytest.approx(-1.1169989967540044, abs=1e-12)


def synthetic_lih():
    """Return a simple four-electron Hamiltonian with exactly known inactive offsets."""
    return ElectronicStructureData(
        2,
        2,
        4,
        np.diag([-2.0, -1.0, 0.0, 1.0]),
        np.zeros((4,) * 4),
        molecular_metadata=MolecularMetadata(("Li", "H"), ((0, 0, 0), (0, 0, 1.6))),
        nuclear_repulsion_energy=1.0,
    )


def test_freeze_then_explicit_original_indices_retains_both_offsets():
    data = synthetic_lih()
    frozen = prepare_problem(data, freeze_core=True)
    assert frozen.active_orbitals == [1, 2, 3]
    assert frozen.num_particles == (1, 1)
    assert frozen.energy_offsets["FreezeCoreTransformer"] == -4
    reduced = prepare_problem(
        data,
        freeze_core=True,
        active_space=ActiveSpaceOptions(electrons=2, orbitals=2, active_orbitals=[1, 3]),
    )
    assert reduced.active_orbitals == [1, 3]
    assert reduced.energy_offsets == {
        "nuclear_repulsion_energy": 1.0,
        "FreezeCoreTransformer": -4.0,
        "ActiveSpaceTransformer": 0.0,
    }
    assert exact_energy(reduced) == pytest.approx(-5.0)
    assert reduced.metadata["transformations"][1]["active_orbitals"] == [1, 3]


def test_count_based_active_space_and_no_core_hydrogen(h2_data):
    reduced = prepare_problem(
        synthetic_lih(), active_space=ActiveSpaceOptions(electrons=2, orbitals=2)
    )
    assert reduced.active_orbitals == [1, 2]
    assert reduced.energy_offsets["ActiveSpaceTransformer"] == -4
    hydrogen = replace(
        h2_data, molecular_metadata=MolecularMetadata(("H", "H"), ((0, 0, 0), (0, 0, 0.735)))
    )
    frozen = prepare_problem(hydrogen, freeze_core=True)
    assert frozen.active_orbitals == [0, 1]
    assert "FreezeCoreTransformer" not in frozen.energy_offsets


@pytest.mark.parametrize(
    "space, message",
    [
        (ActiveSpaceOptions(electrons=1, orbitals=1), "odd active"),
        (ActiveSpaceOptions(electrons=4, orbitals=2), "inactive count"),
        (ActiveSpaceOptions(electrons=(1, 0), orbitals=2), "inactive count"),
        (ActiveSpaceOptions(electrons=2, orbitals=3), "exceeds"),
        (ActiveSpaceOptions(electrons=2, orbitals=1, active_orbitals=[1]), "occupations"),
        (ActiveSpaceOptions(electrons=2, orbitals=1, active_orbitals=[2]), "unavailable"),
    ],
)
def test_invalid_active_space_reports_descriptive_error(h2_data, space, message):
    with pytest.raises(ConfigError, match=message):
        prepare_problem(h2_data, active_space=space)


def test_freeze_core_rejects_missing_metadata_wrong_electrons_and_frozen_indices(h2_data):
    with pytest.raises(ConfigError, match="metadata"):
        prepare_problem(h2_data, freeze_core=True)
    data = synthetic_lih()
    with pytest.raises(ConfigError, match="all-electron"):
        prepare_problem(
            replace(
                data,
                molecular_metadata=MolecularMetadata(
                    ("Li", "H"), ((0, 0, 0), (0, 0, 1.6)), charge=1
                ),
            ),
            freeze_core=True,
        )
    with pytest.raises(ConfigError, match="frozen core"):
        prepare_problem(
            data, freeze_core=True, active_space=ActiveSpaceOptions(active_orbitals=[0, 1])
        )


def test_pyscf_adapter_uses_shared_preparation(h2_data, monkeypatch, tmp_path):
    from qiskit_nature.second_q import drivers

    calls = []
    native = prepare_problem(h2_data).problem

    class Driver:
        """Record geometry adapter options while avoiding a PySCF calculation."""

        def __init__(self, **kwargs):
            calls.append(kwargs)

        def run(self):
            return native

    monkeypatch.setattr(drivers, "PySCFDriver", Driver)
    xyz = tmp_path / "h2.xyz"
    xyz.write_text("2\n\nH 0 0 0\nH 0 0 0.735\n")
    options = QiskitOptions(active_space=ActiveSpaceOptions(electrons=2, orbitals=1))
    prepared = prepare_pyscf_problem(xyz, charge=0, multiplicity=1, options=options)
    assert calls[0]["basis"] == "sto-3g"
    assert prepared.num_spatial_orbitals == 1
    assert prepared.provenance["source"] == "pyscf"
    with pytest.raises(ConfigError, match="multiplicity"):
        prepare_pyscf_problem(xyz, charge=0, multiplicity=0, options=options)

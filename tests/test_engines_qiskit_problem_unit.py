"""Optional-stack-free tests of the integral and transformation API boundary."""

from copy import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_engines_qiskit import _install_module

from chemrefine.engines.qiskit.active_space import _reduce, apply_active_space
from chemrefine.engines.qiskit.data import ElectronicStructureData, MolecularMetadata
from chemrefine.engines.qiskit.options import ActiveSpaceOptions, QiskitOptions
from chemrefine.engines.qiskit.problem import _atom_spec, prepare_problem, prepare_pyscf_problem
from chemrefine.errors import ConfigError


@pytest.fixture
def nature_boundary(monkeypatch):
    """Record the Nature constructor contract without performing integral algebra."""
    state = SimpleNamespace(integrals=[], transforms=[], properties=[], wrong_particles=False)

    class Energy:
        """Minimal electronic-energy protocol, including unmapped constants."""

        def __init__(self, n):
            self.n = n
            self.constants = {}

        @classmethod
        def from_raw_integrals(cls, *args, **kwargs):
            state.integrals.append((args, kwargs))
            return cls(len(args[0]))

        @property
        def nuclear_repulsion_energy(self):
            return self.constants.get("nuclear_repulsion_energy")

        @nuclear_repulsion_energy.setter
        def nuclear_repulsion_energy(self, value):
            self.constants["nuclear_repulsion_energy"] = value

        def second_q_op(self):
            return SimpleNamespace(num_spin_orbitals=2 * self.n)

    class Problem:
        """Keep the metadata assigned by the engine's public construction API."""

        def __init__(self, energy):
            self.hamiltonian = energy
            self.molecule = None
            self.properties = SimpleNamespace(add=state.properties.append)

    class Active:
        """Record transformer arguments and return independent transformed metadata."""

        def __init__(self, particles, orbitals, *, active_orbitals):
            self.particles = particles
            self.orbitals = orbitals
            self.indices = active_orbitals
            state.transforms.append((particles, orbitals, active_orbitals))

        def transform(self, problem):
            result = copy(problem)
            result.hamiltonian = Energy(self.orbitals)
            result.hamiltonian.constants = {
                **problem.hamiltonian.constants,
                "ActiveSpaceTransformer": -10.0,
            }
            result.num_spatial_orbitals = self.orbitals
            result.num_particles = (99, 99) if state.wrong_particles else self.particles
            result.orbital_occupations = problem.orbital_occupations[self.indices]
            result.orbital_occupations_b = problem.orbital_occupations_b[self.indices]
            return result

    class Freeze:
        """Expose the public element-count protocol consumed by freeze-core."""

        def Z(self, symbol):
            return {"Li": 3, "H": 1, "He": 2}[symbol]

        def count_core_orbitals(self, symbols):
            return sum(symbol == "Li" for symbol in symbols)

    _install_module(
        monkeypatch, "qiskit_nature.second_q.formats.molecule_info", MoleculeInfo=SimpleNamespace
    )
    _install_module(monkeypatch, "qiskit_nature.second_q.hamiltonians", ElectronicEnergy=Energy)
    _install_module(
        monkeypatch,
        "qiskit_nature.second_q.problems",
        ElectronicBasis=SimpleNamespace(MO="MO"),
        ElectronicStructureProblem=Problem,
    )
    _install_module(
        monkeypatch,
        "qiskit_nature.second_q.properties",
        AngularMomentum=lambda n, overlap: ("spin", n, overlap),
        Magnetization=lambda n: ("magnetization", n),
        ParticleNumber=lambda n: ("particles", n),
    )
    _install_module(
        monkeypatch,
        "qiskit_nature.second_q.transformers",
        ActiveSpaceTransformer=Active,
        FreezeCoreTransformer=Freeze,
    )
    return state


def input_data(**changes):
    """Construct a four-electron contract fixture with a nontrivial integral tensor."""
    eri = np.zeros((4,) * 4)
    eri[0, 0, 1, 1] = eri[1, 1, 0, 0] = 0.2
    data = ElectronicStructureData(
        2,
        2,
        4,
        np.diag([-2.0, -1.0, 0.0, 1.0]),
        eri,
        nuclear_repulsion_energy=1.0,
        molecular_metadata=MolecularMetadata(("Li", "H"), ((0, 0, 0), (0, 0, 1.6))),
    )
    return replace(data, **changes)


@pytest.mark.parametrize("order", ["chemist", "physicist"])
@pytest.mark.parametrize("unrestricted", [False, True])
def test_prepare_passes_explicit_integral_convention_and_spin_metadata(
    nature_boundary, order, unrestricted
):
    data = input_data()
    eri = np.asarray(data.two_body_integrals)
    if order == "physicist":
        eri = eri.transpose(0, 2, 3, 1)
    changes = {"two_body_order": order, "two_body_integrals": eri}
    if unrestricted:
        changes.update(
            one_body_integrals_beta=data.one_body_integrals,
            two_body_integrals_beta_beta=eri,
            two_body_integrals_beta_alpha=eri,
            overlap_alpha_beta=np.eye(4),
            orbital_energies=np.arange(4),
            orbital_energies_beta=np.arange(4),
        )
    prepared = prepare_problem(replace(data, **changes))
    args, kwargs = nature_boundary.integrals[-1]
    assert kwargs == {"auto_index_order": False}
    expected = eri.transpose(0, 2, 3, 1) if order == "chemist" else eri
    np.testing.assert_array_equal(args[1], expected)
    if unrestricted:
        np.testing.assert_array_equal(args[4], expected)
    else:
        assert args[2:] == (None, None, None)
    assert prepared.problem.basis == "MO"
    assert prepared.problem.molecule.charge == 0
    assert prepared.num_particles == (2, 2)
    assert prepared.num_spatial_orbitals == 4
    assert prepared.num_spin_orbitals == 8
    assert prepared.fermionic_hamiltonian.num_spin_orbitals == 8
    assert prepared.energy_offsets == {"nuclear_repulsion_energy": 1.0}
    assert len(nature_boundary.properties) == 3


def test_missing_optional_molecule_nuclear_energy_and_noop_core(nature_boundary):
    data = input_data(molecular_metadata=None, nuclear_repulsion_energy=None)
    result = prepare_problem(data)
    assert result.energy_offsets == {}
    assert result.problem.molecule is None
    assert result.metadata["transformations"] == []
    hydrogen = input_data(
        molecular_metadata=MolecularMetadata(("H", "H", "H", "H"), ((0, 0, 0),) * 4)
    )
    result = prepare_problem(hydrogen, freeze_core=True)
    assert result.active_orbitals == [0, 1, 2, 3]
    assert nature_boundary.transforms == []


def test_transform_composition_preserves_original_indices_and_accumulates_constants(
    nature_boundary,
):
    result = prepare_problem(
        input_data(), freeze_core=True, active_space=ActiveSpaceOptions(active_orbitals=[1, 3])
    )
    assert nature_boundary.transforms == [((1, 1), 3, [1, 2, 3]), ((1, 1), 2, [0, 2])]
    assert result.active_orbitals == [1, 3]
    assert result.original_num_spatial_orbitals == 4
    assert result.energy_offsets == {
        "nuclear_repulsion_energy": 1.0,
        "FreezeCoreTransformer": -10,
        "ActiveSpaceTransformer": -10,
    }
    again, indices, _ = apply_active_space(result.problem, active_space=ActiveSpaceOptions())
    assert indices == [0, 1]
    assert again.hamiltonian.constants["ActiveSpaceTransformer_2"] == -10
    # The original transformed problem is never mutated by composition.
    assert "ActiveSpaceTransformer_2" not in result.energy_offsets
    default = prepare_problem(input_data(), active_space=ActiveSpaceOptions())
    assert default.active_orbitals == [1, 2]


@pytest.mark.parametrize(
    "indices, particles, match",
    [
        ([], (0, 0), "distinct"),
        ([0, 0], (2, 2), "distinct"),
        ([4], (1, 1), "exceeds"),
        ([0], (0, 0), "occupations"),
    ],
)
def test_reduction_validates_input_before_calling_nature(
    nature_boundary, indices, particles, match
):
    native = prepare_problem(input_data()).problem
    with pytest.raises(ConfigError, match=match):
        _reduce(native, indices, particles, "test")
    assert nature_boundary.transforms == []


def test_reduction_checks_closed_shell_inactive_space_and_output_particle_counts(nature_boundary):
    native = prepare_problem(input_data()).problem
    native.orbital_occupations_b = np.array([1, 0, 0, 0])
    with pytest.raises(ConfigError, match="doubly occupied"):
        _reduce(native, [0], (1, 1), "test")
    nature_boundary.wrong_particles = True
    with pytest.raises(ConfigError, match="inconsistent electron"):
        _reduce(native, [0, 1], (2, 1), "test")


@pytest.mark.parametrize(
    "changes, freeze, active, match",
    [
        ({"molecular_metadata": None}, True, None, "metadata"),
        (
            {
                "molecular_metadata": MolecularMetadata(
                    ("Li", "H"), ((0, 0, 0), (0, 0, 1.6)), charge=1
                )
            },
            True,
            None,
            "all-electron",
        ),
        ({}, True, ActiveSpaceOptions(active_orbitals=[0, 1]), "frozen core"),
        ({}, False, ActiveSpaceOptions(electrons=1, orbitals=1), "odd active"),
        ({}, False, ActiveSpaceOptions(electrons=6, orbitals=3), "inactive count"),
        ({}, False, ActiveSpaceOptions(electrons=(2, 1), orbitals=2), "inactive count"),
    ],
)
def test_active_configuration_failures_are_descriptive(
    nature_boundary, changes, freeze, active, match
):
    with pytest.raises(ConfigError, match=match):
        prepare_problem(input_data(**changes), freeze_core=freeze, active_space=active)


def test_freeze_core_rejects_removing_every_orbital(nature_boundary):
    data = ElectronicStructureData(
        1,
        1,
        1,
        [[-1]],
        [[[[0]]]],
        molecular_metadata=MolecularMetadata(("Li",), ((0, 0, 0),), charge=1),
    )
    with pytest.raises(ConfigError, match="unavailable orbitals"):
        prepare_problem(data, freeze_core=True)


def test_xyz_adapter_and_parser_failures(nature_boundary, monkeypatch, tmp_path):
    native = prepare_problem(input_data()).problem
    calls = []

    def driver(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(run=lambda: native)

    _install_module(monkeypatch, "qiskit_nature.second_q.drivers", PySCFDriver=driver)
    _install_module(
        monkeypatch, "qiskit_nature.units", DistanceUnit=SimpleNamespace(ANGSTROM="angstrom")
    )
    xyz = tmp_path / "input.xyz"
    xyz.write_text("2\n\nLi 0 0 0\nH 0 0 1.6\n")
    options = QiskitOptions(freeze_core=True)
    result = prepare_pyscf_problem(xyz, charge=0, multiplicity=1, options=options)
    assert result.provenance["source"] == "pyscf"
    assert calls == [
        {
            "atom": "Li 0 0 0; H 0 0 1.6",
            "unit": "angstrom",
            "charge": 0,
            "spin": 0,
            "basis": "sto-3g",
        }
    ]
    with pytest.raises(ConfigError, match="multiplicity"):
        prepare_pyscf_problem(xyz, charge=0, multiplicity=0, options=options)
    with pytest.raises(ConfigError, match="cannot read"):
        _atom_spec(tmp_path / "missing.xyz")
    for content, match in [
        ("bad\n", "cannot read"),
        ("2\n\nH 0 0 0\n", "declares"),
        ("1\n\nH 0 0\n", "malformed"),
    ]:
        xyz.write_text(content)
        with pytest.raises(ConfigError, match=match):
            _atom_spec(xyz)

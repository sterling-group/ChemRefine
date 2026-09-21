"""Dependency-free contracts for optional UCC, operator, and ADAPT implementation seams."""

import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_engines_qiskit import _install_module

from chemrefine.engines.qiskit.adapt import AdaptDiagnostics, tracked_adapt_vqe
from chemrefine.engines.qiskit.components.ansatze import (
    UCCOptions,
    UCCSDOptions,
    build_ucc,
    build_uccsd,
)
from chemrefine.engines.qiskit.context import ElectronicStructureContext
from chemrefine.engines.qiskit.operators import OperatorPool, ucc_pool_metadata
from chemrefine.errors import ConfigError


@pytest.fixture
def fake_ucc(monkeypatch):
    """Exercise Nature's public excitation-callable protocol, without circuit algebra."""
    generated = []
    circuits = []

    def generate(rank, orbitals, particles, **options):
        generated.append((rank, orbitals, particles, options))
        return [((0,), (1,)), ((2,), (3,))] if rank == 1 else [((0, 2), (1, 3))]

    class UCC:
        """Record constructor options and lazily evaluate supplied excitations."""

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.excitations = kwargs.get("excitations")
            self.num_spatial_orbitals = args[0] if args else kwargs["num_spatial_orbitals"]
            self.num_particles = args[1] if args else kwargs["num_particles"]
            self.excitation_list = None
            circuits.append(self)

        @property
        def operators(self):
            excitations = (
                self.excitations(self.num_spatial_orbitals, self.num_particles)
                if self.excitations is not None
                else [((0,), (1,)), ((2,), (3,)), ((0, 2), (1, 3))]
            )
            self.excitation_list = [
                excitation
                for excitation in excitations
                for _ in range(2 if self.kwargs.get("include_imaginary") else 1)
            ]
            return tuple(f"generator_{index}" for index in range(len(self.excitation_list)))

        def excitation_ops(self):
            raise AssertionError(
                "The engine must not regenerate Nature's cached excitation metadata"
            )

    _install_module(monkeypatch, "qiskit_nature.second_q.circuit.library", UCC=UCC, UCCSD=UCC)
    _install_module(
        monkeypatch,
        "qiskit_nature.second_q.circuit.library.ansatzes.utils",
        generate_fermionic_excitations=generate,
    )
    return generated, circuits


def context():
    """Use a reordered determinant occupying spin orbitals one and three."""
    return ElectronicStructureContext(
        problem=SimpleNamespace(orbital_occupations=[0, 1], orbital_occupations_b=[0, 1]),
        mapper=object(),
        qubit_hamiltonian=object(),
        num_spatial_orbitals=2,
        num_particles=(1, 1),
        num_qubits=4,
        multiplicity=1,
    )


@pytest.mark.parametrize("include_imaginary", [False, True])
@pytest.mark.parametrize("preserve_spin", [False, True])
def test_reordered_uccsd_maps_generated_excitations_to_actual_reference(
    fake_ucc, include_imaginary, preserve_spin
):
    generated, circuits = fake_ucc
    chemistry = context()
    state = object()
    artifacts = build_uccsd(
        options=UCCSDOptions(
            reps=2, include_imaginary=include_imaginary, preserve_spin=preserve_spin
        ),
        context=chemistry,
        initial_state=state,
    )
    assert artifacts.circuit is circuits[-1]
    assert circuits[-1].kwargs["initial_state"] is state
    assert circuits[-1].kwargs["reps"] == 2
    expected = [((1,), (0,)), ((3,), (2,)), ((1, 3), (0, 2))]
    expected = [excitation for excitation in expected for _ in range(2 if include_imaginary else 1)]
    assert circuits[-1].excitation_list == expected
    assert all(call[3] == {"preserve_spin": preserve_spin} for call in generated)
    assert {call[0] for call in generated} == {1, 2}
    assert len(artifacts.operator_pool) == len(artifacts.pool_metadata) == len(expected)
    for index, metadata in enumerate(artifacts.pool_metadata):
        assert metadata["pool_index"] == index
        assert metadata["excitation"] == {
            "occupied": list(expected[index][0]),
            "unoccupied": list(expected[index][1]),
        }
        assert metadata["generator"] == (
            "symmetric" if include_imaginary and index % 2 else "antisymmetric"
        )


def test_generalized_uccsd_keeps_natures_generator_even_for_reordered_reference(fake_ucc):
    generated, circuits = fake_ucc
    build_uccsd(options=UCCSDOptions(generalized=True), context=context(), initial_state=None)
    assert circuits[-1].excitations is None
    assert generated == []


def test_custom_ucc_preserves_supplied_indices_and_explicit_constructor_settings(fake_ucc):
    _, circuits = fake_ucc
    chemistry = context()
    supplied = (((1, 3), (0, 2)),)
    initial = object()
    artifacts = build_ucc(
        options=UCCOptions(excitations=supplied, reps=2, include_imaginary=True),
        context=chemistry,
        initial_state=initial,
    )
    assert circuits[-1].kwargs["initial_state"] is initial
    assert circuits[-1].kwargs["qubit_mapper"] is chemistry.mapper
    assert circuits[-1].kwargs["reps"] == 2
    assert circuits[-1].excitations(2, (1, 1)) == list(supplied)
    assert artifacts.pool_metadata[0]["excitation"] == {"occupied": [1, 3], "unoccupied": [0, 2]}
    assert artifacts.pool_metadata[1]["generator"] == "symmetric"
    assert len(artifacts.operator_pool) == 2


@pytest.mark.parametrize(
    "excitation, preserve_spin, message",
    [(((0,), (4,)), True, "below 4"), (((0,), (3,)), True, "preserve spin")],
)
def test_custom_ucc_rejects_invalid_runtime_excitation_science(
    fake_ucc, excitation, preserve_spin, message
):
    _, circuits = fake_ucc
    with pytest.raises(ConfigError, match=message):
        build_ucc(
            options=UCCOptions(excitations=(excitation,), preserve_spin=preserve_spin),
            context=context(),
            initial_state=None,
        )
    assert circuits == []


def test_explicit_spin_change_is_accepted_only_when_requested(fake_ucc):
    artifacts = build_ucc(
        options=UCCOptions(excitations=(((0,), (3,)),), preserve_spin=False),
        context=context(),
        initial_state=None,
    )
    assert artifacts.pool_metadata[0]["excitation"] == {"occupied": [0], "unoccupied": [3]}
    assert ucc_pool_metadata(SimpleNamespace(operators=[])) == ()


@pytest.fixture
def pauli_type(monkeypatch):
    """Expose only the SparsePauliOp protocol needed by generator validation."""

    class Pauli:
        """Return caller-controlled simplified coefficients for boundary checking."""

        def __init__(self, coefficients, qubits=2):
            self.num_qubits = qubits
            self.coeffs = np.asarray(coefficients, dtype=complex)
            self.simplified = None

        def simplify(self):
            return self.simplified if self.simplified is not None else self

    _install_module(monkeypatch, "qiskit.quantum_info", SparsePauliOp=Pauli)
    return Pauli


@pytest.mark.parametrize(
    "coefficients, qubits, message",
    [
        ([1.0], 1, "2 qubits"),
        ([np.nan], 2, "finite coefficients"),
        ([np.inf], 2, "finite coefficients"),
        ([1j], 2, "Hermitian"),
        ([0.0], 2, "zero operator"),
    ],
)
def test_external_pool_checks_simplified_numeric_generators(
    pauli_type, coefficients, qubits, message
):
    with pytest.raises(ConfigError, match=message):
        OperatorPool((pauli_type(coefficients, qubits),)).validate(2)


def test_external_pool_accepts_real_simplification_and_rejects_wrong_operator_types(pauli_type):
    operator = pauli_type([1 + 1j, -1j])
    operator.simplified = pauli_type([1])
    OperatorPool((operator,)).validate(2)
    with pytest.raises(ConfigError, match="non-empty"):
        OperatorPool(()).validate(2)
    with pytest.raises(ConfigError, match="SparsePauliOp"):
        OperatorPool((object(),)).validate(2)


@pytest.mark.parametrize("described", [False, True])
def test_adapt_observer_delegates_gradients_and_records_only_retained_sequence(
    monkeypatch, described
):
    records = []
    gradient_rounds: list[list[tuple[float, dict[str, object]]]] = [
        [(0.1, {}), (-0.6, {})],
        [(0.5, {}), (-0.5, {})],
    ]
    returned_result = object()

    class Adapt:
        """Model only the upstream callback and retained-excitation result protocol."""

        def __init__(self, solver, **options):
            records.append((solver, options))
            self._excitation_list = []
            self.round = 0

        def _compute_gradients(self, theta, operator):
            records.append((theta, operator))
            return gradient_rounds[self.round]

        def compute_minimum_eigenvalue(self, operator, aux_operators=None):
            records.append((operator, aux_operators))
            for self.round in range(2):
                assert self._compute_gradients([0.2], operator) is gradient_rounds[self.round]
            # Simulate upstream energy convergence rolling back its final candidate.
            self._excitation_list = ["retained-generator"]
            return returned_result

    _install_module(monkeypatch, "qiskit_algorithms", AdaptVQE=Adapt)
    diagnostics = AdaptDiagnostics(
        gradient_history=[{"stale": True}],
        selected_operator_indices=(9,),
        selected_operators=({"label": "old"},),
    )
    metadata: tuple[dict[str, object], ...] = (
        ({"pool_index": 0, "label": "first"}, {"pool_index": 1, "label": "second"})
        if described
        else ()
    )
    inner, operator, auxiliary = object(), object(), object()
    solver = tracked_adapt_vqe(
        inner, diagnostics=diagnostics, pool_metadata=metadata, max_iterations=5
    )
    assert records[0] == (inner, {"max_iterations": 5})
    for _ in range(2):
        assert solver.compute_minimum_eigenvalue(operator, auxiliary) is returned_result
        assert diagnostics.selected_operator_indices == (1,)
        assert diagnostics.selected_operators == (
            {"pool_index": 1, "label": "second" if described else "operator_1"},
        )
        assert diagnostics.gradient_history == [
            {
                "iteration": 1,
                "pool_index": 1,
                "gradient": -0.6,
                "max_gradient": 0.6,
                "retained": True,
            },
            {
                "iteration": 2,
                "pool_index": 0,
                "gradient": 0.5,
                "max_gradient": 0.5,
                "retained": False,
            },
        ]
    if described:
        assert diagnostics.selected_operators[0] is not metadata[1]


def test_dependency_extras_keep_integral_solver_independent_of_geometry_driver():
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    extras = project["project"]["optional-dependencies"]
    assert len(extras["qiskit-core"]) == 3
    assert all("pyscf" not in dependency.lower() for dependency in extras["qiskit-core"])
    assert {dependency.split(">=")[0] for dependency in extras["qiskit-core"]} == {
        "qiskit",
        "qiskit-nature",
        "qiskit-algorithms",
    }
    assert extras["qiskit"] == ["chemrefine[qiskit-core,pyscf]"]

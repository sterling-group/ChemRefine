"""Reference-sector symmetry reduction, exact state transforms, and chemistry integration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from chemrefine.engines.qiskit.components.initial_states import build_selected_reference
from chemrefine.engines.qiskit.data import ElectronicStructureData
from chemrefine.engines.qiskit.mapping import map_problem
from chemrefine.engines.qiskit.options import ComponentSelection
from chemrefine.engines.qiskit.problem import prepare_problem
from chemrefine.engines.qiskit.registry import ANSATZE, INITIAL_STATES
from chemrefine.engines.qiskit.tapering import Z2TaperingOptions, build_tapering_transform
from chemrefine.engines.qiskit.workflow import run_problem
from chemrefine.errors import ConfigError

pytest.importorskip("qiskit_nature")
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector, random_clifford
from scipy.linalg import expm

pytestmark = pytest.mark.filterwarnings("ignore:.*:DeprecationWarning:qiskit.*")


@pytest.fixture
def h2():
    """Use stored integrals, avoiding any independent classical-reference workflow."""
    path = Path(__file__).parent / "data/engines/qiskit/h2_integrals.json"
    return prepare_problem(ElectronicStructureData(**json.loads(path.read_text())))


@pytest.mark.parametrize("seed", [3, 7, 11])
def test_general_non_z_symmetries_states_observables_and_dynamics_are_consistent(seed):
    """Conjugated Pauli symmetries preserve the full encoded dynamics, not only its spectrum."""
    reference_clifford = random_clifford(3, seed=seed)
    reference = reference_clifford.to_circuit()
    base = SparsePauliOp.from_list(
        [("XII", 0.2), ("YXI", 0.4), ("ZII", -0.3), ("IZI", 0.7), ("IIZ", 0.15)]
    )
    full = SparsePauliOp(base.paulis.evolve(reference_clifford, frame="s"), coeffs=base.coeffs)
    transform = build_tapering_transform(full, reference)
    assert transform.num_tapered == 1
    reduced = transform.map_operator(full)
    initial = Statevector.from_instruction(reference)
    reduced_initial = Statevector.from_instruction(transform.prepare_reference(reference))
    assert abs(np.vdot(transform.lift_statevector(reduced_initial), initial.data)) == pytest.approx(
        1
    )
    evolved = expm(-0.41j * reduced.to_matrix()) @ reduced_initial.data
    expected = expm(-0.41j * full.to_matrix()) @ initial.data
    lifted = transform.lift_statevector(evolved)
    assert abs(np.vdot(lifted, expected)) == pytest.approx(1, abs=1e-12)
    observable = SparsePauliOp.from_list([("XYZ", 0.3), ("IZZ", 0.5), ("YYY", -0.2)])
    projected = transform.map_operator(observable, check_commutes=False)
    assert Statevector(evolved).expectation_value(projected) == pytest.approx(
        Statevector(lifted).expectation_value(observable), abs=1e-12
    )
    roundtrip = transform.taper_statevector(lifted)
    assert roundtrip == pytest.approx(evolved, abs=1e-12)
    assert not roundtrip.flags.writeable
    json.dumps(transform.metadata, allow_nan=False)


def test_explicit_general_reference_sector_and_zero_projected_observables():
    """An arbitrary rotation is supported when its actual symmetry sector is specified."""
    reference = QuantumCircuit(2)
    reference.x(0)
    reference.ry(0.37, 1)
    hamiltonian = SparsePauliOp.from_list([("XI", 0.4), ("ZZ", 0.3)])
    options = Z2TaperingOptions(generators=("IZ",), sectors=(-1,))
    transform = build_tapering_transform(hamiltonian, reference, options=options)
    assert transform.metadata["sectors"] == [-1]
    expected = Statevector.from_instruction(reference)
    actual = Statevector.from_instruction(transform.prepare_reference(reference))
    assert abs(np.vdot(transform.lift_statevector(actual), expected.data)) == pytest.approx(1)
    assert transform.map_operator(SparsePauliOp("IX")) is None
    assert transform.map_operator(
        SparsePauliOp("IX"), check_commutes=False
    ).coeffs == pytest.approx([0])
    mixed = SparsePauliOp.from_list([("IX", 0.8), ("XI", 0.3)])
    assert transform.map_operator(mixed, check_commutes=False).to_list() == [("X", 0.3 + 0j)]
    with pytest.raises(ConfigError, match="Clifford reference"):
        build_tapering_transform(hamiltonian, reference)
    with pytest.raises(ConfigError, match="actual reference"):
        build_tapering_transform(
            hamiltonian, reference, options=Z2TaperingOptions(generators=("IZ",), sectors=(1,))
        )
    with pytest.raises(ConfigError, match="not an eigenstate"):
        build_tapering_transform(
            SparsePauliOp("II"), reference, options=Z2TaperingOptions(generators=("XI",))
        )


@pytest.mark.parametrize("base", ["jordan_wigner", "bravyi_kitaev", "parity"])
def test_molecular_actual_reference_pool_and_observables_share_one_transform(h2, base):
    """Reference-aware Nature mapping filters UCC operators and excitation metadata together."""
    selection = ComponentSelection(name="z2_tapered", options={"base_mapper": base})
    context = map_problem(h2, selection)
    assert context.num_qubits < h2.num_spin_orbitals
    reference = build_selected_reference(context, ComponentSelection.named("hartree_fock"))
    ansatz = ANSATZE.build(
        ComponentSelection.named("uccsd"), context=context, initial_state=reference
    )
    assert ansatz.circuit.num_qubits == context.num_qubits
    assert len(ansatz.operator_pool) == len(ansatz.pool_metadata)
    assert all(operator.num_qubits == context.num_qubits for operator in ansatz.operator_pool)
    full = context.mapper.mapper.map(h2.fermionic_hamiltonian)
    initial = Statevector.from_instruction(reference)
    lift = Statevector(context.mapper.chemrefine_tapering.lift_statevector(initial))
    assert initial.expectation_value(context.qubit_hamiltonian) == pytest.approx(
        lift.expectation_value(full)
    )
    numbers = h2.problem.properties.particle_number.second_q_ops()["ParticleNumber"]
    assert initial.expectation_value(context.mapper.map_observable(numbers)) == pytest.approx(2)
    assert context.mapping_metadata["symmetry_tapering"]["base_mapper"] == base


def test_non_aufbau_selected_determinant_controls_taper_sector_and_rejects_incompatible_reference(
    h2,
):
    """Changing actual orbital occupations changes a spatial symmetry sector when required."""
    selection = ComponentSelection(name="determinant", options={"alpha": [1], "beta": [0]})
    context = map_problem(h2, "z2_tapered", initial_state=selection)
    reference = build_selected_reference(context, selection)
    state = Statevector(
        context.mapper.chemrefine_tapering.lift_statevector(Statevector.from_instruction(reference))
    )
    assert abs(state.data[6]) == pytest.approx(1)
    assert context.mapping_metadata["symmetry_tapering"][
        "reference_selection"
    ] == selection.model_dump(mode="json")
    with pytest.raises(ConfigError, match="selected tapering sector"):
        INITIAL_STATES.build(ComponentSelection.named("hartree_fock"), context=context)
    # The matching explicit-reference builder also works independently of assembly caching.
    explicit = INITIAL_STATES.build(selection, context=context)
    assert Statevector.from_instruction(explicit).equiv(Statevector.from_instruction(reference))


@pytest.mark.parametrize("algorithm", ["exact", "vqe"])
def test_tapered_molecular_energy_agrees_with_untapered_h2(h2, algorithm):
    """A real exact/VQE workflow preserves physical offsets and the molecular ground state."""
    options = {"algorithm": algorithm, "mapper": "z2_tapered"}
    if algorithm == "vqe":
        options["optimizer"] = {"name": "slsqp", "options": {"maxiter": 100, "ftol": 1e-11}}
    result = run_problem(h2, options=options)
    reference = run_problem(h2, options={"algorithm": "exact"})
    assert result.energy_hartree == pytest.approx(reference.energy_hartree, abs=2e-8)


@pytest.mark.parametrize(
    "values",
    [
        {"sectors": [1]},
        {"generators": ["Z"], "sectors": [1, -1]},
        {"min_qubits": 3, "max_qubits": 2},
        {"tolerance": 1},
    ],
)
def test_taper_options_reject_ambiguous_or_unbounded_inputs(values):
    """Reference-sector controls have one clear ordering and finite numerical tolerance."""
    with pytest.raises(ValidationError):
        Z2TaperingOptions(**values)


def test_no_compatible_symmetry_and_explicit_reduction_limits():
    """A trivial subgroup leaves the register intact; discovery limits are recorded."""
    reference = QuantumCircuit(2)
    hamiltonian = SparsePauliOp.from_list([("XI", 0.2), ("IX", 0.4)])
    transform = build_tapering_transform(hamiltonian, reference)
    assert transform.num_tapered == 0
    assert transform.map_operator(hamiltonian) == hamiltonian
    assert Statevector.from_instruction(transform.prepare_reference(reference)).equiv(
        Statevector.from_instruction(reference)
    )
    limited = build_tapering_transform(
        SparsePauliOp("III"), QuantumCircuit(3), options=Z2TaperingOptions(max_symmetries=1)
    )
    assert limited.num_tapered == 1
    assert limited.metadata["compatible_generators_found"] == 3
    with pytest.raises(ConfigError, match="retained min_qubits"):
        build_tapering_transform(
            SparsePauliOp("II"), reference, options=Z2TaperingOptions(generators=("ZI", "IZ"))
        )


@pytest.mark.parametrize(
    "label,message",
    [
        ("not a Pauli", "valid Hermitian"),
        ("Z", "wrong qubit width"),
        ("iZI", "Hermitian"),
        ("II", "identity"),
        ("XI", "does not commute"),
    ],
)
def test_invalid_explicit_symmetry_generators_fail_before_reduction(label, message):
    """Pauli labels, Hermiticity, Hamiltonian invariance and independence are checked."""
    with pytest.raises(ConfigError, match=message):
        build_tapering_transform(
            SparsePauliOp("ZI"), QuantumCircuit(2), options=Z2TaperingOptions(generators=(label,))
        )
    with pytest.raises(ConfigError, match="independent"):
        build_tapering_transform(
            SparsePauliOp("III"),
            QuantumCircuit(3),
            options=Z2TaperingOptions(generators=("ZII", "ZII")),
        )
    with pytest.raises(ConfigError, match="mutually commute"):
        build_tapering_transform(
            SparsePauliOp("III"),
            QuantumCircuit(3),
            options=Z2TaperingOptions(generators=("ZII", "XII")),
        )


def test_reference_parameters_measurements_reset_and_width_are_not_ambiguous():
    """Discovery rejects references whose pure state is unspecified or stochastic."""
    from qiskit.circuit import Parameter

    hamiltonian = SparsePauliOp("ZZ")
    parameterized = QuantumCircuit(2)
    parameterized.ry(Parameter("theta"), 0)
    measured = QuantumCircuit(2, 1)
    measured.measure(0, 0)
    reset = QuantumCircuit(2)
    reset.h(0)
    reset.reset(0)
    for reference, message in [
        (QuantumCircuit(1), "width"),
        (parameterized, "fixed parameters"),
        (measured, "classical bits"),
        (reset, "reset"),
    ]:
        with pytest.raises(ConfigError, match=message):
            build_tapering_transform(hamiltonian, reference)
    for options in [Z2TaperingOptions(max_qubits=1), Z2TaperingOptions(min_qubits=3)]:
        with pytest.raises(ConfigError, match="min_qubits/max_qubits"):
            build_tapering_transform(hamiltonian, QuantumCircuit(2), options=options)
    with pytest.raises(ConfigError, match="Hermitian"):
        build_tapering_transform(SparsePauliOp("ZZ", coeffs=[1j]), QuantumCircuit(2))


def test_state_and_operator_dimensions_normalization_and_budget_guards(h2):
    """Projection never silently accepts lost norm or a state outside the selected sector."""
    reference = QuantumCircuit(2)
    transform = build_tapering_transform(SparsePauliOp("ZZ"), reference)
    for operator in [SparsePauliOp("Z"), object()]:
        with pytest.raises(ConfigError, match="width"):
            transform.transform_operator(operator)
        with pytest.raises(ConfigError, match="width"):
            transform.project_transformed(operator)
    nonfinite = SparsePauliOp("ZZ")
    nonfinite.coeffs[0] = np.nan
    with pytest.raises(ConfigError, match="finite"):
        transform.transform_operator(nonfinite)
    with pytest.raises(ConfigError, match="finite"):
        transform.project_transformed(nonfinite)
    for state in [[1, 0], [0, 0, 0, 0], [1, 1, 0, 0]]:
        with pytest.raises(ConfigError, match="normalized"):
            transform.taper_statevector(state)
    for state in [[1, 0, 0, 0], [0, 0]]:
        with pytest.raises(ConfigError, match="normalized"):
            transform.lift_statevector(state)
    wrong = QuantumCircuit(2)
    wrong.x(0)
    with pytest.raises(ConfigError, match="selected tapering sector"):
        transform.prepare_reference(wrong)
    with pytest.raises(ConfigError, match="selected tapering sector"):
        transform.taper_statevector(Statevector.from_instruction(wrong))
    tiny = build_tapering_transform(
        SparsePauliOp("ZZ"), reference, options=Z2TaperingOptions(max_statevector_bytes=1)
    )
    with pytest.raises(ConfigError, match="max_statevector_bytes"):
        tiny.taper_statevector([1, 0, 0, 0])
    with pytest.raises(ConfigError, match="max_statevector_bytes"):
        tiny.lift_statevector([1, 0])
    nonclifford = QuantumCircuit(2)
    nonclifford.ry(0.3, 1)
    with pytest.raises(ConfigError, match="max_statevector_bytes"):
        build_tapering_transform(
            SparsePauliOp("ZZ"),
            nonclifford,
            options=Z2TaperingOptions(generators=("IZ",), max_statevector_bytes=1),
        )
    with pytest.raises(ConfigError, match="min_qubits/max_qubits"):
        map_problem(h2, ComponentSelection(name="z2_tapered", options={"max_qubits": 1}))


def test_invalid_provider_clifford_is_detected(monkeypatch):
    """A provider returning the wrong coordinate transform cannot silently change energies."""
    monkeypatch.setattr(
        "qiskit.synthesis.synth_circuit_from_stabilizers", lambda *args, **kwargs: QuantumCircuit(2)
    )
    with pytest.raises(ConfigError, match="inconsistent coordinate order"):
        build_tapering_transform(
            SparsePauliOp("ZZ"), QuantumCircuit(2), options=Z2TaperingOptions(generators=("ZZ",))
        )


def test_mapper_container_semantics_keep_pool_metadata_and_project_observables(h2):
    """Lists keep rejected-pool placeholders while observable projection preserves P O P."""
    from qiskit_nature.second_q.operators import FermionicOp

    context = map_problem(h2, "z2_tapered", initial_state="hartree_fock")
    mapper = context.mapper
    preserved = h2.fermionic_hamiltonian
    changing = FermionicOp({"+_0": 1}, num_spin_orbitals=4)
    full = mapper.map_clifford([preserved, changing])
    reduced = mapper.taper_clifford(full, suppress_none=False)
    assert reduced[0] is not None and reduced[1] is None
    assert len(mapper.taper_clifford(full)) == 1
    reduced_dict = mapper.taper_clifford({"h": full[0], "a": full[1]}, suppress_none=False)
    assert reduced_dict["a"] is None
    assert set(mapper.taper_clifford({"h": full[0], "a": full[1]})) == {"h"}
    assert mapper.map_observable(changing).coeffs == pytest.approx([0])
    # A zero initial state is a distinct actual reference and is copied without rebuilding.
    vacuum = map_problem(h2, "z2_tapered", initial_state="zero")
    chosen = build_selected_reference(vacuum, ComponentSelection.named("zero"))
    assert Statevector.from_instruction(chosen).is_valid()
    ordinary = map_problem(h2)
    assert (
        Statevector.from_instruction(
            build_selected_reference(ordinary, ComponentSelection.named("zero"))
        ).data[0]
        == 1
    )


@pytest.mark.filterwarnings("ignore:.*:scipy.sparse.SparseEfficiencyWarning")
def test_tapered_qeom_projects_composed_response_observables_before_estimation(h2):
    """Commutator products retain excursions outside the ground-state symmetry sector."""
    result = run_problem(
        h2,
        options={
            "algorithm": "qeom",
            "mapper": "z2_tapered",
            "optimizer": {"name": "slsqp", "options": {"maxiter": 200, "ftol": 1e-12}},
        },
    )
    full = map_problem(h2).qubit_hamiltonian.to_matrix()
    indices = [5, 6, 9, 10]
    expected = np.linalg.eigvalsh(full[np.ix_(indices, indices)]) + sum(h2.energy_offsets.values())
    assert result.root_energies_hartree == pytest.approx(expected, abs=2e-8)


def test_general_custom_reference_is_transformed_once_and_reused_in_assembly(h2, monkeypatch):
    """A custom non-Clifford state is never rebuilt against an incompatible reduced width."""
    from chemrefine.engines.qiskit.assembly import assemble_components
    from chemrefine.engines.qiskit.options import QiskitOptions
    from chemrefine.engines.qiskit.registry import ComponentSpec, NoComponentOptions

    calls = []

    def build(*, options, context):
        """Prepare a correlated fixed-number reference on the original register."""
        calls.append(context.num_qubits)
        circuit = INITIAL_STATES.build(ComponentSelection.named("hartree_fock"), context=context)
        circuit = circuit.decompose()
        circuit.rxx(0.3, 0, 1)
        circuit.ryy(0.3, 0, 1)
        circuit.metadata = {"custom_reference": True}
        return circuit

    monkeypatch.setitem(
        INITIAL_STATES._specs, "tapering_test_reference", ComponentSpec(NoComponentOptions, build)
    )
    options = QiskitOptions(
        algorithm="vqe",
        initial_state="tapering_test_reference",
        mapper={"name": "z2_tapered", "options": {"generators": ["IIZZ", "ZZII"]}},
    )
    context = map_problem(h2, options.mapper, initial_state=options.initial_state)
    with assemble_components(context, options) as components:
        assert components.initial_state.num_qubits == 2
        assert components.initial_state.metadata["custom_reference"] is True
    assert calls == [4]

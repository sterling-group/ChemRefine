"""Validation and optional real-stack checks for explicit excitation/operator pools."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from chemrefine.engines.qiskit.components.ansatze import (
    UCCOptions,
    UCCSDOptions,
    build_ucc,
    build_uccsd,
)
from chemrefine.engines.qiskit.context import ElectronicStructureContext
from chemrefine.engines.qiskit.operators import OperatorPool, ucc_pool_metadata
from chemrefine.errors import ConfigError

pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning:qiskit"),
    pytest.mark.filterwarnings("ignore::PendingDeprecationWarning:qiskit"),
]


def test_operator_pool_metadata_is_stable_and_json_compatible():
    descriptions = [{"label": "external_double", "pool_index": 99}]
    pool = OperatorPool((object(),), tuple(descriptions))
    assert pool.metadata == ({"label": "external_double", "pool_index": 0},)
    assert descriptions[0]["pool_index"] == 99
    assert OperatorPool((object(),)).metadata[0]["label"] == "operator_0"
    with pytest.raises(ConfigError, match="must match"):
        OperatorPool((object(),), ({}, {}))
    with pytest.raises(ConfigError, match="finite JSON"):
        OperatorPool((object(),), ({"data": object()},))
    with pytest.raises(ConfigError, match="finite JSON"):
        OperatorPool((object(),), ({"data": float("nan")},))


@pytest.mark.parametrize(
    "excitations",
    [[], [((), ())], [((0,), (1, 2))], [((0,), (0,))], [((-1,), (1,))], [((0,), (1,))] * 2],
)
def test_custom_excitation_shape_validation(excitations):
    with pytest.raises(ValidationError):
        UCCOptions(excitations=excitations)


def test_pool_metadata_can_describe_an_external_circuit_without_excitations():
    assert ucc_pool_metadata(SimpleNamespace(operators=["a"])) == (
        {"pool_index": 0, "label": "excitation_0"},
    )


def _real_context():
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_nature")
    from qiskit_nature.second_q.circuit.library import HartreeFock
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    mapper = JordanWignerMapper()
    context = ElectronicStructureContext(None, mapper, None, 2, (1, 1), 4, 1)
    return context, HartreeFock(2, (1, 1), mapper)


def test_custom_ucc_uses_supplied_excitation_and_preserves_interpretation():
    context, initial = _real_context()
    excitation = ((0, 2), (1, 3))
    artifacts = build_ucc(
        options=UCCOptions(excitations=(excitation,)), context=context, initial_state=initial
    )
    assert artifacts.circuit.excitation_list == [excitation]
    assert artifacts.circuit.num_parameters == 1
    assert len(artifacts.operator_pool) == 1
    assert artifacts.pool_metadata[0]["excitation"] == {"occupied": [0, 2], "unoccupied": [1, 3]}
    OperatorPool(tuple(artifacts.operator_pool), artifacts.pool_metadata).validate(4)


def test_ucc_rejects_out_of_range_and_unrequested_spin_changing_excitations():
    context, initial = _real_context()
    with pytest.raises(ConfigError, match="below 4"):
        build_ucc(
            options=UCCOptions(excitations=(((0,), (4,)),)),
            context=context,
            initial_state=initial,
        )
    with pytest.raises(ConfigError, match="preserve spin"):
        build_ucc(
            options=UCCOptions(excitations=(((0,), (3,)),)),
            context=context,
            initial_state=initial,
        )
    artifacts = build_ucc(
        options=UCCOptions(excitations=(((0,), (3,)),), preserve_spin=False),
        context=context,
        initial_state=initial,
    )
    assert len(artifacts.operator_pool) == 1


def test_uccsd_imaginary_generator_metadata_stays_aligned():
    context, initial = _real_context()
    artifacts = build_uccsd(
        options=UCCSDOptions(include_imaginary=True), context=context, initial_state=initial
    )
    assert artifacts.circuit.num_parameters == len(artifacts.pool_metadata) == 6
    for index in range(0, 6, 2):
        real, imaginary = artifacts.pool_metadata[index : index + 2]
        assert real["excitation"] == imaginary["excitation"]
        assert real["generator"] == "antisymmetric"
        assert imaginary["generator"] == "symmetric"
    assert len(artifacts.circuit.excitation_list) == 6


@pytest.mark.parametrize("preserve_spin", [True, False])
@pytest.mark.parametrize("include_imaginary", [False, True])
def test_uccsd_pool_uses_actual_nonprefix_reference(preserve_spin, include_imaginary):
    context, _ = _real_context()
    from qiskit import QuantumCircuit
    from qiskit_nature.second_q.circuit.library import UCCSD

    context = replace(
        context,
        problem=SimpleNamespace(orbital_occupations=[0, 1], orbital_occupations_b=[0, 1]),
    )
    initial = QuantumCircuit(4)
    initial.x([1, 3])
    artifacts = build_uccsd(
        options=UCCSDOptions(
            reps=2, preserve_spin=preserve_spin, include_imaginary=include_imaginary
        ),
        context=context,
        initial_state=initial,
    )
    assert isinstance(artifacts.circuit, UCCSD)
    count = 3 if preserve_spin else 5
    count *= 2 if include_imaginary else 1
    assert artifacts.circuit.num_parameters == count * 2
    assert len(artifacts.operator_pool) == len(artifacts.pool_metadata) == count
    for occupied, unoccupied in artifacts.circuit.excitation_list:
        assert set(occupied) <= {1, 3}
        assert set(unoccupied) <= {0, 2}
    assert artifacts.pool_metadata[-1]["excitation"] == {"occupied": [1, 3], "unoccupied": [0, 2]}


def test_generalized_uccsd_pool_is_independent_of_reference_occupation_order():
    context, initial = _real_context()
    reordered_context = replace(
        context,
        problem=SimpleNamespace(orbital_occupations=[0, 1], orbital_occupations_b=[0, 1]),
    )
    options = UCCSDOptions(generalized=True)
    canonical = build_uccsd(options=options, context=context, initial_state=initial)
    reordered = build_uccsd(options=options, context=reordered_context, initial_state=initial)
    assert canonical.circuit.excitation_list == reordered.circuit.excitation_list


def test_reordered_h2_uccsd_vqe_matches_exact_reference():
    pytest.importorskip("qiskit_algorithms")
    pytest.importorskip("qiskit_nature")
    from qiskit.primitives import StatevectorEstimator
    from qiskit_algorithms import VQE
    from qiskit_algorithms.optimizers import SLSQP
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    from chemrefine.engines.qiskit.components.initial_states import build_hartree_fock
    from chemrefine.engines.qiskit.data import ElectronicStructureData
    from chemrefine.engines.qiskit.options import ActiveSpaceOptions
    from chemrefine.engines.qiskit.problem import prepare_problem
    from chemrefine.engines.qiskit.registry import NoComponentOptions

    path = Path(__file__).parent / "data/engines/qiskit/h2_integrals.json"
    data = ElectronicStructureData(**json.loads(path.read_text()))
    prepared = prepare_problem(data, active_space=ActiveSpaceOptions(active_orbitals=[1, 0]))
    mapper = JordanWignerMapper()
    hamiltonian = mapper.map(prepared.fermionic_hamiltonian)
    context = ElectronicStructureContext(prepared.problem, mapper, hamiltonian, 2, (1, 1), 4, 1)
    initial = build_hartree_fock(options=NoComponentOptions(), context=context)
    artifacts = build_uccsd(options=UCCSDOptions(), context=context, initial_state=initial)
    solver = VQE(
        StatevectorEstimator(),
        artifacts.circuit,
        SLSQP(maxiter=100, ftol=1e-10),
        initial_point=np.zeros(artifacts.circuit.num_parameters),
    )
    energy = solver.compute_minimum_eigenvalue(hamiltonian).eigenvalue
    assert energy.real + sum(prepared.energy_offsets.values()) == pytest.approx(
        -1.1373060357534, abs=1e-8
    )


@pytest.mark.parametrize(
    ("label", "coefficient", "qubits", "message"),
    [
        ("X", 1, 2, "2 qubits"),
        ("X", 1j, 1, "Hermitian"),
        ("X", 0, 1, "zero operator"),
        ("X", float("nan"), 1, "finite coefficients"),
    ],
)
def test_external_operator_validation(label, coefficient, qubits, message):
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import SparsePauliOp

    with pytest.raises(ConfigError, match=message):
        OperatorPool((SparsePauliOp.from_list([(label, coefficient)]),)).validate(qubits)


def test_operator_pool_rejects_empty_and_non_operator_entries():
    pytest.importorskip("qiskit")
    with pytest.raises(ConfigError, match="non-empty"):
        OperatorPool(()).validate(2)
    with pytest.raises(ConfigError, match="SparsePauliOp"):
        OperatorPool((object(),)).validate(2)

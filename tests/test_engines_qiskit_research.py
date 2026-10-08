"""Deterministic public-engine validation using stored H2 integrals, without a driver."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from chemrefine.engines.qiskit.data import ElectronicStructureData
from chemrefine.engines.qiskit.operators import OperatorPool
from chemrefine.engines.qiskit.options import ActiveSpaceOptions
from chemrefine.engines.qiskit.result import QiskitRunResult

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_nature")
pytest.importorskip("qiskit_algorithms")

pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning:qiskit"),
    pytest.mark.filterwarnings("ignore::PendingDeprecationWarning:qiskit"),
]

_H2_EXACT = -1.1373060357534


def _variational_options(*, mapper="jordan_wigner"):
    """Use finite budgets, exact expectations, explicit seeds, and tight tolerances."""
    return {
        "cores": 1,
        "mapper": mapper,
        "estimator": {"name": "statevector", "options": {"default_precision": 0.0, "seed": 17}},
        "optimizer": {"name": "slsqp", "options": {"maxiter": 100, "ftol": 1e-10}},
    }


def _adaptive_options(**kwargs):
    """Bound the adaptive search while retaining a strict stopping threshold."""
    return {
        **_variational_options(**kwargs),
        "algorithm": {
            "name": "adapt_vqe",
            "options": {
                "gradient_threshold": 1e-6,
                "eigenvalue_threshold": 1e-10,
                "max_iterations": 8,
            },
        },
    }


@pytest.fixture(scope="module")
def electronic_data():
    """Read the molecule once; no SCF driver is used anywhere in this module."""
    fixture = Path(__file__).parent / "data/engines/qiskit/h2_integrals.json"
    return ElectronicStructureData(**json.loads(fixture.read_text()))


@pytest.fixture(scope="module")
def prepared(electronic_data):
    from chemrefine.engines.qiskit.api import prepare_problem

    return prepare_problem(electronic_data)


@pytest.fixture(scope="module")
def exact_results(prepared):
    from chemrefine.engines.qiskit.api import solve_exact

    return {
        mapper: solve_exact(prepared, options={"mapper": mapper, "cores": 1})
        for mapper in ("jordan_wigner", "parity", "bravyi_kitaev")
    }


@pytest.fixture(scope="module")
def variational_results(prepared, exact_results):
    from chemrefine.engines.qiskit.api import run_adapt_vqe, run_vqe

    reference = exact_results["jordan_wigner"].energy_hartree
    callbacks: list[dict[str, Any]] = []
    vqe = run_vqe(
        prepared,
        options=_variational_options(),
        initial_point=np.zeros(3),
        callback=callbacks.append,
        reference_energy_hartree=reference,
    )
    adapt = run_adapt_vqe(
        prepared,
        options=_adaptive_options(),
        reference_energy_hartree=reference,
    )
    return vqe, adapt, callbacks


@pytest.mark.parametrize(
    ("mapper", "qubits", "pauli_terms"),
    [("jordan_wigner", 4, 15), ("parity", 2, 5), ("bravyi_kitaev", 4, 15)],
)
def test_public_mapping_and_exact_energy(prepared, exact_results, mapper, qubits, pauli_terms):
    from chemrefine.engines.qiskit.api import map_problem

    context = map_problem(prepared, mapper)
    result = exact_results[mapper]
    assert context.fermionic_hamiltonian is prepared.fermionic_hamiltonian
    assert context.num_qubits_before_reduction == result.num_qubits_before_reduction == 4
    assert context.num_qubits == result.num_qubits == qubits
    assert len(context.qubit_hamiltonian) == result.num_pauli_terms == pauli_terms
    assert result.num_spin_orbitals == 4
    assert result.num_particles == (1, 1)
    assert result.mapping == mapper
    assert result.energy_hartree == pytest.approx(_H2_EXACT, abs=1e-10)
    assert result.solver == "exact"
    assert result.converged is True
    assert result.logical_circuit_metrics is None
    assert result.parameter_count is None


def test_uccsd_vqe_energy_history_callback_and_resource_counts(variational_results):
    vqe, _, callbacks = variational_results
    assert vqe.solver == "vqe"
    assert vqe.ansatz == "uccsd"
    assert vqe.optimizer == "slsqp"
    assert vqe.parameter_count == 3
    assert vqe.energy_hartree == pytest.approx(_H2_EXACT, abs=1e-8)
    assert abs(vqe.energy_error_hartree) < 1e-8
    assert vqe.reference_energy_hartree == pytest.approx(_H2_EXACT, abs=1e-10)
    assert len(callbacks) == vqe.energy_evaluation_count == len(vqe.metadata["evaluations"])
    assert vqe.optimizer_evaluations > 0
    assert vqe.optimizer_evaluations <= vqe.energy_evaluation_count
    assert all(np.isfinite(entry["objective_value_hartree"]) for entry in callbacks)
    assert vqe.logical_circuit_metrics.parameter_count == 3
    assert vqe.logical_circuit_metrics.depth > 0
    assert vqe.logical_circuit_metrics.size > 0
    assert vqe.logical_circuit_metrics.one_qubit_gate_count > 0
    assert vqe.logical_circuit_metrics.two_qubit_gate_count > 0
    assert vqe.logical_circuit_metrics.cx_count == vqe.logical_circuit_metrics.two_qubit_gate_count
    assert vqe.logical_circuit_metrics.representation == "logical"
    assert vqe.transpiled_circuit_metrics is None


def test_adapt_reports_real_selected_excitations_and_terminal_gradient(variational_results):
    _, adapt, _ = variational_results
    assert adapt.solver == "adapt_vqe"
    assert adapt.energy_hartree == pytest.approx(_H2_EXACT, abs=1e-8)
    assert abs(adapt.energy_error_hartree) < 1e-8
    assert adapt.converged is True
    assert adapt.termination_reason == "CONVERGED"
    assert adapt.adapt_pool_size == 3
    assert adapt.adapt_iterations == len(adapt.adapt_gradient_history) >= 2
    assert len(adapt.adapt_selected_operators) == adapt.parameter_count == 1
    assert adapt.adapt_selected_operators[0]["excitation"] == {
        "occupied": [0, 2],
        "unoccupied": [1, 3],
    }
    assert adapt.adapt_selected_operators[0]["pool_index"] == 2
    assert adapt.adapt_gradient_history[-1]["max_gradient"] < 1e-6
    assert adapt.adapt_gradient_history[-1]["retained"] is False
    assert sum(entry["retained"] for entry in adapt.adapt_gradient_history) == 1
    assert adapt.energy_evaluation_count > 0
    assert adapt.logical_circuit_metrics.parameter_count == 1
    assert adapt.logical_circuit_metrics.two_qubit_gate_count > 0


def test_results_are_plain_json_with_consistent_total_energy(
    electronic_data, exact_results, variational_results
):
    vqe, adapt, _ = variational_results
    for result in (*exact_results.values(), vqe, adapt):
        assert isinstance(result, QiskitRunResult)
        assert result.success is True
        assert result.runtime_seconds is not None
        assert result.runtime_seconds >= 0
        assert result.total_energy_hartree == result.energy_hartree
        assert result.electronic_energy_hartree is not None
        assert result.nuclear_repulsion_energy_hartree is not None
        assert (
            result.electronic_energy_hartree + result.nuclear_repulsion_energy_hartree
            == pytest.approx(result.total_energy_hartree, abs=1e-12)
        )
        assert result.nuclear_repulsion_energy_hartree == electronic_data.nuclear_repulsion_energy
        payload = result.as_dict()
        assert json.loads(json.dumps(payload, allow_nan=False)) == payload
        assert payload["num_particles"] == [1, 1]
        assert payload["metadata"]["provenance"]["source"] == "stored_pyscf_integrals"
        assert payload["metadata"]["provenance"]["package_versions"]["qiskit"]


def test_custom_excitation_injection_uses_only_requested_double(prepared):
    from chemrefine.engines.qiskit.api import run_vqe

    result = run_vqe(
        prepared,
        options=_variational_options(),
        excitations=[((0, 2), (1, 3))],
        initial_point=[0.0],
        reference_energy_hartree=_H2_EXACT,
    )
    assert result.ansatz == "ucc"
    assert result.parameter_count == 1
    assert result.energy_hartree == pytest.approx(_H2_EXACT, abs=1e-8)
    assert abs(result.energy_error_hartree) < 1e-8


def test_custom_mapped_operator_pool_bypasses_default_ansatz(prepared):
    from qiskit_nature.second_q.operators import FermionicOp

    from chemrefine.engines.qiskit.api import map_problem, run_adapt_vqe

    context = map_problem(prepared)
    excitation = FermionicOp({"+_0 +_2 -_1 -_3": 1.0}, num_spin_orbitals=4)
    operator = context.mapper.map(1j * (excitation - excitation.adjoint()))
    pool = OperatorPool(
        (operator,),
        ({"label": "external_double", "excitation": {"occupied": [0, 2], "unoccupied": [1, 3]}},),
    )
    callbacks: list[dict[str, Any]] = []
    result = run_adapt_vqe(
        prepared,
        options={**_adaptive_options(), "ansatz": "efficient_su2"},
        operator_pool=pool,
        callback=callbacks.append,
        reference_energy_hartree=_H2_EXACT,
    )
    assert result.energy_hartree == pytest.approx(_H2_EXACT, abs=1e-8)
    assert result.adapt_pool_size == result.parameter_count == 1
    assert result.adapt_selected_operators == pool.metadata
    assert all(entry["pool_index"] == 0 for entry in result.adapt_gradient_history)
    assert len(callbacks) == result.energy_evaluation_count


def test_permuted_active_orbitals_work_through_exact_vqe_and_adapt(electronic_data):
    from chemrefine.engines.qiskit.api import prepare_problem, run_adapt_vqe, run_vqe, solve_exact

    reordered = prepare_problem(
        electronic_data, active_space=ActiveSpaceOptions(active_orbitals=[1, 0])
    )
    exact = solve_exact(reordered, options={"cores": 1})
    vqe = run_vqe(reordered, options=_variational_options())
    adapt = run_adapt_vqe(reordered, options=_adaptive_options())
    for result in (exact, vqe, adapt):
        assert result.energy_hartree == pytest.approx(_H2_EXACT, abs=1e-8)
        assert result.active_space is not None
        assert result.active_space["active_orbitals"] == [1, 0]
    assert adapt.adapt_selected_operators is not None
    assert adapt.adapt_selected_operators[0]["excitation"] == {
        "occupied": [1, 3],
        "unoccupied": [0, 2],
    }

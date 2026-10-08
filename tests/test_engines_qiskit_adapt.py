"""Small deterministic tests of the ADAPT diagnostics compatibility seam."""

from __future__ import annotations

import numpy as np
import pytest

from chemrefine.engines.qiskit.adapt import AdaptDiagnostics, tracked_adapt_vqe

pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning:qiskit"),
    pytest.mark.filterwarnings("ignore::PendingDeprecationWarning:qiskit"),
]


def _solver(*, eigenvalue_threshold=1e-10, max_iterations=8, metadata=True):
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_algorithms")
    from qiskit import QuantumCircuit
    from qiskit.primitives import StatevectorEstimator
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_algorithms import VQE
    from qiskit_algorithms.optimizers import SLSQP

    hamiltonian = SparsePauliOp.from_list(
        [("ZI", 1.0), ("IZ", 0.7), ("XX", 0.4), ("XI", 0.2), ("IX", 0.1)]
    )
    pool = tuple(SparsePauliOp.from_list([(label, 1.0)]) for label in ("YI", "IY", "XY"))
    descriptions = tuple({"pool_index": index, "label": label} for index, label in enumerate("abc"))
    trace = AdaptDiagnostics()
    solver = tracked_adapt_vqe(
        VQE(StatevectorEstimator(), QuantumCircuit(2), SLSQP(maxiter=150, ftol=1e-12)),
        diagnostics=trace,
        pool_metadata=descriptions if metadata else (),
        operators=pool,
        initial_state=QuantumCircuit(2),
        eigenvalue_threshold=eigenvalue_threshold,
        max_iterations=max_iterations,
    )
    return solver, trace, hamiltonian, pool


def test_adapt_captures_gradient_stop_and_retained_pool_sequence():
    solver, trace, hamiltonian, pool = _solver()
    result = solver.compute_minimum_eigenvalue(hamiltonian)
    assert result.termination_criterion.name == "CONVERGED"
    assert result.eigenvalue == pytest.approx(
        np.linalg.eigvalsh(hamiltonian.to_matrix())[0], abs=1e-9
    )
    assert trace.selected_operator_indices == (2, 0, 1)
    assert [entry["label"] for entry in trace.selected_operators] == ["c", "a", "b"]
    assert len(trace.gradient_history) == result.num_iterations == 4
    assert [entry["retained"] for entry in trace.gradient_history] == [True, True, True, False]
    assert trace.gradient_history[-1]["max_gradient"] < solver.gradient_threshold
    assert trace.gradient_history[0]["max_gradient"] == abs(trace.gradient_history[0]["gradient"])
    assert all(
        actual is pool[index]
        for actual, index in zip(
            solver._excitation_list, trace.selected_operator_indices, strict=True
        )
    )
    assert result.optimal_point.size == len(trace.selected_operators) == 3


def test_adapt_energy_rollback_excludes_last_candidate_and_its_parameters():
    solver, trace, hamiltonian, _ = _solver(eigenvalue_threshold=100)
    result = solver.compute_minimum_eigenvalue(hamiltonian)
    assert result.termination_criterion.name == "CONVERGED"
    assert result.num_iterations == 2
    assert trace.selected_operator_indices == (2,)
    assert [entry["pool_index"] for entry in trace.gradient_history] == [2, 0]
    assert [entry["retained"] for entry in trace.gradient_history] == [True, False]
    assert result.optimal_point.size == result.optimal_circuit.num_parameters == 1


def test_adapt_maximum_iteration_stop_keeps_final_candidate_and_resets_between_runs():
    solver, trace, hamiltonian, _ = _solver(max_iterations=1, metadata=False)
    for _ in range(2):
        result = solver.compute_minimum_eigenvalue(hamiltonian)
        assert result.termination_criterion.name == "MAXIMUM"
        assert len(trace.gradient_history) == result.num_iterations == 1
        assert trace.gradient_history[0]["retained"]
        assert trace.selected_operators == ({"pool_index": 2, "label": "operator_2"},)


def test_adapt_cycle_stop_does_not_report_rejected_candidate_as_selected(monkeypatch):
    solver, trace, hamiltonian, _ = _solver()
    from qiskit_algorithms import AdaptVQE

    monkeypatch.setattr(
        AdaptVQE, "_compute_gradients", lambda *_args: [(0.1, {}), (-0.5, {}), (0.1, {})]
    )
    result = solver.compute_minimum_eigenvalue(hamiltonian)
    assert result.termination_criterion.name == "CYCLICITY"
    assert trace.selected_operator_indices == (1,)
    assert len(trace.gradient_history) == 2
    assert trace.gradient_history[1]["gradient"] == -0.5
    assert not trace.gradient_history[1]["retained"]


def test_adapt_preserves_upstream_stationary_initial_state_failure(monkeypatch):
    solver, trace, hamiltonian, _ = _solver()
    from qiskit_algorithms import AdaptVQE, AlgorithmError

    monkeypatch.setattr(AdaptVQE, "_compute_gradients", lambda *_args: [(0.0, {})] * 3)
    with pytest.raises(AlgorithmError, match="first iteration"):
        solver.compute_minimum_eigenvalue(hamiltonian)
    assert trace.selected_operator_indices == ()
    assert trace.selected_operators == ()
    assert trace.gradient_history[0]["retained"] is False


def test_adapt_gradient_ties_preserve_upstream_first_candidate_choice(monkeypatch):
    solver, trace, hamiltonian, _ = _solver(max_iterations=1)
    from qiskit_algorithms import AdaptVQE

    monkeypatch.setattr(
        AdaptVQE, "_compute_gradients", lambda *_args: [(0.5, {}), (-0.5, {}), (0.1, {})]
    )
    solver.compute_minimum_eigenvalue(hamiltonian)
    assert trace.selected_operator_indices == (0,)
    assert trace.gradient_history[0]["gradient"] == 0.5

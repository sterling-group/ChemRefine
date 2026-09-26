"""Owned energy, convergence, and provenance records without optional backend imports."""

from dataclasses import replace
from enum import Enum
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace

import numpy as np
import pytest

from chemrefine.engines.qiskit import reporting
from chemrefine.engines.qiskit.context import (
    AlgorithmArtifacts,
    AnsatzArtifacts,
    ElectronicStructureContext,
)
from chemrefine.engines.qiskit.options import QiskitOptions
from chemrefine.engines.qiskit.result import CircuitMetrics
from chemrefine.errors import ConfigError


def electronic_context(**changes):
    """Supply transformed chemistry facts with independent nuclear/inactive offsets."""
    context = ElectronicStructureContext(
        problem=object(),
        mapper=object(),
        qubit_hamiltonian=object(),
        num_spatial_orbitals=2,
        num_particles=(1, 1),
        num_qubits=2,
        multiplicity=1,
        num_qubits_before_reduction=4,
        num_pauli_terms=5,
        mapping_metadata={"name": "parity", "symmetry_tapering": None},
        active_space_metadata={
            "active_orbitals": [1, 3],
            "energy_offsets": {"nuclear_repulsion_energy": 0.7, "FreezeCoreTransformer": -3.0},
        },
        provenance={"source": "supplied_integrals"},
    )
    return replace(context, **changes)


@pytest.fixture
def report_boundary(monkeypatch):
    """Record circuit selection and package availability without Qiskit."""
    calls = []
    metrics = CircuitMetrics(parameter_count=3, depth=10, size=12)

    def measure(circuit):
        calls.append(circuit)
        return None if circuit is None else metrics

    def version(package):
        if package == "qiskit":
            return "test-version"
        raise PackageNotFoundError(package)

    monkeypatch.setattr(reporting, "logical_circuit_metrics", measure)
    monkeypatch.setattr(reporting, "version", version)
    return calls, metrics


def summarize(result, **changes):
    """Apply one recorded result to an otherwise valid exact-solver context."""
    kwargs = {
        "options": QiskitOptions(mapper="parity"),
        "context": electronic_context(),
        "evaluations": [],
        "ansatz": AnsatzArtifacts(),
        "algorithm": AlgorithmArtifacts(object()),
        "runtime_seconds": 0.25,
        "reference_energy_hartree": None,
    }
    kwargs.update(changes)
    return reporting.summarize_result(result, **kwargs)


def backend_result(**changes):
    """Model Nature's interpreted energies, which already contain inactive offsets."""
    values = {
        "total_energies": [-4.3],
        "electronic_energies": [-5.0],
        "nuclear_repulsion_energy": 0.7,
        "raw_result": None,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_exact_energy_offsets_are_not_added_twice_and_reference_error_is_signed(report_boundary):
    result = summarize(backend_result(), reference_energy_hartree=-4.31)
    assert result.energy_hartree == -4.3
    assert result.electronic_energy_hartree == -5.0
    assert result.total_energy_hartree == -4.3
    assert result.nuclear_repulsion_energy_hartree == 0.7
    assert result.energy_error_hartree == pytest.approx(0.01)
    assert result.reference_energy_hartree == -4.31
    assert result.success is True
    assert result.converged is True
    assert result.termination_reason == "exact_diagonalization"
    assert result.runtime_seconds == 0.25
    assert result.num_qubits == 2
    assert result.num_qubits_before_reduction == 4
    assert result.num_spin_orbitals == 4
    assert result.num_particles == (1, 1)
    assert result.num_pauli_terms == 5
    assert result.ansatz is result.optimizer is result.parameter_count is None
    assert result.energy_evaluation_count == 0
    assert result.adapt_iterations is result.adapt_pool_size is None
    assert result.metadata["provenance"]["package_versions"] == {"qiskit": "test-version"}
    assert result.metadata["multiplicity"] == 1
    assert result.metadata["energy_convention"] == "total"
    assert result.as_dict()["metadata"]["prepared_active_space"]["active_orbitals"] == [1, 3]


@pytest.mark.parametrize("electronic", [None, [], [-2.0]])
def test_missing_nuclear_energy_never_claims_a_total_or_invents_electronic_details(
    report_boundary, electronic
):
    result = summarize(
        backend_result(
            total_energies=[-2.0], electronic_energies=electronic, nuclear_repulsion_energy=None
        )
    )
    assert result.energy_hartree == -2.0
    assert result.total_energy_hartree is None
    assert result.electronic_energy_hartree == (-2.0 if electronic else None)
    assert result.nuclear_repulsion_energy_hartree is None
    assert result.energy_error_hartree is None
    assert result.metadata["energy_convention"] == "electronic"


@pytest.mark.parametrize("verdict", [True, False, np.bool_(True), None, "unknown"])
def test_vqe_reports_only_available_optimizer_convergence_and_original_logical_circuit(
    report_boundary, verdict
):
    calls, metrics = report_boundary
    initial_circuit, optimal_circuit = object(), object()
    raw = SimpleNamespace(
        optimal_circuit=optimal_circuit,
        optimizer_result=SimpleNamespace(success=verdict),
        cost_function_evals=7,
        num_iterations=99,
        optimal_point=np.array([0.1, 0.2]),
        optimal_value=np.float64(-2.0),
    )
    evaluations = [{"evaluation": i + 1, "objective_value_hartree": -2.0} for i in range(4)]
    result = summarize(
        backend_result(raw_result=raw),
        options=QiskitOptions(algorithm="vqe"),
        ansatz=AnsatzArtifacts(circuit=initial_circuit),
        evaluations=evaluations,
    )
    assert calls == [initial_circuit]
    assert result.logical_circuit_metrics == metrics
    assert result.parameter_count == 3
    assert result.converged == (bool(verdict) if isinstance(verdict, bool | np.bool_) else None)
    assert result.termination_reason == "optimizer_returned"
    assert result.ansatz == "uccsd"
    assert result.optimizer == "slsqp"
    assert result.optimizer_evaluations == 7
    assert result.energy_evaluation_count == 4
    assert result.adapt_iterations is None
    assert result.metadata["solver"]["optimal_point"] == [0.1, 0.2]
    assert result.metadata["solver"]["optimal_value"] == -2.0


def test_vqe_fallback_circuit_preserves_custom_termination_and_configured_active_space(
    report_boundary,
):
    calls, _ = report_boundary
    circuit = object()
    result = summarize(
        backend_result(raw_result=SimpleNamespace(termination_criterion="user_stopped")),
        options=QiskitOptions(algorithm="vqe", active_space={"electrons": 2, "orbitals": 2}),
        context=electronic_context(active_space_metadata={}),
        ansatz=AnsatzArtifacts(circuit=circuit),
    )
    assert calls == [circuit]
    assert result.termination_reason == "user_stopped"
    assert result.active_space == {"electrons": 2, "orbitals": 2, "active_orbitals": None}


class AdaptTermination(Enum):
    """Match the observable enum interface, without importing Algorithms."""

    CONVERGED = 1
    MAXIMUM = 2
    CYCLICITY = 3


@pytest.mark.parametrize(
    "termination",
    [AdaptTermination.CONVERGED, AdaptTermination.MAXIMUM, AdaptTermination.CYCLICITY, None],
)
def test_adapt_preserves_selected_pool_and_gradient_trace_without_reporting_full_pool_circuit(
    report_boundary, termination
):
    calls, _ = report_boundary
    selected = ({"pool_index": 2, "label": "double"},)
    gradients = [{"iteration": 1, "max_gradient": 0.1}]
    trace = SimpleNamespace(selected_operators=selected, gradient_history=gradients)
    raw = SimpleNamespace(
        termination_criterion=termination,
        num_iterations=2,
        cost_function_evals=11,
        final_max_gradient=1e-7,
        eigenvalue_history=[-1.0, -1.1],
    )
    result = summarize(
        backend_result(raw_result=raw),
        options=QiskitOptions(algorithm="adapt_vqe"),
        ansatz=AnsatzArtifacts(circuit=object(), operator_pool=[object(), object(), object()]),
        algorithm=AlgorithmArtifacts(object(), diagnostics=trace),
    )
    assert calls == [None]
    assert result.logical_circuit_metrics is result.parameter_count is None
    assert result.converged == (termination == AdaptTermination.CONVERGED if termination else None)
    assert result.adapt_iterations == 2
    assert result.adapt_pool_size == 3
    assert result.adapt_selected_operators == selected
    assert result.adapt_gradient_history == tuple(gradients)
    assert result.metadata["optimizer_evaluations_scope"] == "last_retained_inner_vqe"
    assert result.metadata["solver"]["final_max_gradient"] == 1e-7
    assert result.metadata["solver"]["eigenvalue_history"] == [-1.0, -1.1]


@pytest.mark.parametrize("supplied", [False, True])
def test_adapt_without_pool_or_trace_and_pool_only_ansatz_have_honest_unknowns(
    report_boundary, supplied
):
    result = summarize(backend_result(), options=QiskitOptions(algorithm="adapt_vqe"))
    assert result.adapt_pool_size is result.adapt_selected_operators is None
    assert result.adapt_gradient_history is result.logical_circuit_metrics is None
    result = summarize(
        backend_result(),
        options=QiskitOptions(algorithm="adapt_vqe"),
        ansatz=AnsatzArtifacts(operator_pool=[object()]),
        operator_pool_supplied=supplied,
    )
    assert result.ansatz == ("external_pool" if supplied else "uccsd")
    assert result.metadata["operator_pool_source"] == ("external" if supplied else "uccsd")
    assert result.adapt_pool_size == 1


@pytest.mark.parametrize("algorithm", ["vqe", "adapt_vqe"])
@pytest.mark.parametrize("transpiled", [False, True])
@pytest.mark.parametrize("has_optimal", [False, True])
def test_logical_and_transpiled_metrics_keep_distinct_circuit_provenance(
    report_boundary, monkeypatch, algorithm, transpiled, has_optimal
):
    logical_calls, logical_metrics = report_boundary
    original_circuit = object()
    optimal_circuit = object() if has_optimal else None
    compiled_calls = []
    compiled_metrics = CircuitMetrics(parameter_count=2, depth=30, representation="transpiled")

    def measure_compiled(circuit):
        compiled_calls.append(circuit)
        return compiled_metrics if circuit is not None else None

    monkeypatch.setattr(reporting, "transpiled_circuit_metrics", measure_compiled)
    result = summarize(
        backend_result(raw_result=SimpleNamespace(optimal_circuit=optimal_circuit)),
        options=QiskitOptions(algorithm=algorithm),
        ansatz=AnsatzArtifacts(circuit=original_circuit),
        was_transpiled=transpiled,
    )
    expected_logical = (
        original_circuit if algorithm == "vqe" else (None if transpiled else optimal_circuit)
    )
    assert logical_calls == [expected_logical]
    assert compiled_calls == ([optimal_circuit] if transpiled else [])
    assert result.logical_circuit_metrics == (logical_metrics if expected_logical else None)
    assert result.transpiled_circuit_metrics == (
        compiled_metrics if transpiled and has_optimal else None
    )
    assert result.parameter_count == (
        3 if expected_logical else 2 if transpiled and has_optimal else None
    )
    if transpiled:
        assert result.metadata["transpilation"]["device"] == "cpu"
        assert result.metadata["transpilation"]["estimator"]["name"] == "statevector"
    else:
        assert "transpilation" not in result.metadata


@pytest.mark.parametrize("totals", [None, []])
def test_reporting_rejects_missing_ground_state_energy(report_boundary, totals):
    with pytest.raises(ConfigError, match="no total ground-state energy"):
        summarize(backend_result(total_energies=totals))


@pytest.mark.parametrize(
    "value, message",
    [
        (True, "invalid"),
        (np.bool_(True), "invalid"),
        (object(), "invalid"),
        ("invalid", "invalid"),
        (complex(-1, np.nan), "non-finite"),
        (complex(-1, np.inf), "non-finite"),
        (complex(-1, 1e-3), "complex"),
        (np.nan, "non-finite"),
        (np.inf, "non-finite"),
    ],
)
def test_real_energy_rejects_malformed_nonfinite_or_complex_values(value, message):
    with pytest.raises(ConfigError, match=message):
        reporting.real_energy(value, "test energy")


def test_json_conversion_preserves_numeric_and_container_diagnostics():
    class Description:
        """Supply a stable description for opaque diagnostic values."""

        def __str__(self):
            return "description"

    assert reporting.jsonable(
        {1: (np.float64(2), complex(3, 4), np.array([5, 6]), None, True, Description())}
    ) == {"1": [2.0, {"real": 3.0, "imag": 4.0}, [5, 6], None, True, "description"]}

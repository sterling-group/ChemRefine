"""Joint Pauli measurements reproduce observables and retain shot covariance."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector

from chemrefine.engines.qiskit.measurement import (
    MeasurementOptions,
    group_statistics,
    measure_observable,
    measurement_groups,
)
from chemrefine.engines.qiskit.options import ComponentSelection
from chemrefine.engines.qiskit.sampling import SampleBatch
from chemrefine.errors import ConfigError


def test_measurement_memory_guard_precedes_sampling():
    """Storage guards account for shots, covariance and simulator state arrays."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp

    circuit = QuantumCircuit(20)
    with pytest.raises(ConfigError, match="max_memory_mb"):
        measure_observable(
            circuit, SparsePauliOp("Z" * 20), options=MeasurementOptions(max_memory_mb=1)
        )
    circuit = QuantumCircuit(1)
    with pytest.raises(ConfigError, match="max_memory_mb"):
        measure_observable(
            circuit,
            SparsePauliOp("Z"),
            options=MeasurementOptions(shots=1_000_000, max_memory_mb=1),
        )
    measure_observable(
        circuit,
        SparsePauliOp("I"),
        sampler=ComponentSelection(name="aer", options={"method": "matrix_product_state"}),
    )
    measure_observable(
        circuit,
        SparsePauliOp("I"),
        sampler=ComponentSelection(name="aer", options={"method": "density_matrix"}),
    )
    measure_observable(circuit, SparsePauliOp("I"), sampler=ComponentSelection(name="custom"))


@pytest.mark.parametrize("grouping", ["none", "qwc", "commuting"])
def test_bell_correlators_and_dependent_pauli_signs(grouping):
    """XX*ZZ=-YY must be respected when selecting independent stabilizers."""
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    operator = SparsePauliOp.from_list([("XX", 1), ("YY", 1), ("ZZ", 1), ("II", 0.3)])
    result = measure_observable(circuit, operator, options=MeasurementOptions(grouping=grouping))
    assert result["expectation"] == pytest.approx(1.3)
    assert result["standard_error"] == 0
    assert len(result["groups"]) == (1 if grouping == "commuting" else 3)
    assert sum(group["shots"] + group["pilot_shots"] for group in result["groups"]) == 4096


def test_joint_covariance_prevents_understating_uncertainty():
    """Perfectly correlated Z outcomes have twice the error of independent terms."""
    _, groups = measurement_groups(
        SparsePauliOp.from_list([("ZI", 1), ("IZ", 1)]), MeasurementOptions()
    )
    mean, variance, covariance = group_statistics(
        groups[0], SampleBatch({"00": 50, "11": 50}, 2, 100)
    )
    assert mean == 0
    assert variance == pytest.approx(400 / 99)
    np.testing.assert_allclose(covariance, np.full((2, 2), 100 / 99))
    with pytest.raises(ConfigError, match="two shots"):
        group_statistics(groups[0], SampleBatch({"00": 1}, 2, 1))


@pytest.mark.parametrize("sampler", ["statevector", "basic_backend"])
def test_adaptive_allocation_uses_independent_reproducible_production(sampler):
    """Random expectation estimates agree with an independent exact statevector."""
    circuit = QuantumCircuit(2)
    circuit.ry(0.7, 0)
    circuit.rx(0.4, 1)
    circuit.cx(0, 1)
    operator = SparsePauliOp.from_list([("XI", 0.1), ("YZ", 0.8), ("ZI", 1.2)])
    options = MeasurementOptions(shots=20000, pilot_shots=64, seed=53)
    selected = ComponentSelection.named(sampler)
    first = measure_observable(circuit, operator, options=options, sampler=selected)
    second = measure_observable(circuit, operator, options=options, sampler=selected)
    assert first == second
    exact = Statevector(circuit).expectation_value(operator).real
    assert abs(first["expectation"] - exact) < 5 * first["standard_error"]
    assert first["pilot_policy"] == "independent_allocation_only"


def test_diagonalizing_circuit_conjugates_each_general_commuting_term():
    """The synthesized rotation itself, not just ideal statistics, is correct."""
    operator = SparsePauliOp.from_list([("XZX", 0.4), ("YZY", 0.2), ("ZIZ", -0.7)])
    _, groups = measurement_groups(operator, MeasurementOptions(grouping="commuting"))
    assert len(groups) == 1
    group = groups[0]
    from qiskit.quantum_info import Operator

    rotation = Operator(group.circuit).data
    for label, mask, sign in zip(group.labels, group.z_masks, group.signs, strict=True):
        diagonal = Pauli("".join("Z" if mask >> i & 1 else "I" for i in reversed(range(3))))
        np.testing.assert_allclose(
            rotation @ Pauli(label).to_matrix() @ rotation.conj().T,
            sign * diagonal.to_matrix(),
            atol=1e-12,
        )


def test_constants_need_no_shots_and_invalid_requests_fail():
    """No sampler is built for a constant; invalid budgets and circuits fail early."""
    circuit = QuantumCircuit(1)
    constant = SparsePauliOp.from_list([("I", 2)])
    assert measure_observable(circuit, constant)["shots"] == 0
    with pytest.raises(ConfigError, match="shot budget"):
        measure_observable(circuit, SparsePauliOp("X"), options=MeasurementOptions(shots=1))
    with pytest.raises(ConfigError, match="Hermitian"):
        measure_observable(circuit, SparsePauliOp.from_list([("X", 1j)]))
    with pytest.raises(ConfigError, match="max_terms"):
        measure_observable(
            circuit,
            SparsePauliOp.from_list([("X", 1), ("Z", 1)]),
            options=MeasurementOptions(max_terms=1),
        )
    with pytest.raises(ConfigError, match="matching"):
        measure_observable(QuantumCircuit(2), constant)
    circuit.measure_all()
    with pytest.raises(ConfigError, match="unmeasured"):
        measure_observable(circuit, constant)


def test_incorrect_provider_diagonalization_is_rejected(monkeypatch):
    """A compiler regression cannot silently label X measurements as Z outcomes."""
    from types import SimpleNamespace

    from qiskit.quantum_info import Clifford, StabilizerState

    monkeypatch.setattr(
        StabilizerState,
        "from_stabilizer_list",
        lambda *args, **kwargs: SimpleNamespace(clifford=Clifford(QuantumCircuit(1))),
    )
    with pytest.raises(ConfigError, match="synthesis"):
        measurement_groups(SparsePauliOp("X"), MeasurementOptions(grouping="commuting"))

"""Owned result serialization and circuit-resource semantics for Qiskit."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any, cast

import pytest

from chemrefine.engines.qiskit.metrics import (
    logical_circuit_metrics,
    transpiled_circuit_metrics,
)
from chemrefine.engines.qiskit.result import CircuitMetrics, QiskitRunResult


def test_result_sidecar_preserves_existing_diagnostics() -> None:
    """Typed fields travel beside the original components and solver metadata."""
    result = QiskitRunResult(-1.0, {"components": {"algorithm": "exact"}}, solver="exact")
    sidecar = result.as_metadata()
    assert sidecar["components"] == {"algorithm": "exact"}
    assert sidecar["result"]["energy_hartree"] == -1.0
    assert sidecar["result"]["solver"] == "exact"
    assert "metadata" not in sidecar["result"]
    assert result.metadata == {"components": {"algorithm": "exact"}}


class _Circuit:
    """Small circuit double recording selective decomposition without Qiskit."""

    def __init__(
        self,
        operations: list[tuple[str, int]],
        *,
        num_parameters: int = 2,
        depth: int = 3,
        decomposed: _Circuit | None = None,
    ) -> None:
        self.data = [
            SimpleNamespace(operation=SimpleNamespace(name=name, num_qubits=qubits))
            for name, qubits in operations
        ]
        self.num_parameters = num_parameters
        self._depth = depth
        self._decomposed = decomposed
        self.decomposition_requests: list[list[str]] = []

    def depth(self) -> int:
        """Return the circuit's own depth convention."""
        return self._depth

    def size(self) -> int:
        """Exclude directives from the circuit's own size convention."""
        return sum(instruction.operation.name != "barrier" for instruction in self.data)

    def decompose(self, *, gates_to_decompose: list[str]) -> _Circuit:
        """Return an explicitly supplied expansion, otherwise leave it opaque."""
        self.decomposition_requests.append(gates_to_decompose)
        return self._decomposed if self._decomposed is not None else self


def test_result_preserves_energy_interface_and_unknown_metrics() -> None:
    """Old positional callers work and unsupported solver resources stay unknown."""
    result = QiskitRunResult(-1.25, {"source": "fixture"})
    assert result.energy_hartree == -1.25
    assert result.metadata == {"source": "fixture"}
    assert result.converged is None
    assert result.adapt_iterations is None
    assert result.logical_circuit_metrics is None
    assert result.transpiled_circuit_metrics is None
    assert result.as_dict()["num_particles"] is None
    with pytest.raises(FrozenInstanceError):
        cast("Any", result).energy_hartree = 0.0
    with pytest.raises(FrozenInstanceError):
        cast("Any", CircuitMetrics()).depth = 0


def test_result_serializes_nested_metrics_and_adapt_records_as_plain_data() -> None:
    """The exported snapshot has lists and dictionaries and shares no mutable data."""
    result = QiskitRunResult(
        -1.137,
        {"source": {"name": "fixture"}, "energy_history_hartree": (-1.0, -1.137)},
        solver="adapt_vqe",
        electronic_energy_hartree=-1.85,
        total_energy_hartree=-1.137,
        nuclear_repulsion_energy_hartree=0.713,
        reference_energy_hartree=-1.138,
        energy_error_hartree=0.001,
        converged=True,
        success=True,
        runtime_seconds=0.25,
        num_particles=(1, 1),
        active_space={"active_orbitals": (0, 1)},
        adapt_selected_operators=({"pool_index": 2, "excitation": ((0, 2), (1, 3))},),
        adapt_gradient_history=({"iteration": 1, "max_gradient": 0.2},),
        logical_circuit_metrics=CircuitMetrics(
            parameter_count=1,
            depth=7,
            two_qubit_gate_count=4,
            basis_gates=("u", "cx"),
        ),
    )
    payload = result.as_dict()
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    assert payload["num_particles"] == [1, 1]
    assert payload["logical_circuit_metrics"]["basis_gates"] == ["u", "cx"]
    assert payload["adapt_selected_operators"] == [
        {"pool_index": 2, "excitation": [[0, 2], [1, 3]]}
    ]
    assert payload["active_space"] == {"active_orbitals": [0, 1]}
    payload["metadata"]["source"]["name"] = "changed"
    assert result.metadata["source"]["name"] == "fixture"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_result_rejects_nonfinite_json_numbers(value: float) -> None:
    """NaN and infinity cannot masquerade as interoperable JSON numbers."""
    with pytest.raises(ValueError, match="Out of range float"):
        QiskitRunResult(value).as_dict()


def test_result_rejects_opaque_objects_in_metadata() -> None:
    """A raw provider result cannot silently leak through exported diagnostics."""
    with pytest.raises(TypeError, match="not JSON serializable"):
        QiskitRunResult(-1.0, {"raw_result": object()}).as_dict()


def test_result_metadata_defaults_are_independent() -> None:
    """Result construction does not share mutable metadata between calculations."""
    first = QiskitRunResult(-1.0)
    second = QiskitRunResult(-2.0)
    first.metadata["source"] = "first"
    assert second.metadata == {}


def test_no_circuit_means_no_metrics() -> None:
    """Exact solvers have no circuit rather than a circuit with invented zeros."""
    assert logical_circuit_metrics(None) is None
    assert transpiled_circuit_metrics(None) is None


def test_logical_metrics_count_defined_basis_and_preserve_parameter_count() -> None:
    """Composite ansatze expand without changing the input or optimizing gates."""
    expanded = _Circuit(
        [("u", 1), ("cx", 2), ("u", 1), ("barrier", 2), ("measure", 1)],
        num_parameters=1,
        depth=4,
    )
    circuit = _Circuit([("ansatz", 2)], num_parameters=3, decomposed=expanded)
    metrics = logical_circuit_metrics(circuit)
    assert metrics == CircuitMetrics(
        parameter_count=3,
        depth=4,
        size=4,
        one_qubit_gate_count=2,
        two_qubit_gate_count=1,
        cx_count=1,
        basis_gates=("u", "cx"),
    )
    assert circuit.decomposition_requests == [["ansatz"]]
    assert circuit.data[0].operation.name == "ansatz"
    assert expanded.decomposition_requests == []


def test_empty_circuit_has_zero_resources() -> None:
    """An actual empty circuit can meaningfully report zero gates and depth."""
    circuit = _Circuit([], num_parameters=0, depth=0)
    metrics = logical_circuit_metrics(circuit)
    assert metrics is not None
    assert metrics.parameter_count == metrics.depth == metrics.size == 0
    assert metrics.one_qubit_gate_count == metrics.two_qubit_gate_count == metrics.cx_count == 0


def test_opaque_logical_circuit_retains_only_known_parameter_count() -> None:
    """An unsupported instruction cannot become a fabricated elementary gate."""
    metrics = logical_circuit_metrics(_Circuit([("opaque", 2)], num_parameters=5))
    assert metrics == CircuitMetrics(parameter_count=5, basis_gates=("u", "cx"))


@pytest.mark.parametrize("circuit", [object(), SimpleNamespace(num_parameters=4)])
def test_missing_instruction_data_keeps_resources_unknown(circuit: Any) -> None:
    """Partial third-party artifacts do not produce false zero resource counts."""
    parameter_count = getattr(circuit, "num_parameters", None)
    assert logical_circuit_metrics(circuit) == CircuitMetrics(
        parameter_count=parameter_count, basis_gates=("u", "cx")
    )
    assert transpiled_circuit_metrics(circuit) == CircuitMetrics(
        parameter_count=parameter_count, representation="transpiled"
    )


def test_missing_decomposition_retains_only_known_parameter_count() -> None:
    """An opaque artifact lacking a definition is reported as unknown."""
    circuit = SimpleNamespace(
        num_parameters=2,
        data=[SimpleNamespace(operation=SimpleNamespace(name="custom"))],
    )
    assert logical_circuit_metrics(circuit) == CircuitMetrics(
        parameter_count=2, basis_gates=("u", "cx")
    )


def test_transpiled_metrics_measure_native_circuit_without_decomposing() -> None:
    """Native gate arities remain separate from the logical u/cx representation."""
    circuit = _Circuit(
        [
            ("rz", 1),
            ("sx", 1),
            ("ecr", 2),
            ("cx", 2),
            ("ccx", 3),
            ("barrier", 3),
            ("reset", 1),
            ("delay", 1),
            ("measure", 1),
        ],
        depth=8,
    )
    assert transpiled_circuit_metrics(circuit) == CircuitMetrics(
        parameter_count=2,
        depth=8,
        size=8,
        one_qubit_gate_count=2,
        two_qubit_gate_count=2,
        cx_count=1,
        representation="transpiled",
        basis_gates=("ccx", "cx", "ecr", "rz", "sx"),
    )
    assert circuit.decomposition_requests == []


@pytest.mark.filterwarnings(
    r"ignore:The property .*DAGCircuit\.(duration|unit).*:DeprecationWarning:qiskit"
)
def test_real_circuit_decomposition_and_explicit_transpiled_counts() -> None:
    """Small supported-stack circuits validate selective decomposition semantics."""
    qiskit = pytest.importorskip("qiskit")
    parameter_module = pytest.importorskip("qiskit.circuit")
    circuit = qiskit.QuantumCircuit(3)
    circuit.h(0)
    circuit.ry(parameter_module.Parameter("theta"), 1)
    circuit.ccx(0, 1, 2)
    circuit.barrier()
    logical = logical_circuit_metrics(circuit)
    assert logical is not None
    assert logical.parameter_count == 1
    assert logical.one_qubit_gate_count == 11
    assert logical.two_qubit_gate_count == logical.cx_count == 6
    assert logical.size == 17
    assert circuit.count_ops()["ccx"] == 1

    compiled = qiskit.transpile(circuit, basis_gates=["rz", "sx", "x", "cx"], optimization_level=0)
    native = transpiled_circuit_metrics(compiled)
    assert native is not None
    assert native.parameter_count == 1
    assert native.depth == compiled.depth()
    assert native.size == compiled.size()
    assert native.cx_count == compiled.count_ops()["cx"]
    assert native.basis_gates == ("cx", "rz", "sx")

"""Backend-independent logical counts and explicit transpiled-circuit counts."""

from __future__ import annotations

from typing import Any, Literal

from chemrefine.engines.qiskit.result import CircuitMetrics

_LOGICAL_BASIS = ("u", "cx")
_NON_GATE_OPERATIONS = frozenset({"barrier", "measure", "reset", "delay"})
_MAX_DECOMPOSITIONS = 32


def _parameter_count(circuit: Any) -> int | None:
    """Keep the original symbolic parameter count before any decomposition."""
    value = getattr(circuit, "num_parameters", None)
    return None if value is None else int(value)


def _logical_decomposition(circuit: Any) -> Any | None:
    """Expand known definitions into u/cx without optimizations or a target."""
    for _ in range(_MAX_DECOMPOSITIONS):
        instructions = getattr(circuit, "data", None)
        if instructions is None:
            return None
        names = {
            instruction.operation.name
            for instruction in instructions
            if instruction.operation.name not in _LOGICAL_BASIS
            and instruction.operation.name not in _NON_GATE_OPERATIONS
        }
        if not names:
            return circuit
        decompose = getattr(circuit, "decompose", None)
        if not callable(decompose):
            return None
        circuit = decompose(gates_to_decompose=sorted(names))
    return None


def _measure_circuit(
    circuit: Any,
    *,
    parameter_count: int | None,
    representation: Literal["logical", "transpiled"],
    basis_gates: tuple[str, ...],
) -> CircuitMetrics:
    """Count gate arities on one already defined circuit representation."""
    operations = [
        instruction.operation
        for instruction in circuit.data
        if instruction.operation.name not in _NON_GATE_OPERATIONS
    ]
    return CircuitMetrics(
        parameter_count=parameter_count,
        depth=int(circuit.depth()),
        size=int(circuit.size()),
        one_qubit_gate_count=sum(operation.num_qubits == 1 for operation in operations),
        two_qubit_gate_count=sum(operation.num_qubits == 2 for operation in operations),
        cx_count=sum(operation.name == "cx" for operation in operations),
        representation=representation,
        basis_gates=basis_gates,
    )


def logical_circuit_metrics(circuit: Any | None) -> CircuitMetrics | None:
    """Describe a u/cx logical decomposition, without assuming a hardware target.

    The input is never modified. Opaque instructions, missing definitions, or
    decomposition nesting beyond 32 levels leave resource counts unknown while
    preserving the original parameter count. ``None`` means no circuit exists.
    """
    if circuit is None:
        return None
    parameter_count = _parameter_count(circuit)
    decomposed = _logical_decomposition(circuit)
    if decomposed is None:
        return CircuitMetrics(parameter_count=parameter_count, basis_gates=_LOGICAL_BASIS)
    return _measure_circuit(
        decomposed,
        parameter_count=parameter_count,
        representation="logical",
        basis_gates=_LOGICAL_BASIS,
    )


def transpiled_circuit_metrics(circuit: Any | None) -> CircuitMetrics | None:
    """Describe a caller-supplied compiled circuit without compiling it again.

    The caller must associate this circuit with its backend/target provenance.
    Its native one- and two-qubit instructions count as individual gates; this
    function makes no claim that they form a u/cx basis or fit another target.
    """
    if circuit is None:
        return None
    parameter_count = _parameter_count(circuit)
    instructions = getattr(circuit, "data", None)
    if instructions is None:
        return CircuitMetrics(parameter_count=parameter_count, representation="transpiled")
    basis_gates = tuple(
        sorted(
            {
                instruction.operation.name
                for instruction in instructions
                if instruction.operation.name not in _NON_GATE_OPERATIONS
            }
        )
    )
    return _measure_circuit(
        circuit,
        parameter_count=parameter_count,
        representation="transpiled",
        basis_gates=basis_gates,
    )

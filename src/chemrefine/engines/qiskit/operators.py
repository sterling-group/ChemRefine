"""Externally supplied evolution generators and their interpretable pool identities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from chemrefine.errors import ConfigError

Excitation = tuple[tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True)
class OperatorPool:
    """Mapped Hermitian generators, with parallel JSON-compatible descriptions.

    Operators use the prepared problem's qubit ordering and mapper. Their origin
    is deliberately unrestricted: callers may supply any externally selected
    pool without changing the ADAPT implementation. Validation checks dimensions
    and Hermiticity; preserving a particle or spin sector remains the caller's
    responsibility for arbitrary qubit operators.
    """

    operators: tuple[Any, ...]
    metadata: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        """Normalize sequences and assign stable indices without importing Qiskit."""
        object.__setattr__(self, "operators", tuple(self.operators))
        if self.metadata and len(self.metadata) != len(self.operators):
            raise ConfigError("qiskit operator-pool metadata must match the number of operators")
        descriptions = []
        for index in range(len(self.operators)):
            description = dict(self.metadata[index]) if self.metadata else {}
            description["pool_index"] = index
            description.setdefault("label", f"operator_{index}")
            descriptions.append(description)
        try:
            json.dumps(descriptions, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ConfigError("qiskit operator-pool metadata must be finite JSON data") from exc
        object.__setattr__(self, "metadata", tuple(descriptions))

    def validate(self, num_qubits: int) -> None:
        """Check that every generator can evolve this prepared qubit problem."""
        from qiskit.quantum_info import SparsePauliOp

        if not self.operators:
            raise ConfigError("qiskit ADAPT-VQE requires a non-empty operator pool")
        for index, operator in enumerate(self.operators):
            prefix = f"qiskit operator pool entry {index}"
            if not isinstance(operator, SparsePauliOp):
                raise ConfigError(f"{prefix} must be a mapped SparsePauliOp")
            if operator.num_qubits != num_qubits:
                raise ConfigError(f"{prefix} must act on {num_qubits} qubits")
            coefficients = np.asarray(operator.simplify().coeffs, dtype=complex)
            if not np.all(np.isfinite(coefficients)):
                raise ConfigError(f"{prefix} must have finite coefficients")
            if not np.allclose(coefficients.imag, 0, rtol=0, atol=1e-12):
                raise ConfigError(f"{prefix} must be Hermitian (real Pauli coefficients)")
            if not np.any(np.abs(coefficients) > 0):
                raise ConfigError(f"{prefix} must not be the zero operator")


def ucc_pool_metadata(
    circuit: Any, *, include_imaginary: bool = False
) -> tuple[dict[str, Any], ...]:
    """Describe Nature's already mapped UCC operators in their actual pool order.

    Reading ``operators`` first lets Nature align excitation metadata with any
    mapper filtering. Calling ``excitation_ops`` again would mutate cached
    imaginary-excitation metadata in the supported Nature version.
    """
    operators = tuple(circuit.operators)
    excitations = getattr(circuit, "excitation_list", None)
    descriptions: list[dict[str, Any]] = []
    for index in range(len(operators)):
        description: dict[str, Any] = {"pool_index": index, "label": f"excitation_{index}"}
        if excitations is not None:
            occupied, unoccupied = excitations[index]
            description["excitation"] = {
                "occupied": list(occupied),
                "unoccupied": list(unoccupied),
            }
            description["generator"] = (
                "symmetric" if include_imaginary and index % 2 else "antisymmetric"
            )
        descriptions.append(description)
    return tuple(descriptions)

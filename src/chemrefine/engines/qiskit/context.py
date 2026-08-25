"""Small dependency-light values exchanged by Qiskit component builders."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ElectronicStructureContext:
    """Problem facts available after driver, transformer, and mapper construction."""

    problem: Any
    mapper: Any
    qubit_hamiltonian: Any
    num_spatial_orbitals: int
    num_particles: tuple[int, int]
    num_qubits: int
    multiplicity: int


@dataclass(frozen=True)
class AnsatzArtifacts:
    """The fixed-circuit and adaptive-pool forms an ansatz may provide."""

    circuit: Any | None = None
    operator_pool: Sequence[Any] | None = None


@dataclass(frozen=True)
class SolverComponents:
    """Optional component products consumed by an algorithm builder."""

    estimator: Any | None = None
    optimizer: Any | None = None
    initial_state: Any | None = None
    ansatz: AnsatzArtifacts = field(default_factory=AnsatzArtifacts)
    initial_point: NDArray[np.float64] | None = None
    callback: Callable[[int, NDArray[np.float64], float, dict[str, Any]], None] | None = None
    transpiler: Any | None = None
    transpiler_options: dict[str, Any] | None = None


@dataclass(frozen=True)
class AlgorithmArtifacts:
    """A minimum eigensolver produced by an algorithm factory."""

    solver: Any


@dataclass
class EstimatorResource:
    """A V2 estimator and an optional cleanup callback for provider sessions."""

    estimator: Any
    close: Callable[[], None] = field(default=lambda: None)
    transpiler: Any | None = None
    transpiler_options: dict[str, Any] | None = None

    def __enter__(self) -> Any:
        """Return the estimator to the solver assembly context."""
        return self.estimator

    def __exit__(self, *_exc: object) -> None:
        """Release a provider session when the builder supplied one."""
        self.close()

"""ChemRefine-owned, serializable results for electronic-structure calculations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from chemrefine.engines.qiskit.determinants import DeterminantState


@dataclass(frozen=True)
class CircuitMetrics:
    """Circuit resources at an explicitly identified representation.

    Logical metrics use an unoptimized, backend-independent ``u``/``cx``
    decomposition. Transpiled metrics describe exactly the circuit supplied by
    the caller. Gate counts exclude barriers, measurement, reset, and delay;
    depth and size follow the circuit's own definitions. Unknown resources are
    ``None``, including when logical decomposition leaves opaque operations.
    """

    parameter_count: int | None = None
    depth: int | None = None
    size: int | None = None
    one_qubit_gate_count: int | None = None
    two_qubit_gate_count: int | None = None
    cx_count: int | None = None
    representation: Literal["logical", "transpiled"] = "logical"
    basis_gates: tuple[str, ...] | None = None


@dataclass(frozen=True)
class QiskitRunResult:
    """Stable engine output with energies in hartree and runtime in seconds.

    ``energy_hartree`` is total energy when nuclear repulsion is supplied,
    otherwise electronic energy; ``total_energy_hartree`` then remains unknown.
    Electronic energy includes inactive-space constants. A supplied reference
    must use the same energy convention; energy error is signed. ``success``
    means execution returned a finite energy, separately from ``converged``.
    The first two fields preserve positional callers.
    Metadata and ADAPT records must contain only JSON-compatible plain data.
    """

    energy_hartree: float
    metadata: dict[str, Any] = field(default_factory=dict)
    solver: str | None = None
    electronic_energy_hartree: float | None = None
    total_energy_hartree: float | None = None
    nuclear_repulsion_energy_hartree: float | None = None
    reference_energy_hartree: float | None = None
    energy_error_hartree: float | None = None
    converged: bool | None = None
    success: bool | None = None
    runtime_seconds: float | None = None
    num_qubits: int | None = None
    num_qubits_before_reduction: int | None = None
    num_spin_orbitals: int | None = None
    num_particles: tuple[int, int] | None = None
    num_pauli_terms: int | None = None
    mapping: str | None = None
    active_space: dict[str, Any] | None = None
    ansatz: str | None = None
    optimizer: str | None = None
    parameter_count: int | None = None
    optimizer_evaluations: int | None = None
    energy_evaluation_count: int | None = None
    adapt_iterations: int | None = None
    adapt_pool_size: int | None = None
    adapt_selected_operators: tuple[dict[str, Any], ...] | None = None
    adapt_gradient_history: tuple[dict[str, Any], ...] | None = None
    termination_reason: str | None = None
    logical_circuit_metrics: CircuitMetrics | None = None
    transpiled_circuit_metrics: CircuitMetrics | None = None

    target_root: int = 0
    root_energies_hartree: tuple[float, ...] | None = None
    root_electronic_energies_hartree: tuple[float, ...] | None = None
    root_total_energies_hartree: tuple[float, ...] | None = None
    states: tuple[DeterminantState, ...] = field(default=(), repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        """Return a detached JSON-native snapshot, rejecting objects and NaN/inf."""
        # Numerical states are transient worker payloads, persisted as referenced
        # bundles by the workflow rather than copied into JSON sidecars.
        data = asdict(replace(self, states=()))
        data.pop("states")
        return cast("dict[str, Any]", json.loads(json.dumps(data, allow_nan=False)))

    def as_metadata(self) -> dict[str, Any]:
        """Extend the existing sidecar diagnostics with the standardized result fields."""
        result = self.as_dict()
        metadata = result.pop("metadata")
        return {**metadata, "result": result}

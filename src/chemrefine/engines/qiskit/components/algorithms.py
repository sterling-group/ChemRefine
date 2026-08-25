"""Built-in minimum-eigensolver algorithm factories."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from chemrefine.engines.qiskit.context import (
    AlgorithmArtifacts,
    ElectronicStructureContext,
    SolverComponents,
)
from chemrefine.engines.qiskit.registry import ALGORITHMS, NoComponentOptions
from chemrefine.errors import ConfigError


class AdaptVQEOptions(BaseModel):
    """Convergence controls for the outer ADAPT-VQE loop."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gradient_threshold: float = Field(1e-5, gt=0)
    eigenvalue_threshold: float = Field(1e-5, gt=0)
    max_iterations: int | None = Field(None, ge=1)
    # qiskit-algorithms 0.4 initializes one new ADAPT parameter per iteration;
    # ``reps > 1`` creates more parameters and fails inside VQE with a dimension
    # mismatch. Keep the supported value explicit until that upstream contract changes.
    reps: Literal[1] = 1


def _require(value: Any, name: str) -> Any:
    """Return a component product or raise a composition-focused error."""
    if value is None:
        raise ConfigError(f"qiskit algorithm assembly requires {name}")
    return value


@ALGORITHMS.register("exact", NoComponentOptions)
def build_exact(
    *,
    options: BaseModel,
    context: ElectronicStructureContext,
    components: SolverComponents,
) -> AlgorithmArtifacts:
    """Build the classical exact diagonalizer used as a small-system reference."""
    del options, components
    from qiskit_algorithms import NumPyMinimumEigensolver

    expected_particles = sum(context.num_particles)
    spin = (context.multiplicity - 1) / 2
    expected_angular_momentum = spin * (spin + 1)

    def filter_criterion(
        _eigenstate: object,
        _eigenvalue: float,
        aux_values: dict[str, tuple[float, object]] | None,
    ) -> bool:
        """Keep the configured particle-number and spin-symmetry sector."""
        if aux_values is None:
            return True
        particle_number = aux_values.get("ParticleNumber")
        if particle_number is not None and not np.isclose(particle_number[0], expected_particles):
            return False
        angular_momentum = aux_values.get("AngularMomentum")
        return angular_momentum is None or bool(
            np.isclose(angular_momentum[0], expected_angular_momentum)
        )

    solver = NumPyMinimumEigensolver(filter_criterion=filter_criterion)
    return AlgorithmArtifacts(solver=solver)


@ALGORITHMS.register(
    "vqe",
    NoComponentOptions,
    requires=frozenset({"estimator", "optimizer", "circuit", "initial_point"}),
)
def build_vqe(
    *,
    options: BaseModel,
    context: ElectronicStructureContext,
    components: SolverComponents,
) -> AlgorithmArtifacts:
    """Build ordinary fixed-ansatz VQE."""
    del options, context
    from qiskit_algorithms import VQE

    circuit = _require(components.ansatz.circuit, "a circuit ansatz")
    if int(circuit.num_parameters) == 0:
        raise ConfigError(
            "qiskit VQE requires an ansatz with at least one parameter; "
            "use algorithm 'exact' for this problem or select another ansatz"
        )
    solver = VQE(
        _require(components.estimator, "an estimator"),
        circuit,
        _require(components.optimizer, "an optimizer"),
        initial_point=_require(components.initial_point, "an initial point"),
        callback=components.callback,
        transpiler=components.transpiler,
        transpiler_options=components.transpiler_options,
    )
    return AlgorithmArtifacts(solver=solver)


@ALGORITHMS.register(
    "adapt_vqe",
    AdaptVQEOptions,
    requires=frozenset({"estimator", "optimizer", "operator_pool", "initial_state"}),
)
def build_adapt_vqe(
    *,
    options: AdaptVQEOptions,
    context: ElectronicStructureContext,
    components: SolverComponents,
) -> AlgorithmArtifacts:
    """Build ADAPT-VQE around an inner VQE and an explicit excitation pool."""
    from qiskit import QuantumCircuit
    from qiskit_algorithms import VQE, AdaptVQE

    operator_pool = _require(components.ansatz.operator_pool, "an operator pool")
    if not operator_pool:
        raise ConfigError(
            "qiskit ADAPT-VQE requires a non-empty operator pool; use algorithm "
            "'exact' for this problem or select another ansatz"
        )
    inner = VQE(
        _require(components.estimator, "an estimator"),
        QuantumCircuit(context.num_qubits),
        _require(components.optimizer, "an optimizer"),
        callback=components.callback,
        transpiler=components.transpiler,
        transpiler_options=components.transpiler_options,
    )
    solver = AdaptVQE(
        inner,
        gradient_threshold=options.gradient_threshold,
        eigenvalue_threshold=options.eigenvalue_threshold,
        max_iterations=options.max_iterations,
        operators=operator_pool,
        reps=options.reps,
        initial_state=_require(components.initial_state, "an initial state"),
    )
    return AlgorithmArtifacts(solver=solver)

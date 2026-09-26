"""Estimator-driven variational imaginary and real time on a fixed circuit manifold."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from chemrefine.engines.qiskit.execution import logical_estimator
from chemrefine.engines.qiskit.options import ComponentSelection
from chemrefine.engines.qiskit.registry import ESTIMATORS
from chemrefine.errors import ConfigError


class VariationalDynamicsOptions(BaseModel):
    """Finite-time McLachlan dynamics, numerical integration and metric controls."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    method: Literal["varqite", "varqrte"] = "varqite"
    time: float = Field(1.0, gt=0)
    steps: int = Field(100, ge=1, le=10000)
    integrator: Literal["euler", "rk4"] = "rk4"
    metric_cutoff: float = Field(1e-10, gt=0, lt=1)
    regularization: float = Field(0.0, ge=0)
    numerical_tolerance: float = Field(1e-7, gt=0)
    max_velocity: float = Field(1e6, gt=0)
    max_parameters: int = Field(128, ge=1)
    max_pauli_terms: int = Field(4096, ge=1)
    max_publications: int = Field(100000, ge=1)
    max_memory_mb: int = Field(512, ge=1)


@dataclass(frozen=True)
class VariationalDynamicsResult:
    """Parameter and measured-observable trajectories with metric diagnostics."""

    times: NDArray[np.float64]
    parameters: NDArray[np.float64]
    expectations: NDArray[np.float64]
    metric_diagnostics: NDArray[np.float64]
    metadata: dict[str, Any]


def metric_velocity(
    metric: Any, force: Any, options: VariationalDynamicsOptions
) -> tuple[NDArray[np.float64], list[float]]:
    """Solve a positive-semidefinite metric with explicit truncation and ridge bias."""
    matrix, vector = np.asarray(metric, dtype=complex), np.asarray(force, dtype=complex)
    if (
        vector.ndim != 1
        or matrix.shape != (vector.size, vector.size)
        or not vector.size
        or not np.isfinite(matrix).all()
        or not np.isfinite(vector).all()
        or np.max(np.abs(matrix.imag)) > options.numerical_tolerance
        or np.max(np.abs(vector.imag)) > options.numerical_tolerance
        or not np.allclose(matrix, matrix.T, atol=options.numerical_tolerance, rtol=0)
    ):
        raise ConfigError("variational metric and force must be finite real compatible arrays")
    eigenvalues, vectors = np.linalg.eigh((matrix.real + matrix.real.T) / 2)
    if eigenvalues[0] < -options.numerical_tolerance:
        raise ConfigError("variational metric has a negative eigenvalue beyond tolerance")
    retained = eigenvalues > options.metric_cutoff
    weights = np.zeros(vector.size)
    weights[retained] = 1 / (eigenvalues[retained] + options.regularization)
    velocity = vectors @ (weights * (vectors.T @ vector.real))
    speed = float(np.linalg.norm(velocity))
    if not np.isfinite(velocity).all() or speed > options.max_velocity:
        raise ConfigError("variational parameter velocity exceeds max_velocity")
    residual = float(np.linalg.norm(matrix.real @ velocity - vector.real))
    return velocity, [
        float(sum(retained)),
        float(eigenvalues[0]),
        float(eigenvalues[-1]),
        residual,
        speed,
    ]


def variational_dynamics(
    circuit: Any,
    hamiltonian: Any,
    initial_parameters: Any,
    *,
    options: VariationalDynamicsOptions | None = None,
    estimator: ComponentSelection | None = None,
    observables: tuple[Any, ...] = (),
    cores: int = 1,
    device: Literal["cpu", "cuda"] = "cpu",
) -> VariationalDynamicsResult:
    """Integrate McLachlan equations using the selected estimator for every quantity.

    Qiskit's LCU gradient and geometric tensor generate measurement circuits.
    Their ancillas are compiled at submission, so no exact derivative simulator
    silently replaces the selected provider. The first observable is always H.
    """
    from qiskit_algorithms.gradients import DerivativeType, LinCombEstimatorGradient, LinCombQGT
    from qiskit_algorithms.time_evolvers.variational import (
        ImaginaryMcLachlanPrinciple,
        RealMcLachlanPrinciple,
    )

    controls = options or VariationalDynamicsOptions()
    selection = estimator or ComponentSelection.named("statevector")
    if np.iscomplexobj(initial_parameters):
        raise ConfigError("initial variational parameters must be real")
    point = np.asarray(initial_parameters, dtype=float)
    size = int(circuit.num_parameters)
    if (
        not 0 < size <= controls.max_parameters
        or circuit.num_clbits
        or point.shape != (size,)
        or not np.isfinite(point).all()
    ):
        raise ConfigError(
            "variational dynamics requires a finite initial point and an "
            "unmeasured parameterized circuit"
        )
    operators = (hamiltonian, *observables)
    for operator in operators:
        if (
            operator.num_qubits != circuit.num_qubits
            or len(operator) > controls.max_pauli_terms
            or not np.isfinite(operator.coeffs).all()
            or np.max(np.abs(operator.coeffs.imag)) > controls.numerical_tolerance
        ):
            raise ConfigError(
                "dynamics requires finite Hermitian Pauli operators matching the circuit and budget"
            )
    storage = 64 * size * size + 8 * (controls.steps + 1) * (size + len(operators) + 21)
    if selection.name in {"statevector", "basic_backend", "aer_statevector", "aer_shots"}:
        method = getattr(ESTIMATORS.options_for(selection), "method", "statevector")
        if method in {"automatic", "statevector", "density_matrix"}:
            storage += 64 * (
                1 << ((circuit.num_qubits + 1) * (2 if method == "density_matrix" else 1))
            )
    if storage > controls.max_memory_mb * 1024**2:
        raise ConfigError("variational dynamics storage exceeds max_memory_mb")
    resource = ESTIMATORS.build(selection, device=device, cores=cores)
    times = np.linspace(0, controls.time, controls.steps + 1)
    parameters, values, diagnostics = [], [], []
    with resource:
        execution = logical_estimator(resource, max_publications=controls.max_publications)
        gradient = LinCombEstimatorGradient(
            execution,
            derivative_type=DerivativeType.REAL
            if controls.method == "varqite"
            else DerivativeType.IMAG,
        )
        qgt = LinCombQGT(execution, phase_fix=True)
        principle = (
            ImaginaryMcLachlanPrinciple if controls.method == "varqite" else RealMcLachlanPrinciple
        )(qgt=qgt, gradient=gradient)

        def velocity(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
            """Measure the local geometry and force, recording every actual ODE evaluation."""
            matrix = principle.metric_tensor(circuit, parameters)
            force = principle.evolution_gradient(hamiltonian, circuit, parameters)
            update, diagnostic = metric_velocity(matrix, force, controls)
            diagnostics.append(diagnostic)
            return update

        def observe() -> None:
            """Measure one trajectory point with the same compiled publication boundary."""
            parameters.append(point.copy())
            measured = execution.run([(circuit, list(operators), point)]).result()[0].data.evs
            observed = np.asarray(measured, dtype=complex)
            if (
                observed.shape != (len(operators),)
                or not np.isfinite(observed).all()
                or np.max(np.abs(observed.imag)) > controls.numerical_tolerance
            ):
                raise ConfigError("dynamics estimator returned invalid expectation values")
            values.append(observed.real)

        for time, next_time in itertools.pairwise(times):
            observe()
            interval = next_time - time
            first = velocity(point)
            if controls.integrator == "euler":
                point = point + interval * first
            else:
                second = velocity(point + interval * first / 2)
                third = velocity(point + interval * second / 2)
                fourth = velocity(point + interval * third)
                point = point + interval * (first + 2 * second + 2 * third + fourth) / 6
        observe()
    arrays = [np.asarray(array, dtype=float) for array in (times, parameters, values, diagnostics)]
    for array in arrays:
        array.setflags(write=False)
    return VariationalDynamicsResult(
        times=arrays[0],
        parameters=arrays[1],
        expectations=arrays[2],
        metric_diagnostics=arrays[3],
        metadata={
            "experimental": True,
            "method": controls.method,
            "integrator": controls.integrator,
            "options": controls.model_dump(mode="json"),
            "estimator": selection.model_dump(mode="json"),
            "publications": execution.publications,
            "parameter_order": [str(parameter) for parameter in circuit.parameters],
            "metric_columns": [
                "retained_rank",
                "min_eigenvalue",
                "max_eigenvalue",
                "residual_norm",
                "velocity_norm",
            ],
            "units": {
                "time": "hbar/hamiltonian_energy",
                "expectations": "supplied_operator_units",
                "hbar": 1,
            },
            "regularization_bias": controls.regularization > 0,
            "uncertainty": (
                "No trajectory confidence interval; derivative sampling and integration "
                "errors propagate nonlinearly."
            ),
        },
    )

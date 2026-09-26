"""Measured complex response matrices and diagnostics shared by molecular spectra.

All expectation values use the selected estimator. Non-Hermitian operators are
split into Hermitian real and imaginary parts before submission to a V2 primitive.
No statevector simulation is substituted for a configured provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from chemrefine.engines.qiskit.context import ElectronicStructureContext, SolverComponents
from chemrefine.errors import ConfigError


@dataclass
class ExpectationSession:
    """A bound logical circuit compiled once, with a bounded observable workload."""

    context: ElectronicStructureContext
    components: SolverComponents
    circuit: Any
    max_measurements: int
    max_pauli_terms: int
    measurements: int = 0

    def __post_init__(self) -> None:
        """Compile the state for the selected estimator, retaining its observable layout."""
        if self.components.transpiler is not None:
            self.circuit = self.components.transpiler.run(
                self.circuit, **(self.components.transpiler_options or {})
            )

    def mapped(self, operator: Any) -> complex:
        """Measure both Hermitian parts of an arbitrary logical Pauli operator."""
        from qiskit.quantum_info import SparsePauliOp

        operator = operator.simplify(atol=1e-12)
        if len(operator) > self.max_pauli_terms:
            raise ConfigError("qiskit spectrum exceeds max_pauli_terms")
        if self.circuit.layout is not None:
            operator = operator.apply_layout(self.circuit.layout)
        publications = []
        factors = []
        for factor, coefficients in ((1, operator.coeffs.real), (1j, operator.coeffs.imag)):
            if np.any(np.abs(coefficients) > 1e-12):
                publications.append(
                    (self.circuit, SparsePauliOp(operator.paulis, coefficients).simplify())
                )
                factors.append(factor)
        if self.measurements + len(publications) > self.max_measurements:
            raise ConfigError("qiskit spectrum exceeds max_measurements")
        if not publications:
            return 0j
        self.measurements += len(publications)
        estimator = self.components.estimator
        if estimator is None:
            raise ConfigError("qiskit spectrum requires an estimator")
        values = estimator.run(publications).result()
        if len(values) != len(publications):
            raise ConfigError("qiskit spectrum estimator returned an incompatible result count")
        answer = 0j
        for factor, publication in zip(factors, values, strict=True):
            value = complex(np.asarray(publication.data.evs).item())
            if not np.isfinite(value) or abs(value.imag) > 1e-8:
                raise ConfigError("qiskit spectrum estimator returned a nonfinite/nonreal value")
            answer += factor * value.real
        return answer

    def fermionic(self, operator: Any) -> complex:
        """Map a composed molecular observable using the configured chemistry mapping."""
        project = getattr(self.context.mapper, "map_observable", self.context.mapper.map)
        mapped = project(operator.normal_order().simplify(atol=1e-12))
        if mapped is None:
            raise ConfigError(
                "qiskit spectrum observable leaves the selected tapering sector; "
                "use a mapping retaining that sector"
            )
        return self.mapped(mapped)


def real_value(value: complex, description: str, tolerance: float = 1e-7) -> float:
    """Reject a nonphysical complex or nonfinite quantity rather than discarding it."""
    if not np.isfinite(value) or abs(value.imag) > tolerance:
        raise ConfigError(f"qiskit {description} must be finite and real")
    return float(value.real)


def sector_diagnostics(
    context: ElectronicStructureContext,
    measure: Any,
    *,
    tolerance: float,
    target_s2: float | None,
    spin_tolerance: float,
) -> dict[str, float]:
    """Validate N and Ms eigenstate residuals and report, optionally constrain, S².

    ``measure`` accepts a fermionic observable, optionally evaluated in a reconstructed
    response state. Requiring the second moment prevents a superposition across sectors
    from passing merely because its mean particle count happens to match.
    """
    _, operators = context.problem.second_q_ops()
    expected = {
        "ParticleNumber": float(sum(context.num_particles)),
        "Magnetization": (context.num_particles[0] - context.num_particles[1]) / 2,
    }
    if target_s2 is not None:
        expected["AngularMomentum"] = target_s2
    result = {}
    for name in ("ParticleNumber", "Magnetization", "AngularMomentum"):
        if name not in operators:
            raise ConfigError(f"qiskit spectrum requires the {name} observable")
        observable = operators[name]
        value = real_value(measure(observable), name)
        second = real_value(measure(observable @ observable), f"{name} second moment")
        result[name] = value
        result[f"{name}_variance"] = second - value**2
        if name in expected:
            residual = abs(second - 2 * expected[name] * value + expected[name] ** 2)
            result[f"{name}_squared_residual"] = residual
            limit = spin_tolerance if name == "AngularMomentum" else tolerance
            if abs(value - expected[name]) > limit or residual > limit:
                raise ConfigError(
                    f"qiskit spectrum root is outside target {name} sector "
                    f"(mean={value:.8g}, target={expected[name]:.8g}, residual={residual:.8g})"
                )
    return result


def solve_response_problem(
    hessian: np.ndarray,
    metric: np.ndarray,
    *,
    conditioning_tolerance: float,
    residual_tolerance: float,
    frequency_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Solve a complex Hermitian, indefinite qEOM pencil without clipping bad roots.

    The basis includes excitation and de-excitation operators. Stable roots occur
    in ± pairs. Positive-frequency, positive-metric vectors describe excitations.
    Singular metrics and unstable complex frequencies are reported as failures;
    no pseudoinverse or silent deletion changes the caller's expansion space.
    """
    from scipy.linalg import eig

    if (
        hessian.ndim != 2
        or hessian.shape[0] != hessian.shape[1]
        or hessian.shape != metric.shape
        or hessian.shape[0] == 0
        or hessian.shape[0] % 2
        or not np.isfinite(hessian).all()
        or not np.isfinite(metric).all()
    ):
        raise ConfigError("qEOM matrices must be finite square matrices of the same even size")
    for name, matrix in (("Hessian", hessian), ("metric", metric)):
        if not np.allclose(matrix, matrix.conj().T, atol=residual_tolerance, rtol=0):
            raise ConfigError(f"qEOM {name} is not Hermitian within residual_tolerance")
    singular = np.linalg.svd(metric, compute_uv=False)
    ratio = float(singular[-1] / singular[0]) if singular[0] else 0.0
    if ratio <= conditioning_tolerance:
        raise ConfigError(
            "qEOM metric is singular or ill-conditioned; change the reference/excitation space "
            f"(relative smallest singular value={ratio:.3g})"
        )
    frequencies, vectors = eig(hessian, metric)
    if not np.isfinite(frequencies).all() or not np.isfinite(vectors).all():
        raise ConfigError("qEOM generalized eigensolver returned nonfinite values")
    if np.max(np.abs(frequencies.imag)) > frequency_tolerance:
        raise ConfigError("qEOM has unstable complex excitation frequencies")
    order = np.argsort(frequencies.real)
    frequencies, vectors = frequencies[order].real, vectors[:, order]
    if not np.allclose(frequencies, -frequencies[::-1], atol=frequency_tolerance, rtol=0):
        raise ConfigError("qEOM excitation/de-excitation frequencies do not form ± pairs")
    selected = np.flatnonzero(frequencies > frequency_tolerance)
    if len(selected) != len(frequencies) // 2:
        raise ConfigError(
            "qEOM contains zero/unstable modes; change the reference/excitation space"
        )
    residuals = []
    for index in selected:
        frequency, vector = frequencies[index], vectors[:, index]
        norm = real_value(np.vdot(vector, metric @ vector), "qEOM metric norm")
        if norm <= conditioning_tolerance:
            raise ConfigError("qEOM positive-frequency root has nonpositive metric norm")
        denominator = (
            np.linalg.norm(hessian) + abs(frequency) * np.linalg.norm(metric)
        ) * np.linalg.norm(vector)
        residual = float(
            np.linalg.norm(hessian @ vector - frequency * metric @ vector) / denominator
        )
        if residual > residual_tolerance:
            raise ConfigError("qEOM generalized eigenpair exceeds residual_tolerance")
        residuals.append(residual)
        vectors[:, index] /= np.sqrt(norm)
    return (
        frequencies[selected],
        vectors[:, selected],
        {
            "metric_relative_smallest_singular_value": ratio,
            "generalized_eigenpair_residuals": residuals,
            "matrix_dimension": len(metric),
        },
    )

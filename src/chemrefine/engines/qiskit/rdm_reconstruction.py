"""Complex D/Q/G-constrained reconstruction of measured one- and two-body RDMs.

These are necessary representability constraints, not sufficient certificates.
Energy regularization is optional and does not establish a variational bound.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations, product
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from chemrefine.engines.qiskit.determinants import (
    ComplexArray,
    FermionicHamiltonian,
    ReducedDensityMatrices,
)
from chemrefine.engines.qiskit.shadows import rdm_expectation
from chemrefine.errors import ConfigError


class RDMReconstructionOptions(BaseModel):
    """Explicit physical sector, convex loss, solver accuracy and work budgets."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    num_particles: StrictInt = Field(ge=0)
    constraints: Literal["D", "DQ", "DQG"] = "DQG"
    loss: Literal["frobenius", "nuclear"] = "frobenius"
    energy_weight: float = Field(0, ge=0)
    solver_tolerance: float = Field(1e-6, gt=0)
    feasibility_tolerance: float = Field(1e-5, gt=0)
    max_iterations: int = Field(10_000, ge=1)
    max_memory_mb: int = Field(512, ge=1)
    accept_inaccurate: bool = False


@dataclass(frozen=True)
class RDMReconstructionResult:
    """Raw measured and reconstructed RDMs with numerical/physical diagnostics."""

    raw: ReducedDensityMatrices
    reconstructed: ReducedDensityMatrices
    diagnostics: dict[str, Any]


def _validated_rdms(rdms: ReducedDensityMatrices) -> tuple[ComplexArray, ComplexArray]:
    """Validate and detach finite tensors while allowing noisy symmetry violations."""
    try:
        one = np.array(rdms.one_body, dtype=complex, copy=True)
        two = np.array(rdms.two_body, dtype=complex, copy=True)
    except (TypeError, ValueError) as exc:
        raise ConfigError("RDM reconstruction requires finite numerical tensors") from exc
    if (
        one.ndim != 2
        or one.shape[0] < 1
        or one.shape[0] != one.shape[1]
        or two.shape != (len(one),) * 4
        or not np.isfinite(one).all()
        or not np.isfinite(two).all()
    ):
        raise ConfigError("RDM reconstruction requires finite matching one/two-body tensors")
    one.setflags(write=False)
    two.setflags(write=False)
    return one, two


def _car_matrices(
    one: Any,
    element: Callable[[int, int, int, int], Any],
    num_modes: int,
    stack: Callable[[list[list[Any]]], Any],
) -> tuple[Any, Any]:
    """Apply canonical anticommutation identities to hole and particle-hole Grams.

    Q[(p,q),(r,s)] = <a_q a_p a†r a†s>, p<q and r<s.
    G[(p,q),(r,s)] = <(a†p aq)† (a†r as)>.
    """
    pairs = tuple(combinations(range(num_modes), 2))
    ordered = tuple(product(range(num_modes), repeat=2))
    hole = stack(
        [
            [
                (p == r) * (q == s)
                - (p == s) * (q == r)
                - (p == r) * one[s, q]
                + (p == s) * one[r, q]
                + (q == r) * one[s, p]
                - (q == s) * one[r, p]
                + element(r, s, p, q)
                for r, s in pairs
            ]
            for p, q in pairs
        ]
    )
    particle_hole = stack(
        [[(p == r) * one[q, s] - element(q, r, s, p) for r, s in ordered] for p, q in ordered]
    )
    return hole, particle_hole


def dqg_matrices(rdms: ReducedDensityMatrices) -> dict[str, ComplexArray]:
    """Return D, Q and G with complex CAR signs and explicitly declared pair order."""
    one, two = _validated_rdms(rdms)
    m = len(one)
    pairs = tuple(combinations(range(m), 2))
    particle = np.array(
        [[two[p, q, r, s] for r, s in pairs] for p, q in pairs], dtype=complex
    ).reshape((len(pairs),) * 2)
    hole, particle_hole = _car_matrices(
        one, lambda p, q, r, s: two[p, q, r, s], m, lambda rows: np.array(rows, dtype=complex)
    )
    return {"D": particle, "Q": hole.reshape((len(pairs),) * 2), "G": particle_hole}


def rdm_constraint_diagnostics(rdms: ReducedDensityMatrices, num_particles: int) -> dict[str, Any]:
    """Quantify trace/contraction residuals and necessary positivity constraints."""
    one, two = _validated_rdms(rdms)
    eigenvalues = np.linalg.eigvalsh((one + one.conj().T) / 2)
    matrices = dqg_matrices(rdms)
    return {
        "one_body_trace_residual": float(abs(np.trace(one) - num_particles)),
        "two_body_trace_residual": float(
            abs(np.einsum("pqpq->", two) - num_particles * (num_particles - 1))
        ),
        "contraction_residual": float(
            np.max(abs(np.einsum("pqrq->pr", two) - (num_particles - 1) * one))
        ),
        "one_body_hermiticity_residual": float(np.max(abs(one - one.conj().T))),
        "two_body_hermiticity_residual": float(np.max(abs(two - two.transpose(2, 3, 0, 1).conj()))),
        "two_body_antisymmetry_residual": float(
            max(np.max(abs(two + two.swapaxes(0, 1))), np.max(abs(two + two.swapaxes(2, 3))))
        ),
        "one_body_minimum_eigenvalue": float(eigenvalues.min()),
        "one_body_maximum_eigenvalue": float(eigenvalues.max()),
        "minimum_eigenvalues": {
            name: float(np.linalg.eigvalsh((matrix + matrix.conj().T) / 2).min())
            if matrix.size
            else 0.0
            for name, matrix in matrices.items()
        },
    }


def _weights(value: ArrayLike | None, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Accept finite nonnegative inverse-variance weights; zero masks missing data."""
    if value is None:
        return np.ones(shape)
    try:
        if np.iscomplexobj(value):
            raise ValueError("complex weight")
        weights = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} weights must be finite nonnegative real numbers") from exc
    if weights.shape != shape or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ConfigError(f"{name} weights must be nonnegative and have shape {shape}")
    return weights


def _energy_expression(
    one: Any,
    element: Callable[[int, int, int, int], Any],
    hamiltonian: FermionicHamiltonian,
) -> Any:
    """Contract the same observable convention for noisy numeric and symbolic tensors."""
    value: Any = 0.0
    for term in hamiltonian.terms:
        if not term.creation:
            value += term.coefficient
        elif len(term.creation) == 1:
            value += term.coefficient * one[*term.creation, *term.annihilation]
        else:
            value += term.coefficient * element(*term.creation, *reversed(term.annihilation))
    return value


def reconstruct_rdms(
    raw: ReducedDensityMatrices,
    options: RDMReconstructionOptions,
    *,
    one_body_weights: ArrayLike | None = None,
    two_body_weights: ArrayLike | None = None,
    hamiltonian: FermionicHamiltonian | None = None,
) -> RDMReconstructionResult:
    """Fit measured complex RDMs with CVXPY/SCS and validate the returned primal point.

    The default minimizes weighted squared data error. Nuclear-norm loss and
    energy regularization are explicit alternatives. All observed tensor entries
    contribute, including noisy duplicate symmetry-related measurements.
    """
    one_data, two_data = _validated_rdms(raw)
    m = len(one_data)
    if options.num_particles > m:
        raise ConfigError("RDM particle number exceeds the number of modes")
    if 4096 * m**4 > options.max_memory_mb * 1024**2:
        raise ConfigError("RDM convex-model storage estimate exceeds max_memory_mb")
    if options.energy_weight and hamiltonian is None:
        raise ConfigError("energy regularization requires an explicit Hamiltonian")
    if hamiltonian is not None and hamiltonian.num_modes != m:
        raise ConfigError("RDM and regularization Hamiltonian mode counts differ")
    one_weights = _weights(one_body_weights, one_data.shape, "one-body")
    two_weights = _weights(two_body_weights, two_data.shape, "two-body")
    if not np.any(one_weights) and not np.any(two_weights):
        raise ConfigError("RDM reconstruction requires at least one measured entry")
    try:
        from importlib import import_module

        # CVXPY exposes dynamically assembled atoms across supported provider versions.
        cp: Any = import_module("cvxpy")
    except ImportError as exc:
        raise ConfigError(
            "RDM reconstruction requires the optional cvxpy and SCS providers"
        ) from exc
    if "SCS" not in cp.installed_solvers():
        raise ConfigError("RDM reconstruction requires the SCS solver")
    pairs = tuple(combinations(range(m), 2))
    pair_indices = {pair: index for index, pair in enumerate(pairs)}
    # A one-by-one Hermitian matrix is real; using symmetric=True also avoids
    # CVXPY 1.9's nested-list constant in its scalar complex canonicalizer.
    one = cp.Variable((m, m), **({"symmetric": True} if m == 1 else {"hermitian": True}))
    particle = (
        cp.Variable(
            (len(pairs), len(pairs)),
            **({"symmetric": True} if len(pairs) == 1 else {"hermitian": True}),
        )
        if pairs
        else None
    )

    def element(p: int, q: int, r: int, s: int) -> Any:
        """Expose antisymmetric tensor entries through one Hermitian pair variable."""
        if p == q or r == s:
            return 0.0
        sign = (-1 if p > q else 1) * (-1 if r > s else 1)
        return (
            sign
            * cast("Any", particle)[
                pair_indices[(min(p, q), max(p, q))], pair_indices[(min(r, s), max(r, s))]
            ]
        )

    ordered = tuple(product(range(m), repeat=2))
    embedding = np.zeros((m * m, len(pairs)))
    for index, (p, q) in enumerate(pairs):
        embedding[p * m + q, index] = 1
        embedding[q * m + p, index] = -1
    full_two: Any = (
        embedding @ particle @ embedding.T
        if particle is not None
        else cp.Constant(np.zeros((m * m, m * m)))
    )
    constraints = [one >> 0, np.eye(m) - one >> 0, cp.trace(one) == options.num_particles]
    if particle is not None:
        constraints.append(particle >> 0)
        constraints.append(
            cp.trace(particle) == options.num_particles * (options.num_particles - 1) / 2
        )
    for p, r in ordered:
        constraints.append(
            sum(element(p, q, r, q) for q in range(m)) == (options.num_particles - 1) * one[p, r]
        )
    if m > 1:
        # Vectorized CAR maps keep canonicalization polynomial in tensor size,
        # rather than expanding thousands of nested scalar CVXPY expressions.
        row_first, row_second, col_first, col_second = np.indices((m,) * 4).reshape(4, -1)
        gamma = cp.reshape(one, (m * m,), order="C")
        tensor = cp.reshape(full_two, (m**4,), order="C")
        hole_flat = (
            ((row_first == col_first) & (row_second == col_second)).astype(float)
            - ((row_first == col_second) & (row_second == col_first)).astype(float)
            - cp.multiply(row_first == col_first, gamma[col_second * m + row_second])
            + cp.multiply(row_first == col_second, gamma[col_first * m + row_second])
            + cp.multiply(row_second == col_first, gamma[col_second * m + row_first])
            - cp.multiply(row_second == col_second, gamma[col_first * m + row_first])
            + tensor[(col_first * m + col_second) * m * m + row_first * m + row_second]
        )
        indices = np.array([row_first * m + row_second for row_first, row_second in pairs])
        hole = cp.reshape(hole_flat, (m * m, m * m), order="C")[np.ix_(indices, indices)]
        particle_hole = cp.reshape(
            cp.multiply(row_first == col_first, gamma[row_second * m + col_second])
            - tensor[(row_second * m + col_first) * m * m + col_second * m + row_first],
            (m * m, m * m),
            order="C",
        )
        if "Q" in options.constraints:
            constraints.append(hole >> 0)
        if "G" in options.constraints:
            constraints.append(particle_hole >> 0)
    first_error = cp.multiply(np.sqrt(one_weights), one - one_data)
    second_error = cp.multiply(
        np.sqrt(two_weights.reshape(m * m, m * m)), full_two - two_data.reshape(m * m, m * m)
    )
    if options.loss == "frobenius":
        fit: Any = sum(
            cp.sum_squares(part(error))
            for part in (cp.real, cp.imag)
            for error in (first_error, second_error)
        )
    else:
        fit = cp.normNuc(first_error) + cp.normNuc(second_error)
    energy = 0.0 if hamiltonian is None else _energy_expression(one, element, hamiltonian)
    problem = cp.Problem(cp.Minimize(fit + options.energy_weight * cp.real(energy)), constraints)
    try:
        with warnings.catch_warnings(record=True) as solver_warnings:
            warnings.filterwarnings(
                "always", message=r"Solution may be inaccurate.*", category=UserWarning
            )
            problem.solve(
                solver="SCS",
                eps=options.solver_tolerance,
                max_iters=options.max_iterations,
                use_indirect=False,
                verbose=False,
            )
    except cp.error.SolverError as exc:
        raise ConfigError(f"RDM semidefinite solver failed: {exc}") from exc
    statuses = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} if options.accept_inaccurate else {cp.OPTIMAL}
    if (
        problem.status not in statuses
        or one.value is None
        or (particle is not None and particle.value is None)
    ):
        raise ConfigError(
            f"RDM reconstruction did not return an accepted solution: {problem.status}"
        )
    reconstructed_one = np.asarray(one.value, dtype=np.complex128)
    reconstructed_two = np.asarray(full_two.value, dtype=np.complex128).reshape((m,) * 4)
    reconstructed = ReducedDensityMatrices(reconstructed_one, reconstructed_two)
    checks = rdm_constraint_diagnostics(reconstructed, options.num_particles)
    tolerance = options.feasibility_tolerance
    residuals = [value for name, value in checks.items() if name.endswith("residual")]
    if (
        max(residuals) > tolerance
        or checks["one_body_minimum_eigenvalue"] < -tolerance
        or checks["one_body_maximum_eigenvalue"] > 1 + tolerance
        or any(checks["minimum_eigenvalues"][name] < -tolerance for name in options.constraints)
    ):
        raise ConfigError("RDM solver output violates the requested feasibility tolerance")
    reconstructed_one.setflags(write=False)
    reconstructed_two.setflags(write=False)
    raw_energy = None
    if hamiltonian is not None:
        value = _energy_expression(one_data, lambda p, q, r, s: two_data[p, q, r, s], hamiltonian)
        raw_energy = {"real": float(np.real(value)), "imag": float(np.imag(value))}
    diagnostics = {
        "experimental": True,
        "solver": "SCS",
        "solver_warnings": [str(item.message) for item in solver_warnings],
        "status": problem.status,
        "objective": float(problem.value),
        "data_fit_objective": float(fit.value),
        "iterations": problem.solver_stats.num_iters,
        "solve_time_seconds": problem.solver_stats.solve_time,
        "constraints": options.constraints,
        "necessary_not_sufficient": True,
        "energy_weight": options.energy_weight,
        "loss": options.loss,
        "variational_bound": False,
        "constraint_checks": checks,
        "raw_energy": raw_energy,
        "reconstructed_energy": None
        if hamiltonian is None
        else rdm_expectation(reconstructed, hamiltonian),
    }
    return RDMReconstructionResult(
        ReducedDensityMatrices(one_data, two_data), reconstructed, diagnostics
    )

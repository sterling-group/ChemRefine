"""Variational orbital rotations for a fixed quantum-sampled determinant pool."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from typing import Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from chemrefine.engines.qiskit.determinants import (
    ComplexArray,
    FermionicHamiltonian,
    FermionTerm,
    ProjectedEigensystem,
    projected_eigensystem,
)
from chemrefine.errors import ConfigError


class OrbitalOptimizationOptions(BaseModel):
    """Budgeted alternating CI/orbital optimization within the active modes only."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    max_iterations: int = Field(10, ge=1)
    max_optimizer_iterations: int = Field(50, ge=1)
    energy_tolerance: float = Field(1e-8, gt=0)
    gradient_tolerance: float = Field(1e-6, gt=0)
    spin_mode: Literal["shared", "independent", "spin_orbital"] = "shared"
    complex_rotations: bool = True


@dataclass(frozen=True)
class OrbitalOptimizationResult:
    """Accepted active-space frame, root states, and monotone target-energy history."""

    eigensystem: ProjectedEigensystem
    rotation: ComplexArray
    energy_history: tuple[float, ...]
    converged: bool
    objective_evaluations: int


def _tensors(
    hamiltonian: FermionicHamiltonian, max_memory_mb: int
) -> tuple[float, ComplexArray, ComplexArray]:
    """Expand canonical coefficients under a conservative dense rotation budget."""
    n = hamiltonian.num_modes
    if 128 * (n**4 + n * n) > max_memory_mb * 1024**2:
        raise ConfigError("orbital rotation tensor storage exceeds max_memory_mb")
    one = np.zeros((n, n), dtype=complex)
    two = np.zeros((n,) * 4, dtype=complex)
    constant = 0.0
    for term in hamiltonian.terms:
        if not term.creation:
            constant += term.coefficient.real
        elif len(term.creation) == 1:
            one[term.creation[0], term.annihilation[0]] += term.coefficient
        else:
            two[*term.creation, *term.annihilation] += term.coefficient
    return constant, one, two


def _rotated_tensors(
    one: ComplexArray, two: ComplexArray, rotation: ComplexArray
) -> tuple[ComplexArray, ComplexArray]:
    """Apply the same unitary convention to each creator and annihilator index."""
    return rotation.conj().T @ one @ rotation, np.einsum(
        "ap,bq,cr,ds,abcd->pqrs",
        rotation.conj(),
        rotation.conj(),
        rotation,
        rotation,
        two,
        optimize=True,
    )


def _from_tensors(constant: float, one: ComplexArray, two: ComplexArray) -> FermionicHamiltonian:
    """Canonicalize all rotated terms without dropping small nonzero integrals."""
    terms = [FermionTerm((), (), constant)]
    terms.extend(
        FermionTerm((int(p),), (int(q),), one[p, q]) for p, q in zip(*np.nonzero(one), strict=True)
    )
    terms.extend(
        FermionTerm((int(p), int(q)), (int(r), int(s)), two[p, q, r, s])
        for p, q, r, s in zip(*np.nonzero(two), strict=True)
    )
    return FermionicHamiltonian(one.shape[0], tuple(terms))


def rotate_hamiltonian(
    hamiltonian: FermionicHamiltonian, rotation: ComplexArray, *, max_memory_mb: int = 512
) -> FermionicHamiltonian:
    """Transform H into orbitals whose columns are given in the original mode basis."""
    n = hamiltonian.num_modes
    rotation = np.asarray(rotation, dtype=complex)
    if (
        rotation.shape != (n, n)
        or not np.isfinite(rotation).all()
        or not np.allclose(rotation.conj().T @ rotation, np.eye(n), atol=1e-10, rtol=0)
    ):
        raise ConfigError("orbital rotation must be a finite unitary matching the mode count")
    constant, one, two = _tensors(hamiltonian, max_memory_mb)
    one, two = _rotated_tensors(one, two, rotation)
    return _from_tensors(constant, one, two)


def optimize_orbitals(
    hamiltonian: FermionicHamiltonian,
    determinants: tuple[int, ...],
    *,
    options: OrbitalOptimizationOptions | None = None,
    num_roots: int = 1,
    target_root: int = 0,
    max_subspace_dimension: int = 10_000,
    max_memory_mb: int = 512,
    eigensolver_tolerance: float = 1e-10,
    eigensolver_iterations: int = 1_000,
    seed: int | None = 0,
) -> OrbitalOptimizationResult:
    """Alternate fixed-state orbital minimization and fixed-pool rediagonalization.

    Only energy-improving frames are accepted. The returned states live in the
    returned orbital frame and carry its rotation. Their observable and RDM
    methods default to the original basis. No frozen/core orbitals are introduced.
    """
    from scipy.linalg import expm
    from scipy.optimize import minimize

    controls = options or OrbitalOptimizationOptions()
    n = hamiltonian.num_modes
    if not 0 <= target_root < num_roots:
        raise ConfigError("orbital target_root must identify a requested root")
    if controls.spin_mode != "spin_orbital" and n % 2:
        raise ConfigError("spin-preserving orbital optimization requires alpha-then-beta modes")
    blocks = (
        [tuple(range(n))]
        if controls.spin_mode == "spin_orbital"
        else [tuple(range(n // 2)), tuple(range(n // 2, n))]
    )
    groups = blocks[:1] if controls.spin_mode == "shared" else blocks
    pairs = [pair for block in groups for pair in combinations(block, 2)]
    if not pairs:
        raise ConfigError("orbital optimization requires at least two rotatable orbitals")
    constant, original_one, original_two = _tensors(hamiltonian, max_memory_mb)

    def solve(model: FermionicHamiltonian) -> ProjectedEigensystem:
        """Use identical root and resource controls in every accepted orbital frame."""
        return projected_eigensystem(
            model,
            determinants,
            num_roots=num_roots,
            max_subspace_dimension=max_subspace_dimension,
            max_memory_mb=max_memory_mb,
            tolerance=eigensolver_tolerance,
            max_iterations=eigensolver_iterations,
            seed=seed,
        )

    current = solve(hamiltonian)
    rotation = np.eye(n, dtype=complex)
    history = [float(current.energies[target_root])]
    objective_evaluations = 0
    converged = False
    parameter_count = len(pairs) * (2 if controls.complex_rotations else 1)

    def unitary(parameters: np.ndarray) -> ComplexArray:
        """Exponentiate anti-Hermitian generators, sharing spatial rotations if requested."""
        generator = np.zeros((n, n), dtype=complex)
        for index, (p, q) in enumerate(pairs):
            value = parameters[index]
            if controls.complex_rotations:
                value = value + 1j * parameters[index + len(pairs)]
            generator[p, q], generator[q, p] = value, -value.conjugate()
            if controls.spin_mode == "shared":
                generator[p + n // 2, q + n // 2] = value
                generator[q + n // 2, p + n // 2] = -value.conjugate()
        return cast("ComplexArray", expm(generator))

    for _ in range(controls.max_iterations):
        rdms = current.states[target_root].rdms(max_memory_mb=max_memory_mb)
        frame_one, frame_two = _rotated_tensors(original_one, original_two, rotation)
        two_rdm = cast("ComplexArray", rdms.two_body)

        def objective(
            parameters: np.ndarray,
            frame_one: ComplexArray = frame_one,
            frame_two: ComplexArray = frame_two,
            one_rdm: ComplexArray = rdms.one_body,
            two_rdm: ComplexArray = two_rdm,
        ) -> float:
            """Contract the rotated Hamiltonian with fixed CI one/two-body RDMs."""
            nonlocal objective_evaluations
            objective_evaluations += 1
            one, two = _rotated_tensors(frame_one, frame_two, unitary(parameters))
            value = (
                constant
                + np.einsum("pq,pq->", one, one_rdm)
                + np.einsum("pqrs,pqsr->", two, two_rdm)
            )
            return float(value.real)

        optimum = minimize(
            objective,
            np.zeros(parameter_count),
            method="BFGS",
            options={
                "maxiter": controls.max_optimizer_iterations,
                "gtol": controls.gradient_tolerance,
            },
        )
        if not np.isfinite(optimum.fun) or not np.isfinite(optimum.x).all():
            raise ConfigError("orbital optimizer returned non-finite parameters or energy")
        candidate_rotation = rotation @ unitary(optimum.x)
        one, two = _rotated_tensors(original_one, original_two, candidate_rotation)
        candidate = solve(_from_tensors(constant, one, two))
        energy = float(candidate.energies[target_root])
        improvement = history[-1] - energy
        if improvement <= controls.energy_tolerance:
            converged = bool(optimum.success) and abs(improvement) <= controls.energy_tolerance
            break
        current, rotation = candidate, candidate_rotation
        history.append(energy)
    rotation.setflags(write=False)
    current = replace(
        current, states=tuple(replace(state, orbital_rotation=rotation) for state in current.states)
    )
    return OrbitalOptimizationResult(
        current, rotation, tuple(history), converged, objective_evaluations
    )

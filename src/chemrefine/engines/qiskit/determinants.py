"""Sparse fermionic states and explicit sampled-subspace linear algebra.

Modes are numbered from the least significant bit. Python integers retain modes
above bit 63. A projected basis is precisely the supplied determinants: no spin
factor Cartesian closure and no full Fock-space matrix is constructed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from chemrefine.errors import ConfigError

ComplexArray = NDArray[np.complex128]


def apply_operators(
    bits: int, creation: tuple[int, ...], annihilation: tuple[int, ...]
) -> tuple[int, int] | None:
    """Apply a normal-ordered monomial, with tuples in written operator order."""
    sign = 1
    for mode in reversed(annihilation):
        if not bits & (1 << mode):
            return None
        sign *= -1 if (bits & ((1 << mode) - 1)).bit_count() % 2 else 1
        bits ^= 1 << mode
    for mode in reversed(creation):
        if bits & (1 << mode):
            return None
        sign *= -1 if (bits & ((1 << mode) - 1)).bit_count() % 2 else 1
        bits |= 1 << mode
    return bits, sign


@dataclass(frozen=True)
class FermionTerm:
    """A coefficient times creators followed by annihilators in written order."""

    creation: tuple[int, ...]
    annihilation: tuple[int, ...]
    coefficient: complex


@dataclass(frozen=True)
class FermionicHamiltonian:
    """Hermitian number-conserving one/two-body terms in an orthonormal mode basis.

    Integral tensor prefactors are already included in term coefficients. Scalar
    terms are allowed, but the Nature adapter excludes prepared energy offsets.
    Complex hopping and unrestricted spin blocks need no special representation.
    """

    num_modes: int
    terms: tuple[FermionTerm, ...]

    def __post_init__(self) -> None:
        """Canonicalize fermionic signs and verify Hermiticity without a matrix."""
        if (
            isinstance(self.num_modes, bool)
            or not isinstance(self.num_modes, int)
            or self.num_modes < 1
        ):
            raise ConfigError("fermionic num_modes must be a positive integer")
        combined: dict[tuple[tuple[int, ...], tuple[int, ...]], complex] = {}
        for term in self.terms:
            creators, annihilators = tuple(term.creation), tuple(term.annihilation)
            if len(creators) != len(annihilators) or len(creators) > 2:
                raise ConfigError(
                    "fermionic terms must conserve number and have at most two bodies"
                )
            if any(
                isinstance(mode, bool)
                or not isinstance(mode, int)
                or not 0 <= mode < self.num_modes
                for mode in (*creators, *annihilators)
            ):
                raise ConfigError("fermionic term mode lies outside num_modes")
            try:
                coefficient = complex(term.coefficient)
            except (TypeError, ValueError) as exc:
                raise ConfigError("fermionic coefficients must be finite numbers") from exc
            if not np.isfinite(coefficient):
                raise ConfigError("fermionic coefficients must be finite numbers")
            if len(set(creators)) != len(creators) or len(set(annihilators)) != len(annihilators):
                continue
            swaps = sum(a > b for a, b in combinations(creators, 2))
            swaps += sum(a < b for a, b in combinations(annihilators, 2))
            key = tuple(sorted(creators)), tuple(sorted(annihilators, reverse=True))
            combined[key] = combined.get(key, 0j) + (-1) ** swaps * coefficient
        for (creators, annihilators), coefficient in combined.items():
            adjoint = tuple(reversed(annihilators)), tuple(reversed(creators))
            if not np.isclose(
                coefficient, combined.get(adjoint, 0j).conjugate(), atol=1e-12, rtol=1e-12
            ):
                raise ConfigError("fermionic Hamiltonian must be Hermitian")
        terms = tuple(
            FermionTerm(creators, annihilators, coefficient)
            for (creators, annihilators), coefficient in sorted(combined.items())
            if coefficient != 0
        )
        object.__setattr__(self, "terms", terms)

    @classmethod
    def from_operator(cls, operator: Any, *, num_modes: int) -> FermionicHamiltonian:
        """Adapt Nature's normal-ordered operator without dropping complex coefficients."""
        terms = []
        for actions, coefficient in operator.normal_order().terms():
            if any(action not in {"+", "-"} for action, _ in actions):
                raise ConfigError("unsupported fermionic operator action")
            terms.append(
                FermionTerm(
                    tuple(int(mode) for action, mode in actions if action == "+"),
                    tuple(int(mode) for action, mode in actions if action == "-"),
                    complex(coefficient),
                )
            )
        return cls(num_modes, tuple(terms))

    def apply(
        self, amplitudes: Mapping[int, complex], *, max_entries: int = 1_000_000
    ) -> dict[int, complex]:
        """Apply to a sparse state, retaining contributions outside its input basis."""
        result: dict[int, complex] = {}
        for bits, amplitude in amplitudes.items():
            for term in self.terms:
                moved = apply_operators(bits, term.creation, term.annihilation)
                if moved is not None:
                    target, sign = moved
                    result[target] = result.get(target, 0j) + sign * term.coefficient * amplitude
                    if len(result) > max_entries:
                        raise ConfigError("fermionic operator image exceeds its storage budget")
        return {bits: value for bits, value in result.items() if value != 0}


def _basis(determinants: Iterable[int], num_modes: int) -> tuple[int, ...]:
    """Validate a nonempty, unique basis without narrowing integer width."""
    basis = tuple(determinants)
    if not basis or any(
        isinstance(bits, bool)
        or not isinstance(bits, int)
        or bits < 0
        or bits.bit_length() > num_modes
        for bits in basis
    ):
        raise ConfigError(
            "determinants must be nonempty nonnegative Python integers within num_modes"
        )
    if len(set(basis)) != len(basis):
        raise ConfigError("determinants must be unique")
    if len({bits.bit_count() for bits in basis}) != 1:
        raise ConfigError("determinants must occupy one total-particle sector")
    return basis


@dataclass(frozen=True)
class ReducedDensityMatrices:
    """One/two RDMs: gamma[p,q]=<a†p aq>, Gamma[p,q,r,s]=<a†p a†q a_s a_r>."""

    one_body: ComplexArray
    two_body: ComplexArray | None = None

    def natural_occupations(self) -> NDArray[np.float64]:
        """Return descending one-RDM eigenvalues; transition matrices are not densities."""
        if not np.allclose(self.one_body, self.one_body.conj().T, atol=1e-10, rtol=0):
            raise ConfigError("natural occupations require a Hermitian state one-RDM")
        values = np.array(np.linalg.eigvalsh(self.one_body)[::-1], dtype=float)
        values.setflags(write=False)
        return values

    def number_correlations(self) -> ComplexArray:
        """Return <n_p n_q>, including the n_p squared equals n_p diagonal."""
        if self.two_body is None:
            raise ConfigError("number correlations require a two-body RDM")
        values = np.array(np.einsum("pqpq->pq", self.two_body), dtype=complex)
        values += np.diag(np.diag(self.one_body))
        values.setflags(write=False)
        return values

    def spin_blocks(self) -> dict[str, ComplexArray]:
        """Return alpha/beta blocks for the declared alpha-then-beta convention."""
        n = self.one_body.shape[0]
        if n % 2:
            raise ConfigError("spin blocks require an even alpha-then-beta mode count")
        n //= 2
        blocks = {"alpha": self.one_body[:n, :n], "beta": self.one_body[n:, n:]}
        if self.two_body is not None:
            for left, a in (("alpha", slice(0, n)), ("beta", slice(n, 2 * n))):
                for right, b in (("alpha", slice(0, n)), ("beta", slice(n, 2 * n))):
                    blocks[f"{left}_{right}"] = self.two_body[a, b, a, b]
        return blocks

    def spin_summed(self) -> ReducedDensityMatrices:
        """Sum equal-spatial-index spin blocks without identifying alpha/beta orbitals."""
        blocks = self.spin_blocks()
        two = None
        if self.two_body is not None:
            two = (
                blocks["alpha_alpha"]
                + blocks["alpha_beta"]
                + blocks["beta_alpha"]
                + blocks["beta_beta"]
            )
        return ReducedDensityMatrices(blocks["alpha"] + blocks["beta"], two)


@dataclass(frozen=True)
class DeterminantState:
    """Normalized subspace state with owned immutable coefficients and exact bitstrings."""

    num_modes: int
    determinants: tuple[int, ...]
    amplitudes: ComplexArray
    orbital_rotation: ComplexArray | None = None

    def __post_init__(self) -> None:
        """Copy and validate coefficients, preserving their phase and basis order."""
        if (
            isinstance(self.num_modes, bool)
            or not isinstance(self.num_modes, int)
            or self.num_modes < 1
        ):
            raise ConfigError("state num_modes must be a positive integer")
        basis = _basis(self.determinants, self.num_modes)
        try:
            amplitudes = np.array(self.amplitudes, dtype=complex, copy=True)
        except (TypeError, ValueError) as exc:
            raise ConfigError("state amplitudes must be finite complex numbers") from exc
        if amplitudes.shape != (len(basis),) or not np.isfinite(amplitudes).all():
            raise ConfigError("state amplitudes must be finite and match the determinant count")
        if not np.isclose(np.vdot(amplitudes, amplitudes), 1, atol=1e-9, rtol=0):
            raise ConfigError("state amplitudes must have unit normalization")
        amplitudes.setflags(write=False)
        object.__setattr__(self, "determinants", basis)
        object.__setattr__(self, "amplitudes", amplitudes)
        if self.orbital_rotation is not None:
            try:
                rotation = np.array(self.orbital_rotation, dtype=complex, copy=True)
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    "state orbital_rotation must contain finite complex numbers"
                ) from exc
            if (
                rotation.shape != (self.num_modes, self.num_modes)
                or not np.isfinite(rotation).all()
                or not np.allclose(
                    rotation.conj().T @ rotation, np.eye(self.num_modes), atol=1e-10, rtol=0
                )
            ):
                raise ConfigError(
                    "state orbital_rotation must be a finite unitary matching num_modes"
                )
            rotation.setflags(write=False)
            object.__setattr__(self, "orbital_rotation", rotation)

    @property
    def num_particles(self) -> int:
        """Return the total particle number of this fixed-number state."""
        return self.determinants[0].bit_count()

    def sparse(self) -> dict[int, complex]:
        """Return a detached sparse representation, excluding exact zero amplitudes."""
        return {
            bits: value
            for bits, value in zip(self.determinants, self.amplitudes.tolist(), strict=True)
            if value != 0
        }

    def expectation(
        self,
        operator: FermionicHamiltonian,
        *,
        max_entries: int = 1_000_000,
        basis: Literal["original", "state"] = "original",
        max_memory_mb: int = 512,
    ) -> float:
        """Evaluate an observable on the actual state, including complex integrals."""
        if operator.num_modes != self.num_modes:
            raise ConfigError("observable and state mode counts differ")
        if basis not in {"original", "state"}:
            raise ConfigError("observable basis must be original or state")
        if basis == "original" and self.orbital_rotation is not None:
            from chemrefine.engines.qiskit.orbitals import rotate_hamiltonian

            operator = rotate_hamiltonian(
                operator, self.orbital_rotation, max_memory_mb=max_memory_mb
            )
        state = self.sparse()
        image = operator.apply(state, max_entries=max_entries)
        value = sum(
            amplitude.conjugate() * image.get(bits, 0j) for bits, amplitude in state.items()
        )
        return float(value.real)

    def rdms(
        self,
        *,
        max_order: int = 2,
        max_memory_mb: int = 512,
        bra: DeterminantState | None = None,
        basis: Literal["original", "state"] = "original",
    ) -> ReducedDensityMatrices:
        """Compute state or transition RDMs using sparse annihilated-state overlaps.

        ``bra`` selects <bra|O|self>; omitting it yields ordinary state RDMs.
        The storage estimate is conservative and is not an operating-system limit.
        """
        left = self if bra is None else bra
        if basis not in {"original", "state"}:
            raise ConfigError("RDM basis must be original or state")
        if left.num_modes != self.num_modes or max_order not in {1, 2}:
            raise ConfigError("RDMs require matching modes and max_order 1 or 2")
        if left.orbital_rotation is not None or self.orbital_rotation is not None:
            left_rotation = (
                np.eye(self.num_modes) if left.orbital_rotation is None else left.orbital_rotation
            )
            right_rotation = (
                np.eye(self.num_modes) if self.orbital_rotation is None else self.orbital_rotation
            )
            if not np.allclose(left_rotation, right_rotation, atol=1e-10, rtol=0):
                raise ConfigError("transition RDMs require states in a common orbital frame")
        n = self.num_modes
        pairs = n * (n - 1) // 2 if max_order == 2 else 0
        estimate = 32 * (n * n + (n**4 if max_order == 2 else 0))
        estimate += 192 * (len(self.determinants) + len(left.determinants)) * (n + pairs)
        if estimate > max_memory_mb * 1024**2:
            raise ConfigError("RDM storage estimate exceeds max_memory_mb")

        def removed(
            state: DeterminantState, groups: Sequence[tuple[int, ...]]
        ) -> dict[int, dict[int, complex]]:
            """Group sparse annihilation images by residual determinant."""
            images: dict[int, dict[int, complex]] = {}
            for bits, amplitude in zip(state.determinants, state.amplitudes, strict=True):
                for index, modes in enumerate(groups):
                    moved = apply_operators(bits, (), tuple(reversed(modes)))
                    if moved is not None:
                        residual, sign = moved
                        images.setdefault(residual, {})[index] = sign * amplitude
            return images

        def overlaps(groups: Sequence[tuple[int, ...]]) -> ComplexArray:
            """Contract matching residual determinants into a transition Gram matrix."""
            right_images = removed(self, groups)
            left_images = right_images if left is self else removed(left, groups)
            values = np.zeros((len(groups), len(groups)), dtype=complex)
            for residual, right_values in right_images.items():
                for p, a in left_images.get(residual, {}).items():
                    for q, b in right_values.items():
                        values[p, q] += a.conjugate() * b
            return values

        one = overlaps([(mode,) for mode in range(n)])
        two = None
        if max_order == 2:
            groups = list(combinations(range(n), 2))
            pair_values = overlaps(groups)
            two = np.zeros((n,) * 4, dtype=complex)
            for i, (p, q) in enumerate(groups):
                for j, (r, s) in enumerate(groups):
                    value = pair_values[i, j]
                    two[p, q, r, s] = two[q, p, s, r] = value
                    two[p, q, s, r] = two[q, p, r, s] = -value
        if basis == "original" and self.orbital_rotation is not None:
            rotation = self.orbital_rotation
            one = rotation.conj() @ one @ rotation.T
            if two is not None:
                two = np.einsum(
                    "pi,qj,rk,sl,ijkl->pqrs",
                    rotation.conj(),
                    rotation.conj(),
                    rotation,
                    rotation,
                    two,
                    optimize=True,
                )
        if two is not None:
            two.setflags(write=False)
        one.setflags(write=False)
        return ReducedDensityMatrices(one, two)


@dataclass(frozen=True)
class ProjectedEigensystem:
    """Eigenstates of precisely the supplied subspace and projected residual norms."""

    energies: NDArray[np.float64]
    states: tuple[DeterminantState, ...]
    residuals: tuple[float, ...]


def projected_operator(hamiltonian: FermionicHamiltonian, determinants: Iterable[int]) -> Any:
    """Build a matrix-free Hermitian operator on an explicit determinant basis."""
    from scipy.sparse.linalg import LinearOperator

    basis = _basis(determinants, hamiltonian.num_modes)
    index = {bits: row for row, bits in enumerate(basis)}

    def matvec(vector: ArrayLike) -> ComplexArray:
        """Apply H and retain only matrix elements within the supplied basis."""
        vector = np.asarray(vector).reshape(len(basis))
        output = np.zeros(len(basis), dtype=complex)
        for column, bits in enumerate(basis):
            for term in hamiltonian.terms:
                moved = apply_operators(bits, term.creation, term.annihilation)
                if moved is not None:
                    target, sign = moved
                    row = index.get(target)
                    if row is not None:
                        output[row] += term.coefficient * sign * vector[column]
        return output

    return LinearOperator((len(basis), len(basis)), matvec=matvec, rmatvec=matvec, dtype=complex)


def projected_eigensystem(
    hamiltonian: FermionicHamiltonian,
    determinants: Iterable[int],
    *,
    num_roots: int = 1,
    max_subspace_dimension: int = 10_000,
    max_memory_mb: int = 512,
    tolerance: float = 1e-10,
    max_iterations: int = 1_000,
    seed: int | None = 0,
) -> ProjectedEigensystem:
    """Solve a bounded sampled determinant subspace, retaining every requested root."""
    from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh

    basis = _basis(determinants, hamiltonian.num_modes)
    dim = len(basis)
    if not 1 <= num_roots <= dim or dim > max_subspace_dimension:
        raise ConfigError("projected roots or basis exceed the subspace dimension budget")
    if not np.isfinite(tolerance) or tolerance <= 0 or max_iterations < 1:
        raise ConfigError("projected tolerance and iteration budget must be positive")
    dense = dim <= 64 or num_roots >= dim - 1
    ncv = min(dim, max(2 * num_roots + 1, 20))
    estimate = 64 * dim * (dim if dense else ncv + num_roots + 4)
    estimate += 192 * (dim + len(hamiltonian.terms))
    if estimate > max_memory_mb * 1024**2:
        raise ConfigError("projected eigensolver storage estimate exceeds max_memory_mb")
    operator = projected_operator(hamiltonian, basis)
    if dense:
        energies, vectors = np.linalg.eigh(operator @ np.eye(dim, dtype=complex))
        energies, vectors = energies[:num_roots], vectors[:, :num_roots]
    else:
        rng = np.random.default_rng(seed)
        initial = rng.normal(size=dim) + 1j * rng.normal(size=dim)
        # A negative definite shift keeps an exact zero eigenstate inside the
        # iterative Krylov search instead of losing it through H applied to v0.
        # Each fermionic monomial has norm <= 1, so this is a norm bound.
        shift = 1.0 + sum(abs(term.coefficient) for term in hamiltonian.terms)
        shifted = LinearOperator(
            operator.shape, matvec=lambda vector: operator @ vector - shift * vector, dtype=complex
        )
        try:
            energies, vectors = eigsh(
                shifted,
                k=num_roots,
                which="SA",
                v0=initial,
                tol=tolerance / max(1.0, shift),
                maxiter=max_iterations,
                ncv=ncv,
            )
        except ArpackNoConvergence as exc:
            raise ConfigError(
                "projected eigensolver did not converge within its iteration budget"
            ) from exc
        energies = energies + shift
        order = np.argsort(energies)
        energies, vectors = energies[order], vectors[:, order]
    states, residuals = [], []
    for energy, vector in zip(energies, vectors.T, strict=True):
        pivot = int(np.argmax(np.abs(vector)))
        vector *= np.exp(-1j * np.angle(vector[pivot]))
        residual = float(np.linalg.norm(operator @ vector - energy * vector))
        if not np.isfinite(energy) or residual > max(1e-8, 100 * tolerance):
            raise ConfigError("projected eigensolver returned an invalid energy or residual")
        states.append(DeterminantState(hamiltonian.num_modes, basis, vector))
        residuals.append(residual)
    energies = np.array(energies, dtype=float)
    energies.setflags(write=False)
    return ProjectedEigensystem(energies, tuple(states), tuple(residuals))

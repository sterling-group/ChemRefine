"""Lossless adapters from transformed Nature integrals to fermionic simulators.

The active electronic operator never includes nuclear, frozen-core, or inactive
energy offsets. Those constants remain on ``PreparedProblem`` for reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from chemrefine.engines.qiskit.problem import PreparedProblem
from chemrefine.errors import ConfigError


@dataclass(frozen=True)
class FermionicIntegrals:
    """Real, shared-spatial-orbital active integrals in chemist ordering.

    ``occupations`` contains actual occupied orbital indices, including any
    reordering requested during active-space preparation. No constant is stored.
    """

    h1: NDArray[np.float64]
    h2: NDArray[np.float64]
    norb: int
    nelec: tuple[int, int]
    occupations: tuple[tuple[int, ...], tuple[int, ...]]
    multiplicity: int

    @property
    def target_spin_squared(self) -> float:
        """Return the requested total-spin eigenvalue S(S+1)."""
        spin = (self.multiplicity - 1) / 2
        return spin * (spin + 1)


def _real_dense(tensor: Any, *, name: str) -> NDArray[np.float64]:
    """Copy finite real tensor data without silently dropping imaginary parts."""
    array = np.asarray(tensor.to_dense() if hasattr(tensor, "to_dense") else tensor)
    if np.iscomplexobj(array) and np.any(array.imag != 0):
        raise ConfigError(f"qiskit fermionic adapter requires real {name} integrals")
    array = np.array(array.real, dtype=float, copy=True)
    if not np.isfinite(array).all():
        raise ConfigError(f"qiskit fermionic adapter requires finite {name} integrals")
    return array


def _chemist_tensor(tensor: Any) -> NDArray[np.float64]:
    """Respect Nature's explicit symmetric-integral type instead of guessing order."""
    from qiskit_nature.second_q.operators.symmetric_two_body import (
        SymmetricTwoBodyIntegrals,
        unfold,
    )

    if isinstance(tensor, SymmetricTwoBodyIntegrals):
        return _real_dense(unfold(tensor), name="two-body")
    # Ordinary Nature tensors use physicist ordering. SymmetricTwoBodyIntegrals
    # retain chemist ordering internally and carry a different label template.
    return _real_dense(tensor, name="two-body").transpose(0, 3, 1, 2).copy()


def fermionic_integrals(prepared: PreparedProblem) -> FermionicIntegrals:
    """Extract a supported spin-independent active Hamiltonian and reference.

    Nature may materialize equal beta blocks during active-space transformation;
    these are accepted. Distinct unrestricted blocks or spatial orbitals are
    rejected because the released ffsim ``MolecularHamiltonian`` shares both
    spin species' integrals.
    """
    problem = prepared.problem
    integrals = problem.hamiltonian.electronic_integrals
    norb = prepared.num_spatial_orbitals
    h1 = _real_dense(integrals.alpha["+-"], name="one-body")
    h2 = (
        _chemist_tensor(integrals.alpha["++--"])
        if "++--" in integrals.alpha
        else np.zeros((norb,) * 4)
    )
    if h1.shape != (norb, norb) or h2.shape != (norb,) * 4:
        raise ConfigError("qiskit fermionic integral dimensions do not match active orbitals")
    for block, key, expected in (
        (integrals.beta, "+-", h1),
        (integrals.beta, "++--", h2),
        (integrals.beta_alpha, "++--", h2),
    ):
        if key not in block:
            continue
        actual = (
            _chemist_tensor(block[key])
            if key == "++--"
            else _real_dense(block[key], name="beta one-body")
        )
        if actual.shape != expected.shape or not np.allclose(
            actual, expected, atol=1e-10, rtol=1e-10
        ):
            raise ConfigError(
                "qiskit fermionic algorithms currently require shared spatial orbitals "
                "and spin-independent integrals; unrestricted alpha/beta blocks differ"
            )
    angular_momentum = problem.properties.angular_momentum
    overlap = None if angular_momentum is None else angular_momentum.overlap
    if overlap is not None and not np.allclose(overlap, np.eye(norb), atol=1e-10, rtol=1e-10):
        raise ConfigError("qiskit fermionic algorithms do not support unrestricted orbital overlap")
    if not (
        np.allclose(h1, h1.T, atol=1e-10, rtol=1e-10)
        and np.allclose(h2, h2.swapaxes(0, 1), atol=1e-10, rtol=1e-10)
        and np.allclose(h2, h2.swapaxes(2, 3), atol=1e-10, rtol=1e-10)
        and np.allclose(h2, h2.transpose(2, 3, 0, 1), atol=1e-10, rtol=1e-10)
    ):
        raise ConfigError("qiskit fermionic integrals violate real chemist-order symmetry")
    occupied = []
    for values, count in zip(
        (problem.orbital_occupations, problem.orbital_occupations_b),
        prepared.num_particles,
        strict=True,
    ):
        array = np.asarray(values)
        if (
            array.shape != (norb,)
            or not np.all((array == 0) | (array == 1))
            or np.sum(array) != count
        ):
            raise ConfigError("qiskit fermionic algorithms require actual 0/1 active occupations")
        occupied.append(tuple(int(index) for index in np.flatnonzero(array)))
    h1.setflags(write=False)
    h2.setflags(write=False)
    return FermionicIntegrals(
        h1=h1,
        h2=h2,
        norb=norb,
        nelec=prepared.num_particles,
        occupations=(occupied[0], occupied[1]),
        multiplicity=prepared.multiplicity,
    )


def to_ffsim_hamiltonian(data: FermionicIntegrals) -> Any:
    """Build the released ffsim Hamiltonian with active electronic energy only."""
    import ffsim

    return ffsim.MolecularHamiltonian(
        one_body_tensor=data.h1.copy(), two_body_tensor=data.h2.copy(), constant=0.0
    )


def to_fermion_operator(data: FermionicIntegrals) -> Any:
    """Pack chemist integrals for Qiskit Fermions 0.1's spin-symmetric constructors."""
    from qiskit_fermions.operators import FermionOperator

    rows, columns = np.tril_indices(data.norb)
    pairs = data.h2[rows[:, None], columns[:, None], rows[None, :], columns[None, :]]
    packed_two_body = pairs[np.tril_indices(len(rows))]
    return FermionOperator.from_1body_tril_spin_sym(
        np.ascontiguousarray(data.h1[rows, columns]), data.norb
    ) + FermionOperator.from_2body_tril_spin_sym(np.ascontiguousarray(packed_two_body), data.norb)

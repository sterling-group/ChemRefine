"""Built-in initial-state factories."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from chemrefine.engines.qiskit.context import ElectronicStructureContext
from chemrefine.engines.qiskit.registry import INITIAL_STATES, NoComponentOptions
from chemrefine.errors import ConfigError


def reference_occupations(
    context: ElectronicStructureContext,
) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
    """Return validated alpha/beta occupations, preserving supplied orbital order.

    Problems without occupation metadata use the usual Aufbau prefix. Reference
    states and excitation builders share this convention so they agree on which
    orbitals are occupied, including after an explicit orbital permutation.
    """
    occupations = (
        getattr(context.problem, "orbital_occupations", None),
        getattr(context.problem, "orbital_occupations_b", None),
    )
    if all(occupation is None for occupation in occupations):
        alpha, beta = context.num_particles
        return (
            tuple(orbital < alpha for orbital in range(context.num_spatial_orbitals)),
            tuple(orbital < beta for orbital in range(context.num_spatial_orbitals)),
        )
    spin_occupations: list[tuple[bool, ...]] = []
    for occupation, particles in zip(occupations, context.num_particles, strict=True):
        values = np.asarray(occupation)
        if values.shape != (context.num_spatial_orbitals,) or not np.isin(values, [0, 1]).all():
            raise ConfigError(
                "qiskit Hartree-Fock requires binary alpha and beta orbital occupations "
                "matching the spatial-orbital count"
            )
        if int(values.sum()) != particles:
            raise ConfigError("qiskit Hartree-Fock occupations disagree with the particle count")
        spin_occupations.append(tuple(bool(value) for value in values))
    return spin_occupations[0], spin_occupations[1]


def _build_explicit_reference(
    context: ElectronicStructureContext, occupations: tuple[bool, ...]
) -> object:
    """Map an occupied creation product to its encoded computational-basis state."""
    from qiskit import QuantumCircuit
    from qiskit_nature.second_q.operators import FermionicOp

    creation_product = " ".join(
        f"+_{index}" for index, occupied in enumerate(occupations) if occupied
    )
    mapped = context.mapper.map(
        FermionicOp({creation_product: 1.0}, num_spin_orbitals=len(occupations))
    )
    if mapped is None or mapped.num_qubits != context.num_qubits:
        raise ConfigError("qiskit mapped Hartree-Fock determinant has an incompatible register")
    # For JW, parity (including particle reduction), and BK, every Pauli term
    # flips the same bits. Their Z phases do not affect the prepared determinant.
    patterns = np.asarray(mapped.paulis.x)
    if not len(patterns) or not np.all(patterns == patterns[0]):
        raise ConfigError(
            "qiskit mapper does not encode the Hartree-Fock determinant as one basis state"
        )
    circuit = QuantumCircuit(context.num_qubits)
    for index, occupied in enumerate(patterns[0]):
        if occupied:
            circuit.x(index)
    return circuit


@INITIAL_STATES.register("hartree_fock", NoComponentOptions)
def build_hartree_fock(*, options: BaseModel, context: ElectronicStructureContext) -> object:
    """Prepare the reference occupation dictated by the transformed problem."""
    del options
    occupations = reference_occupations(context)
    if any(
        occupied != (orbital < particles)
        for spin, particles in zip(occupations, context.num_particles, strict=True)
        for orbital, occupied in enumerate(spin)
    ):
        return _build_explicit_reference(context, occupations[0] + occupations[1])
    from qiskit_nature.second_q.circuit.library import HartreeFock

    return HartreeFock(
        context.num_spatial_orbitals,
        context.num_particles,
        context.mapper,
    )


@INITIAL_STATES.register("zero", NoComponentOptions)
def build_zero_state(*, options: BaseModel, context: ElectronicStructureContext) -> object:
    """Return an all-zero computational-basis state circuit."""
    del options
    from qiskit import QuantumCircuit

    return QuantumCircuit(context.num_qubits)

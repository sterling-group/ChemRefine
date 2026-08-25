"""Built-in initial-state factories."""

from __future__ import annotations

from pydantic import BaseModel

from chemrefine.engines.qiskit.context import ElectronicStructureContext
from chemrefine.engines.qiskit.registry import INITIAL_STATES, NoComponentOptions


@INITIAL_STATES.register("hartree_fock", NoComponentOptions)
def build_hartree_fock(*, options: BaseModel, context: ElectronicStructureContext) -> object:
    """Prepare the reference occupation dictated by the transformed problem."""
    del options
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

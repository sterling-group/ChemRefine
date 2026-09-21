"""Map a prepared electronic problem while retaining its chemistry and reduction facts."""

from __future__ import annotations

import logging

from chemrefine.engines.qiskit.context import ElectronicStructureContext
from chemrefine.engines.qiskit.options import ComponentSelection
from chemrefine.engines.qiskit.problem import PreparedProblem
from chemrefine.engines.qiskit.registry import MAPPERS

logger = logging.getLogger(__name__)


def map_problem(
    prepared: PreparedProblem, mapper: ComponentSelection | str = "jordan_wigner"
) -> ElectronicStructureContext:
    """Expose both Hamiltonians and the before/after qubit counts for one mapping.

    The initial register has one qubit per active spin orbital. The mapped
    register may be smaller (for example, parity's particle-aware reduction).
    Additional mapper factories can return their own reduced implementation;
    no symmetry detection or tapering is inferred here.
    """
    selection = ComponentSelection.named(mapper) if isinstance(mapper, str) else mapper
    implementation = MAPPERS.build(selection, problem=prepared.problem)
    hamiltonian = implementation.map(prepared.fermionic_hamiltonian).simplify()
    qubits = int(hamiltonian.num_qubits)
    mapping_metadata = {
        "name": selection.name,
        "options": MAPPERS.options_for(selection).model_dump(mode="json"),
        "num_qubits_before_reduction": prepared.num_spin_orbitals,
        "num_qubits_after_reduction": qubits,
        "symmetry_tapering": None,
    }
    logger.info(
        "Qiskit mapper %s: %d qubits, %d Pauli terms", selection.name, qubits, len(hamiltonian)
    )
    return ElectronicStructureContext(
        problem=prepared.problem,
        mapper=implementation,
        qubit_hamiltonian=hamiltonian,
        num_spatial_orbitals=prepared.num_spatial_orbitals,
        num_particles=prepared.num_particles,
        num_qubits=qubits,
        multiplicity=prepared.multiplicity,
        fermionic_hamiltonian=prepared.fermionic_hamiltonian,
        num_qubits_before_reduction=prepared.num_spin_orbitals,
        num_pauli_terms=len(hamiltonian),
        mapping_metadata=mapping_metadata,
        active_space_metadata={
            "original_num_spatial_orbitals": prepared.original_num_spatial_orbitals,
            "active_orbitals": list(prepared.active_orbitals),
            "num_particles": list(prepared.num_particles),
            "energy_offsets": prepared.energy_offsets,
            "transformations": prepared.metadata.get("transformations", []),
        },
        provenance={**prepared.provenance, "problem_metadata": prepared.metadata},
    )

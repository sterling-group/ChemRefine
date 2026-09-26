"""Built-in fixed-circuit and adaptive operator-pool ansatz factories."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from chemrefine.engines.qiskit.components.initial_states import reference_occupations
from chemrefine.engines.qiskit.context import AnsatzArtifacts, ElectronicStructureContext
from chemrefine.engines.qiskit.operators import Excitation, ucc_pool_metadata
from chemrefine.engines.qiskit.registry import ANSATZE
from chemrefine.errors import ConfigError


class UCCSDOptions(BaseModel):
    """Configuration exposed by Qiskit Nature's UCCSD circuit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reps: int = Field(1, ge=1)
    generalized: bool = False
    preserve_spin: bool = True
    include_imaginary: bool = False


class UCCOptions(UCCSDOptions):
    """Generic UCC using a caller-supplied list of spin-orbital excitations."""

    excitations: tuple[Excitation, ...] = Field(min_length=1)

    @field_validator("excitations")
    @classmethod
    def _validate_excitations(cls, value: tuple[Excitation, ...]) -> tuple[Excitation, ...]:
        """Require distinct, non-negative occupied/unoccupied indices of equal rank."""
        for occupied, unoccupied in value:
            if not occupied or len(occupied) != len(unoccupied):
                raise ValueError(
                    "each excitation must have equal non-empty occupied/unoccupied lists"
                )
            indices = occupied + unoccupied
            if any(index < 0 for index in indices) or len(set(indices)) != len(indices):
                raise ValueError(
                    "excitation spin-orbital indices must be distinct and non-negative"
                )
        if len(set(value)) != len(value):
            raise ValueError("excitations must not contain duplicates")
        return value


class EfficientSU2Options(BaseModel):
    """Configuration for the hardware-efficient EfficientSU2 circuit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reps: int = Field(2, ge=1)
    entanglement: str = "reverse_linear"
    su2_gates: list[str] = Field(default_factory=lambda: ["ry", "rz"])
    skip_final_rotation_layer: bool = False
    flatten: bool = True


def _reference_excitation_permutation(
    context: ElectronicStructureContext, initial_state: object | None = None
) -> tuple[int, ...]:
    """Map canonical occupied/virtual blocks onto the supplied reference determinant."""
    permutation: list[int] = []
    for spin, occupations in enumerate(reference_occupations(context, initial_state)):
        offset = spin * context.num_spatial_orbitals
        permutation.extend(offset + index for index, occupied in enumerate(occupations) if occupied)
        permutation.extend(
            offset + index for index, occupied in enumerate(occupations) if not occupied
        )
    return tuple(permutation)


@ANSATZE.register(
    "uccsd",
    UCCSDOptions,
    capabilities=frozenset({"circuit", "operator_pool"}),
)
def build_uccsd(
    *,
    options: UCCSDOptions,
    context: ElectronicStructureContext,
    initial_state: object,
) -> AnsatzArtifacts:
    """Build UCCSD both as a fixed VQE circuit and as an ADAPT operator pool."""
    from qiskit_nature.second_q.circuit.library import UCCSD

    circuit = UCCSD(
        context.num_spatial_orbitals,
        context.num_particles,
        context.mapper,
        reps=options.reps,
        initial_state=initial_state,
        generalized=options.generalized,
        preserve_spin=options.preserve_spin,
        include_imaginary=options.include_imaginary,
    )
    permutation = _reference_excitation_permutation(context, initial_state)
    if not options.generalized and permutation != tuple(range(2 * context.num_spatial_orbitals)):
        from qiskit_nature.second_q.circuit.library.ansatzes.utils import (
            generate_fermionic_excitations,
        )

        def reference_excitations(
            num_spatial_orbitals: int, num_particles: tuple[int, int]
        ) -> list[Excitation]:
            """Preserve the full singles/doubles pool around the actual occupied orbitals."""
            excitations = [
                excitation
                for rank in (1, 2)
                for excitation in generate_fermionic_excitations(
                    rank,
                    num_spatial_orbitals,
                    num_particles,
                    preserve_spin=options.preserve_spin,
                )
            ]
            return [
                (
                    tuple(permutation[index] for index in occupied),
                    tuple(permutation[index] for index in unoccupied),
                )
                for occupied, unoccupied in excitations
            ]

        # Nature normally assumes prefix occupations. Its public setter clears
        # cached operators before rebuilding against this explicit determinant.
        circuit.excitations = reference_excitations
    # UCCSD already maps and stores its excitation generators. Re-generating the
    # fermionic excitations here mutates Nature 0.8's internal excitation list when
    # ``include_imaginary`` is enabled, leaving its list and parameter counts unequal.
    operator_pool = tuple(circuit.operators)
    return AnsatzArtifacts(
        circuit=circuit,
        operator_pool=operator_pool,
        pool_metadata=ucc_pool_metadata(circuit, include_imaginary=options.include_imaginary),
    )


@ANSATZE.register("ucc", UCCOptions, capabilities=frozenset({"circuit", "operator_pool"}))
def build_ucc(
    *,
    options: UCCOptions,
    context: ElectronicStructureContext,
    initial_state: object,
) -> AnsatzArtifacts:
    """Build a chemistry circuit and pool using only the supplied excitations.

    Spin orbitals follow Nature's block ordering: all alpha spatial orbitals,
    followed by all beta spatial orbitals. No excitation selection is performed.
    """
    from qiskit_nature.second_q.circuit.library import UCC

    orbitals = context.num_spatial_orbitals
    for occupied, unoccupied in options.excitations:
        if any(index >= 2 * orbitals for index in occupied + unoccupied):
            raise ConfigError(f"qiskit UCC excitation indices must be below {2 * orbitals}")
        if options.preserve_spin and sum(index < orbitals for index in occupied) != sum(
            index < orbitals for index in unoccupied
        ):
            raise ConfigError(
                "qiskit UCC excitations must preserve spin when preserve_spin is true"
            )

    def supplied_excitations(
        num_spatial_orbitals: int, num_particles: tuple[int, int]
    ) -> list[Excitation]:
        """Adapt explicit excitations to Nature's generator callable contract."""
        del num_spatial_orbitals, num_particles
        return list(options.excitations)

    circuit = UCC(
        num_spatial_orbitals=orbitals,
        num_particles=context.num_particles,
        excitations=supplied_excitations,
        qubit_mapper=context.mapper,
        reps=options.reps,
        initial_state=initial_state,
        generalized=options.generalized,
        preserve_spin=options.preserve_spin,
        include_imaginary=options.include_imaginary,
    )
    return AnsatzArtifacts(
        circuit=circuit,
        operator_pool=tuple(circuit.operators),
        pool_metadata=ucc_pool_metadata(circuit, include_imaginary=options.include_imaginary),
    )


@ANSATZE.register(
    "efficient_su2",
    EfficientSU2Options,
    capabilities=frozenset({"circuit"}),
)
def build_efficient_su2(
    *,
    options: EfficientSU2Options,
    context: ElectronicStructureContext,
    initial_state: object,
) -> AnsatzArtifacts:
    """Build EfficientSU2 with the functional API, retaining flat or nested representation."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import efficient_su2

    layers = efficient_su2(
        context.num_qubits,
        su2_gates=options.su2_gates,
        entanglement=options.entanglement,
        reps=options.reps,
        skip_final_rotation_layer=options.skip_final_rotation_layer,
    )
    flat = QuantumCircuit(context.num_qubits, name="EfficientSU2")
    if initial_state is not None:
        flat.compose(initial_state, inplace=True)
    flat.compose(layers, inplace=True)
    if options.flatten:
        return AnsatzArtifacts(circuit=flat)
    nested = QuantumCircuit(context.num_qubits)
    nested.append(flat.to_gate(), range(context.num_qubits))
    return AnsatzArtifacts(circuit=nested)

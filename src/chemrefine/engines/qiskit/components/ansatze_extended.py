"""Rank-selected UCC and additional fixed-circuit variational ansatze."""

from __future__ import annotations

from functools import partial
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from chemrefine.engines.qiskit.components.ansatze import (
    UCCSDOptions,
    reference_excitation_permutation,
    reference_excitations,
)
from chemrefine.engines.qiskit.context import AnsatzArtifacts, ElectronicStructureContext
from chemrefine.engines.qiskit.operators import ucc_pool_metadata
from chemrefine.engines.qiskit.registry import ANSATZE
from chemrefine.errors import ConfigError

Entanglement = Literal["full", "linear", "reverse_linear", "circular"]


class UCCRanksOptions(UCCSDOptions):
    """Choose excitation ranks explicitly, such as singles, doubles or triples."""

    ranks: tuple[StrictInt, ...] = Field(default=(1, 2), min_length=1)

    @field_validator("ranks")
    @classmethod
    def _distinct_positive_ranks(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """Keep rank ordering explicit and reject duplicates or nonpositive ranks."""
        if any(rank < 1 for rank in value) or len(set(value)) != len(value):
            raise ValueError("excitation ranks must be distinct positive integers")
        return value


class RealAmplitudesOptions(BaseModel):
    """RY/CX layers; these generally conserve neither electron number nor spin."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reps: int = Field(2, ge=1, strict=True)
    entanglement: Entanglement = "reverse_linear"
    skip_final_rotation_layer: bool = False


class ExcitationPreservingOptions(BaseModel):
    """Number-conserving layers for Jordan-Wigner occupation qubits.

    Separate spin blocks preserve alpha/beta populations by default, with one
    controlled phase between matching alpha/beta orbitals after each layer to
    correlate the spins. Disabling ``preserve_spin`` uses a single block, allowing
    spin transfer while preserving electron number. Neither mode guarantees S².
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    reps: int = Field(2, ge=1, strict=True)
    mode: Literal["iswap", "fsim"] = "iswap"
    entanglement: Entanglement = "linear"
    preserve_spin: bool = True
    skip_final_rotation_layer: bool = False


@ANSATZE.register(
    "ucc_ranks", UCCRanksOptions, capabilities=frozenset({"circuit", "operator_pool"})
)
def build_ucc_ranks(
    *, options: UCCRanksOptions, context: ElectronicStructureContext, initial_state: object
) -> AnsatzArtifacts:
    """Build Nature UCC from requested ranks around the actual reference determinant."""
    if any(rank > context.num_spatial_orbitals for rank in options.ranks):
        raise ConfigError("qiskit excitation rank cannot exceed the spatial-orbital count")
    from qiskit_nature.second_q.circuit.library import UCC

    circuit = UCC(
        num_spatial_orbitals=context.num_spatial_orbitals,
        num_particles=context.num_particles,
        excitations=partial(
            reference_excitations,
            ranks=options.ranks,
            permutation=reference_excitation_permutation(context, initial_state),
            generalized=options.generalized,
            preserve_spin=options.preserve_spin,
        ),
        qubit_mapper=context.mapper,
        reps=options.reps,
        initial_state=initial_state,
        generalized=options.generalized,
        preserve_spin=options.preserve_spin,
        include_imaginary=options.include_imaginary,
    )
    pool = tuple(circuit.operators)
    if not pool:
        raise ConfigError("qiskit requested UCC ranks produce an empty excitation pool")
    return AnsatzArtifacts(
        circuit=circuit,
        operator_pool=pool,
        pool_metadata=ucc_pool_metadata(circuit, include_imaginary=options.include_imaginary),
    )


@ANSATZE.register("real_amplitudes", RealAmplitudesOptions, capabilities=frozenset({"circuit"}))
def build_real_amplitudes(
    *, options: RealAmplitudesOptions, context: ElectronicStructureContext, initial_state: object
) -> AnsatzArtifacts:
    """Build a real-valued hardware-efficient RY/CX circuit with explicit layer controls."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import real_amplitudes

    circuit = QuantumCircuit(context.num_qubits)
    circuit.compose(initial_state, inplace=True)
    circuit.compose(real_amplitudes(context.num_qubits, **options.model_dump()), inplace=True)
    return AnsatzArtifacts(circuit=circuit)


@ANSATZE.register(
    "excitation_preserving", ExcitationPreservingOptions, capabilities=frozenset({"circuit"})
)
def build_excitation_preserving(
    *,
    options: ExcitationPreservingOptions,
    context: ElectronicStructureContext,
    initial_state: object,
) -> AnsatzArtifacts:
    """Build total- or spin-resolved number-conserving layers on an unreduced JW register."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import excitation_preserving
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    if not isinstance(context.mapper, JordanWignerMapper) or context.num_qubits != (
        2 * context.num_spatial_orbitals
    ):
        raise ConfigError("qiskit excitation_preserving requires unreduced Jordan-Wigner mapping")
    circuit = QuantumCircuit(context.num_qubits)
    circuit.compose(initial_state, inplace=True)
    width = context.num_spatial_orbitals if options.preserve_spin else context.num_qubits
    for repetition in range(options.reps):
        for offset in range(0, context.num_qubits, width):
            layer = excitation_preserving(
                width,
                mode=options.mode,
                entanglement=options.entanglement,
                reps=1,
                skip_final_rotation_layer=options.skip_final_rotation_layer,
                parameter_prefix=f"theta_{repetition}_{offset}",
            )
            circuit.compose(layer, qubits=range(offset, offset + width), inplace=True)
        if options.preserve_spin:
            for orbital in range(width):
                circuit.cp(Parameter(f"phi_{repetition}_{orbital}"), orbital, orbital + width)
    return AnsatzArtifacts(circuit=circuit)

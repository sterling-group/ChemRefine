"""Built-in fixed-circuit and adaptive operator-pool ansatz factories."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from chemrefine.engines.qiskit.context import AnsatzArtifacts, ElectronicStructureContext
from chemrefine.engines.qiskit.registry import ANSATZE


class UCCSDOptions(BaseModel):
    """Configuration exposed by Qiskit Nature's UCCSD circuit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reps: int = Field(1, ge=1)
    generalized: bool = False
    preserve_spin: bool = True
    include_imaginary: bool = False


class EfficientSU2Options(BaseModel):
    """Configuration for the hardware-efficient EfficientSU2 circuit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reps: int = Field(2, ge=1)
    entanglement: str = "reverse_linear"
    su2_gates: list[str] = Field(default_factory=lambda: ["ry", "rz"])
    skip_final_rotation_layer: bool = False
    flatten: bool = True


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
    # UCCSD already maps and stores its excitation generators. Re-generating the
    # fermionic excitations here mutates Nature 0.8's internal excitation list when
    # ``include_imaginary`` is enabled, leaving its list and parameter counts unequal.
    operator_pool = tuple(circuit.operators)
    return AnsatzArtifacts(circuit=circuit, operator_pool=operator_pool)


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
    """Build a hardware-efficient circuit for ordinary VQE."""
    from qiskit.circuit.library import EfficientSU2

    circuit = EfficientSU2(
        context.num_qubits,
        su2_gates=options.su2_gates,
        entanglement=options.entanglement,
        reps=options.reps,
        skip_final_rotation_layer=options.skip_final_rotation_layer,
        initial_state=initial_state,
        flatten=options.flatten,
    )
    return AnsatzArtifacts(circuit=circuit)

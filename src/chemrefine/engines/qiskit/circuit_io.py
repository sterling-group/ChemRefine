"""Portable bound preparation circuits with explicit molecular interpretation."""

from __future__ import annotations

import io
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from struct import error as StructError
from typing import Any, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from chemrefine.engines.qiskit.bundles import (
    QuantumBundle,
    bundle_dependencies,
    read_bundle,
    write_bundle,
)
from chemrefine.engines.qiskit.context import ElectronicStructureContext
from chemrefine.errors import ConfigError, OutputParseError


class CircuitDescription(BaseModel):
    """Scientific interpretation of one logical, bound state-preparation circuit."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    version: Literal[1] = 1
    representation: Literal["logical"] = "logical"
    orbital_order: Literal["alpha_then_beta"] = "alpha_then_beta"
    bit_order: Literal["qubit_0_right"] = "qubit_0_right"
    units: Literal["hartree"] = "hartree"
    root: StrictInt = Field(ge=0)
    num_qubits: StrictInt = Field(ge=1)
    num_spin_orbitals: StrictInt = Field(ge=1)
    num_particles: tuple[StrictInt, StrictInt]
    mapping: dict[str, Any]
    active_space: dict[str, Any]
    parameter_order: tuple[str, ...]
    parameter_values: tuple[float, ...]
    energy_offsets: dict[str, float]
    active_hamiltonian: dict[str, float]
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _dimensions(self) -> Self:
        """Require coherent bindings, a declared mapper and register-sized operators."""
        if len(self.parameter_order) != len(self.parameter_values):
            raise ValueError("circuit parameter names and values differ in length")
        if not self.mapping or not self.active_hamiltonian:
            raise ValueError("circuit bundle requires mapping and active Hamiltonian metadata")
        if any(
            len(label) != self.num_qubits or set(label) - set("IXYZ")
            for label in self.active_hamiltonian
        ):
            raise ValueError("circuit Hamiltonian labels must match the logical register")
        if self.num_spin_orbitals % 2 or any(
            count < 0 or count > self.num_spin_orbitals // 2 for count in self.num_particles
        ):
            raise ValueError("circuit particle populations do not fit its spin orbitals")
        return self


@dataclass(frozen=True)
class BoundCircuit:
    """Transient SDK circuit plus owned metadata, excluded from scalar result JSON."""

    circuit: Any
    description: CircuitDescription


def bound_circuit(
    context: ElectronicStructureContext,
    circuit: Any,
    parameters: Mapping[Any, float] | Iterable[float],
    *,
    root: int = 0,
) -> BoundCircuit:
    """Bind in the original logical parameter order and retain mapping/offset facts."""
    order = tuple(circuit.parameters)
    values = (
        tuple(float(parameters[parameter]) for parameter in order)
        if isinstance(parameters, Mapping)
        else tuple(float(value) for value in parameters)
    )
    description = CircuitDescription(
        root=root,
        num_qubits=context.num_qubits,
        num_spin_orbitals=2 * context.num_spatial_orbitals,
        num_particles=context.num_particles,
        mapping=context.mapping_metadata,
        active_space=context.active_space_metadata,
        parameter_order=tuple(str(parameter) for parameter in order),
        parameter_values=values,
        energy_offsets={
            name: float(value) for name, value in context.problem.hamiltonian.constants.items()
        },
        active_hamiltonian={
            label: float(np.real(coefficient))
            for label, coefficient in context.qubit_hamiltonian.simplify().to_list()
        },
        provenance=context.provenance,
    )
    if circuit.num_qubits != description.num_qubits or circuit.num_clbits:
        raise ConfigError("circuit export requires a logical preparation without measurements")
    if any(abs(complex(value).imag) > 1e-10 for _, value in context.qubit_hamiltonian.to_list()):
        raise ConfigError("circuit export requires a Hermitian Pauli Hamiltonian")
    bound = circuit.assign_parameters(values)
    bound.metadata = {}
    return BoundCircuit(bound, description)


class _BoundedStream(io.BytesIO):
    """Stop QPY serialization before its configured byte budget is exceeded."""

    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = limit

    def write(self, data: Any) -> int:
        """Check each provider write without allocating a second large payload."""
        if self.tell() + len(data) > self.limit:
            raise ConfigError("circuit export exceeds max_bytes")
        return super().write(data)


def save_circuit(path: Path, value: BoundCircuit, *, max_bytes: int = 33554432) -> Path:
    """Publish QPY bytes inside a hashed NPZ bundle, preserving its logical conventions."""
    from qiskit import qpy

    if (
        value.circuit.num_parameters
        or value.circuit.num_clbits
        or value.circuit.num_qubits != value.description.num_qubits
    ):
        raise ConfigError("circuit export requires one bound, unmeasured logical preparation")
    stream = _BoundedStream(max_bytes)
    qpy.dump(value.circuit, stream, version=13)
    return write_bundle(
        path,
        kind="bound_circuit",
        arrays={"qpy": np.frombuffer(stream.getvalue(), dtype=np.uint8)},
        metadata=value.description.model_dump(mode="json"),
        max_bytes=max_bytes,
    )


def validate_circuit_bundle(bundle: QuantumBundle) -> CircuitDescription:
    """Check a circuit artifact locally without importing Qiskit or contacting a provider."""
    try:
        description = CircuitDescription.model_validate(bundle.metadata)
        if bundle.description.kind != "bound_circuit" or set(bundle.arrays) != {"qpy"}:
            raise ValueError("expected a bound_circuit artifact with one QPY payload")
        payload = bundle.arrays["qpy"]
        if payload.dtype != np.dtype("uint8") or payload.ndim != 1 or payload.size < 7:
            raise ValueError("circuit QPY payload must be a nonempty byte array")
        if bytes(payload[:6]) != b"QISKIT":
            raise ValueError("circuit payload has no QPY header")
        return description
    except (ValueError, TypeError) as exc:
        raise OutputParseError(f"invalid bound circuit bundle: {exc}") from exc


def load_circuit(path: Path, *, max_bytes: int = 33554432) -> BoundCircuit:
    """Deserialize one validated bound preparation inside a selected worker environment."""
    from qiskit import qpy
    from qiskit.exceptions import QiskitError

    bundle = read_bundle(path, max_bytes=max_bytes)
    description = validate_circuit_bundle(bundle)
    try:
        circuits = qpy.load(io.BytesIO(bundle.arrays["qpy"].tobytes()))
        if len(circuits) != 1:
            raise ValueError("expected exactly one bound circuit")
        circuit = circuits[0]
        if (
            circuit.num_parameters
            or circuit.num_clbits
            or circuit.num_qubits != description.num_qubits
        ):
            raise ValueError("QPY contents differ from the bound logical circuit description")
        circuit.metadata = {"chemrefine_preparation": description.model_dump(mode="json")}
        return BoundCircuit(circuit, description)
    except (QiskitError, ValueError, TypeError, EOFError, StructError) as exc:
        raise OutputParseError(f"invalid circuit QPY in {path}: {exc}") from exc


def circuit_input_dependencies(path: Path) -> dict[str, Path]:
    """Discover a declared QPY-or-bundle input's payload without importing an SDK."""
    try:
        with path.open("rb") as stream:
            if stream.read(6) == b"QISKIT":
                return {}
        return bundle_dependencies(path)
    except OSError as exc:
        raise ConfigError(f"cannot read circuit input {path}: {exc}") from exc

"""Portable bound preparation circuits with explicit molecular interpretation."""

from __future__ import annotations

import io
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from struct import error as StructError
from typing import Any

import numpy as np

from chemrefine.engines.qiskit.bundles import (
    QuantumBundle,
    bundle_dependencies,
    read_bundle,
    write_bundle,
)
from chemrefine.engines.qiskit.circuit_storage import (
    CircuitDescription as CircuitDescription,
)
from chemrefine.engines.qiskit.circuit_storage import (
    decode_circuit_data,
    encode_circuit_data,
    preflight_storage,
)
from chemrefine.engines.qiskit.context import ElectronicStructureContext
from chemrefine.engines.qiskit.qpy_validation import validate_qpy_payload
from chemrefine.errors import ConfigError, OutputParseError


@dataclass(frozen=True)
class BoundCircuit:
    """Transient SDK circuit plus owned metadata, excluded from scalar result JSON."""

    circuit: Any
    description: CircuitDescription


def _description(
    context: ElectronicStructureContext,
    order: tuple[str, ...],
    values: tuple[float, ...],
    *,
    root: int,
) -> CircuitDescription:
    """Collect the same mapped scientific interpretation before and after solving."""
    if any(abs(complex(value).imag) > 1e-10 for _, value in context.qubit_hamiltonian.to_list()):
        raise ConfigError("circuit export requires a Hermitian Pauli Hamiltonian")
    return CircuitDescription(
        root=root,
        num_qubits=context.num_qubits,
        num_spin_orbitals=2 * context.num_spatial_orbitals,
        num_particles=context.num_particles,
        mapping=context.mapping_metadata,
        active_space=context.active_space_metadata,
        parameter_order=order,
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


def preflight_circuit_export(
    context: ElectronicStructureContext, *, max_bytes: int, circuit: Any | None = None
) -> None:
    """Reject predictable export-budget failures before acquiring provider resources.

    Fixed parameter storage and metadata are known before solving. Adaptive growth
    and final serialized QPY bytes remain subject to the bounded writer.
    """
    order = () if circuit is None else tuple(str(parameter) for parameter in circuit.parameters)
    description = _description(context, order, (0.0,) * len(order), root=0)
    preflight_storage(description, max_bytes=max_bytes)


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
    description = _description(
        context, tuple(str(parameter) for parameter in order), values, root=root
    )
    if circuit.num_qubits != description.num_qubits or circuit.num_clbits:
        raise ConfigError("circuit export requires a logical preparation without measurements")
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
    remaining = preflight_storage(value.description, max_bytes=max_bytes, path=path)
    metadata, arrays = encode_circuit_data(value.description)
    stream = _BoundedStream(remaining)
    qpy.dump(value.circuit, stream, version=13)
    return write_bundle(
        path,
        kind="bound_circuit",
        arrays=arrays | {"qpy": np.frombuffer(stream.getvalue(), dtype=np.uint8)},
        metadata=metadata,
        max_bytes=max_bytes,
    )


def validate_circuit_bundle(bundle: QuantumBundle) -> CircuitDescription:
    """Check a circuit artifact locally without importing Qiskit or contacting a provider."""
    try:
        description = decode_circuit_data(bundle)
        if bundle.description.kind != "bound_circuit":
            raise ValueError("expected a bound_circuit artifact with one QPY payload")
        validate_qpy_payload(
            bundle.arrays["qpy"],
            circuits=1,
            num_qubits=description.num_qubits,
            num_clbits=0,
        )
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

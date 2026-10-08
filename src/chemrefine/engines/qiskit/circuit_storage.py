"""Versioned numeric storage for portable molecular circuit interpretations."""

from __future__ import annotations

import math
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from chemrefine.engines.qiskit.bundles import (
    ArrayDescription,
    BundleDescription,
    QuantumBundle,
    encode_bundle_descriptor,
)
from chemrefine.errors import ConfigError


class _CircuitSpace(BaseModel):
    """Scientific interpretation of one logical, bound state-preparation circuit."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
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
    energy_offsets: dict[str, float]
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _space(self) -> Self:
        """Require a mapper and physically possible spin populations."""
        if not self.mapping:
            raise ValueError("circuit bundle requires mapping and active Hamiltonian metadata")
        if self.num_spin_orbitals % 2 or any(
            count < 0 or count > self.num_spin_orbitals // 2 for count in self.num_particles
        ):
            raise ValueError("circuit particle populations do not fit its spin orbitals")
        return self


class CircuitDescription(_CircuitSpace):
    """Scientific interpretation, shared by legacy and numeric circuit bundles."""

    version: Literal[1] = 1
    parameter_order: tuple[str, ...]
    parameter_values: tuple[float, ...]
    active_hamiltonian: dict[str, float]

    @model_validator(mode="after")
    def _dimensions(self) -> Self:
        """Require coherent bindings and register-sized nonempty Pauli operators."""
        if len(self.parameter_order) != len(self.parameter_values):
            raise ValueError("circuit parameter names and values differ in length")
        if not self.active_hamiltonian:
            raise ValueError("circuit bundle requires mapping and active Hamiltonian metadata")
        if any(
            len(label) != self.num_qubits or set(label) - set("IXYZ")
            for label in self.active_hamiltonian
        ):
            raise ValueError("circuit Hamiltonian labels must match the logical register")
        return self


class _StoredCircuit(_CircuitSpace):
    """Version two references with explicit text and Pauli-array conventions."""

    version: Literal[2] = 2
    pauli_encoding: Literal["ascii_IXYZ_qubit_0_right"] = "ascii_IXYZ_qubit_0_right"
    parameter_encoding: Literal["utf8_offsets"] = "utf8_offsets"
    pauli_labels: Literal["pauli_labels"] = "pauli_labels"
    pauli_coefficients: Literal["pauli_coefficients"] = "pauli_coefficients"
    parameter_names: Literal["parameter_names"] = "parameter_names"
    parameter_offsets: Literal["parameter_offsets"] = "parameter_offsets"
    parameter_values: Literal["parameter_values"] = "parameter_values"


def _compact(description: CircuitDescription) -> _StoredCircuit:
    """Retain common interpretation while moving variable-size data to arrays."""
    return _StoredCircuit.model_validate(
        description.model_dump(
            mode="json",
            exclude={"version", "active_hamiltonian", "parameter_order", "parameter_values"},
        )
    )


def storage_specs(description: CircuitDescription) -> dict[str, ArrayDescription]:
    """Describe array allocations without materializing the Hamiltonian payload."""
    terms = len(description.active_hamiltonian)
    parameters = len(description.parameter_order)
    name_bytes = sum(len(name.encode("utf-8")) for name in description.parameter_order)
    return {
        "pauli_labels": ArrayDescription(shape=(terms, description.num_qubits), dtype="|u1"),
        "pauli_coefficients": ArrayDescription(shape=(terms,), dtype="<f8"),
        "parameter_names": ArrayDescription(shape=(name_bytes,), dtype="|u1"),
        "parameter_offsets": ArrayDescription(shape=(parameters + 1,), dtype="<u8"),
        "parameter_values": ArrayDescription(shape=(parameters,), dtype="<f8"),
    }


def preflight_storage(
    description: CircuitDescription, *, max_bytes: int, path: Path | None = None
) -> int:
    """Check known metadata and numeric size, returning remaining QPY capacity.

    The common pre-solve check reserves the worst-case JSON escaping of a
    255-byte payload filename; the writer checks its actual name. Final adaptive
    circuit size remains a runtime guard.
    """
    specs = storage_specs(description)
    used = sum(math.prod(spec.shape) * np.dtype(spec.dtype).itemsize for spec in specs.values())
    remaining = max_bytes - used
    if remaining < 1:
        raise ConfigError("circuit export exceeds max_bytes before QPY serialization")
    payload = "\uffff" * 255 if path is None else f"{path.stem}.{'0' * 32}.npz"
    encode_bundle_descriptor(
        BundleDescription(
            kind="bound_circuit",
            payload=payload,
            sha256="0" * 64,
            arrays=specs | {"qpy": ArrayDescription(shape=(remaining,), dtype="|u1")},
            metadata=_compact(description).model_dump(mode="json"),
        )
    )
    return remaining


def encode_circuit_data(
    description: CircuitDescription,
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    """Encode large interpretation fields as portable, pickle-free numeric arrays."""
    names = [name.encode("utf-8") for name in description.parameter_order]
    offsets = np.zeros(len(names) + 1, dtype="<u8")
    offsets[1:] = np.cumsum([len(name) for name in names], dtype="<u8")
    labels = "".join(description.active_hamiltonian).encode("ascii")
    return _compact(description).model_dump(mode="json"), {
        "pauli_labels": np.frombuffer(labels, dtype=np.uint8).reshape(-1, description.num_qubits),
        "pauli_coefficients": np.asarray(
            list(description.active_hamiltonian.values()), dtype="<f8"
        ),
        "parameter_names": np.frombuffer(b"".join(names), dtype=np.uint8),
        "parameter_offsets": offsets,
        "parameter_values": np.asarray(description.parameter_values, dtype="<f8"),
    }


def decode_circuit_data(bundle: QuantumBundle) -> CircuitDescription:
    """Read legacy JSON or v2 arrays into the stable public interpretation object."""
    metadata = bundle.metadata
    if metadata.get("version", 1) == 1:
        if set(bundle.arrays) != {"qpy"}:
            raise ValueError("legacy circuit bundle requires one QPY payload")
        return CircuitDescription.model_validate(metadata)
    stored = _StoredCircuit.model_validate(metadata)
    expected = {
        "qpy",
        "pauli_labels",
        "pauli_coefficients",
        "parameter_names",
        "parameter_offsets",
        "parameter_values",
    }
    if set(bundle.arrays) != expected:
        raise ValueError("numeric circuit bundle has missing or unknown arrays")
    labels = bundle.arrays[stored.pauli_labels]
    coefficients = bundle.arrays[stored.pauli_coefficients]
    names = bundle.arrays[stored.parameter_names]
    offsets = bundle.arrays[stored.parameter_offsets]
    values = bundle.arrays[stored.parameter_values]
    for array, dtype, rank in (
        (labels, "uint8", 2),
        (coefficients, "<f8", 1),
        (names, "uint8", 1),
        (offsets, "<u8", 1),
        (values, "<f8", 1),
    ):
        if array.dtype != np.dtype(dtype) or array.ndim != rank:
            raise ValueError("invalid numeric circuit array dtype or rank")
    if (
        labels.shape != (len(coefficients), stored.num_qubits)
        or not len(coefficients)
        or not np.isin(labels, [73, 88, 89, 90]).all()
    ):
        raise ValueError("invalid numeric circuit Pauli labels")
    if (
        len(offsets) != len(values) + 1
        or offsets[0] != 0
        or offsets[-1] != len(names)
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise ValueError("invalid circuit parameter offsets")
    hamiltonian = {
        row.tobytes().decode("ascii"): float(coefficient)
        for row, coefficient in zip(labels, coefficients, strict=True)
    }
    if len(hamiltonian) != len(coefficients):
        raise ValueError("duplicate numeric circuit Pauli labels")
    order = tuple(
        names[int(start) : int(end)].tobytes().decode("utf-8") for start, end in pairwise(offsets)
    )
    common = {name: getattr(stored, name) for name in _CircuitSpace.model_fields}
    return CircuitDescription(
        **common,
        parameter_order=order,
        parameter_values=tuple(values),
        active_hamiltonian=hamiltonian,
    )

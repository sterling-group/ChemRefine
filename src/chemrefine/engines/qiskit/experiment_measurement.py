"""YAML adapter for circuit-based Pauli measurements and their native reports."""

from __future__ import annotations

from pathlib import Path
from struct import error as StructError
from typing import Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from chemrefine.engines.qiskit.experiment import EXPERIMENTS, ExperimentResult
from chemrefine.engines.qiskit.measurement import MeasurementOptions, measure_observable
from chemrefine.engines.qiskit.options import ComponentSelection
from chemrefine.errors import ConfigError


def read_circuit(path: Path, *, max_bytes: int = 33554432) -> Any:
    """Read exactly one QPY circuit, with a bound on input bytes."""
    from qiskit import qpy
    from qiskit.exceptions import QiskitError

    try:
        if path.stat().st_size > max_bytes:
            raise ValueError("QPY circuit exceeds its input byte limit")
        with path.open("rb") as stream:
            circuits = qpy.load(stream)
        if len(circuits) != 1:
            raise ValueError("circuit input must contain exactly one QPY circuit")
    except (OSError, ValueError, EOFError, StructError, QiskitError) as exc:
        raise ConfigError(f"cannot load circuit {path}: {exc}") from exc
    return circuits[0]


class PauliCircuitInput(BaseModel):
    """A bounded QPY circuit and SDK-free Hermitian Pauli input validation."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    circuit_path: str = Field(min_length=1, json_schema_extra={"input_file": True})
    max_circuit_bytes: int = Field(33554432, ge=1)
    observable: dict[str, float] = Field(min_length=1)

    @model_validator(mode="after")
    def _observable_labels(self) -> Self:
        """Validate Pauli vocabulary and width before optional SDK loading."""
        widths = {len(label) for label in self.observable}
        if (
            len(widths) != 1
            or 0 in widths
            or any(set(label) - set("IXYZ") for label in self.observable)
        ):
            raise ValueError("observable requires equal-width nonempty I/X/Y/Z Pauli labels")
        return self


class MeasurementExperimentOptions(PauliCircuitInput):
    """A QPY state-preparation circuit, Hermitian Pauli observable and shot controls."""

    measurement: MeasurementOptions = Field(default_factory=MeasurementOptions)
    sampler: ComponentSelection = Field(
        default_factory=lambda: ComponentSelection.named("statevector")
    )

    @field_validator("sampler", mode="before")
    @classmethod
    def _sampler_name(cls, value: Any) -> Any:
        """Use the molecular engine's short component spelling."""
        return {"name": value} if isinstance(value, str) else value


@EXPERIMENTS.register(
    "pauli_measurement",
    MeasurementExperimentOptions,
    requires=frozenset({"sampler"}),
    capabilities=frozenset({"cuda"}),
)
def measurement_experiment(
    *, options: MeasurementExperimentOptions, **context: Any
) -> ExperimentResult:
    """Measure using the configured sampler, returning counts separately from estimates."""
    from qiskit.quantum_info import SparsePauliOp

    circuit = read_circuit(Path(options.circuit_path), max_bytes=options.max_circuit_bytes)
    observable = SparsePauliOp.from_list(list(options.observable.items()))
    result = measure_observable(
        circuit,
        observable,
        options=options.measurement,
        sampler=options.sampler,
        cores=context["cores"],
        device=context["device"],
    )
    arrays = {}
    for index, group in enumerate(result["groups"]):
        prefix = f"group_{index}"
        arrays[prefix + "_covariance"] = np.asarray(group.pop("covariance"), dtype=float)
        counts = group.pop("counts")
        arrays[prefix + "_bitstrings"] = np.packbits(
            np.asarray([[int(bit) for bit in bits] for bits in counts], dtype=np.uint8), axis=1
        )
        arrays[prefix + "_counts"] = np.asarray(list(counts.values()), dtype=np.uint64)
        group.update(
            covariance_array=prefix + "_covariance",
            bitstrings_array=prefix + "_bitstrings",
            counts_array=prefix + "_counts",
        )
    result.update(
        num_qubits=circuit.num_qubits,
        bit_order="Qiskit display order, packed big-endian, trailing zero padding",
        units="observable_coefficient_units",
    )
    return ExperimentResult(kind="pauli_measurement", arrays=arrays, metadata=result)

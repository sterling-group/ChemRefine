"""Native artifact adapter for explicitly noisy endpoint Pauli-check experiments."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from chemrefine.engines.qiskit.experiment import EXPERIMENTS, ExperimentResult
from chemrefine.engines.qiskit.experiment_measurement import read_circuit
from chemrefine.engines.qiskit.options import ComponentSelection
from chemrefine.engines.qiskit.spacetime import (
    SpacetimeOptions,
    build_spacetime_circuit,
    collect_spacetime_counts,
)
from chemrefine.errors import ConfigError


class SpacetimeExperimentOptions(BaseModel):
    """A Clifford QPY payload, optional initial preparation and explicitly noisy sampler."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    circuit_path: str = Field(min_length=1, json_schema_extra={"input_file": True})
    preparation_path: str | None = Field(None, min_length=1, json_schema_extra={"input_file": True})
    max_circuit_bytes: int = Field(33554432, ge=1)
    spacetime: SpacetimeOptions
    sampler: ComponentSelection

    @field_validator("sampler", mode="before")
    @classmethod
    def _sampler_name(cls, value: Any) -> Any:
        """Retain the registry shorthand while requiring explicit noise options."""
        return {"name": value} if isinstance(value, str) else value

    @model_validator(mode="after")
    def _explicit_noise(self) -> Self:
        """Refuse ideal or remote execution in this local noisy research domain."""
        noise = self.sampler.options.get("noise_model")
        if self.sampler.name != "aer" or not isinstance(noise, dict) or not noise.get("errors"):
            raise ValueError(
                "spacetime requires sampler aer with a serialized nonideal noise_model"
            )
        return self


@EXPERIMENTS.register(
    "spacetime_postselection",
    SpacetimeExperimentOptions,
    requires=frozenset({"sampler"}),
    capabilities=frozenset({"cuda"}),
    status="experimental",
    supported_domains=(
        "bound Clifford-unitary payloads with user-chosen endpoint Pauli checks",
        "explicit noisy Aer simulation and arbitrary unchecked input preparations",
        "physical zero-syndrome counts and conditional diagonal-Pauli estimates",
    ),
)
def spacetime_experiment(
    *, options: SpacetimeExperimentOptions, **context: Any
) -> ExperimentResult:
    """Keep circuit checks and physical joint/accepted/rejected records in one bundle."""
    from qiskit import qpy

    payload = read_circuit(Path(options.circuit_path), max_bytes=options.max_circuit_bytes)
    preparation = (
        read_circuit(Path(options.preparation_path), max_bytes=options.max_circuit_bytes)
        if options.preparation_path is not None
        else None
    )
    checked = build_spacetime_circuit(payload, options.spacetime)
    modes, checks = checked.num_data_qubits, len(checked.input_checks)
    stream = io.BytesIO()
    qpy.dump(checked.circuit, stream, version=13)
    circuit_bytes = stream.getvalue()
    joint_bytes = (modes + checks + 7) // 8
    bound = len(circuit_bytes) + 2 * options.spacetime.shots * (joint_bytes + 8)
    if bound > context["max_output_bytes"]:
        raise ConfigError("spacetime physical-count artifact exceeds max_output_bytes")
    result = collect_spacetime_counts(
        payload,
        options.sampler,
        options.spacetime,
        preparation=preparation,
        cores=context["cores"],
        device=context["device"],
    )
    arrays = {"checked_circuit_qpy": np.frombuffer(circuit_bytes, dtype=np.uint8)}
    for name, counts, width in (
        ("raw", result.raw_counts, modes + checks),
        ("accepted", result.accepted_counts, modes),
        ("rejected", result.rejected_counts, modes + checks),
    ):
        strings = np.asarray(
            [[int(bit) for bit in bits.replace(" ", "")] for bits in counts], dtype=np.uint8
        ).reshape((-1, width))
        arrays[name + "_bitstrings"] = np.packbits(strings, axis=1)
        arrays[name + "_counts"] = np.asarray(list(counts.values()), dtype=np.uint64)
    return ExperimentResult(
        kind="spacetime_postselection",
        arrays=arrays,
        metadata={
            **result.metadata,
            "units": "dimensionless",
            "packed_count_order": "syndrome then data, each qubit zero right; big-endian packing",
            "accepted_count_order": "data only; big-endian packing with trailing zero padding",
            "checked_circuit_semantics": (
                "uncompiled unitary checks and payload; excludes input preparation"
            ),
            "qpy_version": 13,
        },
    )

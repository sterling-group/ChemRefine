"""YAML shadow acquisition adapter preserving settings, physical counts and RDMs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator

from chemrefine.engines.api import BackendRequirement
from chemrefine.engines.qiskit.experiment import EXPERIMENTS, ExperimentResult
from chemrefine.engines.qiskit.experiment_measurement import read_circuit
from chemrefine.engines.qiskit.options import ComponentSelection
from chemrefine.engines.qiskit.shadows import FermionicShadowOptions, collect_fermionic_shadows
from chemrefine.errors import ConfigError


class ShadowExperimentOptions(BaseModel):
    """A bound Jordan-Wigner preparation, sampler and distinct fermionic ensemble."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    circuit_path: str = Field(
        min_length=1, json_schema_extra={"input_file": True, "file_format": "quantum_circuit"}
    )
    max_circuit_bytes: int = Field(33554432, ge=1)
    shadows: FermionicShadowOptions
    sampler: ComponentSelection = Field(
        default_factory=lambda: ComponentSelection.named("statevector")
    )

    @field_validator("sampler", mode="before")
    @classmethod
    def _sampler_name(cls, value: Any) -> Any:
        """Accept the established component shorthand without losing nested options."""
        return {"name": value} if isinstance(value, str) else value


@EXPERIMENTS.register(
    "fermionic_shadows",
    ShadowExperimentOptions,
    requires=frozenset({"sampler"}),
    capabilities=frozenset({"cuda"}),
    status="experimental",
    supported_domains=(
        "fixed-N complex Haar U(m) orbital shadows",
        "SO(2m) signed Majorana-Clifford shadows without number postselection",
        "Jordan-Wigner spin-orbital preparations and first/second RDMs",
    ),
    backend_requirement=BackendRequirement(extra="qiskit-fermionic", import_name="ffsim"),
)
def shadow_experiment(*, options: ShadowExperimentOptions, **context: Any) -> ExperimentResult:
    """Collect a shadow dataset and retain enough arrays to reproduce its inversion."""
    circuit = read_circuit(Path(options.circuit_path), max_bytes=options.max_circuit_bytes)
    exported = (circuit.metadata or {}).get("chemrefine_preparation")
    if exported is not None and (
        exported["mapping"].get("name") != "jordan_wigner"
        or exported["num_spin_orbitals"] != circuit.num_qubits
    ):
        raise ConfigError("fermionic shadows require an unreduced Jordan-Wigner circuit bundle")
    modes, controls = circuit.num_qubits, options.shadows
    entries = modes**2 + (modes**4 if controls.max_order == 2 else 0)
    settings = controls.num_settings
    # Bound the full native payload before any sampler (including a QPU) is called.
    required = 16 * entries * (settings + 3) + 64 * settings * modes**2
    required += settings * controls.shots_per_setting * (8 + (modes + 7) // 8)
    required += 8 * (settings + 1)
    if required > context["max_output_bytes"]:
        raise ConfigError("fermionic shadow dataset exceeds max_output_bytes")
    result = collect_fermionic_shadows(
        circuit, options.sampler, controls, device=context["device"], cores=context["cores"]
    )
    arrays: dict[str, NDArray[Any]] = {
        "settings": np.stack([setting.matrix for setting in result.settings]),
        "one_body": result.rdms.one_body,
        "setting_one_body": np.stack([rdm.one_body for rdm in result.setting_rdms]),
    }
    if result.rdms.two_body is not None:
        arrays["two_body"] = result.rdms.two_body
        arrays["setting_two_body"] = np.stack(
            [cast("NDArray[Any]", rdm.two_body) for rdm in result.setting_rdms]
        )
    for name, error in (
        ("standard_error_real", result.standard_errors_real),
        ("standard_error_imag", result.standard_errors_imag),
    ):
        if error is not None:
            arrays[name + "_one_body"] = error.one_body
            if error.two_body is not None:
                arrays[name + "_two_body"] = error.two_body
    bitstrings: list[list[int]] = []
    frequencies: list[int] = []
    offsets = [0]
    for counts in result.counts:
        bitstrings.extend([int(bit) for bit in bits] for bits in counts)
        frequencies.extend(counts.values())
        offsets.append(len(frequencies))
    arrays["bitstrings"] = np.packbits(np.asarray(bitstrings, dtype=np.uint8), axis=1)
    arrays["counts"] = np.asarray(frequencies, dtype=np.uint64)
    arrays["setting_offsets"] = np.asarray(offsets, dtype=np.uint64)
    return ExperimentResult(
        kind="fermionic_shadows",
        arrays=arrays,
        metadata={
            **result.metadata,
            "num_modes": modes,
            "units": "dimensionless",
            "mode_order": "mode i maps to Jordan-Wigner qubit i; no spin ordering assumed",
            "packed_count_order": "Qiskit display order, big-endian packing, trailing zero padding",
            "count_setting_intervals": "[setting_offsets[i], setting_offsets[i+1])",
            "property_source": "measured randomized circuits",
        },
    )

"""Portable numerical storage for reusable sampled quantum states."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from chemrefine.engines.qiskit.bundles import DEFAULT_MAX_BYTES, read_bundle, write_bundle
from chemrefine.engines.qiskit.determinants import DeterminantState
from chemrefine.errors import ConfigError, OutputParseError


class StateDescription(BaseModel):
    """Array references and the orbital space defining a determinant-state basis."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    num_modes: StrictInt = Field(ge=1)
    determinants: str
    amplitudes: str
    orbital_rotation: str | None = None


def save_states(
    path: Path, states: Sequence[DeterminantState], *, max_bytes: int = DEFAULT_MAX_BYTES
) -> Path:
    """Persist all roots, encoding arbitrary-width determinants as little-endian limbs."""
    if not states:
        raise ConfigError("a state bundle requires at least one state")
    arrays: dict[str, Any] = {}
    descriptions = []
    required = 0
    for index, state in enumerate(states):
        prefix = f"root_{index}"
        width = (state.num_modes + 63) // 64
        rotation = state.orbital_rotation
        required += len(state.determinants) * width * 8 + state.amplitudes.nbytes
        required += 0 if rotation is None else rotation.nbytes
        if required > max_bytes:
            raise ConfigError("state bundle exceeds max_bytes")
        arrays[f"{prefix}_determinants"] = np.asarray(
            [
                [(bits >> (64 * limb)) & ((1 << 64) - 1) for limb in range(width)]
                for bits in state.determinants
            ],
            dtype="<u8",
        )
        arrays[f"{prefix}_amplitudes"] = state.amplitudes
        rotation_key = None
        if rotation is not None:
            rotation_key = f"{prefix}_orbital_rotation"
            arrays[rotation_key] = rotation
        descriptions.append(
            StateDescription(
                num_modes=state.num_modes,
                determinants=f"{prefix}_determinants",
                amplitudes=f"{prefix}_amplitudes",
                orbital_rotation=rotation_key,
            ).model_dump(mode="json")
        )
    return write_bundle(
        path,
        kind="determinant_states",
        arrays=arrays,
        metadata={
            "states": descriptions,
            "determinant_encoding": "uint64_little_endian_limbs",
            "amplitude_units": "dimensionless",
            "basis_convention": "mode_0_is_least_significant_bit",
            "property_origin": "projected_subspace",
        },
        max_bytes=max_bytes,
    )


def load_states(path: Path, *, max_bytes: int = DEFAULT_MAX_BYTES) -> tuple[DeterminantState, ...]:
    """Load normalized states, validating numerical integrity and basis semantics."""
    bundle = read_bundle(path, max_bytes=max_bytes)
    try:
        if (
            bundle.description.kind != "determinant_states"
            or bundle.metadata.get("determinant_encoding") != "uint64_little_endian_limbs"
        ):
            raise ValueError("unsupported determinant state bundle")
        descriptions = bundle.metadata["states"]
        if not isinstance(descriptions, list) or not descriptions:
            raise ValueError("state bundle requires a nonempty state list")
        states = []
        for record in descriptions:
            description = StateDescription.model_validate(record)
            limbs = bundle.arrays[description.determinants]
            amplitudes = bundle.arrays[description.amplitudes]
            if (
                limbs.ndim != 2
                or limbs.shape != (len(amplitudes), (description.num_modes + 63) // 64)
                or limbs.dtype != np.dtype("<u8")
            ):
                raise ValueError("invalid determinant limb array")
            determinants = tuple(
                sum(int(word) << (64 * limb) for limb, word in enumerate(row)) for row in limbs
            )
            kwargs = {}
            if description.orbital_rotation is not None:
                kwargs["orbital_rotation"] = bundle.arrays[description.orbital_rotation]
            states.append(
                DeterminantState(description.num_modes, determinants, amplitudes, **kwargs)
            )
        return tuple(states)
    except (KeyError, TypeError, ValueError, ConfigError) as exc:
        raise OutputParseError(f"invalid determinant states in {path}: {exc}") from exc


def validate_state_references(output_path: Path, *, circuit_max_bytes: int = 33554432) -> None:
    """Validate a molecular sidecar's referenced quantum states without provider calls."""
    try:
        raw = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(raw.get("engine_metadata", {}), dict):
            raise ValueError("molecular sidecar metadata must be a mapping")
        references = raw.get("engine_metadata", {}).get("quantum_artifacts", [])
        if not isinstance(references, list):
            raise ValueError("quantum_artifacts must be a list of local filenames")
        for name in references:
            if not isinstance(name, str) or not name or Path(name).name != name or "\\" in name:
                raise ValueError("quantum_artifacts must contain local filenames")
            path = output_path.parent / name
            if path.resolve().parent != output_path.parent.resolve():
                raise ValueError("quantum state descriptor escapes its output directory")
            bundle = read_bundle(path, max_bytes=max(DEFAULT_MAX_BYTES, circuit_max_bytes))
            if bundle.description.kind == "bound_circuit":
                from chemrefine.engines.qiskit.circuit_io import validate_circuit_bundle

                validate_circuit_bundle(bundle)
                if bundle.arrays["qpy"].nbytes > circuit_max_bytes:
                    raise ValueError("bound circuit payload exceeds configured circuit_max_bytes")
            else:
                load_states(path)
    except (OSError, TypeError, ValueError) as exc:
        raise OutputParseError(f"invalid quantum state references in {output_path}: {exc}") from exc

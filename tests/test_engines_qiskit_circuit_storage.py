"""Portable circuit metadata grows in numeric payloads and validates without SDKs."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from chemrefine.engines.qiskit.bundles import (
    MAX_DESCRIPTOR_BYTES,
    ArrayDescription,
    BundleDescription,
    encode_bundle_descriptor,
    read_bundle,
    write_bundle,
)
from chemrefine.engines.qiskit.circuit_storage import (
    CircuitDescription,
    decode_circuit_data,
    encode_circuit_data,
    preflight_storage,
    storage_specs,
)
from chemrefine.errors import ConfigError


@pytest.fixture
def interpretation():
    """Include complex-operator Y labels, Unicode names, and ordered bindings."""
    return CircuitDescription(
        root=2,
        num_qubits=2,
        num_spin_orbitals=4,
        num_particles=(1, 0),
        mapping={"name": "jordan_wigner"},
        active_space={},
        parameter_order=("θ[10]", "θ[2]"),
        parameter_values=(0.125, -0.3),
        active_hamiltonian={"XI": 0.3, "YZ": -1.2},
        energy_offsets={"nuclear": 0.5},
        provenance={"producer": "test"},
    )


def _bundle(tmp_path, metadata, arrays):
    """Give altered interpretation arrays a valid integrity digest."""
    return read_bundle(
        write_bundle(
            tmp_path / "data.json",
            kind="bound_circuit",
            arrays={"qpy": np.zeros(20, dtype=np.uint8)} | arrays,
            metadata=metadata,
        )
    )


@pytest.mark.parametrize("parameters", [True, False])
def test_numeric_storage_roundtrips_conventions_and_parameter_order(
    tmp_path, interpretation, parameters
):
    """V2 stores parameters and Pauli terms outside JSON without changing the public view."""
    if not parameters:
        interpretation = interpretation.model_copy(
            update={"parameter_order": (), "parameter_values": ()}
        )
    metadata, arrays = encode_circuit_data(interpretation)
    assert metadata["version"] == 2
    assert "active_hamiltonian" not in metadata
    assert metadata["pauli_encoding"] == "ascii_IXYZ_qubit_0_right"
    assert metadata["parameter_encoding"] == "utf8_offsets"
    specs = storage_specs(interpretation)
    assert all(specs[name].shape == array.shape for name, array in arrays.items())
    bundle = _bundle(tmp_path, metadata, arrays)
    assert decode_circuit_data(bundle) == interpretation
    used = sum(array.nbytes for array in arrays.values())
    assert preflight_storage(interpretation, max_bytes=used + 20) == 20
    assert preflight_storage(interpretation, max_bytes=used + 20, path=tmp_path / "root.json") == 20
    with pytest.raises(ConfigError, match="max_bytes"):
        preflight_storage(interpretation, max_bytes=used)


def test_legacy_json_description_still_decodes(tmp_path, interpretation):
    """Existing valid v1 handoffs keep their scientific interpretation."""
    bundle = _bundle(tmp_path, interpretation.model_dump(mode="json"), {})
    assert decode_circuit_data(bundle) == interpretation
    with pytest.raises(ValueError, match="one QPY"):
        decode_circuit_data(replace(bundle, arrays=dict(bundle.arrays) | {"extra": np.zeros(1)}))


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "extra",
        "label_dtype",
        "value_rank",
        "label_width",
        "empty_terms",
        "bad_code",
        "duplicate",
        "offset_length",
        "offset_start",
        "offset_end",
        "offset_order",
        "utf8",
        "nan",
        "unknown_version",
        "encoding",
    ],
)
def test_numeric_description_rejects_inconsistent_arrays(tmp_path, interpretation, fault):
    """Checksums cannot establish array semantics, offsets, or Pauli identity."""
    metadata, arrays = encode_circuit_data(interpretation)
    arrays = {name: value.copy() for name, value in arrays.items()}
    if fault == "missing":
        del arrays["pauli_labels"]
    elif fault == "extra":
        arrays["surprise"] = np.zeros(1)
    elif fault == "label_dtype":
        arrays["pauli_labels"] = arrays["pauli_labels"].astype(float)
    elif fault == "value_rank":
        arrays["parameter_values"] = arrays["parameter_values"].reshape(1, -1)
    elif fault == "label_width":
        arrays["pauli_labels"] = arrays["pauli_labels"][:, :1]
    elif fault == "empty_terms":
        arrays["pauli_labels"] = arrays["pauli_labels"][:0]
        arrays["pauli_coefficients"] = arrays["pauli_coefficients"][:0]
    elif fault == "bad_code":
        arrays["pauli_labels"][0, 0] = 255
    elif fault == "duplicate":
        arrays["pauli_labels"][1] = arrays["pauli_labels"][0]
    elif fault == "offset_length":
        arrays["parameter_offsets"] = np.zeros(0, dtype="<u8")
    elif fault == "offset_start":
        arrays["parameter_offsets"][0] = 1
    elif fault == "offset_end":
        arrays["parameter_offsets"][-1] = len(arrays["parameter_names"]) + 1
    elif fault == "offset_order":
        arrays["parameter_offsets"][1] = len(arrays["parameter_names"]) + 1
    elif fault == "utf8":
        arrays["parameter_names"][0] = 255
    elif fault == "nan":
        # Generic readers reject nonfinite arrays first; direct callers also fail.
        bundle = _bundle(tmp_path, metadata, arrays)
        arrays["pauli_coefficients"][0] = np.nan
        with pytest.raises(ValueError):
            decode_circuit_data(replace(bundle, arrays=dict(bundle.arrays) | arrays))
        return
    elif fault == "unknown_version":
        metadata["version"] = 99
    else:
        metadata["pauli_encoding"] = "qubit_0_left"
    with pytest.raises(ValueError):
        decode_circuit_data(_bundle(tmp_path, metadata, arrays))


def test_large_freeform_metadata_fails_before_payload_allocation(interpretation):
    """Binary storage does not remove the independent cap on common JSON metadata."""
    value = interpretation.model_copy(update={"provenance": {"large": "x" * 1048576}})
    with pytest.raises(ConfigError, match="size limit"):
        preflight_storage(value, max_bytes=33554432)


def test_metadata_preflight_reserves_escaped_unicode_output_names(tmp_path, interpretation):
    """A near-limit early success also fits a valid Unicode payload basename."""
    metadata, arrays = encode_circuit_data(interpretation)
    metadata["provenance"] = {"padding": ""}
    specs = storage_specs(interpretation)
    max_bytes = 2048
    remaining = max_bytes - sum(array.nbytes for array in arrays.values())
    document = encode_bundle_descriptor(
        BundleDescription(
            kind="bound_circuit",
            payload="\uffff" * 255,
            sha256="0" * 64,
            arrays=specs | {"qpy": ArrayDescription(shape=(remaining,), dtype="|u1")},
            metadata=metadata,
        )
    )
    value = interpretation.model_copy(
        update={"provenance": {"padding": "x" * (MAX_DESCRIPTOR_BYTES - len(document))}}
    )
    assert preflight_storage(value, max_bytes=max_bytes) == remaining
    beyond_reserve = value.model_copy(
        update={"provenance": {"padding": value.provenance["padding"] + "x"}}
    )
    with pytest.raises(ConfigError, match="size limit"):
        preflight_storage(beyond_reserve, max_bytes=max_bytes)
    path = tmp_path / ("\u03b8" * 70 + ".json")
    payload = f"{path.stem}.{'0' * 32}.npz"
    assert len(payload.encode("utf-8")) <= 255
    assert len(json.dumps(payload)) > 255
    assert preflight_storage(value, max_bytes=max_bytes, path=path) == remaining

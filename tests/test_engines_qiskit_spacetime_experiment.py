"""Native dataset preservation and strict YAML knobs for endpoint checks."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from chemrefine.engines._input_files import typed_input_references
from chemrefine.engines.qiskit.experiment_spacetime import (
    SpacetimeExperimentOptions,
    spacetime_experiment,
)
from chemrefine.errors import ConfigError


def _inputs(tmp_path):
    """Write actual Clifford and non-Clifford initial-preparation QPY inputs."""
    from qiskit import QuantumCircuit, qpy
    from qiskit_aer.noise import NoiseModel, pauli_error

    circuit = QuantumCircuit(1)
    circuit.h(0)
    preparation = QuantumCircuit(1)
    preparation.ry(0.3, 0)
    for name, item in (("circuit.qpy", circuit), ("preparation.qpy", preparation)):
        with (tmp_path / name).open("wb") as stream:
            qpy.dump(item, stream, version=13)
    model = NoiseModel()
    model.add_all_qubit_quantum_error(pauli_error([("X", 0.2), ("I", 0.8)]), ["h"])
    return {
        "circuit_path": str(tmp_path / "circuit.qpy"),
        "preparation_path": str(tmp_path / "preparation.qpy"),
        "sampler": {
            "name": "aer",
            "options": {
                "noise_model": model.to_dict(serializable=True),
                "optimization_level": 0,
                "seed_simulator": 8,
            },
        },
        "spacetime": {"checks": ["Z"], "diagonal_observables": ["Z"], "shots": 256},
    }


def test_artifact_roundtrip_qpy_and_physical_register_arrays(tmp_path):
    """Arrays retain every accepted/rejected raw shot and an executable checked circuit."""
    from qiskit import qpy

    options = SpacetimeExperimentOptions(**_inputs(tmp_path))
    result = spacetime_experiment(options=options, cores=1, device="cpu", max_output_bytes=1000000)
    assert result.kind == "spacetime_postselection"
    assert result.arrays["raw_counts"].sum() == 256
    assert result.arrays["accepted_counts"].sum() + result.arrays["rejected_counts"].sum() == 256
    assert qpy.load(io.BytesIO(result.arrays["checked_circuit_qpy"].tobytes()))[0].num_qubits == 2
    joint = np.unpackbits(result.arrays["raw_bitstrings"], axis=1)[:, :2]
    accepted = np.unpackbits(result.arrays["accepted_bitstrings"], axis=1)[:, :1]
    reconstructed = {
        str(word[0]): int(n)
        for word, n in zip(accepted, result.arrays["accepted_counts"], strict=True)
    }
    for word, n in zip(joint, result.arrays["raw_counts"], strict=True):
        if word[0] == 0:
            assert reconstructed[str(word[1])] == n
    assert {ref.location for ref in typed_input_references(options)} == {
        ("circuit_path",),
        ("preparation_path",),
    }


def test_artifact_size_guard_and_no_preparation(tmp_path):
    """Allocation fails before shots while default zero input is supported."""
    raw = _inputs(tmp_path)
    raw["preparation_path"] = None
    options = SpacetimeExperimentOptions(**raw)
    with pytest.raises(ConfigError, match="max_output_bytes"):
        spacetime_experiment(options=options, cores=1, device="cpu", max_output_bytes=1)
    result = spacetime_experiment(options=options, cores=1, device="cpu", max_output_bytes=1000000)
    assert result.metadata["num_data_qubits"] == 1


def test_noise_requirement_and_sampler_shorthand(tmp_path):
    """Frozen options require an actual serialized-noise declaration at discovery."""
    raw = _inputs(tmp_path)
    for sampler in (
        "aer",
        "statevector",
        {"name": "aer", "options": {"noise_model": {"errors": []}}},
    ):
        with pytest.raises(ValidationError, match="nonideal"):
            SpacetimeExperimentOptions(**{**raw, "sampler": sampler})
    with pytest.raises(ValidationError):
        SpacetimeExperimentOptions.model_validate({**raw, "mystery": 3})


def test_all_public_knobs_are_explicit_in_the_runnable_example():
    """Every local check and sampler control has an example verdict."""
    import yaml

    from chemrefine.engines.qiskit.spacetime import SpacetimeOptions

    path = Path("examples/tutorials/qiskit_experiment/spacetime_postselection.yaml")
    if not path.exists():
        pytest.fail("runnable spacetime example is missing")
    experiment = yaml.safe_load(path.read_text())["steps"][0]["options"]["experiment"]["options"]
    assert set(SpacetimeExperimentOptions.model_fields) == set(experiment)
    assert set(SpacetimeOptions.model_fields) == set(experiment["spacetime"])


def test_actual_complete_rejection_retains_empty_numeric_arrays(tmp_path):
    """Deterministic noisy check readout yields an honest zero-acceptance artifact."""
    raw = _inputs(tmp_path)
    raw["preparation_path"] = None
    raw["sampler"]["options"]["noise_model"] = {
        "errors": [
            {
                "type": "roerror",
                "operations": ["measure"],
                "probabilities": [[0, 1], [1, 0]],
            }
        ]
    }
    result = spacetime_experiment(
        options=SpacetimeExperimentOptions(**raw), cores=1, device="cpu", max_output_bytes=1000000
    )
    assert result.metadata["accepted_shots"] == 0
    assert result.arrays["accepted_bitstrings"].shape == (0, 1)
    assert result.arrays["accepted_counts"].shape == (0,)
    assert result.arrays["rejected_counts"].sum() == 256

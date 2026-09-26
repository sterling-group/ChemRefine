"""External quantum circuits and manifest payloads participate in pipeline identity."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest
from ase import Atoms
from pydantic import BaseModel, Field
from qiskit import QuantumCircuit, qpy

from chemrefine.config import load_config
from chemrefine.engines.api import preflight_steps
from chemrefine.engines.qiskit.bundles import read_bundle, write_bundle
from chemrefine.engines.qiskit.experiment import (
    EXPERIMENTS,
    QiskitExperimentEngine,
    QiskitExperimentOptions,
    run_experiment,
    validate_experiment,
)
from chemrefine.engines.qiskit.experiment_measurement import (
    MeasurementExperimentOptions,
    read_circuit,
)
from chemrefine.errors import ConfigError
from chemrefine.state import PipelineState, Structure
from chemrefine.step import run_step


def _circuit(path, *, flip=False):
    """Write a real QPY state-preparation circuit, optionally editing the same file."""
    circuit = QuantumCircuit(1)
    if flip:
        circuit.x(0)
    else:
        circuit.h(0)
    with path.open("wb") as stream:
        qpy.dump(circuit, stream)


def test_changed_qpy_input_invalidates_artifact_cache(tmp_path):
    """Editing circuit bytes at one filename cannot reuse an old measurement report."""
    source = tmp_path / "state.qpy"
    _circuit(source)
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "cpu.slurm.header").write_text("#!/bin/bash\n")
    config_path = tmp_path / "input.yaml"
    config_path.write_text(
        json.dumps(
            {
                "template_dir": "templates",
                "output_dir": "outputs",
                "charge": 0,
                "multiplicity": 1,
                "max_cores": 1,
                "dispatch": "local",
                "steps": [
                    {
                        "step": 1,
                        "engine": "qiskit-experiment",
                        "options": {
                            "backend_python": sys.executable,
                            "experiment": {
                                "name": "pauli_measurement",
                                "options": {
                                    "circuit_path": "state.qpy",
                                    "observable": {"Z": 1.0},
                                    "measurement": {"shots": 1024, "seed": 23},
                                },
                            },
                        },
                    }
                ],
            }
        )
    )
    config = load_config(config_path)
    preflight_steps(config.steps, charge=0, multiplicity=1)
    seed = PipelineState(structures=(Structure(id="0", atoms=Atoms("H")),))
    first = run_step(config, config.steps[0], seed)
    output = config.output_dir / "step1" / "experiment" / "artifact.json"
    metadata = read_bundle(output).metadata
    assert metadata["groups"][0]["covariance_array"] in read_bundle(output).arrays
    assert sum(read_bundle(output).arrays["group_0_counts"]) == 960
    assert not first.cache_hit
    assert abs(metadata["expectation"]) < 5 * metadata["standard_error"]
    assert run_step(config, config.steps[0], seed).cache_hit
    _circuit(source, flip=True)
    second = run_step(config, config.steps[0], seed)
    assert not second.cache_hit
    assert read_bundle(output).metadata["expectation"] == -1.0
    assert second.state.structures[0].energy_hartree is None


def test_sampler_profile_is_resolved_from_the_experiment_component(tmp_path):
    """A configured Aer sampler selects a worker that contains Aer, before execution."""
    engine = QiskitExperimentEngine()
    options = {
        "experiment": {
            "name": "pauli_measurement",
            "options": {
                "circuit_path": str(tmp_path / "state.qpy"),
                "observable": {"Z": 1},
                "sampler": {"name": "aer"},
            },
        }
    }
    assert engine.backend_requirement(options).extra == "qiskit-aer"
    options["experiment"]["options"]["sampler"] = {"name": "statevector"}
    assert engine.backend_requirement(options).extra == "qiskit"


def test_declared_bundle_payloads_are_transitive_inputs(tmp_path, monkeypatch):
    """File-format declarations, rather than filename guesses, identify dependencies."""
    monkeypatch.setattr(EXPERIMENTS, "_specs", dict(EXPERIMENTS._specs))

    class InputOptions(BaseModel):
        """An extension naming a native artifact input."""

        state_path: str = Field(
            json_schema_extra={"input_file": True, "file_format": "quantum_bundle"}
        )

    @EXPERIMENTS.register("consume", InputOptions)
    def consume(**kwargs):
        """Registration is inspected without executing a builder."""
        pytest.fail("dependency discovery must not execute an experiment")

    path = write_bundle(
        tmp_path / "state.json", kind="state", arrays={"a": np.ones(1)}, metadata={}
    )
    engine = QiskitExperimentEngine()
    pointers = engine.input_file_options(
        {"experiment": {"name": "consume", "options": {"state_path": str(path)}}}
    )
    assert pointers == (("experiment", "options", "state_path"),)
    dependencies = engine.input_file_dependencies({"/experiment/options/state_path": path})
    assert len(dependencies) == 1
    assert next(iter(dependencies.values())).suffix == ".npz"


def test_qpy_input_size_and_circuit_count_are_validated(tmp_path):
    """Malformed and ambiguous circuit files fail with a configuration error."""
    path = tmp_path / "circuit.qpy"
    with pytest.raises(ConfigError):
        read_circuit(path)
    _circuit(path)
    with pytest.raises(ConfigError, match="byte limit"):
        read_circuit(path, max_bytes=1)
    with path.open("wb") as stream:
        qpy.dump([QuantumCircuit(1), QuantumCircuit(1)], stream)
    with pytest.raises(ConfigError, match="exactly one"):
        read_circuit(path)
    path.write_bytes(b"bad qpy")
    with pytest.raises(ConfigError, match="cannot load circuit"):
        read_circuit(path)


@pytest.mark.parametrize("observable", [{"": 1}, {"T": 1}, {"Z": 1, "XX": 1}])
def test_pauli_labels_fail_before_sdk_execution(observable):
    """Schema validation names invalid or inconsistent physical observable labels."""
    with pytest.raises(ValueError, match="Pauli labels"):
        MeasurementExperimentOptions(circuit_path="state.qpy", observable=observable)


def test_measurement_device_validation_and_constant_bundle(tmp_path):
    """Provider capabilities decide GPU grants and constants need no shot arrays."""
    source = tmp_path / "state.qpy"
    _circuit(source)
    raw = {
        "experiment": {
            "name": "pauli_measurement",
            "options": {
                "circuit_path": str(source),
                "observable": {"I": 2},
                "sampler": "statevector",
            },
        }
    }
    selected = QiskitExperimentOptions.from_raw(raw)
    with pytest.raises(ConfigError, match=r"statevector.*cpu"):
        validate_experiment(selected.model_copy(update={"device": "cuda"}))
    aer = {**raw["experiment"], "options": {**raw["experiment"]["options"], "sampler": "aer"}}
    validate_experiment(QiskitExperimentOptions.from_raw({"device": "cuda", "experiment": aer}))
    output = tmp_path / "artifact.json"
    run_experiment(raw, output)
    bundle = read_bundle(output)
    assert bundle.metadata["expectation"] == 2
    assert bundle.arrays == {}


def test_measurement_bundle_reconstructs_actual_joint_counts(tmp_path):
    """Packed bitstrings, frequencies and covariance describe the executed experiment."""
    from chemrefine.engines.qiskit.sampling import SampleBatch

    path = tmp_path / "bell.qpy"
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    with path.open("wb") as stream:
        qpy.dump(circuit, stream)
    output = tmp_path / "artifact.json"
    run_experiment(
        {
            "experiment": {
                "name": "pauli_measurement",
                "options": {
                    "circuit_path": str(path),
                    "observable": {"XX": 1, "YY": 1, "ZZ": 1},
                    "measurement": {"grouping": "commuting", "shots": 256, "pilot_shots": 16},
                },
            }
        },
        output,
    )
    bundle = read_bundle(output)
    bits = np.unpackbits(bundle.arrays["group_0_bitstrings"], axis=1)[:, :2]
    counts = {
        "".join(str(int(bit)) for bit in row): int(count)
        for row, count in zip(bits, bundle.arrays["group_0_counts"], strict=True)
    }
    assert SampleBatch(counts, 2, 240).shots == 240
    assert bundle.metadata["expectation"] == 1
    assert bundle.metadata["standard_error"] == 0
    assert np.array_equal(bundle.arrays["group_0_covariance"], np.zeros((3, 3)))
    assert bundle.metadata["groups"][0]["basis_gate_counts"]["cx"] > 0

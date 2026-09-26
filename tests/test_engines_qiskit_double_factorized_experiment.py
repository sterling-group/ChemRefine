"""Electronic integral dependencies and executable circuits survive native trajectories."""

import io

import numpy as np
import pytest

from chemrefine.engines.qiskit.bundles import read_bundle
from chemrefine.engines.qiskit.data import ElectronicStructureData
from chemrefine.engines.qiskit.experiment import QiskitExperimentEngine, run_experiment
from chemrefine.engines.qiskit.integral_io import save_integrals
from chemrefine.errors import ConfigError


def _options(tmp_path):
    """A single electron initially occupies the second orbital in a complex hopping model."""
    data = ElectronicStructureData(
        1,
        0,
        2,
        np.array([[0, 1j], [-1j, 0]]),
        np.zeros((2,) * 4),
        orbital_occupations=[0, 1],
        nuclear_repulsion_energy=0.5,
    )
    source = save_integrals(tmp_path / "integrals.json", data)
    return {
        "cores": 2,
        "experiment": {
            "name": "double_factorized_evolution",
            "options": {
                "integral_bundle_path": str(source),
                "times": [0, 0.3],
                "evolution": {"exact_reference": True},
            },
        },
    }


def test_double_factorized_worker_preserves_complex_states_and_executable_circuits(tmp_path):
    from qiskit import QuantumCircuit, qpy
    from qiskit.quantum_info import Statevector

    options = _options(tmp_path)
    output = run_experiment(options, tmp_path / "result.json")
    bundle = read_bundle(output)
    assert bundle.description.kind == "double_factorized_trajectory"
    arrays = bundle.arrays
    np.testing.assert_allclose(arrays["energies"], [0.5, 0.5], atol=1e-12)
    circuits = qpy.load(io.BytesIO(arrays["circuits_qpy"].tobytes()))
    reference = QuantumCircuit(4)
    reference.x(1)
    assert len(circuits) == 2
    for circuit, vector in zip(circuits, arrays["statevectors"], strict=True):
        np.testing.assert_allclose(
            Statevector.from_instruction(reference.compose(circuit)).data, vector, atol=1e-12
        )
    np.testing.assert_allclose(arrays["occupations"][0], [0, 1, 0, 0], atol=1e-12)
    assert all(
        record["reference_state_error"] < 1e-12 for record in bundle.metadata["observations"]
    )
    engine = QiskitExperimentEngine()
    assert engine.input_file_options(options) == (
        ("experiment", "options", "integral_bundle_path"),
    )
    from pathlib import Path

    dependencies = engine.input_file_dependencies(
        {
            "/experiment/options/integral_bundle_path": Path(
                options["experiment"]["options"]["integral_bundle_path"]
            )
        }
    )
    assert any(path.suffix == ".npz" for path in dependencies.values())
    assert engine.backend_requirement(options).extra == "qiskit-fermionic"


def test_double_factorized_artifact_refuses_output_allocation_before_evolution(
    tmp_path, monkeypatch
):
    options = _options(tmp_path)
    options["max_output_bytes"] = 1

    def no_execution(*args, **kwargs):
        """Guard against accidental expensive computation after output allocation refusal."""
        pytest.fail("unexpected evolution")

    monkeypatch.setattr(
        "chemrefine.engines.qiskit.experiment_double_factorized.simulate_double_factorized_evolution",
        no_execution,
    )
    with pytest.raises(ConfigError, match="max_output_bytes"):
        run_experiment(options, tmp_path / "refused.json")


def test_double_factorized_example_exercises_every_public_nested_knob():
    from pathlib import Path

    import yaml

    from chemrefine.engines.qiskit.double_factorized import DoubleFactorizedIntegratorOptions
    from chemrefine.engines.qiskit.experiment_double_factorized import (
        DoubleFactorizedExperimentOptions,
    )
    from chemrefine.engines.qiskit.integral_io import load_integrals

    root = Path(__file__).resolve().parents[1] / "examples/tutorials/qiskit_experiment"
    raw = yaml.safe_load((root / "double_factorized.yaml").read_text())["steps"][0]["options"][
        "experiment"
    ]["options"]
    parsed = DoubleFactorizedExperimentOptions.model_validate(raw)
    data = load_integrals(root / parsed.integral_bundle_path)
    assert data.num_spatial_orbitals == 2
    assert set(raw) == set(DoubleFactorizedExperimentOptions.model_fields)
    assert set(raw["evolution"]) == set(DoubleFactorizedIntegratorOptions.model_fields)

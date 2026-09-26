"""Resource workflows integrate with artifact reports and nested input contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from chemrefine.engines.qiskit.bundles import read_bundle
from chemrefine.engines.qiskit.experiment import run_experiment
from chemrefine.engines.qiskit.experiment_resources import (
    FactorizedResourceExperimentOptions,
    surface_code_resource_experiment,
)
from chemrefine.engines.qiskit.resources import SurfaceCodeOptions


def test_pauli_experiment_publishes_durable_report(tmp_path):
    """The normal artifact entry point stores assumptions and costs without a fake energy."""
    path = tmp_path / "resources.json"
    run_experiment(
        {
            "cores": 1,
            "experiment": {
                "name": "pauli_resources",
                "options": {
                    "hamiltonian": {"Z": 1},
                    "budget": {"energy_error_hartree": 0.001},
                },
            },
        },
        output_path=path,
    )
    result = read_bundle(path)
    assert result.description.kind == "resource_estimate"
    assert result.metadata["normalization_hartree"] == 1
    assert result.metadata["estimate_kind"] == "analytical_query_bound"
    assert not result.arrays


def test_physical_experiment_retains_hardware_provenance():
    """The physical adapter delegates its explicit machine assumptions without mutation."""
    options = SurfaceCodeOptions(
        logical_qubits=2,
        logical_cycles=20,
        code_distance=11,
        physical_error_probability=0.001,
        threshold_probability=0.01,
        logical_error_prefactor=0.1,
        physical_qubits_per_patch_d2=2,
        routing_patch_multiplier=1,
        cycle_time_seconds=1e-6,
        failure_budget=0.01,
        hardware_provenance="test device assumptions",
    )
    result = surface_code_resource_experiment(options=options)
    assert result.kind == "resource_estimate"
    assert result.metadata["assumptions"]["hardware_provenance"] == "test device assumptions"


def test_factorized_experiment_file_schema_and_exact_input_consumption():
    """DF and THC declare every payload dependency and reject unconsumed factor files."""
    base = {
        "integral_bundle_path": "integrals.json",
        "budget": {
            "energy_error_hartree": 0.001,
            "synthesis_error_hartree": 0.00001,
        },
    }
    FactorizedResourceExperimentOptions(**base)
    FactorizedResourceExperimentOptions(**base, method="thc", thc_bundle_path="thc.json")
    for extra in ({"method": "thc"}, {"thc_bundle_path": "thc.json"}):
        with pytest.raises(ValidationError, match="required exactly"):
            FactorizedResourceExperimentOptions(**base, **extra)
    schema = FactorizedResourceExperimentOptions.model_json_schema()["properties"]
    for name in ("integral_bundle_path", "thc_bundle_path"):
        assert schema[name]["input_file"] is True
        assert schema[name]["file_format"] == "quantum_bundle"

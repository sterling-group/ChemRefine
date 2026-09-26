"""Native artifact contracts for real orbital and Majorana shadow workers."""

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from chemrefine.engines.qiskit.bundles import read_bundle
from chemrefine.engines.qiskit.experiment import (
    QiskitExperimentEngine,
    run_experiment,
)
from chemrefine.engines.qiskit.experiment_shadows import ShadowExperimentOptions
from chemrefine.errors import ConfigError


@pytest.fixture
def preparation(tmp_path):
    """Write one complex number-conserving state-preparation circuit."""
    from qiskit import QuantumCircuit, qpy

    circuit = QuantumCircuit(3)
    circuit.x(0)
    circuit.ry(0.7, 1)
    circuit.cx(1, 0)
    circuit.s(0)
    path = tmp_path / "state.qpy"
    with path.open("wb") as stream:
        qpy.dump(circuit, stream)
    return path


@pytest.mark.parametrize(
    ("ensemble", "order", "settings"),
    [("orbital_haar", 2, 3), ("majorana_clifford", 1, 3), ("orbital_haar", 1, 1)],
)
def test_shadow_worker_preserves_complete_replayable_dataset(
    preparation, ensemble, order, settings
):
    from chemrefine.engines.qiskit.shadows import (
        FermionicShadowOptions,
        ShadowSetting,
        estimate_fermionic_shadows,
    )

    controls = {
        "ensemble": ensemble,
        "max_order": order,
        "num_settings": settings,
        "shots_per_setting": 8,
    }
    if ensemble == "orbital_haar":
        controls["num_particles"] = 1
    output = preparation.parent / "shadow.json"
    run_experiment(
        {
            "cores": 2,
            "experiment": {
                "name": "fermionic_shadows",
                "options": {
                    "circuit_path": str(preparation),
                    "shadows": controls,
                    "sampler": {"name": "statevector", "options": {"seed": 13}},
                },
            },
        },
        output,
    )
    bundle = read_bundle(output)
    assert bundle.description.kind == "fermionic_shadows"
    assert bundle.metadata["property_source"] == "measured randomized circuits"
    assert bundle.metadata["num_modes"] == 3
    arrays = bundle.arrays
    unpacked = np.unpackbits(arrays["bitstrings"], axis=1, count=3)
    counts = []
    for start, stop in zip(
        arrays["setting_offsets"][:-1], arrays["setting_offsets"][1:], strict=True
    ):
        counts.append(
            {
                "".join(map(str, row)): int(frequency)
                for row, frequency in zip(
                    unpacked[start:stop], arrays["counts"][start:stop], strict=True
                )
            }
        )
    replay = estimate_fermionic_shadows(
        [ShadowSetting(ensemble, matrix) for matrix in arrays["settings"]],
        counts,
        FermionicShadowOptions.model_validate(controls),
    )
    np.testing.assert_allclose(arrays["one_body"], replay.rdms.one_body)
    if order == 2:
        np.testing.assert_allclose(arrays["two_body"], replay.rdms.two_body)
        np.testing.assert_allclose(
            arrays["standard_error_real_two_body"], replay.standard_errors_real.two_body
        )
    else:
        assert "two_body" not in arrays
    assert arrays["setting_one_body"].shape == (settings, 3, 3)
    if settings == 1:
        assert "standard_error_real_one_body" not in arrays
    assert arrays["counts"].sum() == 8 * settings
    assert all(not value.flags.writeable for value in arrays.values())


def test_shadow_worker_validates_output_budget_before_sampling(preparation, monkeypatch):
    def no_sampling(*args, **kwargs):
        """A refused dataset must never submit even one measurement circuit."""
        pytest.fail("called sampler after budget refusal")

    monkeypatch.setattr(
        "chemrefine.engines.qiskit.experiment_shadows.collect_fermionic_shadows", no_sampling
    )
    with pytest.raises(ConfigError, match="dataset exceeds"):
        run_experiment(
            {
                "max_output_bytes": 1,
                "experiment": {
                    "name": "fermionic_shadows",
                    "options": {"circuit_path": str(preparation), "shadows": {"num_particles": 1}},
                },
            },
            preparation.parent / "refused.json",
        )


def test_shadow_catalog_declares_nested_controls_domains_and_file_dependencies():
    options: dict[str, Any] = {
        "experiment": {
            "name": "fermionic_shadows",
            "options": {"circuit_path": "state.qpy", "shadows": {"ensemble": "majorana_clifford"}},
        }
    }
    engine = QiskitExperimentEngine()
    assert engine.input_file_options(options) == (("experiment", "options", "circuit_path"),)
    assert engine.backend_requirement(options).extra == "qiskit-fermionic"
    parsed = ShadowExperimentOptions.model_validate(
        {**options["experiment"]["options"], "sampler": "statevector"}
    )
    assert parsed.sampler.name == "statevector"
    with pytest.raises(ValidationError):
        ShadowExperimentOptions.model_validate({**options["experiment"]["options"], "unknown": 1})


def test_shadow_nested_knobs_have_examples_or_explicit_test_verdicts():
    from pathlib import Path

    import yaml

    from chemrefine.engines.qiskit.shadows import FermionicShadowOptions

    root = Path(__file__).resolve().parents[1] / "examples/tutorials/qiskit_experiment"
    illustrated = set()
    outer = set()
    for name in ("orbital_shadows", "majorana_shadows"):
        options = yaml.safe_load((root / f"{name}.yaml").read_text())["steps"][0]["options"]
        controls = options["experiment"]["options"]
        parsed = ShadowExperimentOptions.model_validate(controls)
        assert (root / parsed.circuit_path).is_file()
        illustrated.update(controls["shadows"])
        outer.update(controls)
    assert set(FermionicShadowOptions.model_fields) == illustrated
    assert set(ShadowExperimentOptions.model_fields) == outer

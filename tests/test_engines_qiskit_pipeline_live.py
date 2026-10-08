"""Real worker execution, circuit handoff and cache reuse through the public pipeline."""

import shutil
import sys
from pathlib import Path

import pytest
import yaml

from chemrefine import pipeline
from chemrefine.config import load_config


@pytest.mark.integration
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_molecular_circuit_to_measurement_pipeline_and_cache(tmp_path, device):
    """A worker-produced molecular preparation remains consumable after pipeline recovery."""
    aer = pytest.importorskip("qiskit_aer")
    pytest.importorskip("qiskit_nature")
    requested = "GPU" if device == "cuda" else "CPU"
    if requested not in aer.AerSimulator().available_devices():
        pytest.skip(f"{requested} unavailable in the installed Aer build")
    from chemrefine.engines.qiskit.bundles import read_bundle

    source = Path(__file__).resolve().parents[1] / "examples/tutorials"
    for directory in ("qiskit_sp", "qiskit_handoffs"):
        shutil.copytree(
            source / directory,
            tmp_path / directory,
            ignore=shutil.ignore_patterns("outputs*", "__pycache__"),
        )
    path = tmp_path / "qiskit_handoffs/input.yaml"
    templates = tmp_path / "qiskit_sp/templates"
    shutil.copyfile(templates / "cpu.slurm.header", templates / "cuda.slurm.header")
    config = yaml.safe_load(path.read_text())
    for step in config["steps"]:
        step["options"].update(device=device, backend_python=sys.executable)
    config["steps"][0]["options"]["estimator"] = {
        "name": "aer_statevector",
        "options": {"seed_simulator": 7, "seed_transpiler": 7},
    }
    config["steps"][1]["options"]["experiment"]["options"]["sampler"] = {
        "name": "aer",
        "options": {"seed_simulator": 7, "seed_transpiler": 7},
    }
    path.write_text(yaml.safe_dump(config))
    resolved = load_config(path)
    outcomes = pipeline.run(resolved)
    assert len(outcomes) == 2
    assert outcomes[-1].state.structures[0].energy_hartree == pytest.approx(-1.1373060358, abs=1e-6)
    artifact = resolved.output_dir / "step2/experiment/artifact.json"
    bundle = read_bundle(artifact)
    assert bundle.description.kind == "pauli_measurement"
    assert list((resolved.output_dir / "step1").rglob("*.result.json"))
    tracked = {p: p.stat().st_mtime_ns for p in resolved.output_dir.rglob("*.npz")}
    assert tracked
    repeated = pipeline.run(load_config(path))
    assert (
        repeated[-1].state.structures[0].energy_hartree
        == outcomes[-1].state.structures[0].energy_hartree
    )
    assert all(p.stat().st_mtime_ns == before for p, before in tracked.items())

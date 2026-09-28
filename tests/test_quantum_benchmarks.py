"""Benchmark evidence must distinguish unsupported work, failures and warmup costs."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture
def benchmark(monkeypatch):
    """Load the standalone CLI as used from a source checkout."""
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "quantum_benchmarks", SCRIPTS / "quantum_benchmarks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matrix_covers_registered_algorithms_ansatze_mappers_and_optimizers(benchmark):
    from chemrefine.engines.qiskit.registry import REGISTRIES

    rows = benchmark.cases("molecular")
    for category in (
        "algorithm",
        "ansatz",
        "mapper",
        "optimizer",
        "initial_state",
        "initial_point",
    ):
        assert {row[category] for row in rows} == REGISTRIES[category].names()
    assert len({benchmark.case_id(row) for row in rows}) == len(rows)


def test_tutorial_matrix_covers_every_registered_experiment(benchmark):
    import yaml

    from chemrefine.engines.qiskit.experiment import EXPERIMENTS

    names = {
        step["options"]["experiment"]["name"]
        for case in benchmark.cases("tutorials")
        for step in yaml.safe_load((benchmark.ROOT / case["path"]).read_text())["steps"]
    }
    assert names == EXPERIMENTS.names()


def test_local_provider_matrix_includes_only_credential_free_runtime_modes(benchmark):
    from chemrefine.engines.qiskit.registry import ESTIMATORS, SAMPLERS

    rows = benchmark.cases("primitives") + benchmark.cases("runtime")
    for family, registry in (("estimator", ESTIMATORS), ("sampler", SAMPLERS)):
        assert {row["provider"] for row in rows if row["family"] == family} == registry.names()
    local = benchmark.cases("runtime")
    assert len(local) == 48
    assert all(row["fake_backend"] == "FakeManilaV2" for row in local)
    assert all("backend_name" not in row for row in local)


def test_regression_matrix_declares_required_ucc_and_spin_controls(benchmark):
    rows = benchmark.cases("regressions")
    explicit = [row for row in rows if row["ansatz"] == "ucc"]
    assert explicit and all(row["ansatz_options"]["excitations"] for row in explicit)
    sampled = [row for row in rows if row["algorithm"] == "skqd"]
    assert len(sampled) == 1 and sampled[0]["algorithm_options"]["symmetrize_spin"] is True


@pytest.mark.filterwarnings("ignore::DeprecationWarning:qiskit")
@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning:qiskit")
def test_explicit_ucc_benchmark_consumes_recorded_controls(benchmark):
    pytest.importorskip("qiskit_nature")
    pytest.importorskip("qiskit_aer")
    case = next(
        row
        for row in benchmark.cases("regressions")
        if row["ansatz"] == "ucc"
        and row["mapper"] == "jordan_wigner"
        and row["optimizer"] == "slsqp"
    )
    row = benchmark.worker({"case": case, "device": "cpu", "warmups": 0, "repeats": 1})["rows"][0]
    assert row["status"] == "ok", row.get("traceback", row)
    assert row["resolved_options"]["ansatz"]["options"] == case["ansatz_options"]
    assert row["energy_hartree"] == pytest.approx(-1.1373060358, abs=2e-3)


def test_summary_excludes_warmups_and_keeps_failed_repetitions(benchmark):
    base = {"case_id": "x", "device": "cpu", "family": "sampler", "status": "ok"}
    rows = [
        {**base, "phase": "warmup", "seconds": 1000},
        {**base, "phase": "measure", "seconds": 1},
        {**base, "phase": "measure", "seconds": 3},
        {**base, "phase": "failure", "status": "error", "seconds": None},
    ]
    summary = benchmark.summarize(rows)[0]
    assert summary["median_seconds"] == 2
    assert summary["samples"] == 2
    assert summary["status"] == "error,ok"


def test_worker_does_not_report_failure_as_speed_or_skip(benchmark, monkeypatch):
    import quantum_workloads

    def prepare(*args):
        def run():
            raise AssertionError("wrong quantum answer")

        return run, {}

    monkeypatch.setattr(quantum_workloads, "prepare", prepare)
    result = benchmark.worker(
        {"case": {"family": "sampler"}, "device": "cpu", "warmups": 0, "repeats": 2}
    )
    assert result["rows"][0]["status"] == "error"
    assert result["rows"][0]["seconds"] is None
    assert "wrong quantum answer" in result["rows"][0]["reason"]


def test_worker_records_unsupported_gpu_without_claiming_execution(benchmark):
    result = benchmark.worker(
        {
            "case": {"family": "sampler", "provider": "statevector"},
            "device": "cuda",
            "warmups": 0,
            "repeats": 1,
        }
    )
    row = result["rows"][0]
    assert row["status"] == "unsupported"
    assert row["seconds"] is None


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_actual_primitive_benchmark_checks_completed_provider_result(benchmark, device):
    aer = pytest.importorskip("qiskit_aer")
    requested = "GPU" if device == "cuda" else "CPU"
    if requested not in aer.AerSimulator().available_devices():
        pytest.skip(f"{requested} absent from installed Aer build")
    case = next(
        c
        for c in benchmark.cases("primitives")
        if c["provider"] == "aer_statevector" and c["precision"] == "double"
    )
    result = benchmark.worker({"case": case, "device": device, "warmups": 1, "repeats": 2})
    rows = result["rows"]
    assert len(rows) == 3
    assert all(row["status"] == "ok" and row["seconds"] > 0 for row in rows)
    assert all(row["absolute_error"] <= row["tolerance"] for row in rows)
    json.dumps(result, allow_nan=False)


def test_command_timeout_retains_failed_case(benchmark, monkeypatch, tmp_path):
    import subprocess

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("worker", 1)

    monkeypatch.setattr(benchmark, "snapshot", lambda: {})
    monkeypatch.setattr(benchmark, "cases", lambda _: [{"family": "sampler"}])
    monkeypatch.setattr(benchmark.subprocess, "run", timeout)
    output = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", ["benchmark", "--output", str(output)])
    assert benchmark.main() == 1
    row = json.loads((output / "samples.jsonl").read_text())
    assert row["status"] == "timeout" and row["seconds"] is None
    assert (output / "summary.csv").is_file()

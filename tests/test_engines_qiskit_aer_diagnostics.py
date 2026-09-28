"""Aer failures retain their provider status across sampler and estimator boundaries."""

import pickle
from types import SimpleNamespace

import pytest

from chemrefine.engines.qiskit import aer_diagnostics as diagnostics
from chemrefine.errors import ConfigError


class Backend:
    """Picklable backend with a synchronous public job for protocol tests."""

    def run(self, result, *, fail=False):
        """Either reject submission or return a controllable job."""
        if fail:
            raise ValueError("submission failed")
        return Job(result)


class Job:
    """Expose result and cancellation without any optional SDK."""

    def __init__(self, outcome):
        """Retain the exact outcome for pass-through and failure assertions."""
        self.outcome = outcome

    def result(self, **kwargs):
        """Return a result or reproduce a provider exception."""
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome

    def cancel(self):
        """Model the job-control method delegated by the wrapper."""
        return True


def test_aer_job_preserves_success_control_and_backend_pickling(monkeypatch):
    backend = Backend()
    monkeypatch.setattr(backend, "run", diagnostics.AerRun(backend, "statevector", "cpu", "double"))
    restored = pickle.loads(pickle.dumps(backend))
    result = SimpleNamespace(success=True, results=[SimpleNamespace(success=True)])
    job = restored.run(result)
    assert job.result(timeout=1) is result
    assert job.cancel() is True
    assert isinstance(restored.run, diagnostics.AerRun)


@pytest.mark.parametrize("top_success", [False, True])
def test_aer_failure_retains_experiment_error_and_configuration(monkeypatch, top_success):
    monkeypatch.setattr(
        diagnostics, "aer_build_identity", lambda: "qiskit-aer=0.17.2; build=cuda129"
    )
    backend = Backend()
    run = diagnostics.AerRun(backend, "tensor_network", "cuda", "single")
    result = SimpleNamespace(
        success=top_success,
        status="ERROR",
        results=[SimpleNamespace(success=False, status="CUTENSORNET_STATUS_INTERNAL_ERROR")],
    )
    with pytest.raises(ConfigError, match="CUTENSORNET_STATUS_INTERNAL_ERROR") as exc:
        run(result).result()
    message = str(exc.value)
    for expected in (
        "method=tensor_network",
        "device=cuda",
        "precision=single",
        "cuda129",
        "no fallback",
        "statevector",
    ):
        assert expected in message


def test_aer_submission_and_job_exceptions_remain_chained():
    run = diagnostics.AerRun(Backend(), "statevector", "cpu", "double")
    with pytest.raises(ConfigError, match="submission failed") as submitted:
        run(None, fail=True)
    assert isinstance(submitted.value.__cause__, ValueError)
    error = RuntimeError("provider failed")
    with pytest.raises(ConfigError, match="provider failed") as executed:
        run(error).result()
    assert executed.value.__cause__ is error
    with pytest.raises(ConfigError, match="no successful result"):
        run(object()).result()


def test_aer_build_identity_is_explicit_without_distribution_metadata(monkeypatch, tmp_path):
    def missing(_):
        raise diagnostics.PackageNotFoundError

    monkeypatch.setattr(diagnostics, "version", missing)
    monkeypatch.setattr(diagnostics.sys, "prefix", str(tmp_path))
    assert "qiskit-aer=unknown; build=pip/source" in diagnostics.aer_build_identity()
    metadata = tmp_path / "conda-meta"
    metadata.mkdir()
    record = metadata / "qiskit-aer-0.17.2.json"
    record.write_text('{"build": "cuda129_py312"}')
    assert "cuda129_py312" in diagnostics.aer_build_identity()
    record.write_text("invalid")
    assert "unknown Conda build" in diagnostics.aer_build_identity()


@pytest.mark.parametrize("provider", ["sampler", "estimator", "aer_statevector"])
def test_real_aer_primitive_reports_failure_before_postprocessing(monkeypatch, provider):
    aer = pytest.importorskip("qiskit_aer")
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp

    from chemrefine.engines.qiskit.components.estimators import (
        AerShotsEstimatorOptions,
        AerStatevectorEstimatorOptions,
        build_aer_shots_estimator,
        build_aer_statevector_estimator,
    )
    from chemrefine.engines.qiskit.components.samplers import AerSamplerOptions, build_aer_sampler

    def failed(self, *args, **kwargs):
        return SimpleNamespace(
            success=False, status="CUTENSORNET_STATUS_INTERNAL_ERROR", results=[]
        )

    monkeypatch.setattr(aer.AerSimulator, "_execute_circuits_job", failed)
    circuit = QuantumCircuit(1)
    if provider == "sampler":
        resource = build_aer_sampler(options=AerSamplerOptions(method="statevector"))
        circuit.measure_all()
        job = resource.sampler.run([circuit], shots=16)
    else:
        resource = (
            build_aer_shots_estimator(options=AerShotsEstimatorOptions())
            if provider == "estimator"
            else build_aer_statevector_estimator(options=AerStatevectorEstimatorOptions())
        )
        job = resource.estimator.run([(circuit, SparsePauliOp("Z"))])
    with pytest.raises(ConfigError, match="CUTENSORNET_STATUS_INTERNAL_ERROR"):
        job.result()

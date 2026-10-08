"""Check Aer results before primitive post-processing can obscure provider failures."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from chemrefine.errors import ConfigError


def aer_build_identity() -> str:
    """Describe the installed wheel or Conda build without importing an optional SDK."""
    try:
        installed = version("qiskit-aer")
    except PackageNotFoundError:
        installed = "unknown"
    record = next((Path(sys.prefix) / "conda-meta").glob("qiskit-aer-*.json"), None)
    build = "pip/source (compiled variant not recorded)"
    if record is not None:
        try:
            build = str(json.loads(record.read_text())["build"])
        except (OSError, ValueError, KeyError):
            build = "unknown Conda build"
    return f"qiskit-aer={installed}; build={build}"


@dataclass(frozen=True)
class AerRun:
    """Wrap only an owned backend's public run method, retaining its BackendV2 identity."""

    backend: Any
    method: str
    device: str
    precision: str

    def failure(self, status: str) -> ConfigError:
        """Keep original provider status and give method-specific remediation."""
        advice = "Verify the requested method with this Aer build; no fallback was performed."
        if self.method == "tensor_network" and self.device == "cuda":
            advice += (
                " For affected GPU tensor-network jobs, explicitly select statevector or "
                "density_matrix within its memory budget; inspect the CUDA/cuTensorNet stack."
            )
        return ConfigError(
            f"Aer simulation failed (method={self.method}, device={self.device}, "
            f"precision={self.precision}; {aer_build_identity()}): {status}. {advice}"
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Dispatch through the backend class, avoiding a bound-method pickle cycle."""
        try:
            job = type(self.backend).run(self.backend, *args, **kwargs)
        except Exception as exc:
            raise self.failure(str(exc)) from exc
        return CheckedAerJob(job, self)


@dataclass(frozen=True)
class CheckedAerJob:
    """Delegate job control while checking both job-level and experiment-level success."""

    job: Any
    context: AerRun

    def __getattr__(self, name: str) -> Any:
        """Preserve cancellation, status and other provider job operations."""
        return getattr(self.job, name)

    def result(self, *args: Any, **kwargs: Any) -> Any:
        """Raise the original failed Aer status before counts or memory are decoded."""
        try:
            result = self.job.result(*args, **kwargs)
        except Exception as exc:
            raise self.context.failure(str(exc)) from exc
        failures = [
            str(getattr(experiment, "status", "experiment reported failure"))
            for experiment in getattr(result, "results", ())
            if not experiment.success
        ]
        if not getattr(result, "success", False) or failures:
            status = str(getattr(result, "status", "backend returned no successful result"))
            raise self.context.failure("; ".join([status, *failures]))
        return result

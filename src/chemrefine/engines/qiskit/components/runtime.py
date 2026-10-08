"""Explicit, journaled IBM Runtime execution providers with lazy SDK construction."""

from __future__ import annotations

from typing import Literal, cast

from chemrefine.engines.api import BackendRequirement
from chemrefine.engines.qiskit.context import EstimatorResource, SamplerResource
from chemrefine.engines.qiskit.registry import ESTIMATORS, SAMPLERS
from chemrefine.engines.qiskit.runtime_options import RuntimeEstimatorOptions, RuntimeSamplerOptions


@ESTIMATORS.register(
    "ibm_runtime",
    RuntimeEstimatorOptions,
    backend_requirement=BackendRequirement(
        extra="qiskit-runtime", import_name="qiskit_ibm_runtime"
    ),
    status="experimental",
    supported_domains=(
        "IBM Runtime0.50 executor or legacyV2",
        "ISA circuits",
        "journaled remote jobs",
    ),
)
def build_runtime_estimator(
    *,
    options: RuntimeEstimatorOptions,
    device: Literal["cpu", "cuda"] = "cpu",
    cores: int = 1,
) -> EstimatorResource:
    """Build the configured estimator without reading credentials in the orchestrator."""
    from chemrefine.engines.qiskit.runtime import build_runtime_resource

    del cores
    return cast(
        "EstimatorResource", build_runtime_resource(options, kind="estimator", device=device)
    )


@SAMPLERS.register(
    "ibm_runtime",
    RuntimeSamplerOptions,
    backend_requirement=BackendRequirement(
        extra="qiskit-runtime", import_name="qiskit_ibm_runtime"
    ),
    status="experimental",
    supported_domains=(
        "IBM Runtime0.50 executor or legacyV2",
        "classified shots",
        "journaled remote jobs",
    ),
)
def build_runtime_sampler(
    *,
    options: RuntimeSamplerOptions,
    device: Literal["cpu", "cuda"] = "cpu",
    cores: int = 1,
) -> SamplerResource:
    """Build a real BaseSamplerV2 adapter usable by sampling and fidelity algorithms."""
    from chemrefine.engines.qiskit.runtime import build_runtime_resource

    del cores
    return cast("SamplerResource", build_runtime_resource(options, kind="sampler", device=device))

"""Local V2 sampler providers with explicit finite-shot semantics."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from chemrefine.engines.api import BackendRequirement
from chemrefine.engines.qiskit.components.estimators import (
    AerShotsEstimatorOptions,
    build_aer_backend,
)
from chemrefine.engines.qiskit.context import SamplerResource
from chemrefine.engines.qiskit.registry import SAMPLERS
from chemrefine.errors import ConfigError


class StatevectorSamplerOptions(BaseModel):
    """Seeded ideal sampling using Qiskit's reference statevector simulator."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    seed: int | None = None


class BackendSamplerOptions(BaseModel):
    """Reproducible simulation and compilation settings for the basic sampler."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    seed_simulator: int | None = None
    seed_transpiler: int | None = None
    optimization_level: int = Field(1, ge=0, le=3)


class AerSamplerOptions(BackendSamplerOptions):
    """Aer sampling with a declared simulation method and optional noise model."""

    method: Literal[
        "automatic", "statevector", "density_matrix", "matrix_product_state", "tensor_network"
    ] = "automatic"
    simulation_precision: Literal["single", "double"] = "double"
    noise_model: dict[str, object] | None = None


@SAMPLERS.register("statevector", StatevectorSamplerOptions)
def build_statevector_sampler(
    *, options: StatevectorSamplerOptions, device: Literal["cpu", "cuda"] = "cpu", cores: int = 1
) -> SamplerResource:
    """Build an ideal terminal sampler without an additional provider dependency."""
    del cores
    if device != "cpu":
        raise ConfigError("qiskit sampler 'statevector' supports device: cpu only")
    from qiskit.primitives import StatevectorSampler

    return SamplerResource(StatevectorSampler(seed=options.seed))


@SAMPLERS.register("basic_backend", BackendSamplerOptions)
def build_basic_sampler(
    *, options: BackendSamplerOptions, device: Literal["cpu", "cuda"] = "cpu", cores: int = 1
) -> SamplerResource:
    """Wrap the bundled basic simulator in the V2 sampler interface."""
    del cores
    if device != "cpu":
        raise ConfigError("qiskit sampler 'basic_backend' supports device: cpu only")
    from qiskit.primitives import BackendSamplerV2
    from qiskit.providers.basic_provider import BasicSimulator
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    backend = BasicSimulator()
    return SamplerResource(
        BackendSamplerV2(backend=backend, options={"seed_simulator": options.seed_simulator}),
        transpiler=generate_preset_pass_manager(
            backend=backend,
            optimization_level=options.optimization_level,
            seed_transpiler=options.seed_transpiler,
        ),
    )


@SAMPLERS.register(
    "aer",
    AerSamplerOptions,
    capabilities=frozenset({"cuda"}),
    backend_requirement=BackendRequirement(extra="qiskit-aer", import_name="qiskit_aer"),
)
def build_aer_sampler(
    *, options: AerSamplerOptions, device: Literal["cpu", "cuda"] = "cpu", cores: int = 1
) -> SamplerResource:
    """Use actual Aer shots and preserve logical measurement wiring when compiling."""
    from qiskit.primitives import BackendSamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    # Share Aer method/device/noise validation with the existing estimator provider.
    validated = AerShotsEstimatorOptions(**options.model_dump())
    backend = build_aer_backend(
        method=validated.method,
        device=device,
        cores=cores,
        simulation_precision=validated.simulation_precision,
        noise_model=validated.noise_model,
    )
    return SamplerResource(
        BackendSamplerV2(backend=backend, options={"seed_simulator": options.seed_simulator}),
        transpiler=generate_preset_pass_manager(
            backend=backend,
            optimization_level=options.optimization_level,
            seed_transpiler=options.seed_transpiler,
        ),
    )

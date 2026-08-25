"""Built-in V2 estimator factories and resource lifecycles."""

from __future__ import annotations

import warnings
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from chemrefine.engines.api import BackendRequirement
from chemrefine.engines.qiskit.context import EstimatorResource
from chemrefine.engines.qiskit.registry import ESTIMATORS
from chemrefine.errors import ConfigError

SimulationPrecision = Literal["single", "double"]
AerShotMethod = Literal[
    "automatic",
    "statevector",
    "density_matrix",
    "matrix_product_state",
    "tensor_network",
]


class StatevectorEstimatorOptions(BaseModel):
    """Options for the exact local statevector estimator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    default_precision: float = Field(0.0, ge=0)
    seed: int | None = None


class BasicBackendEstimatorOptions(BaseModel):
    """Shot-based options for Qiskit's bundled BasicSimulator backend."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    backend_name: str = "basic_simulator"
    default_precision: float = Field(0.015625, gt=0)
    abelian_grouping: bool = True
    seed_simulator: int | None = None
    optimization_level: int = Field(1, ge=0, le=3)


class AerStatevectorEstimatorOptions(BaseModel):
    """Options for Aer expectation values calculated from a statevector."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    default_precision: float = Field(0.0, ge=0)
    seed_simulator: int | None = None
    simulation_precision: SimulationPrecision = "double"
    optimization_level: int = Field(1, ge=0, le=3)
    seed_transpiler: int | None = None


class AerShotsEstimatorOptions(BaseModel):
    """Options for finite-shot measurements performed by AerSimulator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: AerShotMethod = "automatic"
    default_precision: float = Field(0.015625, gt=0)
    abelian_grouping: bool = True
    seed_simulator: int | None = None
    simulation_precision: SimulationPrecision = "double"
    optimization_level: int = Field(1, ge=0, le=3)
    seed_transpiler: int | None = None
    noise_model: dict[str, Any] | None = None


@ESTIMATORS.register("statevector", StatevectorEstimatorOptions)
def build_statevector_estimator(
    *,
    options: StatevectorEstimatorOptions,
    device: Literal["cpu", "cuda"] = "cpu",
    cores: int = 1,
) -> EstimatorResource:
    """Build Qiskit's deterministic local BaseEstimatorV2 implementation."""
    from qiskit.primitives import StatevectorEstimator

    if device != "cpu":
        raise ConfigError("qiskit estimator 'statevector' supports device: cpu only")
    return EstimatorResource(
        StatevectorEstimator(
            default_precision=options.default_precision,
            seed=options.seed,
        )
    )


@ESTIMATORS.register("basic_backend", BasicBackendEstimatorOptions)
def build_basic_backend_estimator(
    *,
    options: BasicBackendEstimatorOptions,
    device: Literal["cpu", "cuda"] = "cpu",
    cores: int = 1,
) -> EstimatorResource:
    """Wrap Qiskit's dependency-free BasicSimulator in BackendEstimatorV2."""
    from qiskit.primitives import BackendEstimatorV2
    from qiskit.providers.basic_provider import BasicProvider
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    if device != "cpu":
        raise ConfigError("qiskit estimator 'basic_backend' supports device: cpu only")
    backend = BasicProvider().get_backend(options.backend_name)
    estimator = BackendEstimatorV2(
        backend=backend,
        options={
            "default_precision": options.default_precision,
            "abelian_grouping": options.abelian_grouping,
            "seed_simulator": options.seed_simulator,
        },
    )
    return EstimatorResource(
        estimator,
        transpiler=generate_preset_pass_manager(
            backend=backend,
            optimization_level=options.optimization_level,
        ),
    )


def _aer_backend(
    *,
    method: AerShotMethod | Literal["statevector"],
    device: Literal["cpu", "cuda"],
    cores: int,
    simulation_precision: SimulationPrecision,
    noise_model: dict[str, Any] | None = None,
) -> Any:
    """Build and validate one AerSimulator without importing Aer at discovery time."""
    from qiskit_aer import AerError, AerSimulator

    backend_options: dict[str, Any] = {
        "method": method,
        "device": {"cpu": "CPU", "cuda": "GPU"}[device],
        "precision": simulation_precision,
        "max_parallel_threads": cores,
    }
    if noise_model is not None:
        from qiskit_aer.noise import NoiseModel

        try:
            # Aer 0.17 retains this serialized-model boundary but deprecates the
            # constructor.  Keep the suppression narrow; the <0.18 dependency cap
            # prevents a silent removal before this adapter is migrated.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*from_dict.*deprecated.*",
                    category=DeprecationWarning,
                )
                backend_options["noise_model"] = NoiseModel.from_dict(noise_model)
        except (AerError, TypeError, ValueError) as exc:
            raise ConfigError(f"invalid qiskit Aer noise_model: {exc}") from exc
    try:
        backend = AerSimulator(**backend_options)
        requested_device = backend_options["device"]
        available_devices = backend.available_devices()
        available_methods = backend.available_methods()
    except (AerError, TypeError, ValueError) as exc:
        raise ConfigError(f"cannot configure qiskit Aer simulator: {exc}") from exc
    if requested_device not in available_devices:
        raise ConfigError(
            f"qiskit Aer device {requested_device!r} is unavailable "
            f"(available: {list(available_devices)}); install a compatible Aer GPU package "
            "or set device: cpu"
        )
    if method not in available_methods:
        raise ConfigError(
            f"qiskit Aer method {method!r} is unavailable in this Aer installation "
            f"(available: {list(available_methods)})"
        )
    return backend


def _preset_pass_manager(
    backend: Any,
    *,
    optimization_level: int,
    seed_transpiler: int | None,
) -> Any:
    """Build the transpiler shared by both Aer estimator resources."""
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    options: dict[str, Any] = {
        "backend": backend,
        "optimization_level": optimization_level,
    }
    if seed_transpiler is not None:
        options["seed_transpiler"] = seed_transpiler
    return generate_preset_pass_manager(**options)


@ESTIMATORS.register(
    "aer_statevector",
    AerStatevectorEstimatorOptions,
    capabilities=frozenset({"cuda"}),
    backend_requirement=BackendRequirement(extra="qiskit-aer", import_name="qiskit_aer"),
)
def build_aer_statevector_estimator(
    *,
    options: AerStatevectorEstimatorOptions,
    device: Literal["cpu", "cuda"] = "cpu",
    cores: int = 1,
) -> EstimatorResource:
    """Build Aer's statevector expectation-value EstimatorV2."""
    from qiskit_aer.primitives import EstimatorV2

    backend = _aer_backend(
        method="statevector",
        device=device,
        cores=cores,
        simulation_precision=options.simulation_precision,
    )
    run_options = (
        {"seed_simulator": options.seed_simulator} if options.seed_simulator is not None else {}
    )
    estimator = EstimatorV2.from_backend(
        backend,
        options={
            "default_precision": options.default_precision,
            "run_options": run_options,
        },
    )
    return EstimatorResource(
        estimator,
        transpiler=_preset_pass_manager(
            backend,
            optimization_level=options.optimization_level,
            seed_transpiler=options.seed_transpiler,
        ),
    )


@ESTIMATORS.register(
    "aer_shots",
    AerShotsEstimatorOptions,
    capabilities=frozenset({"cuda"}),
    backend_requirement=BackendRequirement(extra="qiskit-aer", import_name="qiskit_aer"),
)
def build_aer_shots_estimator(
    *,
    options: AerShotsEstimatorOptions,
    device: Literal["cpu", "cuda"] = "cpu",
    cores: int = 1,
) -> EstimatorResource:
    """Build a finite-shot BackendEstimatorV2 around AerSimulator."""
    from qiskit.primitives import BackendEstimatorV2

    backend = _aer_backend(
        method=options.method,
        device=device,
        cores=cores,
        simulation_precision=options.simulation_precision,
        noise_model=options.noise_model,
    )
    estimator = BackendEstimatorV2(
        backend=backend,
        options={
            "default_precision": options.default_precision,
            "abelian_grouping": options.abelian_grouping,
            "seed_simulator": options.seed_simulator,
        },
    )
    return EstimatorResource(
        estimator,
        transpiler=_preset_pass_manager(
            backend,
            optimization_level=options.optimization_level,
            seed_transpiler=options.seed_transpiler,
        ),
    )

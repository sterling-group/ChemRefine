"""Durable YAML resource-estimation experiments, separate from energy solvers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

from pydantic import Field, model_validator

from chemrefine.engines.api import BackendRequirement
from chemrefine.engines.qiskit.experiment import EXPERIMENTS, ExperimentResult
from chemrefine.engines.qiskit.factorized_resources import (
    FactorizedResourceOptions,
    estimate_factorized_resources,
    load_thc_factors,
)
from chemrefine.engines.qiskit.resources import (
    PauliResourceOptions,
    SurfaceCodeOptions,
    estimate_pauli_resources,
    estimate_surface_code,
)


class FactorizedResourceExperimentOptions(FactorizedResourceOptions):
    """External integral and optional THC bundles with relocation-safe dependencies."""

    integral_bundle_path: str = Field(
        min_length=1, json_schema_extra={"input_file": True, "file_format": "quantum_bundle"}
    )
    thc_bundle_path: str | None = Field(
        None,
        min_length=1,
        json_schema_extra={"input_file": True, "file_format": "quantum_bundle"},
    )

    @model_validator(mode="after")
    def _factor_input(self) -> Self:
        """Make the chosen representation consume exactly its advertised inputs."""
        if (self.method == "thc") != (self.thc_bundle_path is not None):
            raise ValueError("thc_bundle_path is required exactly when method=thc")
        return self


@EXPERIMENTS.register(
    "pauli_resources",
    PauliResourceOptions,
    status="experimental",
    supported_domains=("finite Hermitian Pauli sums", "analytical QPE query bounds"),
)
def pauli_resource_experiment(
    *, options: PauliResourceOptions, **_context: Any
) -> ExperimentResult:
    """Persist numerical query bounds without generating a circuit or energy result."""
    return ExperimentResult(
        kind="resource_estimate", arrays={}, metadata=estimate_pauli_resources(options)
    )


@EXPERIMENTS.register(
    "factorized_resources",
    FactorizedResourceExperimentOptions,
    status="experimental",
    supported_domains=(
        "real restricted spatial integrals",
        "DF and supplied THC factors",
        "OpenFermion cost formulae and optional Qualtran DF cost graphs",
    ),
    backend_requirement=BackendRequirement(extra="qiskit-resources", import_name="openfermion"),
)
def factorized_resource_experiment(
    *, options: FactorizedResourceExperimentOptions, **context: Any
) -> ExperimentResult:
    """Load tracked numeric bundles inside the isolated resource worker."""
    from pyscf.lib import with_omp_threads

    from chemrefine.engines.qiskit.integral_io import load_integrals

    data = load_integrals(Path(options.integral_bundle_path), max_bytes=options.max_working_bytes)
    factors = (
        None
        if options.thc_bundle_path is None
        else load_thc_factors(Path(options.thc_bundle_path), max_bytes=options.max_working_bytes)
    )
    numerical_options = FactorizedResourceOptions.model_validate(
        options.model_dump(exclude={"integral_bundle_path", "thc_bundle_path"})
    )
    with with_omp_threads(context["cores"]):
        result = estimate_factorized_resources(data, numerical_options, thc_factors=factors)
    return ExperimentResult(kind="resource_estimate", arrays={}, metadata=result)


@EXPERIMENTS.register(
    "surface_code_resources",
    SurfaceCodeOptions,
    status="experimental",
    supported_domains=(
        "explicit workload and below-threshold phenomenological surface code",
        "declared T/CCZ factory footprints and throughputs",
    ),
)
def surface_code_resource_experiment(
    *, options: SurfaceCodeOptions, **_context: Any
) -> ExperimentResult:
    """Persist a machine-assumption-dependent physical resource estimate."""
    return ExperimentResult(
        kind="resource_estimate", arrays={}, metadata=estimate_surface_code(options)
    )

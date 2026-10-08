"""Artifact adapter for bounded circuit cutting and durable signed reconstruction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from chemrefine.engines.api import BackendRequirement
from chemrefine.engines.qiskit.cutting import CuttingOptions, execute_cutting, plan_cutting
from chemrefine.engines.qiskit.experiment import EXPERIMENTS, ExperimentResult
from chemrefine.engines.qiskit.experiment_measurement import PauliCircuitInput, read_circuit
from chemrefine.engines.qiskit.options import ComponentSelection
from chemrefine.errors import ConfigError


class CuttingExperimentOptions(PauliCircuitInput):
    """A bound preparation circuit, Hermitian Pauli sum and explicit cutting controls."""

    cutting: CuttingOptions = Field(default_factory=CuttingOptions)
    sampler: ComponentSelection = Field(
        default_factory=lambda: ComponentSelection.named("basic_backend")
    )

    @field_validator("sampler", mode="before")
    @classmethod
    def _sampler_name(cls, value: Any) -> Any:
        """Accept the standard named-component shorthand."""
        return {"name": value} if isinstance(value, str) else value


@EXPERIMENTS.register(
    "circuit_cutting",
    CuttingExperimentOptions,
    requires=frozenset({"sampler"}),
    capabilities=frozenset({"cuda"}),
    status="experimental",
    supported_domains=(
        "bound qubit circuits and Hermitian Pauli observables",
        "manual gate/wire, named partitions and automatic width-constrained cuts",
    ),
    backend_requirement=BackendRequirement(
        extra="qiskit-cutting", import_name="qiskit_addon_cutting"
    ),
)
def cutting_experiment(*, options: CuttingExperimentOptions, **context: Any) -> ExperimentResult:
    """Plan before execution, enforcing output capacity before any quantum submission."""
    circuit = read_circuit(Path(options.circuit_path), max_bytes=options.max_circuit_bytes)
    plan = plan_cutting(circuit, options.observable, options.cutting)
    if (
        plan.metadata["shot_record_bytes_bound"]
        + plan.metadata["generated_qpy_bytes"]
        + 8 * (len(plan.coefficients) + 2 * len(plan.observable))
        > context["max_output_bytes"]
    ):
        raise ConfigError("prospective cutting artifact exceeds max_output_bytes")
    result = execute_cutting(
        plan, options.sampler, device=context["device"], cores=context["cores"]
    )
    return ExperimentResult(kind="circuit_cutting", arrays=result.arrays, metadata=result.metadata)

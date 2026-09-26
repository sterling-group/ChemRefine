"""Typed artifact-engine configuration for variational circuit trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from chemrefine.engines.qiskit.dynamics import VariationalDynamicsOptions, variational_dynamics
from chemrefine.engines.qiskit.experiment import EXPERIMENTS, ExperimentResult
from chemrefine.engines.qiskit.experiment_measurement import PauliCircuitInput, read_circuit
from chemrefine.engines.qiskit.options import ComponentSelection


class VariationalExperimentOptions(PauliCircuitInput):
    """A parameterized QPY circuit and a Hermitian Hamiltonian for McLachlan dynamics."""

    initial_parameters: tuple[float, ...]
    observables: tuple[dict[str, float], ...] = Field((), max_length=128)
    dynamics: VariationalDynamicsOptions = Field(default_factory=VariationalDynamicsOptions)
    estimator: ComponentSelection = Field(
        default_factory=lambda: ComponentSelection.named("statevector")
    )

    @field_validator("estimator", mode="before")
    @classmethod
    def _estimator_name(cls, value: Any) -> Any:
        """Accept the same provider shorthand as other quantum workflows."""
        return {"name": value} if isinstance(value, str) else value

    @field_validator("observables")
    @classmethod
    def _observable_inputs(
        cls, values: tuple[dict[str, float], ...]
    ) -> tuple[dict[str, float], ...]:
        """Apply the same SDK-free Pauli contract to every additional observable."""
        for value in values:
            PauliCircuitInput(circuit_path="unused", observable=value)
        return values


@EXPERIMENTS.register(
    "variational_dynamics",
    VariationalExperimentOptions,
    requires=frozenset({"estimator"}),
    capabilities=frozenset({"cuda"}),
    status="experimental",
    supported_domains=(
        "fixed parameterized qubit circuits",
        "time-independent Hermitian Pauli Hamiltonians",
        "VarQITE and VarQRTE with McLachlan geometry",
    ),
)
def variational_experiment(
    *, options: VariationalExperimentOptions, **context: Any
) -> ExperimentResult:
    """Run the selected estimator and retain trajectory arrays without molecular energies."""
    from qiskit.quantum_info import SparsePauliOp

    circuit = read_circuit(Path(options.circuit_path), max_bytes=options.max_circuit_bytes)
    result = variational_dynamics(
        circuit,
        SparsePauliOp.from_list(list(options.observable.items())),
        options.initial_parameters,
        options=options.dynamics,
        observables=tuple(
            SparsePauliOp.from_list(list(item.items())) for item in options.observables
        ),
        estimator=options.estimator,
        cores=context["cores"],
        device=context["device"],
    )
    return ExperimentResult(
        "variational_trajectory",
        arrays={
            "times": result.times,
            "parameters": result.parameters,
            "expectations": result.expectations,
            "metric_diagnostics": result.metric_diagnostics,
        },
        metadata=result.metadata,
    )

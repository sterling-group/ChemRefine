"""Execution contracts for solvers that consume a prepared fermionic problem."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from chemrefine.engines.qiskit.options import QiskitOptions
from chemrefine.engines.qiskit.problem import PreparedProblem
from chemrefine.engines.qiskit.registry import REGISTRIES
from chemrefine.engines.qiskit.reporting import real_energy
from chemrefine.engines.qiskit.result import QiskitRunResult
from chemrefine.errors import ConfigError


@dataclass(frozen=True)
class NativeSolveRequest:
    """Prepared chemistry and caller controls, without mandatory Pauli mapping."""

    prepared: PreparedProblem
    options: QiskitOptions
    initial_point: Sequence[float] | None = None
    callback: Callable[[dict[str, Any]], None] | None = None
    reference_energy_hartree: float | None = None


@dataclass(frozen=True)
class NativeOutcome:
    """An active-Hamiltonian energy with algorithm-specific diagnostics.

    The energy excludes every scalar constant in ``PreparedProblem.energy_offsets``.
    The workflow restores those constants exactly once. ``converged`` describes the
    numerical stopping criterion, never a guarantee of chemical accuracy.
    """

    active_energy_hartree: float
    diagnostics: dict[str, Any] = field(default_factory=dict)
    converged: bool | None = None
    termination_reason: str | None = None
    num_qubits: int | None = None
    ansatz: str | None = None
    optimizer: str | None = None
    parameter_count: int | None = None
    optimizer_evaluations: int | None = None
    evaluations: list[dict[str, Any]] = field(default_factory=list)


def summarize_native(
    outcome: NativeOutcome, request: NativeSolveRequest, *, runtime_seconds: float
) -> QiskitRunResult:
    """Normalize native solvers to the existing pipeline energy convention."""
    if not isinstance(outcome, NativeOutcome):
        raise ConfigError("qiskit native algorithm must return NativeOutcome")
    prepared, options = request.prepared, request.options
    active = real_energy(outcome.active_energy_hartree, "active electronic energy")
    offsets = {
        name: real_energy(value, f"{name} offset")
        for name, value in prepared.energy_offsets.items()
    }
    nuclear = offsets.get("nuclear_repulsion_energy")
    electronic = active + sum(
        value for name, value in offsets.items() if name != "nuclear_repulsion_energy"
    )
    energy = real_energy(electronic + (nuclear or 0.0), "reported energy")
    versions = {}
    for package in (
        "qiskit",
        "qiskit-nature",
        "qiskit-algorithms",
        "qiskit-aer",
        "pyscf",
        "ffsim",
        "qiskit-fermions",
        "qiskit-addon-sqd",
    ):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            continue
    active_space = {
        "original_num_spatial_orbitals": prepared.original_num_spatial_orbitals,
        "active_orbitals": list(prepared.active_orbitals),
        "num_particles": list(prepared.num_particles),
        "energy_offsets": offsets,
        "transformations": prepared.metadata.get("transformations", []),
    }
    metadata = {
        "components": {
            category: {
                "name": selection.name,
                "options": REGISTRIES[category].options_for(selection).model_dump(mode="json"),
            }
            for category, selection in options.component_selections().items()
        },
        "execution": "native",
        "energy_convention": "total" if nuclear is not None else "electronic",
        "active_energy_hartree": active,
        "evaluations": outcome.evaluations,
        "solver": outcome.diagnostics,
        "prepared_active_space": active_space,
        "provenance": {
            **prepared.provenance,
            "problem_metadata": prepared.metadata,
            "package_versions": versions,
        },
    }
    result = QiskitRunResult(
        energy_hartree=energy,
        metadata=metadata,
        solver=options.algorithm.name,
        electronic_energy_hartree=electronic,
        total_energy_hartree=energy if nuclear is not None else None,
        nuclear_repulsion_energy_hartree=nuclear,
        reference_energy_hartree=request.reference_energy_hartree,
        energy_error_hartree=energy - request.reference_energy_hartree
        if request.reference_energy_hartree is not None
        else None,
        converged=outcome.converged,
        success=True,
        runtime_seconds=runtime_seconds,
        num_qubits=outcome.num_qubits,
        num_spin_orbitals=prepared.num_spin_orbitals,
        num_particles=prepared.num_particles,
        active_space=active_space,
        ansatz=outcome.ansatz,
        optimizer=outcome.optimizer,
        parameter_count=outcome.parameter_count,
        optimizer_evaluations=outcome.optimizer_evaluations,
        energy_evaluation_count=len(outcome.evaluations),
        termination_reason=outcome.termination_reason,
    )
    # Check the complete artifact before it can be written or enter the result cache.
    try:
        result.as_dict()
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            "qiskit native result must contain finite JSON-compatible diagnostics"
        ) from exc
    return result

"""Translate backend results into ChemRefine energies, diagnostics, and resource records."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np

from chemrefine.engines.qiskit.context import (
    AlgorithmArtifacts,
    AnsatzArtifacts,
    ElectronicStructureContext,
)
from chemrefine.engines.qiskit.metrics import logical_circuit_metrics, transpiled_circuit_metrics
from chemrefine.engines.qiskit.options import QiskitOptions
from chemrefine.engines.qiskit.registry import REGISTRIES
from chemrefine.engines.qiskit.result import QiskitRunResult
from chemrefine.errors import ConfigError


def jsonable(value: Any) -> Any:
    """Convert Qiskit/numpy diagnostics to plain JSON-compatible values."""
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    return str(value)


def result_metadata(
    result: Any,
    options: QiskitOptions,
    context: ElectronicStructureContext,
    evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Preserve the existing sidecar shape and append problem/mapping provenance."""
    raw = getattr(result, "raw_result", None)
    diagnostics = {
        name: jsonable(value)
        for name in (
            "cost_function_evals",
            "num_iterations",
            "optimal_point",
            "optimal_value",
            "termination_criterion",
            "final_max_gradient",
            "eigenvalue_history",
        )
        if (value := getattr(raw, name, None)) is not None
    }
    versions = {}
    for package in ("qiskit", "qiskit-nature", "qiskit-algorithms", "qiskit-aer", "pyscf"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            continue
    return {
        "components": {
            category: {
                "name": selection.name,
                "options": REGISTRIES[category].options_for(selection).model_dump(mode="json"),
            }
            for category, selection in options.component_selections().items()
        },
        "basis": options.basis,
        "device": options.device,
        "cores": options.cores,
        "active_space": options.active_space.model_dump(mode="json")
        if options.active_space
        else None,
        "freeze_core": options.freeze_core,
        "num_spatial_orbitals": context.num_spatial_orbitals,
        "num_particles": list(context.num_particles),
        "multiplicity": context.multiplicity,
        "num_qubits": context.num_qubits,
        "evaluations": evaluations,
        "solver": diagnostics,
        "mapping": context.mapping_metadata,
        "prepared_active_space": context.active_space_metadata,
        "provenance": {**context.provenance, "package_versions": versions},
    }


def _real_energy(value: Any, label: str) -> float:
    """Require one finite real energy instead of silently accepting solver failure."""
    try:
        if isinstance(value, bool | np.bool_):
            raise ValueError("boolean energy")
        energy = complex(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Qiskit solver returned an invalid {label}: {value!r}") from exc
    if not np.isfinite(energy.imag):
        raise ConfigError(f"Qiskit solver returned non-finite {label} {energy!r}")
    if abs(energy.imag) > 1e-10:
        raise ConfigError(f"Qiskit solver returned a complex {label} {energy!r}")
    if not np.isfinite(energy.real):
        raise ConfigError(f"Qiskit solver returned non-finite {label} {energy.real!r}")
    return float(energy.real)


def summarize_result(
    result: Any,
    *,
    options: QiskitOptions,
    context: ElectronicStructureContext,
    evaluations: list[dict[str, Any]],
    ansatz: AnsatzArtifacts,
    algorithm: AlgorithmArtifacts,
    runtime_seconds: float,
    reference_energy_hartree: float | None,
    was_transpiled: bool = False,
    operator_pool_supplied: bool = False,
) -> QiskitRunResult:
    """Extract owned result fields, leaving unavailable convergence/resources unknown."""
    totals = getattr(result, "total_energies", None)
    if totals is None or len(totals) == 0:
        raise ConfigError("Qiskit solver returned no total ground-state energy")
    energy = _real_energy(totals[0], "total energy")
    electronic = getattr(result, "electronic_energies", None)
    electronic_energy = (
        _real_energy(electronic[0], "electronic energy")
        if electronic is not None and len(electronic)
        else None
    )
    nuclear = getattr(result, "nuclear_repulsion_energy", None)
    nuclear = _real_energy(nuclear, "nuclear repulsion energy") if nuclear is not None else None
    raw = getattr(result, "raw_result", None)
    termination = getattr(raw, "termination_criterion", None)
    reason = getattr(termination, "name", str(termination)) if termination is not None else None
    converged: bool | None = None
    if options.algorithm.name == "exact":
        converged, reason = True, "exact_diagonalization"
    elif options.algorithm.name == "adapt_vqe":
        converged = reason == "CONVERGED" if reason is not None else None
    else:
        verdict = getattr(getattr(raw, "optimizer_result", None), "success", None)
        converged = bool(verdict) if isinstance(verdict, bool | np.bool_) else None
        reason = "optimizer_returned" if reason is None else reason
    optimal_circuit = getattr(raw, "optimal_circuit", None)
    # VQE's result contains the compiled circuit when a transpiler was supplied.
    # ADAPT does not expose its final uncompiled circuit in the supported API.
    circuit = (
        (None if was_transpiled else optimal_circuit)
        if options.algorithm.name == "adapt_vqe"
        else ansatz.circuit
    )
    metrics = logical_circuit_metrics(circuit)
    compiled_metrics = transpiled_circuit_metrics(optimal_circuit) if was_transpiled else None
    available_metrics = metrics or compiled_metrics
    trace = algorithm.diagnostics
    metadata = result_metadata(result, options, context, evaluations)
    metadata["energy_convention"] = "total" if nuclear is not None else "electronic"
    metadata["optimizer_evaluations_scope"] = (
        "last_retained_inner_vqe" if options.algorithm.name == "adapt_vqe" else "solver"
    )
    if options.algorithm.name == "adapt_vqe":
        metadata["operator_pool_source"] = (
            "external" if operator_pool_supplied else options.ansatz.name
        )
    if was_transpiled:
        metadata["transpilation"] = {
            "estimator": metadata["components"]["estimator"],
            "device": options.device,
            "scope": "configured estimator target; not a hardware resource estimate",
        }
    return QiskitRunResult(
        energy_hartree=energy,
        metadata=metadata,
        solver=options.algorithm.name,
        electronic_energy_hartree=electronic_energy,
        total_energy_hartree=energy if nuclear is not None else None,
        nuclear_repulsion_energy_hartree=nuclear,
        reference_energy_hartree=reference_energy_hartree,
        energy_error_hartree=energy - reference_energy_hartree
        if reference_energy_hartree is not None
        else None,
        converged=converged,
        success=True,
        runtime_seconds=runtime_seconds,
        num_qubits=context.num_qubits,
        num_qubits_before_reduction=context.num_qubits_before_reduction,
        num_spin_orbitals=2 * context.num_spatial_orbitals,
        num_particles=context.num_particles,
        num_pauli_terms=context.num_pauli_terms,
        mapping=options.mapper.name,
        active_space=context.active_space_metadata or metadata["active_space"],
        ansatz=("external_pool" if operator_pool_supplied else options.ansatz.name)
        if ansatz.circuit is not None or ansatz.operator_pool is not None
        else None,
        optimizer=options.optimizer.name if evaluations else None,
        parameter_count=available_metrics.parameter_count if available_metrics else None,
        optimizer_evaluations=getattr(raw, "cost_function_evals", None),
        energy_evaluation_count=len(evaluations),
        adapt_iterations=getattr(raw, "num_iterations", None)
        if options.algorithm.name == "adapt_vqe"
        else None,
        adapt_pool_size=len(ansatz.operator_pool)
        if options.algorithm.name == "adapt_vqe" and ansatz.operator_pool is not None
        else None,
        adapt_selected_operators=trace.selected_operators if trace is not None else None,
        adapt_gradient_history=tuple(trace.gradient_history) if trace is not None else None,
        termination_reason=reason,
        logical_circuit_metrics=metrics,
        transpiled_circuit_metrics=compiled_metrics,
    )

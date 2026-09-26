"""Compose existing registries around independently prepared electronic problems."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from numbers import Real
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from chemrefine.engines.qiskit import components as _builtins  # noqa: F401
from chemrefine.engines.qiskit.assembly import assemble_components
from chemrefine.engines.qiskit.context import ElectronicStructureContext
from chemrefine.engines.qiskit.mapping import map_problem
from chemrefine.engines.qiskit.native import NativeSolveRequest, summarize_native
from chemrefine.engines.qiskit.operators import OperatorPool
from chemrefine.engines.qiskit.options import QiskitOptions
from chemrefine.engines.qiskit.problem import PreparedProblem, prepare_pyscf_problem
from chemrefine.engines.qiskit.registry import (
    ALGORITHMS,
    validate_component_graph,
)
from chemrefine.engines.qiskit.reporting import jsonable, summarize_result
from chemrefine.engines.qiskit.result import QiskitRunResult as QiskitRunResult
from chemrefine.errors import ConfigError

logger = logging.getLogger(__name__)
EvaluationCallback = Callable[[dict[str, Any]], None]


def validate_options(options: QiskitOptions) -> None:
    """Fail fast on unknown components, bad knobs, or incompatible artifacts."""
    validate_component_graph(options)


def _solve_context(
    context: ElectronicStructureContext,
    resolved: QiskitOptions,
    *,
    operator_pool: OperatorPool | None,
    supplied_initial_point: Sequence[float] | None,
    user_callback: EvaluationCallback | None,
    reference_energy_hartree: float | None,
    started: float,
) -> QiskitRunResult:
    """Assemble components once and own the estimator lifecycle for a single solve."""
    evaluations: list[dict[str, Any]] = []
    previous_algorithm_evaluation: int | None = None
    inner_run = 0

    def callback(
        evaluation: int,
        _parameters: np.ndarray,
        mean: float,
        metadata: dict[str, Any],
    ) -> None:
        nonlocal inner_run, previous_algorithm_evaluation
        algorithm_evaluation = int(evaluation)
        if (
            previous_algorithm_evaluation is None
            or algorithm_evaluation <= previous_algorithm_evaluation
        ):
            inner_run += 1
        evaluations.append(
            {
                # The callback counter resets for each inner VQE in ADAPT. Keep a
                # workflow-global index while retaining the algorithm-provided value.
                "evaluation": len(evaluations) + 1,
                "inner_run": inner_run,
                "algorithm_evaluation": algorithm_evaluation,
                # VQE evaluates the mapped electronic Hamiltonian here; nuclear
                # repulsion and transformer constants are added to the final total.
                "objective_value_hartree": float(mean),
                "metadata": jsonable(metadata),
            }
        )
        previous_algorithm_evaluation = algorithm_evaluation
        logger.debug("Qiskit energy evaluation %d: %.12f hartree", len(evaluations), mean)
        if user_callback is not None:
            user_callback(deepcopy(evaluations[-1]))

    with assemble_components(
        context,
        resolved,
        operator_pool=operator_pool,
        initial_point=supplied_initial_point,
        callback=callback,
    ) as assembled:
        algorithm = ALGORITHMS.build(
            resolved.algorithm,
            context=context,
            components=assembled,
        )
        from qiskit_nature.second_q.algorithms import GroundStateEigensolver

        result = GroundStateEigensolver(context.mapper, algorithm.solver).solve(context.problem)

    summary = summarize_result(
        result,
        options=resolved,
        context=context,
        evaluations=evaluations,
        ansatz=assembled.ansatz,
        algorithm=algorithm,
        runtime_seconds=perf_counter() - started,
        reference_energy_hartree=reference_energy_hartree,
        was_transpiled=assembled.transpiler is not None,
        operator_pool_supplied=operator_pool is not None,
    )
    if resolved.circuit_export is not None:
        from chemrefine.engines.qiskit.circuit_io import bound_circuit

        raw: Any = result.raw_result
        if resolved.algorithm.name == "adapt_vqe":
            logical = algorithm.solver.retained_logical_circuit()
            parameters = raw.optimal_point
        else:
            logical = assembled.ansatz.circuit
            parameters = raw.optimal_parameters
        summary = replace(summary, circuits=(bound_circuit(context, logical, parameters),))
    logger.info(
        "Qiskit %s finished: %.12f hartree", resolved.algorithm.name, summary.energy_hartree
    )
    if summary.energy_error_hartree is not None:
        logger.info(
            "Qiskit error against supplied reference: %.8g hartree", summary.energy_error_hartree
        )
    return summary


def run_problem(
    prepared: PreparedProblem,
    *,
    options: QiskitOptions | Mapping[str, Any] | None = None,
    operator_pool: OperatorPool | None = None,
    initial_point: Sequence[float] | None = None,
    callback: EvaluationCallback | None = None,
    reference_energy_hartree: float | None = None,
) -> QiskitRunResult:
    """Solve a prepared problem using the same component graph as a pipeline job.

    Supplied pools replace the configured ansatz's pool for ADAPT only. Reference
    energy is an explicitly supplied comparison, never a request to perform an
    unbounded exact solve. Callback records use electronic-Hamiltonian objective
    energies, excluding inactive and nuclear constants; they are evaluation
    records rather than guaranteed optimizer iterations or parameter trajectories.
    """
    started = perf_counter()
    resolved = options if isinstance(options, QiskitOptions) else QiskitOptions.from_raw(options)
    if operator_pool is not None and resolved.algorithm.name not in {"adapt_vqe", "tetris_adapt"}:
        raise ConfigError(
            "qiskit supplied operator_pool requires algorithm 'adapt_vqe' or 'tetris_adapt'"
        )
    algorithm_spec = ALGORITHMS.spec(resolved.algorithm.name)
    if (
        initial_point is not None
        and resolved.algorithm.name != "vqe"
        and algorithm_spec.execution != "native"
    ):
        raise ConfigError("qiskit supplied initial_point requires algorithm 'vqe'")
    if reference_energy_hartree is not None and (
        isinstance(reference_energy_hartree, bool)
        or not isinstance(reference_energy_hartree, Real)
        or not np.isfinite(reference_energy_hartree)
    ):
        raise ConfigError("qiskit reference energy must be a finite number in hartree")
    validate_component_graph(resolved, operator_pool_supplied=operator_pool is not None)
    if algorithm_spec.execution == "native":
        request = NativeSolveRequest(
            prepared=prepared,
            options=resolved,
            initial_point=initial_point,
            callback=callback,
            reference_energy_hartree=reference_energy_hartree,
            operator_pool=operator_pool,
        )
        outcome = ALGORITHMS.build(resolved.algorithm, request=request)
        return summarize_native(outcome, request, runtime_seconds=perf_counter() - started)
    context = map_problem(prepared, resolved.mapper, initial_state=resolved.initial_state)
    logger.info(
        "Qiskit %s starting: ansatz=%s optimizer=%s",
        resolved.algorithm.name,
        resolved.ansatz.name,
        resolved.optimizer.name,
    )
    return _solve_context(
        context,
        resolved,
        operator_pool=operator_pool,
        supplied_initial_point=initial_point,
        user_callback=callback,
        reference_energy_hartree=reference_energy_hartree,
        started=started,
    )


def run_job(
    xyz_path: str | Path,
    *,
    charge: int,
    multiplicity: int,
    options: QiskitOptions | Mapping[str, Any],
    artifact_dir: str | Path | None = None,
) -> QiskitRunResult:
    """Adapt a geometry/PySCF pipeline job to the driver-independent solver API."""
    started = perf_counter()
    resolved = options if isinstance(options, QiskitOptions) else QiskitOptions.from_raw(options)
    validate_options(resolved)
    if resolved.integral_source is None:
        prepared = prepare_pyscf_problem(
            xyz_path,
            charge=charge,
            multiplicity=multiplicity,
            options=resolved,
        )
    else:
        from chemrefine.engines.qiskit.integral_source import prepare_integral_job

        prepared = prepare_integral_job(
            Path(xyz_path), charge=charge, multiplicity=multiplicity, options=resolved
        )
    result = run_problem(prepared, options=resolved)
    # Pipeline runtime includes its classical electronic-structure preparation.
    artifacts = []
    if artifact_dir is not None and result.states:
        from chemrefine.engines.qiskit.state_io import save_states

        path = Path(artifact_dir) / f"{Path(xyz_path).stem}.states.json"
        save_states(path, result.states)
        artifacts.append(path.name)
    if artifact_dir is not None and result.circuits:
        from chemrefine.engines.qiskit.circuit_io import save_circuit

        for circuit in result.circuits:
            path = (
                Path(artifact_dir)
                / f"{Path(xyz_path).stem}.root{circuit.description.root}.circuit.json"
            )
            limit = resolved.circuit_export.max_bytes if resolved.circuit_export else 33554432
            save_circuit(path, circuit, max_bytes=limit)
            artifacts.append(path.name)
    if artifacts:
        result = replace(result, metadata={**result.metadata, "quantum_artifacts": artifacts})
    return replace(result, runtime_seconds=perf_counter() - started)

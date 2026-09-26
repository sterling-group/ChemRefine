"""Compose existing registries around independently prepared electronic problems."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from copy import deepcopy
from numbers import Real
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from chemrefine.engines.qiskit import components as _builtins  # noqa: F401
from chemrefine.engines.qiskit.context import (
    AnsatzArtifacts,
    ElectronicStructureContext,
    EstimatorResource,
    SolverComponents,
)
from chemrefine.engines.qiskit.mapping import map_problem
from chemrefine.engines.qiskit.native import NativeSolveRequest, summarize_native
from chemrefine.engines.qiskit.operators import OperatorPool
from chemrefine.engines.qiskit.options import QiskitOptions
from chemrefine.engines.qiskit.problem import PreparedProblem, prepare_pyscf_problem
from chemrefine.engines.qiskit.registry import (
    ALGORITHMS,
    ANSATZE,
    ESTIMATORS,
    INITIAL_POINTS,
    INITIAL_STATES,
    OPTIMIZERS,
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
    requirements = ALGORITHMS.spec(resolved.algorithm.name).requires

    initial_state: Any | None = None
    ansatz = AnsatzArtifacts()
    initial_point: Any | None = None
    optimizer: Any | None = None
    if requirements & {"initial_state", "circuit", "operator_pool"}:
        initial_state = INITIAL_STATES.build(resolved.initial_state, context=context)
    if operator_pool is not None:
        operator_pool.validate(context.num_qubits)
        ansatz = AnsatzArtifacts(
            operator_pool=operator_pool.operators, pool_metadata=operator_pool.metadata
        )
    elif requirements & {"circuit", "operator_pool"}:
        ansatz = ANSATZE.build(
            resolved.ansatz,
            context=context,
            initial_state=initial_state,
        )
    if "initial_point" in requirements:
        if supplied_initial_point is None:
            initial_point = INITIAL_POINTS.build(resolved.initial_point, ansatz=ansatz)
        else:
            try:
                if np.iscomplexobj(supplied_initial_point):
                    raise ValueError("complex initial point")
                initial_point = np.asarray(supplied_initial_point, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ConfigError("qiskit initial point must contain finite real numbers") from exc
            if (
                ansatz.circuit is None
                or initial_point.shape != (int(ansatz.circuit.num_parameters),)
                or not np.isfinite(initial_point).all()
            ):
                raise ConfigError(
                    "qiskit initial point must have one finite value per ansatz parameter"
                )
    if "optimizer" in requirements:
        optimizer = OPTIMIZERS.build(resolved.optimizer)

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

    estimator_resource: EstimatorResource | nullcontext[None]
    if "estimator" in requirements:
        estimator_resource = ESTIMATORS.build(
            resolved.estimator,
            device=resolved.device,
            cores=resolved.cores,
        )
    else:
        estimator_resource = nullcontext(None)

    transpiler = (
        estimator_resource.transpiler if isinstance(estimator_resource, EstimatorResource) else None
    )
    transpiler_options = (
        estimator_resource.transpiler_options
        if isinstance(estimator_resource, EstimatorResource)
        else None
    )
    with estimator_resource as estimator:
        assembled = SolverComponents(
            estimator=estimator,
            optimizer=optimizer,
            initial_state=initial_state,
            ansatz=ansatz,
            initial_point=initial_point,
            callback=callback,
            transpiler=transpiler,
            transpiler_options=transpiler_options,
        )
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
        ansatz=ansatz,
        algorithm=algorithm,
        runtime_seconds=perf_counter() - started,
        reference_energy_hartree=reference_energy_hartree,
        was_transpiled=transpiler is not None,
        operator_pool_supplied=operator_pool is not None,
    )
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
    if operator_pool is not None and resolved.algorithm.name != "adapt_vqe":
        raise ConfigError("qiskit supplied operator_pool requires algorithm 'adapt_vqe'")
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
        )
        outcome = ALGORITHMS.build(resolved.algorithm, request=request)
        return summarize_native(outcome, request, runtime_seconds=perf_counter() - started)
    context = map_problem(prepared, resolved.mapper)
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
    prepared = prepare_pyscf_problem(
        xyz_path,
        charge=charge,
        multiplicity=multiplicity,
        options=resolved,
    )
    result = run_problem(prepared, options=resolved)
    # Pipeline runtime includes its classical electronic-structure preparation.
    from dataclasses import replace

    if artifact_dir is not None and result.states:
        from chemrefine.engines.qiskit.state_io import save_states

        path = Path(artifact_dir) / f"{Path(xyz_path).stem}.states.json"
        save_states(path, result.states)
        result = replace(result, metadata={**result.metadata, "quantum_artifacts": [path.name]})
    return replace(result, runtime_seconds=perf_counter() - started)

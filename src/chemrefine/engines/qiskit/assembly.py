"""Shared lifecycle for fixed circuits, context-aware optimizers and V2 primitives."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from typing import Any

import numpy as np

from chemrefine.engines.qiskit.components.initial_states import build_selected_reference
from chemrefine.engines.qiskit.context import (
    AnsatzArtifacts,
    ElectronicStructureContext,
    SolverComponents,
)
from chemrefine.engines.qiskit.operators import OperatorPool
from chemrefine.engines.qiskit.options import QiskitOptions
from chemrefine.engines.qiskit.registry import (
    ANSATZE,
    ESTIMATORS,
    INITIAL_POINTS,
    OPTIMIZERS,
    SAMPLERS,
    consumed_component_categories,
)
from chemrefine.errors import ConfigError


def validated_initial_point(point: Sequence[float], ansatz: AnsatzArtifacts) -> np.ndarray:
    """Require one finite real initial parameter per circuit parameter."""
    try:
        if np.iscomplexobj(point):
            raise ValueError("complex initial point")
        values = np.asarray(point, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ConfigError("qiskit initial point must contain finite real numbers") from exc
    if (
        ansatz.circuit is None
        or values.shape != (int(ansatz.circuit.num_parameters),)
        or not np.isfinite(values).all()
    ):
        raise ConfigError("qiskit initial point must have one finite value per ansatz parameter")
    return values


@contextmanager
def assemble_components(
    context: ElectronicStructureContext,
    options: QiskitOptions,
    *,
    operator_pool: OperatorPool | None = None,
    initial_point: Sequence[float] | None = None,
    callback: Any = None,
    defer_optimizer: bool = False,
) -> Iterator[SolverComponents]:
    """Construct consumed resources once and close each on every exit path.

    Circuit construction and supplied-parameter validation precede provider creation.
    An optimizer declaring resource requirements receives the assembled context; legacy
    optimizers retain their options-only builder signature.
    """
    required = consumed_component_categories(options)
    state = (
        build_selected_reference(context, options.initial_state)
        if "initial_state" in required
        else None
    )
    ansatz = AnsatzArtifacts()
    if operator_pool is not None:
        operator_pool.validate(context.num_qubits)
        ansatz = AnsatzArtifacts(
            operator_pool=operator_pool.operators, pool_metadata=operator_pool.metadata
        )
    elif "ansatz" in required:
        ansatz = ANSATZE.build(options.ansatz, context=context, initial_state=state)
    point = None
    if "initial_point" in required:
        point = (
            INITIAL_POINTS.build(options.initial_point, ansatz=ansatz)
            if initial_point is None
            else validated_initial_point(initial_point, ansatz)
        )
    with ExitStack() as stack:
        estimator = None
        sampler = None
        estimator_resource = None
        sampler_resource = None
        if "estimator" in required:
            estimator_resource = ESTIMATORS.build(
                options.estimator, device=options.device, cores=options.cores
            )
            estimator = stack.enter_context(estimator_resource)
        if "sampler" in required:
            sampler_resource = SAMPLERS.build(
                options.sampler, device=options.device, cores=options.cores
            )
            sampler = stack.enter_context(sampler_resource)
        components = SolverComponents(
            initial_state=state,
            ansatz=ansatz,
            initial_point=point,
            callback=callback,
            estimator=estimator,
            sampler=sampler,
            transpiler=estimator_resource.transpiler if estimator_resource else None,
            transpiler_options=estimator_resource.transpiler_options
            if estimator_resource
            else None,
            sampler_transpiler=sampler_resource.transpiler if sampler_resource else None,
            sampler_transpiler_options=(
                sampler_resource.transpiler_options if sampler_resource else None
            ),
        )
        if "optimizer" in required and not defer_optimizer:
            components = replace(
                components, optimizer=build_optimizer(context, options, components)
            )
        yield components


def build_optimizer(
    context: ElectronicStructureContext, options: QiskitOptions, components: SolverComponents
) -> Any:
    """Bind one optimizer to the current circuit and the already acquired primitives.

    Owned adaptive drivers call this after every circuit growth. A metric from a
    smaller circuit is never reused for a larger parameter space.
    """
    extra = (
        {"context": context, "components": components}
        if OPTIMIZERS.spec(options.optimizer.name).requires
        else {}
    )
    return OPTIMIZERS.build(options.optimizer, **extra)

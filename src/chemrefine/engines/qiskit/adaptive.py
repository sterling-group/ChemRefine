"""Measured-gradient adaptive growth with explicit block identities and retained-state checks."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from chemrefine.engines.qiskit.assembly import assemble_components, build_optimizer
from chemrefine.engines.qiskit.ceo import (
    mvp_circuit,
    ovp_circuit,
    pauli_support,
    single_exchange_circuit,
)
from chemrefine.engines.qiskit.context import AnsatzArtifacts
from chemrefine.engines.qiskit.mapping import map_problem
from chemrefine.engines.qiskit.metrics import logical_circuit_metrics
from chemrefine.engines.qiskit.native import NativeOutcome, NativeSolveRequest
from chemrefine.engines.qiskit.operators import OperatorPool
from chemrefine.engines.qiskit.options import ComponentSelection
from chemrefine.engines.qiskit.reporting import jsonable
from chemrefine.engines.qiskit.spectra import ExpectationSession, real_value, sector_diagnostics
from chemrefine.errors import ConfigError

if TYPE_CHECKING:
    from chemrefine.engines.qiskit.components.adaptive import AdaptiveOptions


def select_blocks(
    gradients: np.ndarray,
    operators: tuple[Any, ...],
    metadata: tuple[dict[str, Any], ...],
    *,
    ceo_variant: str | None,
    tetris: bool,
    threshold: float,
) -> list[dict[str, Any]]:
    """Prioritize gradients, enforcing disjoint *mapped* support when TETRIS is enabled."""
    candidates: list[dict[str, Any]] = []
    seen = set()
    for index, gradient in enumerate(gradients):
        item = metadata[index]
        if ceo_variant is not None and not item["candidate"]:
            continue
        kind = "generic"
        indices = [index]
        score = abs(float(gradient))
        if ceo_variant is not None:
            group = item["group_indices"]
            if ceo_variant == "mvp":
                key = tuple(group)
                if key in seen:
                    continue
                seen.add(key)
                indices = group.copy()
                score = sum(abs(float(gradients[parent])) for parent in group)
                kind = "mvp"
            elif ceo_variant == "adaptive":
                active = [parent for parent in group if abs(gradients[parent]) > threshold]
                if len(active) > 1:
                    indices = active
                    kind = "mvp"
                else:
                    kind = item["role"]
            else:
                kind = item["role"]
        if score <= threshold:
            continue
        support = frozenset().union(*(pauli_support(operators[parent]) for parent in indices))
        candidates.append(
            {
                "candidate_index": index,
                "indices": indices,
                "kind": kind,
                "score": score,
                "support": sorted(support),
            }
        )
    candidates.sort(key=lambda item: (-item["score"], item["candidate_index"]))
    selected = []
    occupied: set[int] = set()
    for candidate in candidates:
        candidate_support = set(candidate["support"])
        if occupied.isdisjoint(candidate_support):
            selected.append(candidate)
            occupied.update(candidate_support)
            if not tetris:
                break
    return selected


def append_block(
    circuit: Any,
    block: dict[str, Any],
    operators: tuple[Any, ...],
    metadata: tuple[dict[str, Any], ...],
    parameters: list[Any],
    *,
    optimized_occupation: bool,
    options: AdaptiveOptions,
) -> None:
    """Append exact CEO networks where applicable or an explicitly selected Pauli evolution."""
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.synthesis import LieTrotter, SuzukiTrotter

    indices = block["indices"]
    records = [metadata[index] for index in indices]
    generators = [operators[index] for index in indices]
    if (
        optimized_occupation
        and options.evolution == "exact_commuting"
        and all(item.get("quadrature") == "antisymmetric" for item in records)
    ):
        first = records[0]
        if len(indices) == 1 and first["role"] == "ovp":
            left, right = first["parents"]
            layer = ovp_circuit(
                circuit.num_qubits,
                metadata[left],
                metadata[right],
                first["weights"][1],
                parameters[0],
            )
        elif len(indices) == 1 and len(first["source"]) == 1:
            layer = single_exchange_circuit(
                circuit.num_qubits, first["source"][0], first["target"][0], parameters[0]
            )
        else:
            layer = mvp_circuit(circuit.num_qubits, generators, parameters)
        circuit.compose(layer, inplace=True)
        return
    for generator, parameter in zip(generators, parameters, strict=True):
        if options.evolution == "exact_commuting":
            for pauli in generator.paulis:
                if not np.all(generator.paulis.commutes(pauli)):
                    raise ConfigError(
                        "adaptive generator has noncommuting Pauli terms; "
                        "select an explicit product formula"
                    )
        synthesis = (
            SuzukiTrotter(order=options.suzuki_order, reps=options.repetitions)
            if options.evolution == "suzuki"
            else LieTrotter(reps=options.repetitions)
        )
        evolution = PauliEvolutionGate(generator, time=parameter, synthesis=synthesis)
        # Expose the selected synthesis to every primitive. An opaque evolution
        # gate's matrix protocol can otherwise evaluate the exact exponential even
        # when a caller explicitly requested an approximate product formula.
        circuit.compose(evolution.definition, inplace=True)


def adaptive_solve(
    request: NativeSolveRequest, options: AdaptiveOptions, *, ceo: bool
) -> NativeOutcome:
    """Run bounded TETRIS/CEO growth with optimizer rebuilding and energy rollback."""
    from qiskit.circuit import Parameter
    from qiskit_algorithms import VQE
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    if (
        request.initial_point is not None
        or request.options.initial_point != ComponentSelection.named("zeros")
    ):
        raise ConfigError(
            "owned adaptive solvers start with no parameters; initial_point must be zeros"
        )
    context = map_problem(
        request.prepared, request.options.mapper, initial_state=request.options.initial_state
    )
    evaluations: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    retained: list[dict[str, Any]] = []
    measurements = 0
    iteration = 0

    def callback(count: int, parameters: Any, energy: float, metadata: Any) -> None:
        """Keep every inner search evaluation and detach records sent to user callbacks."""
        del parameters
        if len(evaluations) >= options.max_evaluations:
            raise ConfigError("adaptive solve exceeds max_evaluations")
        record = {
            "evaluation": len(evaluations) + 1,
            "inner_run": iteration,
            "algorithm_evaluation": count,
            "objective_value_hartree": real_value(complex(energy), "adaptive energy"),
            "metadata": jsonable(metadata),
        }
        evaluations.append(record)
        if request.callback is not None:
            request.callback(deepcopy(record))

    with assemble_components(
        context,
        request.options,
        operator_pool=request.operator_pool,
        callback=callback,
        defer_optimizer=True,
    ) as components:
        raw_pool = components.ansatz.operator_pool
        if raw_pool is None:
            raise ConfigError("adaptive solve requires an operator pool")
        pool = OperatorPool(tuple(raw_pool), components.ansatz.pool_metadata)
        pool.validate(context.num_qubits)
        operators, metadata = pool.operators, pool.metadata
        if len(operators) > options.max_pool_size:
            raise ConfigError("adaptive solve exceeds max_pool_size")
        if ceo and any(item.get("family") != "ceo" for item in metadata):
            raise ConfigError("CEO-ADAPT requires a declared coupled-exchange pool")
        if components.initial_state is None:
            raise ConfigError("adaptive solve requires a prepared initial state")
        circuit = components.initial_state.copy()
        if circuit.num_parameters:
            raise ConfigError("adaptive initial state must be fully bound")
        point = np.empty(0)
        hamiltonian = context.qubit_hamiltonian
        commutators = {}
        for index, generator in enumerate(operators):
            if ceo and metadata[index]["role"] == "ovp":
                continue
            if len(hamiltonian) * len(generator) > options.max_product_terms:
                raise ConfigError("adaptive gradient construction exceeds max_product_terms")
            commutators[index] = (
                1j * (generator @ hamiltonian - hamiltonian @ generator)
            ).simplify(atol=1e-12)
        session = ExpectationSession(
            context, components, circuit, options.max_measurements, options.max_pauli_terms
        )
        energy = real_value(session.mapped(hamiltonian), "adaptive reference energy")
        measurements += session.measurements
        converged = False
        reason = "maximum_iterations"
        previous_selection = None
        for iteration in range(1, options.max_iterations + 1):
            bound = circuit.assign_parameters(point)
            session = ExpectationSession(
                context,
                components,
                bound,
                options.max_measurements - measurements,
                options.max_pauli_terms,
            )
            gradients = np.zeros(len(operators))
            for index, commutator in commutators.items():
                gradients[index] = real_value(session.mapped(commutator), "adaptive gradient")
            if ceo:
                for index, item in enumerate(metadata):
                    if item["role"] == "ovp":
                        gradients[index] = sum(
                            weight * gradients[parent]
                            for weight, parent in zip(item["weights"], item["parents"], strict=True)
                        )
            measurements += session.measurements
            selection_gradients = gradients[
                [i for i, item in enumerate(metadata) if not ceo or item["candidate"]]
            ]
            gradient_norm = float(
                np.linalg.norm(
                    selection_gradients, ord=2 if options.gradient_norm == "l2" else np.inf
                )
            )
            entry = {
                "iteration": iteration,
                "gradients": gradients.tolist(),
                "gradient_norm": gradient_norm,
                "energy_before_hartree": energy,
                "retained": False,
                "blocks": [],
            }
            history.append(entry)
            if gradient_norm < options.gradient_threshold:
                converged, reason = True, "gradient_converged"
                break
            variant = getattr(options, "variant", "adaptive") if ceo else None
            blocks = select_blocks(
                gradients,
                operators,
                metadata,
                ceo_variant=variant,
                tetris=getattr(options, "tetris", True),
                threshold=options.selection_threshold,
            )
            entry["blocks"] = blocks
            if not blocks:
                reason = "no_candidate_above_selection_threshold"
                break
            identity = tuple(tuple(block["indices"]) for block in blocks)
            if identity == previous_selection:
                reason = "repeated_selection"
                break
            added = sum(len(block["indices"]) for block in blocks)
            if len(point) + added > options.max_parameters:
                raise ConfigError("adaptive solve exceeds max_parameters")
            grown = circuit.copy()
            for block in blocks:
                start = grown.num_parameters
                parameters = [
                    Parameter(f"theta_{start + i:08d}") for i in range(len(block["indices"]))
                ]
                append_block(
                    grown,
                    block,
                    operators,
                    metadata,
                    parameters,
                    optimized_occupation=(
                        request.operator_pool is None
                        and request.options.ansatz.name in {"qe", "ceo"}
                        and isinstance(context.mapper, JordanWignerMapper)
                    ),
                    options=options,
                )
            new_point = np.concatenate((point, np.zeros(added)))
            assembled = replace(
                components, ansatz=AnsatzArtifacts(circuit=grown), initial_point=new_point
            )
            optimizer = build_optimizer(context, request.options, assembled)
            solver = VQE(
                components.estimator,
                grown,
                optimizer,
                initial_point=new_point,
                callback=callback,
                transpiler=components.transpiler,
                transpiler_options=components.transpiler_options,
            )
            result = solver.compute_minimum_eigenvalue(hamiltonian)
            candidate_point = np.asarray(
                [result.optimal_parameters[p] for p in grown.parameters], dtype=float
            )
            session = ExpectationSession(
                context,
                components,
                grown.assign_parameters(candidate_point),
                options.max_measurements - measurements,
                options.max_pauli_terms,
            )
            candidate_energy = real_value(session.mapped(hamiltonian), "adaptive optimized energy")
            measurements += session.measurements
            entry["energy_after_hartree"] = candidate_energy
            improvement = energy - candidate_energy
            if improvement < -options.energy_increase_tolerance:
                reason = "energy_increase_rollback"
                break
            circuit, point, energy = grown, candidate_point, candidate_energy
            retained.extend(deepcopy(blocks))
            entry["retained"] = True
            previous_selection = identity
            if options.eigenvalue_threshold and abs(improvement) < options.eigenvalue_threshold:
                converged, reason = True, "energy_converged"
                break
        session = ExpectationSession(
            context,
            components,
            circuit.assign_parameters(point),
            options.max_measurements - measurements,
            options.max_pauli_terms,
        )
        spin = (context.multiplicity - 1) / 2
        sector = sector_diagnostics(
            context,
            session.fermionic,
            tolerance=options.sector_tolerance,
            target_s2=spin * (spin + 1) if options.spin_constraint == "require" else None,
            spin_tolerance=options.spin_tolerance,
        )
        measurements += session.measurements
        metrics = logical_circuit_metrics(circuit)
        from chemrefine.engines.qiskit.circuit_io import bound_circuit

        return NativeOutcome(
            energy,
            converged=converged,
            termination_reason=reason,
            num_qubits=context.num_qubits,
            ansatz="external_pool" if request.operator_pool else request.options.ansatz.name,
            optimizer=request.options.optimizer.name,
            parameter_count=len(point),
            optimizer_evaluations=len(evaluations),
            evaluations=evaluations,
            circuits=(bound_circuit(context, circuit, point),)
            if request.options.circuit_export is not None
            else (),
            diagnostics={
                "experimental": True,
                "method": "ceo_adapt" if ceo else "tetris_adapt",
                "gradient_history": history,
                "selected_blocks": retained,
                "pool_metadata": metadata,
                "pool_size": len(operators),
                "optimal_point": point.tolist(),
                "final_sector": sector,
                "measurements": measurements,
                "logical_circuit_metrics": asdict(metrics) if metrics else None,
                "optimizer_rebuilt_each_inner_solve": True,
                "evolution": options.evolution,
                "repetitions": options.repetitions,
            },
        )

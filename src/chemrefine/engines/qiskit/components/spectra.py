"""Native molecular VQD and complex qEOM using the configured component graph."""

from __future__ import annotations

from copy import deepcopy
from math import comb
from typing import Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from chemrefine.engines.qiskit.assembly import assemble_components, validated_initial_point
from chemrefine.engines.qiskit.components.ansatze import reference_excitation_permutation
from chemrefine.engines.qiskit.mapping import map_problem
from chemrefine.engines.qiskit.native import NativeOutcome, NativeSolveRequest
from chemrefine.engines.qiskit.registry import ALGORITHMS
from chemrefine.engines.qiskit.reporting import jsonable
from chemrefine.engines.qiskit.spectra import (
    ExpectationSession,
    energy_residual,
    optimizer_termination,
    real_value,
    residual_hamiltonian,
    sector_diagnostics,
    solve_response_problem,
)
from chemrefine.errors import ConfigError


class SpectrumOptions(BaseModel):
    """Root selection, physical-sector checks and explicit experimental budgets."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    target_root: int = Field(0, ge=0)
    max_evaluations: int = Field(10000, ge=1)
    max_measurements: int = Field(100000, ge=1)
    max_pauli_terms: int = Field(100000, ge=1)
    sector_tolerance: float = Field(1e-3, gt=0)
    target_s2: float | None = Field(None, ge=0)
    spin_tolerance: float = Field(1e-3, gt=0)
    measure_residuals: bool = False
    max_residual_product_terms: int = Field(1000000, ge=1)
    residual_variance_tolerance: float = Field(1e-8, gt=0)


class VQDOptions(SpectrumOptions):
    """Deflation penalties and measured-overlap validation for fixed-ansatz roots."""

    k: int = Field(2, ge=1, le=64)
    betas: tuple[float, ...] | None = None
    initial_points: tuple[tuple[float, ...], ...] | None = None
    fidelity_shots: int = Field(16384, ge=1)
    overlap_tolerance: float = Field(0.05, gt=0, lt=1)

    @model_validator(mode="after")
    def _root_controls(self) -> Self:
        """Check root-specific input lengths without importing a quantum backend."""
        if self.target_root >= self.k:
            raise ValueError("target_root must be smaller than k")
        if self.betas is not None and (
            len(self.betas) != self.k - 1 or any(beta <= 0 for beta in self.betas)
        ):
            raise ValueError("betas must contain k-1 positive deflation penalties")
        if self.initial_points is not None and len(self.initial_points) != self.k:
            raise ValueError("initial_points must contain one parameter vector per root")
        return self


class QEOMOptions(SpectrumOptions):
    """Reference-relative excitation ranks and tolerances for complex linear response."""

    excitation_ranks: tuple[int, ...] = (1, 2)
    max_excitations: int = Field(64, ge=1)
    max_product_terms: int = Field(1000000, ge=1)
    conditioning_tolerance: float = Field(1e-10, gt=0, lt=1)
    residual_tolerance: float = Field(1e-7, gt=0)
    frequency_tolerance: float = Field(1e-6, gt=0)

    @model_validator(mode="after")
    def _ranks(self) -> Self:
        """Require a nonempty, distinct set of positive excitation ranks."""
        if not self.excitation_ranks or any(rank < 1 for rank in self.excitation_ranks):
            raise ValueError("excitation_ranks must be positive and nonempty")
        if len(set(self.excitation_ranks)) != len(self.excitation_ranks):
            raise ValueError("excitation_ranks must be distinct")
        return self


def _recorder(
    request: NativeSolveRequest, options: SpectrumOptions
) -> tuple[list[dict[str, Any]], Any]:
    """Record detached evaluation history and enforce the configured objective budget."""
    evaluations: list[dict[str, Any]] = []

    def callback(count: int, parameters: Any, value: float, metadata: Any, step: int = 1) -> None:
        """Keep penalties distinct from physical post-optimization energies."""
        del parameters
        if len(evaluations) >= options.max_evaluations:
            raise ConfigError("qiskit spectrum exceeds max_evaluations")
        value = real_value(complex(value), "spectrum objective")
        record = {
            "evaluation": len(evaluations) + 1,
            "algorithm_evaluation": int(count),
            "root": int(step) - 1,
            "objective_value_hartree": value,
            "objective_includes_deflation": request.options.algorithm.name == "vqd" and step > 1,
            "metadata": jsonable(metadata),
        }
        evaluations.append(record)
        if request.callback is not None:
            request.callback(deepcopy(record))

    return evaluations, callback


def _check_circuit(circuit: Any) -> Any:
    """Reject an empty variational search before contacting a primitive."""
    if circuit is None or circuit.num_parameters < 1:
        raise ConfigError("qiskit spectrum requires a parameterized fixed-circuit ansatz")
    return circuit


def _root_sector(context: Any, measure: Any, options: SpectrumOptions) -> dict[str, float]:
    """Apply the same physical-sector contract to VQD and reconstructed qEOM roots."""
    return sector_diagnostics(
        context,
        measure,
        tolerance=options.sector_tolerance,
        target_s2=options.target_s2,
        spin_tolerance=options.spin_tolerance,
    )


def _residual_operator(context: Any, options: SpectrumOptions) -> Any:
    """Keep expensive second-moment construction and measurements explicitly opt-in."""
    return (
        residual_hamiltonian(
            context.fermionic_hamiltonian, max_product_terms=options.max_residual_product_terms
        )
        if options.measure_residuals
        else None
    )


@ALGORITHMS.register(
    "vqd",
    VQDOptions,
    capabilities=frozenset({"bound_circuit"}),
    status="experimental",
    supported_domains=(
        "fixed parameterized circuits",
        "fixed molecular particle/magnetization sector",
        "complex Hamiltonians",
    ),
    execution="native",
    requires=frozenset({"mapper", "estimator", "sampler", "optimizer", "circuit", "initial_point"}),
)
def build_vqd(*, options: VQDOptions, request: NativeSolveRequest) -> NativeOutcome:
    """Optimize roots with VQD, then independently measure energies, overlaps and sectors."""
    from qiskit_algorithms import VQD
    from qiskit_algorithms.state_fidelities import ComputeUncompute

    if request.initial_point is not None and options.initial_points is not None:
        raise ConfigError("supply either initial_point or VQD initial_points, not both")
    context = map_problem(
        request.prepared, request.options.mapper, initial_state=request.options.initial_state
    )
    squared = _residual_operator(context, options)
    if options.k > comb(context.num_spatial_orbitals, context.num_particles[0]) * comb(
        context.num_spatial_orbitals, context.num_particles[1]
    ):
        raise ConfigError("VQD k exceeds the fixed N/Ms sector dimension")
    evaluations, callback = _recorder(request, options)
    with assemble_components(
        context, request.options, initial_point=request.initial_point, callback=callback
    ) as components:
        circuit = _check_circuit(components.ansatz.circuit)
        initial = components.initial_point
        if options.initial_points is not None:
            initial = np.asarray(
                [
                    validated_initial_point(point, components.ansatz)
                    for point in options.initial_points
                ]
            )
        fidelity = ComputeUncompute(
            components.sampler,
            shots=options.fidelity_shots,
            transpiler=components.sampler_transpiler,
            transpiler_options=components.sampler_transpiler_options,
        )
        solver = VQD(
            components.estimator,
            fidelity,
            circuit,
            components.optimizer,
            k=options.k,
            betas=np.asarray(options.betas) if options.betas is not None else None,
            initial_point=initial,
            callback=callback,
            transpiler=components.transpiler,
            transpiler_options=components.transpiler_options,
        )
        result = solver.compute_eigenvalues(context.qubit_hamiltonian)
        circuits = [
            circuit.assign_parameters(parameters) for parameters in result.optimal_parameters
        ]
        energies = []
        residuals = []
        sectors = []
        measurements = 0
        for state in circuits:
            session = ExpectationSession(
                context,
                components,
                state,
                options.max_measurements - measurements,
                options.max_pauli_terms,
            )
            energies.append(real_value(session.mapped(context.qubit_hamiltonian), "VQD energy"))
            sectors.append(_root_sector(context, session.fermionic, options))
            if squared is not None:
                residuals.append(
                    energy_residual(
                        energies[-1],
                        session.fermionic(squared),
                        variance_tolerance=options.residual_variance_tolerance,
                    )
                )
            measurements += session.measurements
        pairs = [(left, right) for left in range(len(circuits)) for right in range(left)]
        overlaps = np.eye(len(circuits))
        if pairs:
            values = (
                fidelity.run(
                    [circuits[left] for left, _ in pairs],
                    [circuits[right] for _, right in pairs],
                )
                .result()
                .fidelities
            )
            for (left, right), value in zip(pairs, values, strict=True):
                overlap = real_value(complex(value), "VQD overlap")
                if overlap < 0 or overlap > options.overlap_tolerance:
                    raise ConfigError(
                        "VQD roots exceed overlap_tolerance; increase penalties or change "
                        "initial_points/ansatz/optimizer"
                    )
                overlaps[left, right] = overlaps[right, left] = overlap
        # Sort by measured physical energy; keep corresponding root facts aligned.
        order = np.argsort(energies, kind="stable")
        roots = tuple(float(energies[index]) for index in order)
        raw_optimizers = getattr(result, "optimizer_results", None)
        terminations = [
            optimizer_termination(None if raw_optimizers is None else raw_optimizers[index])
            for index in order
        ]
        verdicts = [item["converged"] for item in terminations]
        from chemrefine.engines.qiskit.circuit_io import bound_circuit

        return NativeOutcome(
            roots[options.target_root],
            active_energies_hartree=roots,
            target_root=options.target_root,
            converged=None if None in verdicts else all(verdicts),
            num_qubits=context.num_qubits,
            ansatz=request.options.ansatz.name,
            optimizer=request.options.optimizer.name,
            parameter_count=circuit.num_parameters,
            optimizer_evaluations=len(evaluations),
            evaluations=evaluations,
            circuits=tuple(
                bound_circuit(context, circuit, result.optimal_parameters[index], root=root)
                for root, index in enumerate(order)
            )
            if request.options.circuit_export is not None
            else (),
            termination_reason="optimizer_returned_and_roots_validated",
            diagnostics={
                "experimental": True,
                "method": "variational_quantum_deflation",
                "energy_convention": "measured_active_hamiltonian_without_deflation_penalty",
                "root_sectors": [sectors[index] for index in order],
                "root_overlap_matrix": overlaps[np.ix_(order, order)].tolist(),
                "root_optimization_order": order.tolist(),
                "optimal_points": np.asarray(result.optimal_points)[order].tolist(),
                "penalized_objectives": np.asarray(result.optimal_values)[order].tolist(),
                "post_optimization_measurements": measurements,
                "fidelity_shots": options.fidelity_shots,
                "optimizer_termination": terminations,
                "root_hamiltonian_residuals": [residuals[index] for index in order]
                if options.measure_residuals
                else None,
            },
        )


def _response_basis(
    context: Any, initial_state: Any, options: QEOMOptions
) -> tuple[list[Any], list[Any]]:
    """Build reference-relative excitation/de-excitation pairs in fermionic space."""
    from qiskit_nature.second_q.circuit.library.ansatzes.utils import generate_fermionic_excitations
    from qiskit_nature.second_q.operators import FermionicOp

    n = context.num_spatial_orbitals
    alpha, beta = context.num_particles
    count = sum(
        comb(alpha, a) * comb(n - alpha, a) * comb(beta, rank - a) * comb(n - beta, rank - a)
        for rank in options.excitation_ranks
        for a in range(max(0, rank - beta), min(rank, alpha) + 1)
    )
    if not 1 <= count <= options.max_excitations:
        raise ConfigError("qEOM requires 1..max_excitations reference-relative excitations")
    permutation = reference_excitation_permutation(context, initial_state)
    excitations = [
        (tuple(permutation[i] for i in occupied), tuple(permutation[i] for i in virtual))
        for rank in options.excitation_ranks
        for occupied, virtual in generate_fermionic_excitations(
            rank, context.num_spatial_orbitals, context.num_particles, preserve_spin=True
        )
    ]
    if not excitations or len(excitations) > options.max_excitations:
        raise ConfigError("qEOM requires 1..max_excitations reference-relative excitations")
    if options.target_root > len(excitations):
        raise ConfigError("qEOM target_root exceeds the excitation-space root count")
    operators = []
    for occupied, virtual in excitations:
        label = " ".join([*(f"+_{i}" for i in virtual), *(f"-_{i}" for i in occupied)])
        operators.append(
            FermionicOp({label: 1}, num_spin_orbitals=2 * context.num_spatial_orbitals)
        )
    return operators + [operator.adjoint() for operator in operators], excitations


def _product(left: Any, right: Any, options: QEOMOptions) -> Any:
    """Bound intermediate fermionic products before multiplication and normal ordering."""
    if len(left) * len(right) > options.max_product_terms:
        raise ConfigError("qEOM exceeds max_product_terms")
    product = (left @ right).normal_order().simplify(atol=1e-12)
    if len(product) > options.max_product_terms:
        raise ConfigError("qEOM exceeds max_product_terms after normal ordering")
    return product


def _commutator(left: Any, right: Any, options: QEOMOptions) -> Any:
    """Compute a bounded operator commutator without discarding complex coefficients."""
    return (_product(left, right, options) - _product(right, left, options)).simplify(atol=1e-12)


@ALGORITHMS.register(
    "qeom",
    QEOMOptions,
    status="experimental",
    supported_domains=(
        "stable nonsingular response pencils",
        "reference-relative molecular excitations",
        "complex Hamiltonians",
    ),
    execution="native",
    requires=frozenset({"mapper", "estimator", "optimizer", "circuit", "initial_point"}),
)
def build_qeom(*, options: QEOMOptions, request: NativeSolveRequest) -> NativeOutcome:
    """Measure the complex symmetrized double-commutator response on a VQE reference.

    H_ij = <[[B_i†, H], B_j] + [B_i†, [H, B_j]]>/2 and
    S_ij = <[B_i†, B_j]>. Both matrices are Hermitian, while S is indefinite.
    Root energies are the VQE reference plus positive response frequencies; they
    are approximate linear-response energies, not variational upper bounds.
    """
    from qiskit_algorithms import VQE

    context = map_problem(
        request.prepared, request.options.mapper, initial_state=request.options.initial_state
    )
    squared = _residual_operator(context, options)
    evaluations, callback = _recorder(request, options)
    with assemble_components(
        context, request.options, initial_point=request.initial_point, callback=callback
    ) as components:
        circuit = _check_circuit(components.ansatz.circuit)
        basis, excitations = _response_basis(context, components.initial_state, options)
        solver = VQE(
            components.estimator,
            circuit,
            components.optimizer,
            initial_point=components.initial_point,
            callback=callback,
            transpiler=components.transpiler,
            transpiler_options=components.transpiler_options,
        )
        ground = solver.compute_minimum_eigenvalue(context.qubit_hamiltonian)
        state = circuit.assign_parameters(ground.optimal_parameters)
        session = ExpectationSession(
            context, components, state, options.max_measurements, options.max_pauli_terms
        )
        energy = real_value(session.mapped(context.qubit_hamiltonian), "qEOM reference energy")
        sectors = [_root_sector(context, session.fermionic, options)]
        residuals = []
        if squared is not None:
            residuals.append(
                energy_residual(
                    energy,
                    session.fermionic(squared),
                    variance_tolerance=options.residual_variance_tolerance,
                )
            )
        hamiltonian = context.fermionic_hamiltonian
        hessian = np.zeros((len(basis), len(basis)), dtype=complex)
        metric = np.zeros_like(hessian)
        commutators = [_commutator(hamiltonian, operator, options) for operator in basis]
        for row in range(len(basis)):
            left = basis[row].adjoint()
            left_h = _commutator(left, hamiltonian, options)
            for column in range(row, len(basis)):
                right = basis[column]
                h_element = (
                    _commutator(left_h, right, options)
                    + _commutator(left, commutators[column], options)
                ) * 0.5
                h_value = session.fermionic(h_element)
                s_value = session.fermionic(_commutator(left, right, options))
                if row == column:
                    h_value = real_value(h_value, "qEOM Hessian diagonal")
                    s_value = real_value(s_value, "qEOM metric diagonal")
                hessian[row, column], hessian[column, row] = h_value, np.conjugate(h_value)
                metric[row, column], metric[column, row] = s_value, np.conjugate(s_value)
        gaps, vectors, diagnostics = solve_response_problem(
            hessian,
            metric,
            conditioning_tolerance=options.conditioning_tolerance,
            residual_tolerance=options.residual_tolerance,
            frequency_tolerance=options.frequency_tolerance,
        )
        rayleigh_energies = [energy]
        root_norms = []
        for vector in vectors.T:
            excitation = sum(
                (
                    coefficient * operator
                    for coefficient, operator in zip(vector, basis, strict=True)
                )
            )
            overlap = session.fermionic(excitation)
            # Remove its ground-state component: response eigenvectors describe transitions.
            excitation = excitation - overlap * type(excitation).one()
            norm = real_value(
                session.fermionic(_product(excitation.adjoint(), excitation, options)),
                "qEOM reconstructed root norm",
            )
            if norm <= options.conditioning_tolerance:
                raise ConfigError("qEOM reconstructed root has zero norm")
            root_norms.append(norm)

            def root_measure(
                operator: Any, excitation: Any = excitation, norm: float = norm
            ) -> complex:
                """Evaluate normalized observables in the reconstructed response state."""
                composed = _product(
                    _product(excitation.adjoint(), operator, options), excitation, options
                )
                return session.fermionic(composed) / norm

            sectors.append(_root_sector(context, root_measure, options))
            rayleigh_energies.append(
                real_value(root_measure(hamiltonian), "qEOM reconstructed energy")
            )
            if squared is not None:
                residuals.append(
                    energy_residual(
                        rayleigh_energies[-1],
                        root_measure(squared),
                        variance_tolerance=options.residual_variance_tolerance,
                    )
                )
        roots = (energy, *(float(energy + gap) for gap in gaps))
        termination = optimizer_termination(getattr(ground, "optimizer_result", None))
        diagnostics.update(
            {
                "experimental": True,
                "method": "complex_symmetric_double_commutator_qeom",
                "energy_convention": "active_reference_plus_response_frequency_nonvariational",
                "excitation_indices": excitations,
                "excitation_gaps_hartree": gaps.tolist(),
                "root_sectors": sectors,
                "reconstructed_rayleigh_energies_hartree": rayleigh_energies,
                "reconstructed_root_norms": root_norms,
                "post_optimization_measurements": session.measurements,
                "hessian_real": hessian.real.tolist(),
                "hessian_imag": hessian.imag.tolist(),
                "metric_real": metric.real.tolist(),
                "metric_imag": metric.imag.tolist(),
                "expansion_coefficients_real": vectors.real.tolist(),
                "expansion_coefficients_imag": vectors.imag.tolist(),
                "reference_optimal_point": np.asarray(ground.optimal_point).tolist(),
                "reference_optimizer_termination": termination,
                "reconstructed_root_hamiltonian_residuals": residuals
                if options.measure_residuals
                else None,
            }
        )
        return NativeOutcome(
            roots[options.target_root],
            active_energies_hartree=roots,
            target_root=options.target_root,
            converged=termination["converged"],
            diagnostics=diagnostics,
            num_qubits=context.num_qubits,
            ansatz=request.options.ansatz.name,
            optimizer=request.options.optimizer.name,
            parameter_count=circuit.num_parameters,
            optimizer_evaluations=len(evaluations),
            evaluations=evaluations,
            termination_reason="optimizer_returned_and_response_problem_validated",
        )

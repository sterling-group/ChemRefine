"""Sampled Krylov powers and excitation-expanded sampled quantum diagonalization."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from itertools import combinations
from math import comb
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
from pydantic import Field, model_validator

from chemrefine.engines.api import BackendRequirement
from chemrefine.engines.qiskit.components.subspace_algorithms import (
    SpinSector,
    SQDOptions,
    SubspaceOptions,
    build_sqd,
    check_sampling_memory,
    solve_explicit_samples,
    validate_subspace_options,
)
from chemrefine.engines.qiskit.determinants import apply_operators
from chemrefine.engines.qiskit.registry import ALGORITHMS, INITIAL_STATES
from chemrefine.errors import ConfigError

if TYPE_CHECKING:
    from chemrefine.engines.qiskit.native import NativeOutcome, NativeSolveRequest


class SKQDOptions(SubspaceOptions):
    """Sample powers of one fixed product-formula step, including the zero-time state."""

    projection: Literal["explicit"] = "explicit"
    time_step: float = Field(0.1, gt=0)
    num_steps: int = Field(4, ge=1)
    product_formula: Literal["lie", "suzuki"] = "suzuki"
    suzuki_order: Literal[2, 4, 6] = 2
    repetitions: int = Field(1, ge=1)
    max_circuits: int = Field(256, ge=2)

    @model_validator(mode="after")
    def _circuit_budget(self) -> Self:
        """Account for every evolution power and refuse an ignored formula order."""
        if self.num_steps + 1 > self.max_circuits:
            raise ValueError("SKQD circuit count exceeds max_circuits")
        if (self.num_steps + 1) * self.shots > self.max_total_shots:
            raise ValueError("SKQD shots exceed max_total_shots")
        if self.product_formula == "lie" and self.suzuki_order != 2:
            raise ValueError("suzuki_order is not consumed by the Lie formula")
        return self


class ExtendedSQDOptions(SQDOptions):
    """Extend sampled determinants with occupied-to-virtual reference excitations."""

    projection: Literal["explicit"] = "explicit"
    spin_constraint: Literal["require", "report"] = "report"
    excitation_ranks: tuple[Literal[1, 2, 3], ...] = (1, 2)
    minimum_probability: float = Field(0, ge=0, lt=1)
    max_excitation_operators: int = Field(10_000, ge=1)
    max_generated_determinants: int = Field(100_000, ge=1)

    @model_validator(mode="after")
    def _extension_budget(self) -> Self:
        """Keep excitation ranks distinct and reserve one final expanded solve."""
        if not self.excitation_ranks or len(set(self.excitation_ranks)) != len(
            self.excitation_ranks
        ):
            raise ValueError("excitation_ranks must be nonempty and distinct")
        iterations = self.max_iterations if self.configuration_recovery else 1
        extra = (
            0 if self.orbital_optimization is None else self.orbital_optimization.max_iterations + 1
        )
        if iterations * self.num_batches + extra + 1 > self.max_total_diagonalizations:
            raise ValueError("extended SQD solves exceed max_total_diagonalizations")
        return self


def krylov_circuits(request: NativeSolveRequest, options: SKQDOptions) -> tuple[Any, ...]:
    """Synthesize one approximate unitary once, then repeat exactly that circuit."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.synthesis import LieTrotter, SuzukiTrotter

    from chemrefine.engines.qiskit.mapping import map_problem

    context = map_problem(
        request.prepared, request.options.mapper, initial_state=request.options.initial_state
    )
    reference = INITIAL_STATES.build(request.options.initial_state, context=context)
    synthesis = (
        LieTrotter(reps=options.repetitions)
        if options.product_formula == "lie"
        else SuzukiTrotter(order=options.suzuki_order, reps=options.repetitions)
    )
    step = QuantumCircuit(context.num_qubits)
    step.append(
        PauliEvolutionGate(context.qubit_hamiltonian, time=options.time_step, synthesis=synthesis),
        range(context.num_qubits),
    )
    # Explicit synthesis prevents a simulator from substituting an exact matrix
    # exponential for the fixed approximate evolution gate.
    step = step.decompose()
    circuit_bytes = options.num_steps * (options.num_steps + 1) // 2 * step.size() * 256
    if circuit_bytes > options.max_memory_mb * 1024**2:
        raise ConfigError("SKQD circuit storage estimate exceeds max_memory_mb")
    circuits = []
    current = reference.copy()
    for power in range(options.num_steps + 1):
        circuits.append(current.copy())
        if power < options.num_steps:
            current.compose(step, inplace=True)
    return tuple(circuits)


@ALGORITHMS.register(
    "skqd",
    SKQDOptions,
    status="experimental",
    supported_domains=(
        "number-conserving molecular Hamiltonians",
        "fixed alpha/beta sector",
        "complex integrals",
    ),
    execution="native",
    requires=frozenset({"sampler", "initial_state", "mapper"}),
    backend_requirement=BackendRequirement(
        extra="qiskit-fermionic", import_name="qiskit_addon_sqd"
    ),
)
def build_skqd(*, options: SKQDOptions, request: NativeSolveRequest) -> NativeOutcome:
    """Acquire Krylov samples and diagonalize the original active Hamiltonian."""
    from chemrefine.engines.qiskit.sampling import SamplingSession

    validate_subspace_options(request.options)
    if request.initial_point is not None:
        raise ConfigError("SKQD does not accept variational initial parameters")
    data = SpinSector(request.prepared.num_spatial_orbitals, request.prepared.num_particles)
    check_sampling_memory(
        request, data, options, total_shots=(options.num_steps + 1) * options.shots
    )
    counts: Counter[str] = Counter()
    records = []
    circuits = krylov_circuits(request, options)
    with SamplingSession(
        request.options.sampler, device=request.options.device, cores=request.options.cores
    ) as session:
        for power, circuit in enumerate(circuits):
            batch = session.sample(circuit, shots=options.shots)
            counts.update(batch.counts)
            records.append(
                {
                    "power": power,
                    "time": power * options.time_step,
                    "shots": batch.shots,
                    "sampling": batch.metadata,
                }
            )
    return solve_explicit_samples(
        request,
        dict(counts),
        options,
        {
            "source": "skqd",
            "experimental": True,
            "fixed_step_operator": True,
            "time_step": options.time_step,
            "product_formula": options.product_formula,
            "suzuki_order": options.suzuki_order if options.product_formula == "suzuki" else None,
            "repetitions": options.repetitions,
            "includes_time_zero": True,
            "circuits": records,
        },
    )


def _reference_excitations(
    request: NativeSolveRequest, options: ExtendedSQDOptions
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Build a bounded spin-preserving pool from actual, possibly reordered occupations."""
    n = request.prepared.num_spatial_orbitals
    values = np.concatenate(
        (
            request.prepared.problem.orbital_occupations,
            request.prepared.problem.orbital_occupations_b,
        )
    )
    if values.shape != (2 * n,) or not np.all((values == 0) | (values == 1)):
        raise ConfigError("extended SQD requires actual binary reference occupations")
    occupied = tuple(int(mode) for mode in np.flatnonzero(values))
    virtual = tuple(int(mode) for mode in np.flatnonzero(1 - values))
    populations = request.prepared.num_particles
    size = sum(
        comb(populations[0], alpha)
        * comb(n - populations[0], alpha)
        * comb(populations[1], rank - alpha)
        * comb(n - populations[1], rank - alpha)
        for rank in options.excitation_ranks
        for alpha in range(rank + 1)
        if alpha <= min(populations[0], n - populations[0])
        and rank - alpha <= min(populations[1], n - populations[1])
    )
    if size > options.max_excitation_operators:
        raise ConfigError("extended SQD excitation pool exceeds max_excitation_operators")
    return tuple(
        (creators, tuple(reversed(annihilators)))
        for rank in options.excitation_ranks
        for annihilators in combinations(occupied, rank)
        for creators in combinations(virtual, rank)
        if sum(mode < n for mode in creators) == sum(mode < n for mode in annihilators)
    )


@ALGORITHMS.register(
    "extended_sqd",
    ExtendedSQDOptions,
    status="experimental",
    supported_domains=(
        "sampled molecular determinants",
        "reference singles/doubles/triples",
        "complex explicit projection",
    ),
    execution="native",
    requires=frozenset({"sampler"}),
    backend_requirement=BackendRequirement(
        extra="qiskit-fermionic", import_name="qiskit_addon_sqd"
    ),
)
def build_extended_sqd(
    *, options: ExtendedSQDOptions, request: NativeSolveRequest
) -> NativeOutcome:
    """Apply reference excitations to the sampled state and solve its explicit union."""
    validate_subspace_options(request.options)
    pool = _reference_excitations(request, options)
    base_options = {
        key: value for key, value in options.model_dump().items() if key in SQDOptions.model_fields
    }
    base_options.update(num_roots=1, target_root=0, orbital_optimization=None)
    source = build_sqd(options=SQDOptions(**base_options), request=replace(request, callback=None))
    state = source.states[0]
    selected = [
        bits
        for bits, amplitude in zip(state.determinants, state.amplitudes, strict=True)
        if abs(amplitude) ** 2 > options.minimum_probability
    ]
    if len(selected) * len(pool) > options.max_generated_determinants:
        raise ConfigError("extended SQD excitation applications exceed max_generated_determinants")
    basis = set(state.determinants)
    for bits in selected:
        for creators, annihilators in pool:
            moved = apply_operators(bits, creators, annihilators)
            if moved is not None:
                basis.add(moved[0])
                if (
                    len(basis) > options.max_subspace_dimension
                    or len(basis) * 192 > options.max_memory_mb * 1024**2
                ):
                    raise ConfigError("extended SQD basis exceeds its subspace or memory budget")
    # This is a deterministic generated-basis solve. Unit weights are an internal
    # adapter input, never reported as quantum measurements or acquired shots.
    generated = {format(bits, f"0{state.num_modes}b"): 1 for bits in sorted(basis)}
    final_options = SQDOptions(
        **{
            **base_options,
            "counts": generated,
            "parameter_values": None,
            "num_roots": options.num_roots,
            "target_root": options.target_root,
            "orbital_optimization": options.orbital_optimization,
            "configuration_recovery": False,
            "samples_per_batch": len(basis),
            "num_batches": 1,
            "max_total_shots": max(options.max_total_shots, len(basis)),
        }
    )
    outcome = solve_explicit_samples(
        request,
        generated,
        final_options,
        {
            "source": "extended_sqd",
            "experimental": True,
            "acquisition": source.diagnostics["sampling"],
            "source_recovery": source.diagnostics["iterations"],
            "excitation_ranks": list(options.excitation_ranks),
            "excitation_operators": len(pool),
            "source_dimension": len(state.determinants),
            "expanded_dimension": len(basis),
            "minimum_probability": options.minimum_probability,
        },
    )
    diagnostics = dict(outcome.diagnostics)
    for key in (
        "input_shots",
        "input_unique_bitstrings",
        "valid_shots",
        "valid_unique_bitstrings",
        "invalid_particle_fraction",
    ):
        diagnostics[key] = source.diagnostics[key]
    diagnostics["generated_determinants"] = len(basis) - len(state.determinants)
    return replace(outcome, diagnostics=diagnostics, ansatz=source.ansatz)

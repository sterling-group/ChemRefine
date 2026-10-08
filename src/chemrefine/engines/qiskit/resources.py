"""Analytical qubitization/QPE resources with explicit conditional assumptions.

These functions produce numerical estimates, never executable circuits. Query
counts use textbook finite-register phase estimation of a qubitized walk, whose
eigenphase obeys E/lambda = cos(2*pi*phase). Hardware costs require a separate,
fully specified workload and machine model.
"""

from __future__ import annotations

import math
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator


class ResourceModel(BaseModel):
    """Frozen, finite and typo-intolerant resource assumptions."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class QPEBudget(ResourceModel):
    """Absolute energy errors and a conditional target-sampling probability.

    ``target_overlap`` is a supplied lower bound on squared overlap, not an
    overlap computed from a trial state. Independent preparations are assumed.
    """

    energy_error_hartree: float = Field(gt=0)
    representation_error_hartree: float = Field(0.0, ge=0)
    synthesis_error_hartree: float = Field(0.0, ge=0)
    failure_probability: float = Field(0.01, ge=1e-15, lt=1)
    target_overlap: float = Field(1.0, ge=1e-15, le=1)
    max_phase_bits: StrictInt = Field(128, ge=1, le=4096)
    max_repetitions: StrictInt = Field(1000000, ge=1)

    @model_validator(mode="after")
    def _remaining_precision(self) -> Self:
        """Reserve positive precision for phase estimation itself."""
        if self.qpe_error_hartree <= 0:
            raise ValueError("representation and synthesis errors exhaust the energy error budget")
        return self

    @property
    def qpe_error_hartree(self) -> float:
        """Return the energy-error allowance remaining for phase estimation."""
        return (
            self.energy_error_hartree
            - self.representation_error_hartree
            - self.synthesis_error_hartree
        )


class WalkOracleCost(ResourceModel):
    """User-supplied costs of one controlled walk, including its inverse prepares.

    Logical qubits include the system, selection and workspace registers, but
    exclude the QPE phase register. Rotations remain unsynthesized counts.
    """

    logical_qubits: StrictInt = Field(ge=1)
    t: StrictInt = Field(0, ge=0)
    toffoli: StrictInt = Field(0, ge=0)
    clifford: StrictInt = Field(0, ge=0)
    rotations: StrictInt = Field(0, ge=0)
    provenance: str = Field(min_length=1)


class PauliResourceOptions(ResourceModel):
    """A Hermitian Pauli sum in Hartree and optional explicit oracle costs."""

    hamiltonian: dict[str, float] = Field(min_length=1)
    budget: QPEBudget
    oracle: WalkOracleCost | None = None
    max_terms: StrictInt = Field(1000000, ge=1)

    @model_validator(mode="after")
    def _pauli_domain(self) -> Self:
        """Reject malformed labels and oversized Hamiltonians without any SDK."""
        widths = {len(label) for label in self.hamiltonian}
        if (
            len(widths) != 1
            or 0 in widths
            or any(set(label) - set("IXYZ") for label in self.hamiltonian)
        ):
            raise ValueError("hamiltonian requires equal-width nonempty I/X/Y/Z Pauli labels")
        if len(self.hamiltonian) > self.max_terms:
            raise ValueError("Pauli Hamiltonian exceeds max_terms")
        if self.oracle is not None and self.oracle.logical_qubits < next(iter(widths)):
            raise ValueError("oracle logical_qubits must include all system qubits")
        return self


def qpe_queries(normalization: float, budget: QPEBudget) -> dict[str, Any]:
    """Bound standard QPE queries and independent repetitions.

    With m accuracy bits and t=m+ceil(log2(2+1/(2*delta))) total
    bits, standard QPE estimates circular phase within 2**(-m) except
    with probability delta. The cosine is 2*pi*lambda Lipschitz in phase.
    Half the requested failure budget covers missing the target in all
    preparations; half covers any inaccurate QPE sample by a union bound.
    """
    if not math.isfinite(normalization) or normalization < 0:
        raise ValueError("normalization must be finite and nonnegative")
    if normalization == 0:
        return {
            "normalization_hartree": 0.0,
            "accuracy_bits": 0,
            "phase_bits": 0,
            "repetitions": 0,
            "controlled_walk_queries_per_run": 0,
            "controlled_walk_queries": 0,
            "conditional_failure_bound": 0.0,
            "budget": budget.model_dump(),
            "protocol": "constant Hamiltonian; no phase estimation",
        }
    overlap = budget.target_overlap
    repeats = (
        1
        if overlap == 1
        else math.ceil(math.log(budget.failure_probability / 2) / math.log1p(-overlap))
    )
    if repeats > budget.max_repetitions:
        raise ValueError("target overlap requires more than max_repetitions")
    delta = budget.failure_probability / (2 * repeats)
    # Taking logs separately avoids overflow of lambda / epsilon.
    bits = max(
        1,
        math.ceil(
            math.log2(2 * math.pi) + math.log2(normalization) - math.log2(budget.qpe_error_hartree)
        ),
    )
    total_bits = bits + math.ceil(math.log2(2 + 1 / (2 * delta)))
    if total_bits > budget.max_phase_bits:
        raise ValueError("requested precision exceeds max_phase_bits")
    queries = (1 << total_bits) - 1
    return {
        "normalization_hartree": normalization,
        "accuracy_bits": bits,
        "phase_bits": total_bits,
        "repetitions": repeats,
        "qpe_failure_per_run": delta,
        "controlled_walk_queries_per_run": queries,
        "controlled_walk_queries": queries * repeats,
        "conditional_failure_bound": (1 - overlap) ** repeats + repeats * delta,
        "budget": budget.model_dump(),
        "protocol": "standard inverse-QFT QPE on a qubitized walk; independent preparations",
        "success_event": "all samples accurate and target eigenvalue sampled at least once",
        "assumptions": [
            "supplied squared target-overlap lower bound holds for each preparation",
            "representation and synthesis energy-error bounds hold",
            "ideal controlled walk and inverse QFT within those supplied error bounds",
        ],
    }


def estimate_pauli_resources(options: PauliResourceOptions) -> dict[str, Any]:
    """Compute the exact Pauli-LCU one-norm and conservative QPE query bound."""
    width = len(next(iter(options.hamiltonian)))
    scalar = options.hamiltonian.get("I" * width, 0.0)
    terms = {
        label: value
        for label, value in options.hamiltonian.items()
        if label != "I" * width and value != 0
    }
    normalization = math.fsum(abs(value) for value in terms.values())
    result = qpe_queries(normalization, options.budget)
    result.update(
        estimate_kind="analytical_query_bound",
        system_qubits=width,
        nonidentity_terms=len(terms),
        identity_offset_hartree=scalar,
        lcu_selection_qubits=max(0, (len(terms) - 1).bit_length()) if terms else 0,
        normalization_convention="sum absolute nonidentity Pauli coefficients",
        executable_circuit=False,
        exclusions=[
            "state preparation",
            "inverse QFT gate synthesis",
            "routing",
            "error correction",
            "magic-state factories",
        ],
    )
    if options.oracle is not None:
        oracle = options.oracle
        result["oracle_assumptions"] = oracle.model_dump()
        result["controlled_walk_gate_counts"] = {
            key: getattr(oracle, key) * result["controlled_walk_queries"]
            for key in ("t", "toffoli", "clifford", "rotations")
        }
        result["logical_qubits_including_phase_register"] = (
            oracle.logical_qubits + result["phase_bits"] if normalization else 0
        )
    return result


class MagicFactory(ResourceModel):
    """Explicit factory throughput and output error, including its internal faults."""

    count: StrictInt = Field(ge=1)
    physical_qubits_per_factory: StrictInt = Field(ge=1)
    cycles_per_state: float = Field(gt=0)
    output_error_probability: float = Field(ge=0, lt=1)
    provenance: str = Field(min_length=1)


class SurfaceCodeOptions(ResourceModel):
    """A declared workload and phenomenological surface-code machine model.

    The logical error model is A*(p/p_threshold)**((d+1)/2) per
    data patch per cycle. No scheduling or factory design is inferred.
    """

    logical_qubits: StrictInt = Field(ge=1)
    logical_cycles: StrictInt = Field(ge=1)
    t_states: StrictInt = Field(0, ge=0)
    ccz_states: StrictInt = Field(0, ge=0)
    code_distance: StrictInt = Field(ge=3)
    physical_error_probability: float = Field(gt=0, lt=1)
    threshold_probability: float = Field(gt=0, lt=1)
    logical_error_prefactor: float = Field(gt=0)
    physical_qubits_per_patch_d2: float = Field(gt=0)
    routing_patch_multiplier: float = Field(ge=1)
    cycle_time_seconds: float = Field(gt=0)
    failure_budget: float = Field(gt=0, lt=1)
    t_factory: MagicFactory | None = None
    ccz_factory: MagicFactory | None = None
    hardware_provenance: str = Field(min_length=1)

    @model_validator(mode="after")
    def _hardware_domain(self) -> Self:
        """Require below-threshold operation and a factory for each consumed state."""
        if self.code_distance % 2 != 1:
            raise ValueError("surface-code distance must be odd")
        if self.physical_error_probability >= self.threshold_probability:
            raise ValueError("physical error must be below the supplied threshold")
        for name in ("t", "ccz"):
            if getattr(self, name + "_states") and getattr(self, name + "_factory") is None:
                raise ValueError(f"{name} states require an explicit {name}_factory")
        return self


def estimate_surface_code(options: SurfaceCodeOptions) -> dict[str, Any]:
    """Evaluate a throughput lower bound and a stated-model failure union bound."""
    patches = math.ceil(options.logical_qubits * options.routing_patch_multiplier)
    physical_qubits = math.ceil(
        patches * options.physical_qubits_per_patch_d2 * options.code_distance**2
    )
    cycles = options.logical_cycles
    factory_failure = 0.0
    for name in ("t", "ccz"):
        factory = getattr(options, name + "_factory")
        if factory is not None:
            states = getattr(options, name + "_states")
            cycles = max(cycles, math.ceil(states * factory.cycles_per_state / factory.count))
            physical_qubits += factory.count * factory.physical_qubits_per_factory
            factory_failure += states * factory.output_error_probability
    logical_per_cycle = options.logical_error_prefactor * (
        options.physical_error_probability / options.threshold_probability
    ) ** ((options.code_distance + 1) / 2)
    data_failure = patches * cycles * logical_per_cycle
    failure = min(1.0, data_failure + factory_failure)
    return {
        "estimate_kind": "phenomenological_surface_code_model",
        "executable_circuit": False,
        "assumptions": options.model_dump(),
        "physical_qubits": physical_qubits,
        "data_and_routing_patches": patches,
        "runtime_cycles_lower_bound": cycles,
        "runtime_seconds_lower_bound": cycles * options.cycle_time_seconds,
        "logical_error_per_patch_cycle": logical_per_cycle,
        "data_failure_union_bound": min(1.0, data_failure),
        "factory_failure_union_bound": min(1.0, factory_failure),
        "failure_union_bound_at_runtime_lower_bound": failure,
        "meets_failure_budget_at_runtime_lower_bound": failure <= options.failure_budget,
        "exclusions": [
            "schedule and dependency stalls",
            "factory warm-up and buffering",
            "decoder latency",
            "correlated faults beyond the supplied error model",
        ],
    }

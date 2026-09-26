"""Owned TETRIS and coupled-exchange adaptive VQE with explicit experimental domains."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from chemrefine.engines.qiskit.ceo import exchange_pool
from chemrefine.engines.qiskit.context import AnsatzArtifacts, ElectronicStructureContext
from chemrefine.engines.qiskit.native import NativeOutcome, NativeSolveRequest
from chemrefine.engines.qiskit.options import ComponentSelection, QiskitOptions
from chemrefine.engines.qiskit.registry import ALGORITHMS, ANSATZE
from chemrefine.errors import ConfigError


class ExchangePoolOptions(BaseModel):
    """Generalized N/Ms-preserving exchanges and optional complex-amplitude quadratures."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    include_imaginary: bool = False
    max_pool_size: int = Field(4096, ge=1)


class AdaptiveOptions(BaseModel):
    """Bounded adaptive growth; final-sector checks concern the retained optimized state."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    max_iterations: int = Field(50, ge=1)
    max_parameters: int = Field(512, ge=1)
    max_pool_size: int = Field(4096, ge=1)
    max_evaluations: int = Field(10000, ge=1)
    max_measurements: int = Field(100000, ge=1)
    max_pauli_terms: int = Field(100000, ge=1)
    max_product_terms: int = Field(1000000, ge=1)
    gradient_threshold: float = Field(1e-5, gt=0)
    gradient_norm: Literal["l2", "max"] = "l2"
    selection_threshold: float = Field(1e-12, ge=0)
    eigenvalue_threshold: float = Field(1e-8, ge=0)
    energy_increase_tolerance: float = Field(1e-7, ge=0)
    evolution: Literal["exact_commuting", "lie_trotter", "suzuki"] = "exact_commuting"
    repetitions: int = Field(1, ge=1)
    suzuki_order: Literal[2, 4, 6] = 2
    sector_tolerance: float = Field(1e-3, gt=0)
    spin_constraint: Literal["require", "report"] = "require"
    spin_tolerance: float = Field(1e-3, gt=0)


class CEOOptions(AdaptiveOptions):
    """Published gradient-based OVP/MVP choice, or explicitly selected restricted variants."""

    variant: Literal["adaptive", "ovp", "mvp"] = "adaptive"
    tetris: bool = True


def validate_adaptive_options(options: QiskitOptions) -> None:
    """Reject a fixed-circuit initializer that cannot seed an initially empty adaptive circuit."""
    if options.initial_point != ComponentSelection.named("zeros"):
        raise ConfigError(
            "owned adaptive solvers start with no parameters; initial_point must be zeros"
        )


@ANSATZE.register(
    "qe",
    ExchangePoolOptions,
    capabilities=frozenset({"operator_pool"}),
    status="experimental",
    supported_domains=("occupation_qubits", "jordan_wigner", "tapered_jordan_wigner"),
)
def build_qe_pool(
    *, options: ExchangePoolOptions, context: ElectronicStructureContext, initial_state: object
) -> AnsatzArtifacts:
    """Build generalized qubit exchanges, retaining actual mapped support metadata."""
    del initial_state
    return exchange_pool(context, coupled=False, **options.model_dump())


@ANSATZE.register(
    "ceo",
    ExchangePoolOptions,
    capabilities=frozenset({"operator_pool", "ceo_pool"}),
    status="experimental",
    supported_domains=("occupation_qubits", "jordan_wigner", "tapered_jordan_wigner"),
)
def build_ceo_pool(
    *, options: ExchangePoolOptions, context: ElectronicStructureContext, initial_state: object
) -> AnsatzArtifacts:
    """Build OVP candidates and their independent QE parents for coupled-exchange selection."""
    del initial_state
    return exchange_pool(context, coupled=True, **options.model_dump())


@ALGORITHMS.register(
    "tetris_adapt",
    AdaptiveOptions,
    execution="native",
    capabilities=frozenset({"rebuilds_optimizer", "bound_circuit"}),
    requires=frozenset({"mapper", "estimator", "optimizer", "operator_pool", "initial_state"}),
    status="experimental",
    supported_domains=("mapped_hermitian_pools", "fixed_particle_and_spin_projection"),
)
def build_tetris_adapt(*, options: AdaptiveOptions, request: NativeSolveRequest) -> NativeOutcome:
    """Grow disjoint-support operator blocks and jointly reoptimize all retained parameters."""
    from chemrefine.engines.qiskit.adaptive import adaptive_solve

    return adaptive_solve(request, options, ceo=False)


@ALGORITHMS.register(
    "ceo_adapt",
    CEOOptions,
    execution="native",
    capabilities=frozenset({"rebuilds_optimizer", "bound_circuit"}),
    requires=frozenset(
        {"mapper", "estimator", "optimizer", "operator_pool", "ceo_pool", "initial_state"}
    ),
    status="experimental",
    supported_domains=("occupation_qubits", "jordan_wigner", "tapered_jordan_wigner"),
)
def build_ceo_adapt(*, options: CEOOptions, request: NativeSolveRequest) -> NativeOutcome:
    """Select coupled exchanges with optional TETRIS batching and per-iteration metric rebuilds."""
    from chemrefine.engines.qiskit.adaptive import adaptive_solve

    return adaptive_solve(request, options, ceo=True)

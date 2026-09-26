"""Native UCJ/LUCJ variational energy minimization in a fixed electron sector."""

from __future__ import annotations

from copy import deepcopy
from math import comb
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from chemrefine.engines.api import BackendRequirement
from chemrefine.engines.qiskit.fermionic import fermionic_integrals, to_ffsim_hamiltonian
from chemrefine.engines.qiskit.options import QiskitOptions
from chemrefine.engines.qiskit.registry import ALGORITHMS, OPTIMIZERS
from chemrefine.errors import ConfigError

if TYPE_CHECKING:
    from chemrefine.engines.qiskit.native import NativeOutcome, NativeSolveRequest

InteractionBlock = tuple[tuple[StrictInt, StrictInt], ...] | None


class FfsimVQEOptions(BaseModel):
    """Numeric UCJ controls for ffsim's released spinful operator families.

    LUCJ defaults to nearest-neighbor same-spin and onsite opposite-spin
    interactions. Custom pairs explicitly replace this pattern. Random numeric
    initialization avoids the stationary all-zero UCJ parameterization; callers
    may instead supply a complete parameter vector. No classical amplitudes are
    calculated. A final spin-eigenstate check enforces the requested multiplicity.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    ansatz: Literal["ucj", "lucj"] = "lucj"
    spin_variant: Literal["balanced", "unbalanced"] = "balanced"
    n_reps: int = Field(1, ge=1)
    interaction_pairs: tuple[InteractionBlock, ...] | None = None
    with_final_orbital_rotation: bool = False
    initial_parameters: tuple[float, ...] | None = None
    initialization: Literal["random", "zeros"] = "random"
    seed: int | None = Field(0, ge=0)
    initial_scale: float = Field(0.1, gt=0)
    spin_tolerance: float = Field(1e-5, gt=0)
    max_statevector_dimension: int = Field(1_000_000, ge=1, strict=True)
    max_parameters: int = Field(100_000, ge=1, strict=True)
    max_evaluations: int = Field(10_000, ge=1, strict=True)
    max_memory_mb: int = Field(512, ge=1, strict=True)

    @model_validator(mode="after")
    def _validate_interactions(self) -> Self:
        """Validate pair conventions before problem dimensions are available."""
        if self.interaction_pairs is None:
            return self
        count = 2 if self.spin_variant == "balanced" else 3
        if len(self.interaction_pairs) != count:
            raise ValueError(f"{self.spin_variant} UCJ requires {count} interaction blocks")
        for block_index, pairs in enumerate(self.interaction_pairs):
            if pairs is None:
                continue
            if len(set(pairs)) != len(pairs):
                raise ValueError("interaction pairs must be distinct within each spin block")
            for first, second in pairs:
                if min(first, second) < 0:
                    raise ValueError("interaction orbital indices must be non-negative")
                if first > second and (self.spin_variant == "balanced" or block_index != 1):
                    raise ValueError(
                        "same-spin/balanced interaction pairs must be upper triangular"
                    )
        return self


def validate_ffsim_options(resolved: QiskitOptions) -> None:
    """Reject unused circuit components without importing a simulation provider."""
    defaults = QiskitOptions()
    for category in ("ansatz", "mapper", "initial_state", "estimator", "sampler", "initial_point"):
        if getattr(resolved, category) != getattr(defaults, category):
            raise ConfigError(
                f"qiskit ffsim_vqe does not consume the {category} component; "
                "use algorithm options for UCJ/LUCJ and numeric initialization. "
                "The reference uses the actual hartree_fock occupations"
            )
    if resolved.device != "cpu":
        raise ConfigError("qiskit ffsim_vqe supports only device: cpu")


def _interaction_pairs(options: FfsimVQEOptions, norb: int) -> Any:
    """Resolve a documented LUCJ pattern or the user's numeric interaction graph."""
    if options.interaction_pairs is not None:
        pairs = tuple(None if block is None else list(block) for block in options.interaction_pairs)
        if any(
            index >= norb
            for block in pairs
            if block is not None
            for pair in block
            for index in pair
        ):
            raise ConfigError(f"qiskit ffsim interaction indices must be below {norb}")
        return pairs
    if options.ansatz == "ucj":
        return None
    same_spin = [(index, index + 1) for index in range(norb - 1)]
    opposite_spin = [(index, index) for index in range(norb)]
    if options.spin_variant == "balanced":
        return same_spin, opposite_spin
    return same_spin, opposite_spin, list(same_spin)


def _parameters(value: Any, size: int, *, label: str) -> np.ndarray:
    """Require a finite real vector with the released ansatz's exact dimension."""
    try:
        if np.iscomplexobj(value):
            raise ValueError("complex parameters")
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"qiskit ffsim {label} must contain finite real parameters") from exc
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ConfigError(f"qiskit ffsim {label} requires exactly {size} finite real parameters")
    return result.copy()


@ALGORITHMS.register(
    "ffsim_vqe",
    FfsimVQEOptions,
    execution="native",
    backend_requirement=BackendRequirement(extra="qiskit-fermionic", import_name="ffsim"),
    requires=frozenset({"optimizer"}),
)
def build_ffsim_vqe(*, options: FfsimVQEOptions, request: NativeSolveRequest) -> NativeOutcome:
    """Optimize an exact fixed-sector objective through the optimizer registry.

    This is classical noiseless fermionic simulation, not a shot estimator or a
    hardware submission. Its determinant basis still grows combinatorially.
    """
    validate_ffsim_options(request.options)

    from pyscf.lib import with_omp_threads

    with with_omp_threads(request.options.cores):
        return _run_ffsim_vqe(options=options, request=request)


def _run_ffsim_vqe(*, options: FfsimVQEOptions, request: NativeSolveRequest) -> NativeOutcome:
    """Execute numeric UCJ optimization within the caller's CPU thread budget."""

    import ffsim
    from pyscf.fci.spin_op import contract_ss

    from chemrefine.engines.qiskit.native import NativeOutcome

    data = fermionic_integrals(request.prepared)
    dimension = comb(data.norb, data.nelec[0]) * comb(data.norb, data.nelec[1])
    if dimension > options.max_statevector_dimension:
        raise ConfigError("ffsim determinant-sector dimension exceeds max_statevector_dimension")
    # Allow several simultaneous complex working vectors and integral/rotation
    # temporaries. This is a planning bound, not an operating-system memory limit.
    estimated_bytes = 128 * dimension + 64 * data.norb**4
    if estimated_bytes > options.max_memory_mb * 1024**2:
        raise ConfigError("ffsim estimated working storage exceeds max_memory_mb")
    operator_type = (
        ffsim.UCJOpSpinBalanced if options.spin_variant == "balanced" else ffsim.UCJOpSpinUnbalanced
    )
    interaction_pairs = _interaction_pairs(options, data.norb)
    settings = {
        "norb": data.norb,
        "n_reps": options.n_reps,
        "interaction_pairs": interaction_pairs,
        "with_final_orbital_rotation": options.with_final_orbital_rotation,
    }
    parameter_count = operator_type.n_params(**settings)
    if parameter_count > options.max_parameters:
        raise ConfigError("ffsim ansatz size exceeds max_parameters")
    if request.initial_point is not None and options.initial_parameters is not None:
        raise ConfigError(
            "supply ffsim initial parameters through the API or algorithm options, not both"
        )
    supplied = (
        request.initial_point if request.initial_point is not None else options.initial_parameters
    )
    if supplied is not None:
        initial = _parameters(supplied, parameter_count, label="initial point")
        initialization = "supplied"
    elif options.initialization == "zeros":
        initial = np.zeros(parameter_count)
        initialization = "zeros"
    else:
        initial = np.random.default_rng(options.seed).uniform(
            -options.initial_scale, options.initial_scale, parameter_count
        )
        initialization = "random"
    reference = ffsim.slater_determinant(data.norb, data.occupations)
    hamiltonian = ffsim.linear_operator(to_ffsim_hamiltonian(data), data.norb, data.nelec)
    evaluations: list[dict[str, Any]] = []

    def statevector(parameters: np.ndarray) -> np.ndarray:
        """Apply one numeric UCJ operator to the actual active reference determinant."""
        operator = operator_type.from_parameters(parameters, **settings)
        return np.asarray(ffsim.apply_unitary(reference, operator, data.norb, data.nelec))

    def energy(parameters: np.ndarray) -> float:
        """Evaluate and record only the active electronic energy."""
        if len(evaluations) >= options.max_evaluations:
            raise ConfigError("ffsim objective calls exceed max_evaluations")
        state = statevector(_parameters(parameters, parameter_count, label="objective point"))
        value = np.vdot(state, hamiltonian @ state)
        if not np.isfinite(value) or abs(value.imag) > 1e-8:
            raise ConfigError("qiskit ffsim objective returned a non-finite or complex energy")
        real_value = float(value.real)
        record = {
            "evaluation": len(evaluations) + 1,
            "objective_value_hartree": real_value,
            "metadata": {"simulation": "ffsim", "shots": None},
        }
        evaluations.append(record)
        if request.callback is not None:
            request.callback(deepcopy(record))
        return real_value

    optimizer = OPTIMIZERS.build(request.options.optimizer)
    optimization = optimizer.minimize(energy, initial)
    optimal_parameters = _parameters(optimization.x, parameter_count, label="optimizer result")
    final_state = statevector(optimal_parameters)
    active_energy = np.vdot(final_state, hamiltonian @ final_state)
    if not np.isfinite(active_energy) or abs(active_energy.imag) > 1e-8:
        raise ConfigError("qiskit ffsim optimizer returned an invalid final electronic energy")
    # Check the eigenstate residual, not merely <S²>: a mixture of lower and
    # higher spin sectors can have the requested mean without belonging to it.
    spin_applied = contract_ss(final_state.real, data.norb, data.nelec).astype(complex)
    spin_applied += 1j * contract_ss(final_state.imag, data.norb, data.nelec)
    spin_applied = spin_applied.reshape(-1)
    spin_squared = float(np.vdot(final_state, spin_applied).real)
    spin_residual = float(np.linalg.norm(spin_applied - data.target_spin_squared * final_state))
    if not np.isfinite(spin_squared) or not np.isfinite(spin_residual):
        raise ConfigError("qiskit ffsim returned non-finite spin diagnostics")
    if spin_residual > options.spin_tolerance:
        raise ConfigError(
            "qiskit ffsim optimized state does not have the requested spin: "
            f"S²={spin_squared:.8g}, target={data.target_spin_squared:.8g}, "
            f"spin residual={spin_residual:.8g} exceeds {options.spin_tolerance:.8g}; "
            "UCJ conserves alpha/beta counts but does not guarantee total spin"
        )
    success = getattr(optimization, "success", None)
    converged = None if success is None else bool(success)
    termination = getattr(optimization, "message", None)
    return NativeOutcome(
        active_energy_hartree=float(active_energy.real),
        converged=converged,
        termination_reason=str(termination) if termination else "optimizer_returned_without_status",
        num_qubits=None,
        ansatz=f"{options.ansatz}_{options.spin_variant}",
        optimizer=request.options.optimizer.name,
        parameter_count=parameter_count,
        optimizer_evaluations=getattr(optimization, "nfev", None),
        evaluations=evaluations,
        diagnostics={
            "simulation": "ffsim_fixed_particle_sector",
            "statevector_dimension": dimension,
            "estimated_working_bytes": estimated_bytes,
            "num_parameters": parameter_count,
            "n_reps": options.n_reps,
            "interaction_pairs": interaction_pairs,
            "initialization": initialization,
            "seed": options.seed,
            "initial_parameters": initial.tolist(),
            "optimal_parameters": optimal_parameters.tolist(),
            "reference_occupations": data.occupations,
            "num_particles": data.nelec,
            "spin_squared": spin_squared,
            "target_spin_squared": data.target_spin_squared,
            "spin_eigenstate_residual": spin_residual,
            "spin_tolerance": options.spin_tolerance,
            "optimizer_nfev": getattr(optimization, "nfev", None),
            "optimizer_nit": getattr(optimization, "nit", None),
        },
    )

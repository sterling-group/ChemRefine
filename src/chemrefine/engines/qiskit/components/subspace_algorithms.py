"""Sample-based molecular solvers with bounded classical subspace processing."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from math import isqrt
from typing import TYPE_CHECKING, Any, Self, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from chemrefine.engines.api import BackendRequirement
from chemrefine.engines.qiskit.registry import (
    ALGORITHMS,
    ANSATZE,
    INITIAL_POINTS,
    INITIAL_STATES,
    SAMPLERS,
)
from chemrefine.errors import ConfigError

if TYPE_CHECKING:
    from chemrefine.engines.qiskit.components.samplers import AerSamplerOptions
    from chemrefine.engines.qiskit.fermionic import FermionicIntegrals
    from chemrefine.engines.qiskit.native import NativeOutcome, NativeSolveRequest
    from chemrefine.engines.qiskit.options import QiskitOptions


class SubspaceOptions(BaseModel):
    """Budgets and numerical controls shared by sampled-subspace algorithms."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    shots: int = Field(4096, ge=1)
    samples_per_batch: int = Field(100, ge=1)
    num_batches: int = Field(3, ge=1)
    max_iterations: int = Field(10, ge=1)
    max_subspace_dimension: int = Field(10_000, ge=1)
    max_total_diagonalizations: int = Field(1_000, ge=1)
    max_total_shots: int = Field(1_000_000, ge=1)
    max_memory_mb: int = Field(512, ge=1)
    sci_max_cycle: int = Field(100, ge=1)
    sci_max_space: int = Field(12, ge=2)
    energy_tol: float = Field(1e-8, gt=0)
    occupancies_tol: float = Field(1e-5, gt=0)
    spin_tolerance: float = Field(1e-5, gt=0)
    configuration_recovery: bool = True
    symmetrize_spin: bool = False
    seed: int | None = Field(0, ge=0)

    @model_validator(mode="after")
    def _diagonalization_budget(self) -> Self:
        """Bound the total requested batch solves, including recovery iterations."""
        iterations = self.max_iterations if self.configuration_recovery else 1
        if iterations * self.num_batches > self.max_total_diagonalizations:
            raise ValueError("SQD batch solves exceed max_total_diagonalizations")
        return self


class SQDOptions(SubspaceOptions):
    """SQD from supplied canonical counts or a fixed-parameter registered ansatz."""

    counts: dict[str, StrictInt] | None = None
    parameter_values: list[float] | None = None

    @field_validator("counts")
    @classmethod
    def _valid_counts(cls, value: dict[str, int] | None) -> dict[str, int] | None:
        """Require positive integer frequencies and canonical binary strings."""
        if value is not None and (
            not value
            or any(not bits or set(bits) - {"0", "1"} or count < 1 for bits, count in value.items())
        ):
            raise ValueError("counts must contain nonempty binary strings with positive counts")
        return value

    @model_validator(mode="after")
    def _separate_inputs(self) -> Self:
        """Avoid an ignored parameter specification when external counts are given."""
        if self.counts is not None and self.parameter_values is not None:
            raise ValueError("counts and parameter_values are mutually exclusive")
        return self


class SqDRIFTOptions(SubspaceOptions):
    """Experimental grouped-qDRIFT sampler followed by the ordinary SQD solver."""

    times: tuple[float, ...] = (0.1, 0.2, 0.3)
    num_groups: int = Field(10, ge=1)
    randomizations: int = Field(10, ge=1)
    max_circuits: int = Field(256, ge=1)

    @field_validator("times")
    @classmethod
    def _nonempty_times(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        """Require at least one finite nonnegative propagation time."""
        if not value or any(time < 0 or not np.isfinite(time) for time in value):
            raise ValueError("times must contain finite nonnegative evolution times")
        return value

    @model_validator(mode="after")
    def _sampling_budget(self) -> Self:
        """Reject excessive circuit and shot requests before any execution."""
        circuits = len(self.times) * self.randomizations
        if circuits > self.max_circuits:
            raise ValueError("SqDRIFT circuit count exceeds max_circuits")
        if circuits * self.shots > self.max_total_shots:
            raise ValueError("SqDRIFT total shots exceed max_total_shots")
        return self


def validate_subspace_options(resolved: QiskitOptions) -> None:
    """Reject explicitly configured components the selected subspace path ignores."""
    from chemrefine.engines.qiskit.options import QiskitOptions

    if resolved.algorithm.name not in {"sqd", "sqdrift"}:
        return
    if resolved.mapper.name != "jordan_wigner":
        raise ConfigError("SQD and SqDRIFT require the untapered jordan_wigner mapper")
    if (
        resolved.algorithm.name == "sqd"
        and resolved.algorithm.options.get("counts") is None
        and "circuit" not in ANSATZE.spec(resolved.ansatz.name).capabilities
    ):
        raise ConfigError("SQD sampling requires an ansatz supplying a circuit")
    unused = {"estimator", "optimizer"}
    if resolved.algorithm.name == "sqdrift":
        unused.update({"ansatz", "initial_state", "initial_point"})
    elif resolved.algorithm.options.get("counts") is not None:
        unused.update({"ansatz", "initial_state", "initial_point", "sampler"})
        if resolved.device != "cpu":
            raise ConfigError("SQD supplied-count processing requires device: cpu")
    defaults = QiskitOptions()
    for category in sorted(unused):
        if getattr(resolved, category) != getattr(defaults, category):
            raise ConfigError(
                f"{resolved.algorithm.name} does not consume the configured {category}; "
                "leave this component at its default"
            )


def _chemistry(request: NativeSolveRequest) -> FermionicIntegrals:
    """Restrict occupation bitstrings to the supported untapered JW representation."""
    from chemrefine.engines.qiskit.fermionic import fermionic_integrals

    if request.options.mapper.name != "jordan_wigner":
        raise ConfigError("SQD and SqDRIFT require the untapered jordan_wigner mapper")
    data = fermionic_integrals(request.prepared)
    if min(data.nelec) < 1:
        raise ConfigError(
            "the released PySCF SQD solver requires at least one alpha and one beta electron"
        )
    return data


def _checked_counts(
    counts: dict[str, int], data: FermionicIntegrals, options: SubspaceOptions
) -> tuple[dict[str, int], dict[str, Any]]:
    """Validate sample width, frequencies, budget and spin-resolved particle counts."""
    if not counts:
        raise ConfigError("sample-based diagonalization requires nonempty counts")
    width = 2 * data.norb
    if any(
        not isinstance(bits, str)
        or len(bits) != width
        or set(bits) - {"0", "1"}
        or isinstance(count, bool)
        or not isinstance(count, (int, np.integer))
        or count < 1
        for bits, count in counts.items()
    ):
        raise ConfigError(
            f"samples must be {width}-bit canonical binary strings with positive counts"
        )
    total = sum(counts.values())
    if total > options.max_total_shots:
        raise ConfigError("sample count exceeds max_total_shots")
    valid = {
        bits: int(count)
        for bits, count in counts.items()
        if (bits[data.norb :].count("1"), bits[: data.norb].count("1")) == data.nelec
    }
    valid_shots = sum(valid.values())
    if not valid:
        raise ConfigError("samples contain no determinants in the requested alpha/beta sector")
    metadata = {
        "input_shots": int(total),
        "input_unique_bitstrings": len(counts),
        "valid_shots": valid_shots,
        "valid_unique_bitstrings": len(valid),
        "invalid_particle_fraction": 1 - valid_shots / total,
        "bit_order": "beta_then_alpha_most_significant_first",
    }
    return (dict(counts) if options.configuration_recovery else valid), metadata


def _check_sampling_memory(
    request: NativeSolveRequest,
    data: FermionicIntegrals,
    options: SubspaceOptions,
    *,
    total_shots: int,
) -> None:
    """Check dense simulation and retained-shot estimates before any circuit executes."""
    if data.norb > 63:
        raise ConfigError("the PySCF SQD adapter supports at most 63 spatial orbitals")
    if options.symmetrize_spin and data.nelec[0] != data.nelec[1]:
        raise ConfigError("SQD symmetrize_spin requires equal alpha and beta populations")
    sampler = request.options.sampler
    state_bytes = 0
    if sampler.name in {"statevector", "basic_backend"}:
        state_bytes = 32 * 2 ** (2 * data.norb)
    elif sampler.name == "aer":
        sampler_options = cast("AerSamplerOptions", SAMPLERS.options_for(sampler))
        method = sampler_options.method
        if method == "density_matrix" or (
            method == "automatic" and sampler_options.noise_model is not None
        ):
            state_bytes = 32 * 4 ** (2 * data.norb)
        elif method in {"automatic", "statevector"}:
            state_bytes = 32 * 2 ** (2 * data.norb)
    storage = total_shots * (8 * data.norb + 32) + 32 * data.norb**4
    if state_bytes + storage > options.max_memory_mb * 1024**2:
        raise ConfigError("SQD sampling storage estimate exceeds max_memory_mb")


def _spin_diagnostics(
    state: Any, data: FermionicIntegrals, *, memory_bytes: int
) -> tuple[float, float]:
    """Apply S² through sparse spin ladders, including determinants outside the SCI space.

    Checking only a projected spin operator or its mean can miss a mixture of
    different total-spin sectors. S² = S+ S- + Sz² - Sz gives the full residual
    without allocating the complete FCI sector.
    """
    original = {
        int(alpha) | (int(beta) << data.norb): complex(state.amplitudes[i, j])
        for i, alpha in enumerate(state.ci_strs_a)
        for j, beta in enumerate(state.ci_strs_b)
        if state.amplitudes[i, j] != 0
    }
    # Python dict/int/complex overhead depends on the interpreter. Reserve a
    # conservative planning allowance per sparse element, not a hard RSS cap.
    max_entries = memory_bytes // 192

    def move(bits: int, source: int, target: int) -> tuple[int, int]:
        """Apply a†_target a_source and return the fermionic sign."""
        parity = (bits & ((1 << source) - 1)).bit_count()
        bits ^= 1 << source
        parity += (bits & ((1 << target) - 1)).bit_count()
        return bits | (1 << target), -1 if parity % 2 else 1

    lowered: dict[int, complex] = {}
    for bits, amplitude in original.items():
        for orbital in range(data.norb):
            alpha, beta = orbital, data.norb + orbital
            if bits & (1 << alpha) and not bits & (1 << beta):
                moved, sign = move(bits, alpha, beta)
                lowered[moved] = lowered.get(moved, 0j) + sign * amplitude
                if len(original) + len(lowered) > max_entries:
                    raise ConfigError("SQD spin-validation storage estimate exceeds max_memory_mb")
    ms = (data.nelec[0] - data.nelec[1]) / 2
    applied = {bits: (ms * ms - ms) * amplitude for bits, amplitude in original.items()}
    for bits, amplitude in lowered.items():
        for orbital in range(data.norb):
            alpha, beta = orbital, data.norb + orbital
            if bits & (1 << beta) and not bits & (1 << alpha):
                moved, sign = move(bits, beta, alpha)
                applied[moved] = applied.get(moved, 0j) + sign * amplitude
                if len(original) + len(lowered) + len(applied) > max_entries:
                    raise ConfigError("SQD spin-validation storage estimate exceeds max_memory_mb")
    norm = float(sum(abs(amplitude) ** 2 for amplitude in original.values()))
    if not np.isfinite(norm) or abs(norm - 1) > 1e-7:
        raise ConfigError("SQD returned an invalid CI-state normalization")
    expectation = sum(amplitude.conjugate() * applied[bits] for bits, amplitude in original.items())
    residual = np.sqrt(
        sum(
            abs(amplitude - data.target_spin_squared * original.get(bits, 0j)) ** 2
            for bits, amplitude in applied.items()
        )
    )
    return float(expectation.real), float(residual)


def _solve_samples(
    request: NativeSolveRequest,
    data: FermionicIntegrals,
    counts: dict[str, int],
    options: SubspaceOptions,
    sampling_metadata: dict[str, Any],
    *,
    ansatz: str | None = None,
) -> NativeOutcome:
    """Use released SQD recovery and SCI APIs with pre-solve resource checks."""
    from pyscf.lib import num_threads, with_omp_threads
    from qiskit.primitives.containers import BitArray
    from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian, solve_sci_batch

    from chemrefine.engines.qiskit.native import NativeOutcome

    counts, sample_metadata = _checked_counts(counts, data, options)
    if data.norb > 63:
        raise ConfigError("the PySCF SQD adapter supports at most 63 spatial orbitals")
    if options.symmetrize_spin and data.nelec[0] != data.nelec[1]:
        raise ConfigError("SQD symmetrize_spin requires equal alpha and beta populations")
    # The addon's max_dim limits each spin factor, not their Cartesian product.
    side = isqrt(options.max_subspace_dimension)
    iteration_records: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    memory_limit = options.max_memory_mb * 1024**2
    # The released addon unpacks all shots and calls numpy.unique on boolean
    # rows. Include copies and indices rather than only packed BitArray bytes.
    sample_bytes = sum(counts.values()) * (8 * data.norb + 32)
    sample_bytes += (
        options.num_batches * min(options.samples_per_batch, len(counts)) * (2 * data.norb + 16)
    )
    integral_bytes = 32 * data.norb**4
    if sample_bytes + integral_bytes > memory_limit:
        raise ConfigError("SQD sample/integral storage estimate exceeds max_memory_mb")

    def bounded_solver(spaces: Any, h1: Any, h2: Any, norb: int, nelec: tuple[int, int]) -> Any:
        """Check actual spin-factor dimensions before invoking PySCF."""
        dimensions = [len(alpha) * len(beta) for alpha, beta in spaces]
        if not dimensions or any(
            dim < 1 or dim > options.max_subspace_dimension for dim in dimensions
        ):
            raise ConfigError(
                "SQD diagonalization subspace exceeds its configured dimension budget"
            )
        # Account for retained batch vectors and Davidson working vectors. This
        # is a conservative planning estimate, not an operating-system memory cap.
        work_bytes = 8 * (sum(dimensions) + max(dimensions) * (2 * options.sci_max_space + 12))
        if work_bytes + sample_bytes + integral_bytes > memory_limit:
            raise ConfigError("SQD estimated working storage exceeds max_memory_mb")
        # Do not reconfigure an already-correct thread count. Serial PySCF
        # builds warn even when asked to keep their existing single thread.
        threads = (
            nullcontext()
            if num_threads() == request.options.cores
            else with_omp_threads(request.options.cores)
        )
        with threads:
            return solve_sci_batch(
                spaces,
                h1,
                h2,
                norb,
                nelec,
                spin_sq=data.target_spin_squared,
                max_cycle=options.sci_max_cycle,
                max_space=options.sci_max_space,
                tol=options.energy_tol,
                max_memory=options.max_memory_mb,
            )

    def record_iteration(results: Any) -> None:
        """Record sampled-subspace energies without asserting exact-state convergence."""
        iteration_records.append(
            {
                "iteration": len(iteration_records) + 1,
                "energies_hartree": [float(result.energy) for result in results],
                "subspace_dimensions": [
                    int(result.sci_state.amplitudes.size) for result in results
                ],
            }
        )
        for batch, result in enumerate(results):
            record = {
                "evaluation": len(evaluations) + 1,
                "objective_value_hartree": float(result.energy),
                "metadata": {"iteration": len(iteration_records), "batch": batch},
            }
            evaluations.append(record)
            if request.callback is not None:
                request.callback(deepcopy(record))

    result = diagonalize_fermionic_hamiltonian(
        data.h1,
        data.h2,
        BitArray.from_counts(counts, num_bits=2 * data.norb),
        options.samples_per_batch,
        data.norb,
        data.nelec,
        num_batches=options.num_batches,
        energy_tol=options.energy_tol,
        occupancies_tol=options.occupancies_tol,
        max_iterations=options.max_iterations if options.configuration_recovery else 1,
        sci_solver=bounded_solver,
        max_dim=side,
        symmetrize_spin=options.symmetrize_spin,
        callback=record_iteration,
        seed=options.seed,
    )
    energy = float(result.energy)
    spin_squared, spin_residual = _spin_diagnostics(
        result.sci_state, data, memory_bytes=memory_limit - sample_bytes - integral_bytes
    )
    if not np.isfinite(energy) or not np.isfinite(spin_squared) or not np.isfinite(spin_residual):
        raise ConfigError("SQD returned a non-finite energy or spin expectation")
    if spin_residual > options.spin_tolerance:
        raise ConfigError(
            f"SQD state has S^2={spin_squared:.8g}, expected {data.target_spin_squared:.8g}, "
            f"spin residual={spin_residual:.8g}; "
            "enlarge the sampled subspace or change state preparation"
        )
    return NativeOutcome(
        active_energy_hartree=energy,
        converged=None,
        termination_reason="sampled_subspace_completed",
        num_qubits=2 * data.norb,
        ansatz=ansatz,
        evaluations=evaluations,
        diagnostics={
            **sample_metadata,
            "sampling": sampling_metadata,
            "configuration_recovery": options.configuration_recovery,
            "iterations": iteration_records,
            "subspace_dimension": int(result.sci_state.amplitudes.size),
            "spin_factor_dimension_limit": side,
            "max_subspace_dimension": options.max_subspace_dimension,
            "spin_squared": spin_squared,
            "target_spin_squared": data.target_spin_squared,
            "spin_eigenstate_residual": spin_residual,
            "orbital_occupancies": [np.asarray(x).tolist() for x in result.orbital_occupancies],
            "seed": options.seed,
            "energy_convention": "active_electronic_without_offsets",
            "convergence_note": (
                "Recovery stopping does not certify ground-state or chemical accuracy."
            ),
        },
    )


@ALGORITHMS.register(
    "sqd",
    SQDOptions,
    execution="native",
    requires=frozenset({"sampler"}),
    backend_requirement=BackendRequirement(
        extra="qiskit-fermionic", import_name="qiskit_addon_sqd"
    ),
)
def build_sqd(*, options: SQDOptions, request: NativeSolveRequest) -> NativeOutcome:
    """Sample a fixed registered circuit or consume external counts, then run SQD."""
    from chemrefine.engines.qiskit.mapping import map_problem
    from chemrefine.engines.qiskit.sampling import sample_circuit

    validate_subspace_options(request.options)
    data = _chemistry(request)
    if options.counts is not None:
        if request.initial_point is not None:
            raise ConfigError("SQD supplied counts cannot be combined with an initial point")
        return _solve_samples(request, data, options.counts, options, {"source": "supplied_counts"})
    if options.shots > options.max_total_shots:
        raise ConfigError("SQD shots exceed max_total_shots")
    _check_sampling_memory(request, data, options, total_shots=options.shots)
    context = map_problem(request.prepared, request.options.mapper)
    reference = INITIAL_STATES.build(request.options.initial_state, context=context)
    ansatz = ANSATZE.build(request.options.ansatz, context=context, initial_state=reference)
    if ansatz.circuit is None:
        raise ConfigError("SQD requires an ansatz supplying a circuit when counts are not supplied")
    if options.parameter_values is not None and request.initial_point is not None:
        raise ConfigError("SQD parameter_values and explicit initial_point are mutually exclusive")
    parameters = (
        options.parameter_values if options.parameter_values is not None else request.initial_point
    )
    parameter_source = (
        "parameter_values" if options.parameter_values is not None else "initial_point"
    )
    if parameters is None:
        parameters = INITIAL_POINTS.build(request.options.initial_point, ansatz=ansatz)
        parameter_source = "configured_initial_point"
    try:
        if np.iscomplexobj(parameters):
            raise ValueError("complex parameters")
        values = np.asarray(parameters, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ConfigError("SQD parameters must be finite real numbers") from exc
    if values.shape != (ansatz.circuit.num_parameters,) or not np.all(np.isfinite(values)):
        raise ConfigError("SQD parameters must be finite and match the ansatz parameter count")
    batch = sample_circuit(
        ansatz.circuit,
        request.options.sampler,
        shots=options.shots,
        device=request.options.device,
        cores=request.options.cores,
        parameter_values=values.tolist(),
    )
    return _solve_samples(
        request,
        data,
        batch.counts,
        options,
        {
            "source": "fixed_ansatz",
            "parameter_source": parameter_source,
            "parameter_values": values.tolist(),
            **batch.metadata,
        },
        ansatz=request.options.ansatz.name,
    )

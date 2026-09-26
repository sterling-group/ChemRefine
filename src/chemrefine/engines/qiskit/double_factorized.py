"""Domain-validated double-factorized fermionic evolution using released ffsim."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from chemrefine.engines.qiskit.data import ElectronicStructureData
from chemrefine.engines.qiskit.determinants import ComplexArray
from chemrefine.errors import ConfigError


class DoubleFactorizedIntegratorOptions(BaseModel):
    """Separate integral compression, product-formula and actual synthesis controls."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    steps: StrictInt = Field(1, ge=1)
    order: Literal[1, 2, 4] = 2
    factorization: Literal["eigh", "cholesky", "compressed"] = "eigh"
    factorization_tolerance: float = Field(1e-8, gt=0)
    max_factors: StrictInt | None = Field(None, ge=1)
    max_optimizer_iterations: StrictInt = Field(100, ge=1)
    max_integral_error: float | None = Field(None, ge=0)
    z_representation: bool = False
    givens_tolerance: float = Field(1e-12, gt=0)
    energy_shift_hartree: float = 0.0
    optimization_level: Literal[0, 1, 2, 3] = 1
    seed_transpiler: int = Field(0, ge=0)
    max_operations: StrictInt = Field(1_000_000, ge=1)
    max_memory_mb: StrictInt = Field(512, ge=1)
    max_statevector_qubits: StrictInt = Field(16, ge=1)
    exact_reference: bool = False


class DoubleFactorizedOptions(DoubleFactorizedIntegratorOptions):
    """The shared factorization/synthesis controls at one signed evolution time."""

    time: float = 1.0


@dataclass(frozen=True)
class DoubleFactorizedEvolution:
    """Executable compiled circuit, factorization tensors and independent error accounting."""

    circuit: Any
    one_body: ComplexArray
    diagonal_coulomb: ComplexArray
    orbital_rotations: ComplexArray
    metadata: dict[str, Any]
    statevector: ComplexArray | None = None


def _supported_integrals(data: ElectronicStructureData) -> tuple[ComplexArray, np.ndarray]:
    """Accept complex shared hopping with real shared-spin chemist interactions."""
    one = np.asarray(data.one_body_integrals, dtype=complex)
    raw_two = np.asarray(data.two_body_integrals)
    two = raw_two if data.two_body_order == "chemist" else raw_two.transpose(0, 3, 1, 2)
    if np.iscomplexobj(two) and np.any(two.imag != 0):
        raise ConfigError("double factorization requires real two-body integrals")
    two = np.array(two.real, dtype=float, copy=True)
    if not (
        np.allclose(two, two.swapaxes(0, 1), atol=1e-10, rtol=0)
        and np.allclose(two, two.swapaxes(2, 3), atol=1e-10, rtol=0)
    ):
        raise ConfigError("double factorization requires real chemist pair symmetry")
    for name, expected in (
        ("one_body_integrals_beta", one),
        ("two_body_integrals_beta_beta", two),
        ("two_body_integrals_beta_alpha", two),
    ):
        value = getattr(data, name)
        if value is not None:
            actual = np.asarray(value)
            if name.startswith("two_body") and data.two_body_order == "physicist":
                actual = actual.transpose(0, 3, 1, 2)
            if not np.allclose(actual, expected, atol=1e-10, rtol=0):
                raise ConfigError("double factorization requires shared alpha/beta integrals")
    if data.overlap_alpha_beta is not None and not np.allclose(
        data.overlap_alpha_beta, np.eye(data.num_spatial_orbitals), atol=1e-10, rtol=0
    ):
        raise ConfigError("double factorization requires shared alpha/beta spatial orbitals")
    return one, two


def build_double_factorized_evolution(
    data: ElectronicStructureData, options: DoubleFactorizedOptions, *, cores: int = 1
) -> DoubleFactorizedEvolution:
    """Factorize supported integrals and synthesize exp(-itH), including constant phases.

    Complex one-body terms are retained. Released ffsim factorization requires real
    shared-spin interactions. Open-shell references remain supported. Error bounds
    reported here cover factorization only, not product-formula or synthesis error.
    """
    if isinstance(cores, bool) or not isinstance(cores, int) or cores < 1:
        raise ConfigError("double-factorized evolution cores must be a positive integer")
    n = data.num_spatial_orbitals
    factors = n * (n + 1) // 2 if options.max_factors is None else options.max_factors
    suzuki_terms = {1: 1, 2: 2, 4: 10}[options.order]
    estimated_operations = options.steps * suzuki_terms * (2 * factors + 2) * (20 * n * n + 1)
    if estimated_operations > options.max_operations:
        raise ConfigError("double-factorized synthesis estimate exceeds max_operations")
    required = 512 * n**4 + 256 * estimated_operations
    if required > options.max_memory_mb * 1024**2:
        raise ConfigError("double-factorized storage estimate exceeds max_memory_mb")
    one, two = _supported_integrals(data)
    import ffsim
    from pyscf.lib import with_omp_threads
    from qiskit import QuantumCircuit, transpile

    nuclear = data.nuclear_repulsion_energy or 0.0
    constant = nuclear + options.energy_shift_hartree
    original = ffsim.MolecularHamiltonian(one, two, constant=constant)
    with with_omp_threads(cores):
        if options.factorization == "cholesky":
            minimum = float(np.linalg.eigvalsh(two.reshape(n * n, n * n)).min())
            if minimum < -options.factorization_tolerance:
                raise ConfigError(
                    "Cholesky double factorization requires positive semidefinite Coulomb tensor"
                )
        optimizer = None
        if options.factorization == "compressed" and np.any(two):
            matrices, rotations, optimized = ffsim.linalg.double_factorized(
                two,
                tol=options.factorization_tolerance,
                max_vecs=options.max_factors,
                optimize=True,
                return_optimize_result=True,
                options={"maxiter": options.max_optimizer_iterations},
            )
            factorized = ffsim.DoubleFactorizedHamiltonian(
                one - 0.5 * np.einsum("prrq->pq", two), matrices, rotations, constant=constant
            )
            if options.z_representation:
                factorized = factorized.to_z_representation()
            optimizer = {
                "success": bool(optimized.success),
                "status": int(optimized.status),
                "message": str(optimized.message),
                "iterations": int(optimized.nit),
                "function_evaluations": int(optimized.nfev),
                "objective": float(optimized.fun),
            }
        else:
            factorized = ffsim.DoubleFactorizedHamiltonian.from_molecular_hamiltonian(
                original,
                z_representation=options.z_representation,
                tol=options.factorization_tolerance,
                max_vecs=options.max_factors,
                cholesky=options.factorization == "cholesky",
            )
        reconstructed = factorized.to_molecular_hamiltonian()
        delta_one = np.asarray(reconstructed.one_body_tensor) - one
        delta_two = np.asarray(reconstructed.two_body_tensor) - two
        one_error = float(np.max(abs(delta_one)))
        two_error = float(np.max(abs(delta_two)))
        constant_error = abs(float(reconstructed.constant) - constant)
        if (
            options.max_integral_error is not None
            and max(one_error, two_error) > options.max_integral_error
        ):
            raise ConfigError("double factorization exceeds max_integral_error")
        norm_bound = float(2 * np.sum(abs(delta_one)) + 2 * np.sum(abs(delta_two)) + constant_error)
        gate = ffsim.qiskit.SimulateTrotterDoubleFactorizedJW(
            factorized,
            options.time,
            n_steps=options.steps,
            order={1: 0, 2: 1, 4: 2}[options.order],
            tol=options.givens_tolerance,
        )
        circuit = QuantumCircuit(2 * n)
        circuit.append(gate, range(2 * n))
        compiled = transpile(
            circuit,
            basis_gates=["rz", "sx", "x", "cx"],
            optimization_level=options.optimization_level,
            seed_transpiler=options.seed_transpiler,
            num_processes=cores,
        )
    if compiled.size() > options.max_operations:
        raise ConfigError("compiled double-factorized circuit exceeds max_operations")
    one_factor = np.array(factorized.one_body_tensor, dtype=complex, copy=True)
    coulomb = np.array(factorized.diag_coulomb_mats, dtype=complex, copy=True)
    rotations = np.array(factorized.orbital_rotations, dtype=complex, copy=True)
    for array in (one_factor, coulomb, rotations):
        array.setflags(write=False)
    return DoubleFactorizedEvolution(
        compiled,
        one_factor,
        coulomb,
        rotations,
        {
            "experimental": True,
            "num_spatial_orbitals": n,
            "factor_count": len(coulomb),
            "factorization": options.factorization,
            "compression_optimizer": optimizer,
            "z_representation": options.z_representation,
            "factorized_constant_hartree": float(factorized.constant),
            "energy_offsets_hartree": {
                "nuclear_repulsion": nuclear,
                "supplied_shift": options.energy_shift_hartree,
            },
            "one_body_max_absolute_error": one_error,
            "two_body_max_absolute_error": two_error,
            "constant_absolute_error": constant_error,
            "requested_tensor_tolerance_met": two_error <= options.factorization_tolerance,
            "factorization_hamiltonian_norm_error_bound": norm_bound,
            "factorization_unitary_error_bound": min(2.0, abs(options.time) * norm_bound),
            "product_formula_error_bound": None,
            "givens_synthesis_error_bound": None,
            "time": options.time,
            "physical_product_formula_order": options.order,
            "steps": options.steps,
            "synthesis_operation_estimate": estimated_operations,
            "compiled_operations": compiled.size(),
            "compiled_depth": compiled.depth(),
            "compiled_two_qubit_gates": sum(
                instruction.operation.num_qubits == 2 for instruction in compiled.data
            ),
            "basis_gates": ["rz", "sx", "x", "cx"],
            "cores": cores,
            "units": {"energy": "hartree", "time": "hbar/hartree", "hbar": 1},
            "orbital_order": "alpha_then_beta",
        },
    )


def simulate_double_factorized_evolution(
    data: ElectronicStructureData, options: DoubleFactorizedOptions, *, cores: int = 1
) -> DoubleFactorizedEvolution:
    """Evolve the declared occupied orbitals and optionally compare the original Hamiltonian."""
    width = 2 * data.num_spatial_orbitals
    if width > options.max_statevector_qubits:
        raise ConfigError("double-factorized trajectory exceeds max_statevector_qubits")
    memory = 16 * (1 << width) * 8
    if options.exact_reference:
        memory += 64 * (1 << (2 * width))
    if memory > options.max_memory_mb * 1024**2:
        raise ConfigError("double-factorized reference simulation exceeds max_memory_mb")
    result = build_double_factorized_evolution(data, options, cores=cores)
    from pyscf.lib import with_omp_threads
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    from chemrefine.engines.qiskit.problem import prepare_problem

    reference = QuantumCircuit(width)
    occupations = np.concatenate([data.orbital_occupations, data.orbital_occupations_beta])
    for mode in np.flatnonzero(occupations):
        reference.x(int(mode))
    prepared = prepare_problem(data)
    operator = JordanWignerMapper().map(prepared.fermionic_hamiltonian)
    constant = (data.nuclear_repulsion_energy or 0) + options.energy_shift_hartree
    with with_omp_threads(cores):
        state = Statevector.from_instruction(reference.compose(result.circuit))
        vector = np.asarray(state.data, dtype=complex)
        energy = float(state.expectation_value(operator).real) + constant
        probabilities = abs(vector) ** 2
        addresses = np.arange(1 << width, dtype=np.uint64)
        expectation = np.array([probabilities @ ((addresses >> mode) & 1) for mode in range(width)])
        metadata = {
            **result.metadata,
            "total_energy_hartree": energy,
            "mode_occupations": expectation.tolist(),
            "alpha_particle_number": float(sum(expectation[: width // 2])),
            "beta_particle_number": float(sum(expectation[width // 2 :])),
            "norm": float(np.linalg.norm(vector)),
        }
        if options.exact_reference:
            from scipy.linalg import expm

            matrix = operator.to_matrix() + constant * np.eye(1 << width)
            initial = Statevector.from_instruction(reference).data
            exact = expm(-1j * options.time * matrix) @ initial
            metadata["reference_state_error"] = float(np.linalg.norm(vector - exact))
            metadata["reference_infidelity"] = float(max(0.0, 1 - abs(np.vdot(exact, vector)) ** 2))
            metadata["reference_energy_hartree"] = float(np.vdot(initial, matrix @ initial).real)
    vector.setflags(write=False)
    return replace(result, statevector=cast("ComplexArray", vector), metadata=metadata)

"""Restricted-real DF/THC estimates through public optional provider APIs.

OpenFermion supplies factorization and paper cost formulae; Qualtran supplies an
additional DF cost graph. Integral normalization and residual bounds are owned
here so neither provider needs a fabricated PySCF mean-field calculation.
"""

from __future__ import annotations

import importlib.metadata
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import Field, StrictInt, model_validator

from chemrefine.engines.qiskit.bundles import DEFAULT_MAX_BYTES, read_bundle, write_bundle
from chemrefine.engines.qiskit.data import ElectronicStructureData
from chemrefine.engines.qiskit.resources import QPEBudget, ResourceModel, qpe_queries
from chemrefine.errors import ConfigError


class FactorizedResourceOptions(ResourceModel):
    """Explicit precision and allocation controls for chemistry cost models."""

    method: Literal["df", "thc"] = "df"
    budget: QPEBudget
    factorization_threshold: float = Field(1e-8, gt=0)
    coefficient_bits: StrictInt = Field(20, ge=2, le=128)
    rotation_bits: StrictInt = Field(30, ge=3, le=128)
    initial_cost_guess: StrictInt = Field(20000, ge=1)
    qualtran_cost_graph: bool = False
    amplitude_bits_outer: StrictInt = Field(8, ge=1, le=128)
    amplitude_bits_inner: StrictInt = Field(8, ge=1, le=128)
    max_working_bytes: StrictInt = Field(DEFAULT_MAX_BYTES, ge=1)

    @model_validator(mode="after")
    def _precision_contract(self) -> Self:
        """Never mistake finite coefficient precision for exact synthesis."""
        if self.budget.synthesis_error_hartree <= 0:
            raise ValueError(
                "finite provider bit precisions require a supplied synthesis error bound"
            )
        if self.method == "thc" and self.qualtran_cost_graph:
            raise ValueError("qualtran_cost_graph currently supports DF only")
        if self.method == "thc" and self.factorization_threshold != 1e-8:
            raise ValueError(
                "factorization_threshold applies only to DF; THC consumes supplied factors"
            )
        if not self.qualtran_cost_graph and (
            self.amplitude_bits_outer != 8 or self.amplitude_bits_inner != 8
        ):
            raise ValueError("amplitude_bits controls require qualtran_cost_graph")
        return self


@dataclass(frozen=True)
class THCFactors:
    """Real THC leaves eta[P,p] and symmetric central matrix zeta[P,Q]."""

    leaf: NDArray[np.float64]
    central: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Own finite validated copies and prohibit mutating cached factors."""
        leaf = _real(self.leaf, "THC leaf")
        central = _real(self.central, "THC central")
        if leaf.ndim != 2 or min(leaf.shape) < 1:
            raise ConfigError("THC leaf must have nonempty shape (rank, spatial orbitals)")
        if central.shape != (leaf.shape[0],) * 2 or not np.allclose(
            central, central.T, atol=1e-12, rtol=1e-12
        ):
            raise ConfigError("THC central matrix must be symmetric with shape (rank, rank)")
        if np.any(np.linalg.norm(leaf, axis=1) == 0):
            raise ConfigError("THC factors must not contain zero leaf rows")
        leaf.setflags(write=False)
        central.setflags(write=False)
        object.__setattr__(self, "leaf", leaf)
        object.__setattr__(self, "central", central)


def _real(value: ArrayLike, name: str) -> NDArray[np.float64]:
    """Refuse complex coefficients rather than dropping their imaginary parts."""
    array = np.asarray(value)
    if np.iscomplexobj(array) and np.any(array.imag != 0):
        raise ConfigError(f"factorized resource estimates require real {name}")
    result = np.array(array.real, dtype=float, copy=True)
    if not np.isfinite(result).all():
        raise ConfigError(f"factorized resource estimates require finite {name}")
    return result


def save_thc_factors(
    path: Path, factors: THCFactors, *, max_bytes: int = DEFAULT_MAX_BYTES
) -> Path:
    """Publish portable numeric factors using the common checksummed bundle."""
    write_bundle(
        path,
        kind="thc_factors",
        arrays={"leaf": factors.leaf, "central": factors.central},
        metadata={
            "convention": "eri[p,q,r,s]=sum_PQ eta[P,p]eta[P,q]zeta[P,Q]eta[Q,r]eta[Q,s]",
            "units": "reconstructed ERI in hartree",
        },
        max_bytes=max_bytes,
    )
    return path


def load_thc_factors(path: Path, *, max_bytes: int = DEFAULT_MAX_BYTES) -> THCFactors:
    """Read only the declared THC bundle shape and check its mathematical domain."""
    bundle = read_bundle(path, max_bytes=max_bytes)
    if bundle.description.kind != "thc_factors" or set(bundle.arrays) != {"leaf", "central"}:
        raise ConfigError("expected a thc_factors bundle with leaf and central arrays")
    return THCFactors(bundle.arrays["leaf"], bundle.arrays["central"])


def restricted_resource_integrals(
    data: ElectronicStructureData, *, max_bytes: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate real shared-orbital input and bound dense factorization workspace."""
    n = data.num_spatial_orbitals
    if n < 2:
        raise ConfigError("DF/THC provider cost models require at least two spatial orbitals")
    if 8 * (10 * n**4 + 4 * n**2) > max_bytes:
        raise ConfigError("factorization workspace exceeds max_working_bytes")
    h1 = _real(data.one_body_integrals, "one-body integrals")
    eri = _real(data.two_body_integrals, "two-body integrals")
    if data.two_body_order == "physicist":
        eri = eri.transpose(0, 3, 1, 2).copy()
    for name, reference in (
        ("one_body_integrals_beta", h1),
        ("two_body_integrals_beta_beta", eri),
        ("two_body_integrals_beta_alpha", eri),
    ):
        value = getattr(data, name)
        if value is not None:
            array = _real(value, name)
            if array.ndim == 4 and data.two_body_order == "physicist":
                array = array.transpose(0, 3, 1, 2)
            if not np.allclose(array, reference, atol=1e-12, rtol=1e-12):
                raise ConfigError("DF/THC resource models require equal alpha/beta integral blocks")
    if data.overlap_alpha_beta is not None and not np.allclose(
        data.overlap_alpha_beta, np.eye(n), atol=1e-12, rtol=1e-12
    ):
        raise ConfigError("DF/THC resource models require shared spatial orbitals")
    if not (
        np.allclose(eri, eri.swapaxes(0, 1), atol=1e-12, rtol=1e-12)
        and np.allclose(eri, eri.swapaxes(2, 3), atol=1e-12, rtol=1e-12)
        and np.allclose(eri, eri.transpose(2, 3, 0, 1), atol=1e-12, rtol=1e-12)
    ):
        raise ConfigError("DF/THC resource models require real chemist ERI symmetry")
    return h1, eri


def factorization_metrics(
    h1: NDArray[np.float64], eri: NDArray[np.float64], reconstructed: NDArray[np.float64]
) -> dict[str, float]:
    """Compute a conservative full-Fock operator error and the shifted one-body norm.

    Keeping h1 fixed and replacing the spatial ERI by g' changes the Hamiltonian
    norm by at most 2*sum(abs(g-g')): four spin combinations, the 1/2 prefactor,
    and a unit norm bound for each fermionic monomial. The normal-order
    correction in T is built from g', consistently with that approximate
    Hamiltonian, rather than mixing exact and approximate integral tensors.
    """
    difference = eri - reconstructed
    t_matrix = (
        h1 - 0.5 * np.einsum("illj->ij", reconstructed) + np.einsum("llij->ij", reconstructed)
    )
    return {
        "eri_frobenius_residual_hartree": float(np.linalg.norm(difference)),
        "representation_error_bound_hartree": 2 * float(np.sum(np.abs(difference))),
        "one_body_normalization_hartree": float(np.sum(np.abs(np.linalg.eigvalsh(t_matrix)))),
    }


def qualtran_df_cost_graph(
    nspin: int, rank: int, eigenvectors: int, options: FactorizedResourceOptions
) -> dict[str, Any]:
    """Count a Qualtran DF block encoding without claiming its circuit is complete."""
    from qualtran.bloqs.chemistry.df.double_factorization import (
        DoubleFactorizationBlockEncoding,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    bloq = DoubleFactorizationBlockEncoding(
        num_spin_orb=nspin,
        num_aux=rank,
        num_eig=eigenvectors,
        num_bits_state_prep=options.coefficient_bits,
        num_bits_rot=options.rotation_bits,
        num_bits_rot_aa_outer=options.amplitude_bits_outer,
        num_bits_rot_aa_inner=options.amplitude_bits_inner,
    )
    counts = get_cost_value(bloq, QECGatesCost()).asdict()
    return {
        "provider": "qualtran",
        "version": importlib.metadata.version("qualtran"),
        "object": "DoubleFactorizationBlockEncoding",
        "gate_counts_per_block_encoding": {name: int(value) for name, value in counts.items()},
        "signature_qubits": int(sum(register.total_bits() for register in bloq.signature)),
        "executable_circuit": False,
        "limitations": [
            "cost graph, not a synthesized circuit",
            "provider alpha and epsilon are not certified by this graph",
            "external qubitization reflection and QPE overhead excluded",
            "signature width need not equal peak decomposition workspace",
        ],
    }


def provider_qrom_tables(
    norb: int, parameters: dict[str, int], options: FactorizedResourceOptions
) -> dict[str, dict[str, int]]:
    """Validate released OpenFermion QR table-size assumptions before cost evaluation.

    The provider's QR helper exits its process for table_size < payload_bits.
    This is a restriction of that cost implementation, not of quantum chemistry
    or qubitization. Keep the actual scientific factors and precision unchanged.
    """
    if options.method == "df":
        rank, eigenvectors = parameters["L"], parameters["Lxi"]
        orbital_bits = math.ceil(math.log2(norb))
        tables = {
            "outer_coefficients": (
                rank + 1,
                math.ceil(math.log2(rank + 1)) + options.coefficient_bits,
            ),
            "outer_offsets": (
                rank + 1,
                orbital_bits + math.ceil(math.log2(eigenvectors + norb)) + 8,
            ),
            "inner_coefficients": (eigenvectors, orbital_bits + options.coefficient_bits + 2),
            "inner_rotations": (eigenvectors, norb * options.rotation_bits),
        }
    else:
        rank = parameters["M"]
        tables = {
            "thc_coefficients": (
                rank * (rank + 1) // 2 + norb,
                2 * math.ceil(math.log2(rank + 1)) + 2 + options.coefficient_bits,
            )
        }
    for name, (size, bits) in tables.items():
        if size < bits:
            raise ConfigError(
                f"OpenFermion {options.method} QROM domain unsupported: {name} has "
                f"{size} entries but requires at least {bits} for the selected precision. "
                "Use pauli_resources for a general query estimate; factors or precision "
                "must not be changed merely to make this cost model accept the input."
            )
    return {name: {"entries": size, "payload_bits": bits} for name, (size, bits) in tables.items()}


def estimate_factorized_resources(
    data: ElectronicStructureData,
    options: FactorizedResourceOptions,
    *,
    thc_factors: THCFactors | None = None,
) -> dict[str, Any]:
    """Run public OpenFermion cost APIs on a validated DF or supplied THC factorization."""
    h1, eri = restricted_resource_integrals(data, max_bytes=options.max_working_bytes)
    n = data.num_spatial_orbitals
    parameters: dict[str, int] = {}
    if options.method == "df":
        if thc_factors is not None:
            raise ConfigError("THC factors are incompatible with method=df")
        eigenvalues = np.linalg.eigvalsh(eri.reshape(n * n, n * n))
        if eigenvalues[0] < -1e-10 * max(1.0, float(np.max(np.abs(eigenvalues)))):
            raise ConfigError("DF factorization requires a positive-semidefinite Coulomb matrix")
        from openfermion.resource_estimates.df import compute_cost, factorize

        try:
            reconstructed, factors, _provider_rank, eigenvectors = factorize(
                eri, options.factorization_threshold
            )
        except (ValueError, UnboundLocalError) as exc:
            # Released factorize cannot form its einsum when no factor survives.
            raise ConfigError(
                "DF factorization failed; check nonzero interactions and lower the threshold"
            ) from exc
        # Released factorize reports a loop index as 'rank'; use retained data shape.
        rank = factors.shape[2]
        if rank < 1 or eigenvectors < 1:
            raise ConfigError("DF factorization retained no factors; lower the threshold")
        parameters.update(L=rank, Lxi=int(eigenvectors))
        two_body_lambda = 0.25 * sum(
            float(np.sum(np.abs(np.linalg.eigvalsh(factors[:, :, index])))) ** 2
            for index in range(rank)
        )
    else:
        if thc_factors is None or thc_factors.leaf.shape[1] != n:
            raise ConfigError("method=thc requires factors matching the spatial orbital count")
        rank = thc_factors.leaf.shape[0]
        if 8 * (10 * n**4 + 4 * rank**2 + 4 * rank * n**2) > options.max_working_bytes:
            raise ConfigError("THC reconstruction workspace exceeds max_working_bytes")
        from openfermion.resource_estimates.thc import compute_cost

        leaf, central = thc_factors.leaf, thc_factors.central
        reconstructed = np.einsum(
            "Pp,Pq,PQ,Qr,Qs->pqrs", leaf, leaf, central, leaf, leaf, optimize=True
        )
        norms = np.sum(leaf**2, axis=1)
        two_body_lambda = 0.5 * float(np.sum(np.abs(central * norms[:, None] * norms[None, :])))
        parameters["M"] = rank
    metrics = factorization_metrics(h1, eri, reconstructed)
    residual = metrics["representation_error_bound_hartree"]
    if residual > options.budget.representation_error_hartree:
        raise ConfigError(
            f"factorization error bound {residual:.8g} exceeds representation_error_hartree; "
            "tighten the factorization or explicitly allocate a larger error budget"
        )
    normalization = metrics["one_body_normalization_hartree"] + two_body_lambda
    if not math.isfinite(normalization) or normalization <= 0:
        raise ConfigError("factorized provider estimates require positive finite normalization")
    qpe = qpe_queries(normalization, options.budget)
    qrom = provider_qrom_tables(n, parameters, options)
    cost_arguments = dict(
        n=2 * n,
        lam=normalization,
        dE=options.budget.qpe_error_hartree,
        chi=options.coefficient_bits,
        beta=options.rotation_bits,
        **parameters,
    )
    try:
        first_cost = compute_cost(**cost_arguments, stps=options.initial_cost_guess)
        step_cost, total_cost, logical_qubits = compute_cost(
            **cost_arguments, stps=int(first_cost[0])
        )
    except (SystemExit, AssertionError, ValueError, OverflowError) as exc:
        raise ConfigError(f"OpenFermion resource cost model rejected this input: {exc}") from exc
    if any(
        not np.isfinite(value) or value <= 0 for value in (step_cost, total_cost, logical_qubits)
    ):
        raise ConfigError(
            "provider returned nonpositive or nonfinite resource costs outside its valid domain"
        )
    result: dict[str, Any] = {
        "estimate_kind": "provider_analytical_cost_model",
        "executable_circuit": False,
        "provider": "openfermion",
        "version": importlib.metadata.version("openfermion"),
        "method": options.method,
        "parameters": options.model_dump(),
        "system_qubits": 2 * n,
        "factorization_parameters": parameters,
        "provider_qrom_tables": qrom,
        "normalization_hartree": normalization,
        "two_body_normalization_hartree": two_body_lambda,
        **metrics,
        "normalization_convention": "shifted DF/THC LCU; T computed from reconstructed ERI",
        "nuclear_repulsion_energy_hartree": data.nuclear_repulsion_energy,
        "identity_offsets_in_query_cost": False,
        "provider_toffoli_per_step": int(step_cost),
        "provider_toffoli_total_single_run": int(total_cost),
        "provider_logical_qubits_including_system_and_phase": int(logical_qubits),
        "provider_phase_iterations": math.ceil(
            math.pi * normalization / (2 * options.budget.qpe_error_hartree)
        ),
        "provider_failure_probability": None,
        "provider_confidence_note": (
            "provider pi*lambda/(2*epsilon) formula does not certify "
            "the requested failure probability"
        ),
        "conservative_standard_qpe": qpe,
        "synthesis_note": (
            "finite bit counts are explicit inputs; supplied synthesis error is an "
            "assumption, not a provider certification"
        ),
        "exclusions": [
            "initial-state preparation",
            "routing",
            "physical error correction",
            "magic-state factories",
            "independent repetition overhead in provider totals",
        ],
    }
    if options.qualtran_cost_graph:
        result["qualtran_cost_graph"] = qualtran_df_cost_graph(
            2 * n, parameters["L"], parameters["Lxi"], options
        )
    return result

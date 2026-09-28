"""Distinct orbital-Haar and Majorana-Clifford shadow channels for fermionic RDMs.

The orbital estimator uses Low, arXiv:2208.08964v2, Eq. (4). The Majorana
estimator inverts even-monomial eigenvalues C(m,k)/C(2m,2k). Their measurement
circuits and inverses are intentionally separate; only aggregation is shared.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import comb, sqrt
from typing import Any, Literal, Self, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from chemrefine.engines.qiskit.determinants import (
    ComplexArray,
    FermionicHamiltonian,
    ReducedDensityMatrices,
)
from chemrefine.engines.qiskit.options import ComponentSelection
from chemrefine.errors import ConfigError

Ensemble = Literal["orbital_haar", "majorana_clifford"]


class FermionicShadowOptions(BaseModel):
    """Bound settings, physical shots, dense RDM storage and explicit postselection."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    ensemble: Ensemble = "orbital_haar"
    num_particles: int | None = Field(None, ge=0)
    num_settings: int = Field(100, ge=1)
    shots_per_setting: int = Field(1, ge=1)
    max_order: Literal[1, 2] = 2
    postselect_particles: bool = False
    max_total_shots: int = Field(1_000_000, ge=1)
    max_memory_mb: int = Field(512, ge=1)
    seed: int | None = Field(0, ge=0)

    @model_validator(mode="after")
    def _ensemble_contract(self) -> Self:
        """Keep the inverse-channel assumptions consistent with acquisition controls."""
        if self.num_settings * self.shots_per_setting > self.max_total_shots:
            raise ValueError("shadow acquisition exceeds max_total_shots")
        if self.ensemble == "orbital_haar" and self.num_particles is None:
            raise ValueError("orbital Haar shadows require the fixed num_particles")
        if self.ensemble == "majorana_clifford" and (
            self.num_particles is not None or self.postselect_particles
        ):
            raise ValueError("Majorana shadows do not support total-particle postselection")
        return self


@dataclass(frozen=True)
class ShadowSetting:
    """Actual measurement basis: orbital U or Majorana R with V†gamma V = R gamma."""

    ensemble: Ensemble
    matrix: ComplexArray

    def __post_init__(self) -> None:
        """Own an immutable unitary or orientation-preserving signed permutation."""
        try:
            matrix = np.array(self.matrix, dtype=complex, copy=True)
        except (TypeError, ValueError) as exc:
            raise ConfigError("shadow setting must be a finite numerical matrix") from exc
        if self.ensemble not in {"orbital_haar", "majorana_clifford"}:
            raise ConfigError("unknown fermionic shadow ensemble")
        if (
            matrix.ndim != 2
            or matrix.shape[0] < 1
            or matrix.shape[0] != matrix.shape[1]
            or not np.isfinite(matrix).all()
            or not np.allclose(matrix.conj().T @ matrix, np.eye(len(matrix)), atol=1e-10, rtol=0)
        ):
            raise ConfigError("shadow setting must be a finite square unitary")
        if self.ensemble == "majorana_clifford" and (
            len(matrix) % 2
            or not np.isin(matrix, [-1, 0, 1]).all()
            or not np.isclose(np.linalg.det(matrix), 1, atol=1e-10)
        ):
            raise ConfigError("Majorana setting must be an even SO signed permutation")
        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)

    @property
    def num_modes(self) -> int:
        """Return the number of measured fermionic modes."""
        return len(self.matrix) if self.ensemble == "orbital_haar" else len(self.matrix) // 2


def _storage(num_modes: int, options: FermionicShadowOptions) -> None:
    """Bound retained clusters, settings and temporary fourth-order tensor work."""
    if isinstance(num_modes, bool) or not isinstance(num_modes, int) or num_modes < 1:
        raise ConfigError("shadows require a positive integer number of modes")
    if options.num_particles is not None and options.num_particles > num_modes:
        raise ConfigError("shadow particle number exceeds the number of modes")
    entries = num_modes**2 + (num_modes**4 if options.max_order == 2 else 0)
    estimate = (
        16 * entries * (2 * options.num_settings + 24) + 64 * options.num_settings * num_modes**2
    )
    estimate += options.num_settings * options.shots_per_setting * (num_modes + 128)
    if estimate > options.max_memory_mb * 1024**2:
        raise ConfigError("shadow storage estimate exceeds max_memory_mb")


def random_shadow_settings(
    num_modes: int, options: FermionicShadowOptions
) -> tuple[ShadowSetting, ...]:
    """Draw reproducible complex Haar orbitals or uniform SO signed permutations."""
    _storage(num_modes, options)
    rng = np.random.default_rng(options.seed)
    settings = []
    for _ in range(options.num_settings):
        if options.ensemble == "orbital_haar":
            raw = rng.normal(size=(num_modes, num_modes)) + 1j * rng.normal(
                size=(num_modes, num_modes)
            )
            matrix, upper = np.linalg.qr(raw)
            diagonal = np.diag(upper)
            matrix *= diagonal / np.abs(diagonal)
        else:
            width = 2 * num_modes
            permutation = rng.permutation(width)
            signs = rng.choice([-1, 1], size=width)
            parity = (-1) ** sum(
                permutation[i] > permutation[j] for i in range(width) for j in range(i + 1, width)
            )
            signs[-1] = parity * np.prod(signs[:-1])
            matrix = np.zeros((width, width), dtype=complex)
            matrix[np.arange(width), permutation] = signs
        settings.append(ShadowSetting(options.ensemble, matrix))
    return tuple(settings)


def shadow_circuit(setting: ShadowSetting) -> Any:
    """Synthesize the declared measurement basis, preserving its Majorana orientation."""
    from qiskit import QuantumCircuit

    m = setting.num_modes
    circuit = QuantumCircuit(m)
    if setting.ensemble == "orbital_haar":
        from ffsim.qiskit import OrbitalRotationSpinlessJW

        circuit.append(OrbitalRotationSpinlessJW(m, setting.matrix), range(m))
        return circuit.decompose()
    from qiskit.quantum_info import Pauli

    def majorana(index: int) -> Any:
        """Represent one Majorana in Qiskit's most-significant-first Pauli labels."""
        labels = ["Z"] * (index // 2) + ["X" if index % 2 == 0 else "Y"]
        labels += ["I"] * (m - len(labels))
        return Pauli("".join(reversed(labels)))

    target = np.argmax(np.abs(setting.matrix), axis=1).tolist()
    target_signs = setting.matrix[np.arange(2 * m), target].real.astype(int)
    current = list(range(2 * m))
    signs = np.ones(2 * m, dtype=int)
    for row, target_mode in enumerate(target):
        position = current.index(target_mode)
        while position > row:
            lower = position - 1
            if lower % 2 == 0:
                circuit.rz(-np.pi / 2, lower // 2)
            else:
                circuit.rxx(-np.pi / 2, lower // 2, lower // 2 + 1)
            current[lower], current[position] = current[position], current[lower]
            signs[lower], signs[position] = signs[position], -signs[lower]
            position -= 1
    flips = np.flatnonzero(signs != target_signs)
    for first, second in zip(flips[::2], flips[1::2], strict=True):
        pauli = majorana(int(first)) @ majorana(int(second))
        pauli.phase = 0
        circuit.pauli(pauli.to_label(), range(m))
    return circuit


def _wedge(one: ComplexArray) -> ComplexArray:
    """Antisymmetrize the product of two one-body matrices in RDM index order."""
    return cast(
        "ComplexArray", np.einsum("pr,qs->pqrs", one, one) - np.einsum("ps,qr->pqrs", one, one)
    )


def _lift(one: ComplexArray) -> ComplexArray:
    """Lift a one-body matrix to the antisymmetric pair space without dense pair products."""
    identity = np.eye(len(one))
    return cast(
        "ComplexArray",
        (
            np.einsum("pr,qs->pqrs", one, identity)
            + np.einsum("pr,qs->pqrs", identity, one)
            - np.einsum("ps,qr->pqrs", one, identity)
            - np.einsum("ps,qr->pqrs", identity, one)
        ),
    )


def _snapshot_storage(num_modes: int, max_order: int, max_memory_mb: int) -> None:
    """Bound standalone inverse-channel temporaries before allocating dense tensors."""
    if isinstance(max_memory_mb, bool) or not isinstance(max_memory_mb, int) or max_memory_mb < 1:
        raise ConfigError("shadow snapshot max_memory_mb must be a positive integer")
    entries = num_modes**2 + (num_modes**4 if max_order == 2 else 0)
    if 384 * entries > max_memory_mb * 1024**2:
        raise ConfigError("shadow snapshot storage estimate exceeds max_memory_mb")


def orbital_shadow_snapshot(
    setting: ShadowSetting,
    bits: int,
    *,
    num_particles: int,
    max_order: int = 2,
    max_memory_mb: int = 512,
) -> ReducedDensityMatrices:
    """Invert Low's fixed-N orbital channel, retaining complex off-diagonal RDMs."""
    m = setting.num_modes
    if (
        setting.ensemble != "orbital_haar"
        or bits < 0
        or bits.bit_length() > m
        or bits.bit_count() != num_particles
        or max_order not in {1, 2}
    ):
        raise ConfigError("orbital shadow snapshot requires a matching fixed-N outcome")
    _snapshot_storage(m, max_order, max_memory_mb)
    occupied = np.array([(bits >> mode) & 1 for mode in range(m)])
    # rho_snapshot = U†|bits><bits|U; gamma[p,q] is rho[q,p].
    normal = (setting.matrix.conj().T @ (occupied[:, None] * setting.matrix)).T
    one = (m + 1) * normal - num_particles * np.eye(m)
    two = None
    if max_order == 2:
        two = np.zeros((m,) * 4, dtype=complex)
        if num_particles >= 2:
            two = (
                comb(m + 1, 2) * _wedge(normal)
                - (num_particles - 1) * (m + 1) / 2 * _lift(normal)
                + comb(num_particles, 2) * _wedge(np.eye(m, dtype=complex))
            )
        two.setflags(write=False)
    one.setflags(write=False)
    return ReducedDensityMatrices(one, two)


def majorana_shadow_snapshot(
    setting: ShadowSetting, bits: int, *, max_order: int = 2, max_memory_mb: int = 512
) -> ReducedDensityMatrices:
    """Invert the signed-Majorana channel; occupations are never N-postselected."""
    m = setting.num_modes
    if (
        setting.ensemble != "majorana_clifford"
        or bits < 0
        or bits.bit_length() > m
        or max_order not in {1, 2}
    ):
        raise ConfigError("Majorana shadow snapshot requires a matching occupation outcome")
    _snapshot_storage(m, max_order, max_memory_mb)
    measured = np.zeros((2 * m, 2 * m))
    for mode in range(m):
        measured[2 * mode, 2 * mode + 1] = 2 * ((bits >> mode) & 1) - 1
        measured[2 * mode + 1, 2 * mode] = -measured[2 * mode, 2 * mode + 1]
    covariance = setting.matrix.T @ measured @ setting.matrix
    annihilator = np.zeros((m, 2 * m), dtype=complex)
    annihilator[np.arange(m), 2 * np.arange(m)] = 0.5
    annihilator[np.arange(m), 2 * np.arange(m) + 1] = 0.5j
    normal = 0.5 * np.eye(m) - 1j * annihilator.conj() @ covariance @ annihilator.T
    pairing = -1j * annihilator @ covariance @ annihilator.T
    inverse_two_majoranas = 2 * m - 1
    one = 0.5 * np.eye(m) + inverse_two_majoranas * (normal - 0.5 * np.eye(m))
    two = None
    if max_order == 2:
        two = np.zeros((m,) * 4, dtype=complex)
        if m >= 2:
            gaussian = _wedge(normal) + np.einsum("pq,rs->pqrs", pairing.conj(), pairing)
            scalar = 0.25 * _wedge(np.eye(m, dtype=complex))
            quadratic = 0.5 * _lift(normal - 0.5 * np.eye(m))
            inverse_four_majoranas = comb(2 * m, 4) / comb(m, 2)
            two = (
                scalar
                + inverse_two_majoranas * quadratic
                + inverse_four_majoranas * (gaussian - scalar - quadratic)
            )
        two.setflags(write=False)
    one.setflags(write=False)
    return ReducedDensityMatrices(one, two)


@dataclass(frozen=True)
class RDMStandardErrors:
    """Real standard errors for one component of complex one/two-RDM estimates."""

    one_body: NDArray[np.float64]
    two_body: NDArray[np.float64] | None = None


@dataclass(frozen=True)
class ShadowResult:
    """Raw settings/outcomes, cluster estimates and aggregate experimental RDMs."""

    settings: tuple[ShadowSetting, ...]
    counts: tuple[dict[str, int], ...]
    setting_rdms: tuple[ReducedDensityMatrices, ...]
    rdms: ReducedDensityMatrices
    standard_errors_real: RDMStandardErrors | None
    standard_errors_imag: RDMStandardErrors | None
    metadata: dict[str, Any]

    def observable(self, operator: FermionicHamiltonian) -> dict[str, float | None]:
        """Preserve covariance by contracting each complete setting before estimating error."""
        estimates = np.array([rdm_expectation(rdm, operator) for rdm in self.setting_rdms])
        return {
            "mean": float(estimates.mean()),
            "standard_error": float(estimates.std(ddof=1) / sqrt(len(estimates)))
            if len(estimates) > 1
            else None,
        }


def rdm_expectation(rdms: ReducedDensityMatrices, operator: FermionicHamiltonian) -> float:
    """Contract an owned Hermitian observable with one/two-RDM conventions."""
    if rdms.one_body.shape != (operator.num_modes,) * 2:
        raise ConfigError("RDM and observable mode counts differ")
    value = 0j
    for term in operator.terms:
        if not term.creation:
            value += term.coefficient
        elif len(term.creation) == 1:
            value += term.coefficient * rdms.one_body[*term.creation, *term.annihilation]
        else:
            if rdms.two_body is None:
                raise ConfigError("two-body observable requires a two-body RDM")
            value += term.coefficient * rdms.two_body[*term.creation, *reversed(term.annihilation)]
    if not np.isfinite(value) or abs(value.imag) > 1e-8:
        raise ConfigError("RDM observable contraction must be finite and real")
    return float(value.real)


def estimate_fermionic_shadows(
    settings: Sequence[ShadowSetting],
    counts: Sequence[Mapping[str, int]],
    options: FermionicShadowOptions,
) -> ShadowResult:
    """Reconstruct measured batches, treating each randomized setting as one cluster."""
    if len(settings) != options.num_settings or len(counts) != len(settings):
        raise ConfigError("shadow settings and count batches must match num_settings")
    m = settings[0].num_modes
    _storage(m, options)
    clusters = []
    acceptance = []
    copied_counts = []
    total = 0
    for setting, batch in zip(settings, counts, strict=True):
        if setting.num_modes != m or setting.ensemble != options.ensemble:
            raise ConfigError("shadow settings must share the declared ensemble and mode count")
        if not batch or any(
            not isinstance(bits, str)
            or len(bits) != m
            or set(bits) - {"0", "1"}
            or isinstance(frequency, bool)
            or not isinstance(frequency, int)
            or frequency < 1
            for bits, frequency in batch.items()
        ):
            raise ConfigError(
                "shadow counts require fixed-width binary strings and positive integer frequencies"
            )
        shots = sum(batch.values())
        total += shots
        if shots > options.shots_per_setting or total > options.max_total_shots:
            raise ConfigError("shadow supplied counts exceed acquisition shot budgets")
        one = np.zeros((m, m), dtype=complex)
        two = None if options.max_order == 1 else np.zeros((m,) * 4, dtype=complex)
        accepted = 0
        for binary, frequency in batch.items():
            bits = int(binary, 2)
            if setting.ensemble == "orbital_haar":
                if bits.bit_count() != options.num_particles:
                    if options.postselect_particles:
                        continue
                    raise ConfigError(
                        "orbital shadow counts violate fixed N; enable explicit postselection"
                    )
                snapshot = orbital_shadow_snapshot(
                    setting,
                    bits,
                    num_particles=int(options.num_particles),
                    max_order=options.max_order,
                    max_memory_mb=options.max_memory_mb,
                )
            else:
                snapshot = majorana_shadow_snapshot(
                    setting, bits, max_order=options.max_order, max_memory_mb=options.max_memory_mb
                )
            accepted += frequency
            one += frequency * snapshot.one_body
            if two is not None:
                two += frequency * cast("ComplexArray", snapshot.two_body)
        if accepted == 0:
            raise ConfigError(
                "a shadow setting has no accepted outcomes; its channel cannot be estimated"
            )
        one /= accepted
        if two is not None:
            two /= accepted
            two.setflags(write=False)
        one.setflags(write=False)
        clusters.append(ReducedDensityMatrices(one, two))
        acceptance.append(
            {"shots": shots, "accepted_shots": accepted, "acceptance_fraction": accepted / shots}
        )
        copied_counts.append(dict(batch))
    one_values = np.array([rdm.one_body for rdm in clusters], dtype=np.complex128)
    two_values = (
        None
        if options.max_order == 1
        else np.array([rdm.two_body for rdm in clusters], dtype=np.complex128)
    )
    mean = ReducedDensityMatrices(
        one_values.mean(axis=0), None if two_values is None else two_values.mean(axis=0)
    )
    real = imag = None
    if len(clusters) > 1:
        real = RDMStandardErrors(
            one_values.real.std(axis=0, ddof=1) / sqrt(len(clusters)),
            None
            if two_values is None
            else two_values.real.std(axis=0, ddof=1) / sqrt(len(clusters)),
        )
        imag = RDMStandardErrors(
            one_values.imag.std(axis=0, ddof=1) / sqrt(len(clusters)),
            None
            if two_values is None
            else two_values.imag.std(axis=0, ddof=1) / sqrt(len(clusters)),
        )
    for rdms in (mean, real, imag):
        if rdms is not None:
            rdms.one_body.setflags(write=False)
            if rdms.two_body is not None:
                rdms.two_body.setflags(write=False)
    return ShadowResult(
        tuple(settings),
        tuple(copied_counts),
        tuple(clusters),
        mean,
        real,
        imag,
        {
            "experimental": True,
            "ensemble": options.ensemble,
            "num_particles": options.num_particles,
            "uncertainty_cluster": "randomized_measurement_setting",
            "num_settings": len(settings),
            "total_shots": total,
            "postselect_particles": options.postselect_particles,
            "acceptance": acceptance,
            "seed": options.seed,
            "bit_order": "mode_0_right",
            "rdm_convention": "gamma[p,q]=<a†p aq>; Gamma[p,q,r,s]=<a†p a†q a_s a_r>",
        },
    )


def collect_fermionic_shadows(
    preparation: Any,
    sampler: ComponentSelection,
    options: FermionicShadowOptions,
    *,
    device: Literal["cpu", "cuda"] = "cpu",
    cores: int = 1,
) -> ShadowResult:
    """Execute real randomized circuits through the selected local/provider sampler."""
    from chemrefine.engines.qiskit.sampling import SamplingSession

    if preparation.num_clbits or preparation.num_parameters:
        raise ConfigError("shadow preparation must have no classical bits or unbound parameters")
    settings = random_shadow_settings(preparation.num_qubits, options)
    simulation_bytes = 0
    if sampler.name in {"statevector", "basic_backend"}:
        simulation_bytes = 32 * 2**preparation.num_qubits
    elif sampler.name == "aer":
        from chemrefine.engines.qiskit.registry import SAMPLERS

        controls = SAMPLERS.options_for(sampler).model_dump()
        if controls["method"] == "density_matrix" or (
            controls["method"] == "automatic" and controls["noise_model"] is not None
        ):
            simulation_bytes = 32 * 4**preparation.num_qubits
        elif controls["method"] in {"statevector", "automatic"}:
            simulation_bytes = 32 * 2**preparation.num_qubits
    if simulation_bytes > options.max_memory_mb * 1024**2:
        raise ConfigError("shadow sampler storage estimate exceeds max_memory_mb")
    counts, sampling = [], []
    with SamplingSession(sampler, device=device, cores=cores) as session:
        for setting in settings:
            circuit = preparation.compose(shadow_circuit(setting))
            batch = session.sample(circuit, shots=options.shots_per_setting)
            counts.append(batch.counts)
            sampling.append(batch.metadata)
    result = estimate_fermionic_shadows(settings, counts, options)
    result.metadata["sampling"] = sampling
    return result

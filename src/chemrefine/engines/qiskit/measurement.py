"""Commuting Pauli measurements with pilot allocation and joint-shot uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from chemrefine.engines.qiskit.options import ComponentSelection
from chemrefine.engines.qiskit.registry import SAMPLERS
from chemrefine.engines.qiskit.sampling import SampleBatch, SamplingSession
from chemrefine.errors import ConfigError


class MeasurementOptions(BaseModel):
    """Total shot budget, independent pilots and the chosen commuting partition."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    grouping: Literal["none", "qwc", "commuting"] = "qwc"
    shots: StrictInt = Field(4096, ge=1)
    pilot_shots: StrictInt = Field(64, ge=2)
    seed: StrictInt = Field(0, ge=0)
    max_terms: StrictInt = Field(4096, ge=1)
    max_memory_mb: StrictInt = Field(512, ge=1)


@dataclass(frozen=True)
class MeasurementGroup:
    """A simultaneous diagonalizing circuit and each Pauli's signed Z image."""

    circuit: Any
    labels: tuple[str, ...]
    coefficients: NDArray[np.float64]
    z_masks: tuple[int, ...]
    signs: tuple[int, ...]
    tableau: NDArray[np.uint8]


def _independent(paulis: Any) -> list[str]:
    """Choose independent symplectic vectors before asking for a stabilizer frame."""
    pivots: dict[int, int] = {}
    labels = []
    for pauli in paulis:
        value = sum(int(bit) << index for index, bit in enumerate(np.r_[pauli.x, pauli.z]))
        while value:
            pivot = value.bit_length() - 1
            if pivot in pivots:
                value ^= pivots[pivot]
            else:
                pivots[pivot] = value
                labels.append(pauli.to_label())
                break
    return labels


def measurement_groups(
    observable: Any, options: MeasurementOptions
) -> tuple[float, tuple[MeasurementGroup, ...]]:
    """Separate constants and synthesize actual joint measurement rotations."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Clifford, SparsePauliOp, StabilizerState

    operator = observable.simplify(atol=0)
    if len(operator) > options.max_terms:
        raise ConfigError("observable exceeds measurement max_terms")
    if not np.isfinite(operator.coeffs).all() or np.any(np.abs(operator.coeffs.imag) > 1e-12):
        raise ConfigError("measurement observable must be finite and Hermitian")
    constants = 0.0
    terms = []
    for label, coefficient in operator.to_list():
        if set(label) <= {"I"}:
            constants += float(coefficient.real)
        else:
            terms.append((label, coefficient))
    if not terms:
        return constants, ()
    nonconstant = SparsePauliOp.from_list(terms)
    groups = (
        [SparsePauliOp.from_list([term]) for term in terms]
        if options.grouping == "none"
        else nonconstant.group_commuting(qubit_wise=options.grouping == "qwc")
    )
    result = []
    for group in groups:
        if options.grouping == "commuting":
            frame = StabilizerState.from_stabilizer_list(
                _independent(group.paulis), allow_underconstrained=True
            ).clifford
            rotation = frame.adjoint().to_circuit()
        else:
            rotation = QuantumCircuit(operator.num_qubits)
            for qubit in range(operator.num_qubits):
                axes = {label[-1 - qubit] for label in group.paulis.to_labels()} - {"I"}
                if "Y" in axes:
                    rotation.sdg(qubit)
                    rotation.h(qubit)
                elif "X" in axes:
                    rotation.h(qubit)
            frame = Clifford(rotation).adjoint()
        diagonal = group.paulis.evolve(frame, frame="h")
        if np.any(diagonal.x) or any(int(phase) not in (0, 2) for phase in diagonal.phase):
            raise ConfigError("commuting measurement synthesis did not produce signed Z operators")
        result.append(
            MeasurementGroup(
                rotation,
                tuple(group.paulis.to_labels()),
                np.asarray(group.coeffs.real, dtype=float),
                tuple(
                    sum(int(bit) << index for index, bit in enumerate(row)) for row in diagonal.z
                ),
                tuple(1 if phase == 0 else -1 for phase in diagonal.phase),
                np.asarray(Clifford(rotation).tableau, dtype=np.uint8),
            )
        )
    return constants, tuple(result)


def group_statistics(
    group: MeasurementGroup, samples: SampleBatch
) -> tuple[float, float, NDArray[np.float64]]:
    """Estimate a weighted observable using the covariance of jointly measured terms."""
    if samples.shots < 2:
        raise ConfigError("measurement covariance requires at least two shots")
    means = np.zeros(len(group.labels))
    for bits, count in samples.counts.items():
        values = np.asarray(
            [
                sign * (-1 if (int(bits, 2) & mask).bit_count() % 2 else 1)
                for mask, sign in zip(group.z_masks, group.signs, strict=True)
            ],
            dtype=float,
        )
        means += count * values
    means /= samples.shots
    covariance = np.zeros((len(means), len(means)))
    for bits, count in samples.counts.items():
        centered = (
            np.asarray(
                [
                    sign * (-1 if (int(bits, 2) & mask).bit_count() % 2 else 1)
                    for mask, sign in zip(group.z_masks, group.signs, strict=True)
                ],
                dtype=float,
            )
            - means
        )
        covariance += count * np.outer(centered, centered) / (samples.shots - 1)
    variance = float(group.coefficients @ covariance @ group.coefficients)
    return float(group.coefficients @ means), max(0.0, variance), covariance


def measure_observable(
    circuit: Any,
    observable: Any,
    *,
    options: MeasurementOptions | None = None,
    sampler: ComponentSelection | None = None,
    cores: int = 1,
    device: Literal["cpu", "cuda"] = "cpu",
) -> dict[str, Any]:
    """Measure with independent pilots, variance allocation and covariance-aware errors.

    Pilots determine production allocations and are not reused in the final mean:
    adaptive sample counts otherwise couple pilot fluctuations to their own weight.
    Reported errors are empirical standard errors, not confidence guarantees.
    """
    options = options or MeasurementOptions()
    sampler = sampler or ComponentSelection.named("statevector")
    if circuit.num_qubits != observable.num_qubits or circuit.num_clbits or circuit.num_parameters:
        raise ConfigError(
            "measurement requires a bound, unmeasured circuit matching the observable"
        )
    width = int(circuit.num_qubits)
    storage = (
        24 * len(observable) ** 2
        + options.shots * (width + 32)
        + len(observable) * 2 * width * (2 * width + 1)
    )
    if sampler.name in {"statevector", "basic_backend", "aer"}:
        method = SAMPLERS.options_for(sampler).model_dump().get("method", "statevector")
        if method in {"automatic", "statevector", "density_matrix"}:
            storage += 64 * (1 << (width * (2 if method == "density_matrix" else 1)))
    if storage > options.max_memory_mb * 1024**2:
        raise ConfigError("measurement storage estimate exceeds max_memory_mb")
    constant, groups = measurement_groups(observable, options)
    if not groups:
        return {
            "measurement_format_version": 2,
            "expectation": constant,
            "standard_error": 0.0,
            "shots": 0,
            "grouping": options.grouping,
            "pilot_policy": "independent_allocation_only",
            "groups": [],
        }
    remaining = options.shots - len(groups) * options.pilot_shots
    if remaining < 2 * len(groups):
        raise ConfigError(
            "measurement shot budget requires pilots plus two production shots per group"
        )
    pilots, pilot_metadata = [], []
    records = []
    expectation, variance_of_mean = constant, 0.0
    with SamplingSession(sampler, cores=cores, device=device, seed=options.seed) as session:
        for group in groups:
            pilot = session.sample(circuit.compose(group.circuit), shots=options.pilot_shots)
            pilots.append(group_statistics(group, pilot)[1])
            pilot_metadata.append(pilot.metadata)
        scale = np.sqrt(pilots)
        if not np.any(scale):
            scale = np.ones(len(groups))
        fractional = (remaining - 2 * len(groups)) * scale / scale.sum()
        allocation = np.floor(fractional).astype(int) + 2
        for index in np.argsort(-(fractional - np.floor(fractional)), kind="stable")[
            : remaining - sum(allocation)
        ]:
            allocation[index] += 1
        for index, (group, shots) in enumerate(zip(groups, allocation, strict=True)):
            samples = session.sample(circuit.compose(group.circuit), shots=int(shots))
            mean, variance, covariance = group_statistics(group, samples)
            expectation += mean
            variance_of_mean += variance / int(shots)
            records.append(
                {
                    "paulis": list(group.labels),
                    "coefficients": group.coefficients.tolist(),
                    "shots": int(shots),
                    "pilot_shots": options.pilot_shots,
                    "pilot_variance": pilots[index],
                    "pilot_sampling": pilot_metadata[index],
                    "sampling": samples.metadata,
                    "expectation": mean,
                    "covariance": covariance.tolist(),
                    "counts": samples.counts,
                    "z_masks": list(group.z_masks),
                    "signs": list(group.signs),
                    "clifford_tableau": group.tableau.tolist(),
                    "basis_gate_counts": dict(group.circuit.count_ops()),
                    "basis_depth": group.circuit.depth(),
                }
            )
    return {
        "measurement_format_version": 2,
        "expectation": expectation,
        "standard_error": float(np.sqrt(variance_of_mean)),
        "shots": options.shots,
        "grouping": options.grouping,
        "pilot_policy": "independent_allocation_only",
        "groups": records,
    }

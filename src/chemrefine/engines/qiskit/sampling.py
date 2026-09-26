"""Canonical terminal-measurement samples shared by quantum subspace workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from chemrefine.engines.qiskit.options import ComponentSelection
from chemrefine.engines.qiskit.registry import SAMPLERS
from chemrefine.errors import ConfigError


@dataclass(frozen=True)
class SampleBatch:
    """Physical counts in Qiskit display order, with logical qubit zero at right.

    Counts represent observed shots, never negative mitigation quasiprobabilities.
    The measured classical bits retain their logical meaning after routing.
    """

    counts: dict[str, int]
    num_qubits: int
    shots: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject malformed sample registers or inconsistent shot totals."""
        if (
            isinstance(self.num_qubits, bool)
            or not isinstance(self.num_qubits, int)
            or self.num_qubits < 1
            or isinstance(self.shots, bool)
            or not isinstance(self.shots, int)
            or self.shots < 1
        ):
            raise ConfigError("qiskit sample qubit and shot counts must be positive integers")
        if not isinstance(self.counts, Mapping) or not self.counts:
            raise ConfigError("qiskit samples require non-empty bitstring counts")
        for bits, count in self.counts.items():
            if not isinstance(bits, str) or len(bits) != self.num_qubits or set(bits) - {"0", "1"}:
                raise ConfigError("qiskit sample bitstrings must match the logical qubit count")
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ConfigError("qiskit sample counts must be positive integers")
        if sum(self.counts.values()) != self.shots:
            raise ConfigError("qiskit sample counts do not sum to the requested shots")
        object.__setattr__(self, "counts", dict(self.counts))
        object.__setattr__(self, "metadata", dict(self.metadata))


def sample_circuit(
    circuit: Any,
    selection: ComponentSelection | str = "statevector",
    *,
    shots: int,
    device: Literal["cpu", "cuda"] = "cpu",
    cores: int = 1,
    parameter_values: Sequence[float] | None = None,
) -> SampleBatch:
    """Measure every logical qubit once using the selected managed V2 sampler.

    Input circuits must have no classical registers. This restriction excludes
    ambiguous existing measurements and conditional circuits. A fresh measurement
    register is added before target compilation, preserving occupation decoding.
    """
    if isinstance(shots, bool) or not isinstance(shots, int) or shots < 1:
        raise ConfigError("qiskit sampler shots must be a positive integer")
    if circuit.num_clbits:
        raise ConfigError("qiskit sampling requires a circuit without classical registers")
    measured = circuit.copy()
    if parameter_values is not None:
        try:
            if np.iscomplexobj(parameter_values):
                raise ValueError("complex parameters")
            values = np.asarray(parameter_values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ConfigError("qiskit sample parameters must be finite real numbers") from exc
        if values.shape != (measured.num_parameters,) or not np.isfinite(values).all():
            raise ConfigError("qiskit sampling requires one finite value per circuit parameter")
        measured = measured.assign_parameters(values)
    if measured.num_parameters:
        raise ConfigError("qiskit sampling requires a fully bound circuit")
    if measured.num_qubits < 1:
        raise ConfigError("qiskit sampling requires at least one qubit")
    measured.measure_all()
    selected = ComponentSelection.named(selection) if isinstance(selection, str) else selection
    resource = SAMPLERS.build(selected, device=device, cores=cores)
    with resource as sampler:
        if resource.transpiler is not None:
            measured = resource.transpiler.run(measured)
        result = sampler.run([measured], shots=shots).result()
        try:
            counts = result[0].data.meas.get_counts()
        except (AttributeError, IndexError, TypeError) as exc:
            raise ConfigError("qiskit sampler returned no terminal measurement counts") from exc
    return SampleBatch(
        counts=counts,
        num_qubits=int(circuit.num_qubits),
        shots=shots,
        metadata={
            "sampler": selected.name,
            "options": SAMPLERS.options_for(selected).model_dump(mode="json"),
            "bit_order": "qubit_0_right",
            "transpiled": resource.transpiler is not None,
        },
    )

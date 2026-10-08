"""Canonical terminal-measurement samples shared by quantum subspace workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Self

import numpy as np

from chemrefine.engines.qiskit.context import SamplerResource
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


def _measured_circuit(circuit: Any, shots: int, parameter_values: Sequence[float] | None) -> Any:
    """Validate and bind an unmeasured circuit before opening a provider resource."""
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
    return measured


class SamplingSession:
    """Own one sampler, provider budget and retrieval cursor across an experiment.

    Each local request receives a reproducible child seed. The provider callback
    must apply that seed through a public RNG/options interface. Providers without
    local seed controls retain responsibility for their own sampling randomness.
    Calls are synchronous; a session must not be shared across concurrent threads.
    """

    def __init__(
        self,
        selection: ComponentSelection | str = "statevector",
        *,
        device: Literal["cpu", "cuda"] = "cpu",
        cores: int = 1,
        seed: int | None = None,
    ) -> None:
        """Resolve controls once and reserve an independent sequence of request streams."""
        self.selection = (
            ComponentSelection.named(selection) if isinstance(selection, str) else selection
        )
        resolved = SAMPLERS.options_for(self.selection)
        self.options = resolved.model_dump(mode="json")
        seeds = [
            value
            for name in ("seed", "seed_simulator")
            if (value := getattr(resolved, name, None)) is not None
        ]
        self._configured_seed = bool(seeds)
        if seed is not None:
            seeds.append(seed)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in seeds
        ):
            raise ConfigError("sampling stream seeds must be nonnegative integers")
        self._streams = np.random.SeedSequence(seeds or None)
        self.device, self.cores = device, cores
        self._resource: SamplerResource | None = None
        self._used = False
        self._request = 0

    def __enter__(self) -> Self:
        """Build exactly one provider and reject unsupported seeded custom adapters."""
        if self._used:
            raise ConfigError("sampling sessions may only be entered once")
        self._used = True
        resource = SAMPLERS.build(self.selection, device=self.device, cores=self.cores)
        if self._configured_seed and resource.set_sampling_seed is None:
            resource.close()
            raise ConfigError("a seeded sampler requires a set_sampling_seed resource callback")
        resource.__enter__()
        self._resource = resource
        return self

    def __exit__(self, *exc: object) -> None:
        """Release the provider after all requests, including a failed intermediate request."""
        if self._resource is not None:
            resource, self._resource = self._resource, None
            resource.__exit__(*exc)

    def sample(
        self,
        circuit: Any,
        *,
        shots: int,
        parameter_values: Sequence[float] | None = None,
    ) -> SampleBatch:
        """Sample a bound copy with one fresh local stream and cumulative provider limits."""
        return self._sample_measured(_measured_circuit(circuit, shots, parameter_values), shots)

    def _sample_measured(self, measured: Any, shots: int) -> SampleBatch:
        """Compile and execute one canonical register while the owned resource is open."""
        resource = self._resource
        if resource is None:
            raise ConfigError("sampling requires an open SamplingSession context")
        width = int(measured.num_qubits)
        child_seed = None
        if resource.set_sampling_seed is not None:
            child_seed = int(self._streams.spawn(1)[0].generate_state(1)[0]) % (2**32 - 1) + 1
            resource.set_sampling_seed(child_seed)
        request = self._request
        self._request += 1
        if resource.transpiler is not None:
            measured = resource.transpiler.run(measured, **(resource.transpiler_options or {}))
        result = resource.sampler.run([measured], shots=shots).result()
        try:
            counts = result[0].data.meas.get_counts()
        except (AttributeError, IndexError, TypeError) as exc:
            raise ConfigError("qiskit sampler returned no terminal measurement counts") from exc
        return SampleBatch(
            counts=counts,
            num_qubits=width,
            shots=shots,
            metadata={
                "sampler": self.selection.name,
                "options": dict(self.options),
                "bit_order": "qubit_0_right",
                "transpiled": resource.transpiler is not None,
                "request_index": request,
                "sampling_seed": child_seed,
                "random_stream": "independent_child_seed"
                if child_seed is not None
                else "provider_managed",
            },
        )


def sample_circuit(
    circuit: Any,
    selection: ComponentSelection | str = "statevector",
    *,
    shots: int,
    device: Literal["cpu", "cuda"] = "cpu",
    cores: int = 1,
    parameter_values: Sequence[float] | None = None,
) -> SampleBatch:
    """Sample one unmeasured circuit with an owned provider and canonical logical bits.

    Use ``SamplingSession`` for multiple requests so provider budgets, recovery
    cursors and independent random streams cover the entire experiment.
    """
    measured = _measured_circuit(circuit, shots, parameter_values)
    with SamplingSession(selection, device=device, cores=cores) as session:
        return session._sample_measured(measured, shots)

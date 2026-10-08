"""Experimental endpoint Pauli checks with physical, conditional shot statistics."""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from chemrefine.engines.qiskit.options import ComponentSelection
from chemrefine.engines.qiskit.registry import SAMPLERS
from chemrefine.errors import ConfigError


def _label(value: str, *, diagonal: bool = False) -> tuple[int, str]:
    """Parse a real signed Pauli without discarding its controlled phase."""
    sign = -1 if value.startswith("-") else 1
    bare = value[1:] if value.startswith(("-", "+")) else value
    if not bare or set(bare) - set("IZ" if diagonal else "IXYZ"):
        raise ValueError("Pauli labels require an optional real sign and I/X/Y/Z letters")
    return sign, bare


class SpacetimeOptions(BaseModel):
    """Chosen endpoint checks and finite-shot conditional estimation controls."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    checks: tuple[str, ...] = Field(min_length=1)
    diagonal_observables: tuple[str, ...] = ()
    shots: StrictInt = Field(10000, ge=1, le=10000000)
    confidence: float = Field(0.95, gt=0, lt=1)
    max_qubits: StrictInt = Field(128, ge=2)
    max_circuit_operations: StrictInt = Field(100000, ge=1)
    max_memory_mb: StrictInt = Field(512, ge=1)

    @model_validator(mode="after")
    def _paulis(self) -> Self:
        """Reject ambiguous widths, empty checks and non-diagonal readout requests."""
        labels = [_label(value)[1] for value in self.checks]
        if len({len(value) for value in labels}) != 1:
            raise ValueError("spacetime checks must have equal widths")
        if any(set(value) == {"I"} for value in labels):
            raise ValueError("spacetime checks must act on at least one data qubit")
        for value in self.diagonal_observables:
            if len(_label(value, diagonal=True)[1]) != len(labels[0]):
                raise ValueError("spacetime observables must match the check width")
        return self


@dataclass(frozen=True)
class CheckedCircuit:
    """A unitary checked payload with explicit data and syndrome qubit locations."""

    circuit: Any
    num_data_qubits: int
    input_checks: tuple[str, ...]
    output_checks: tuple[str, ...]


@dataclass(frozen=True)
class SpacetimeResult:
    """Physical joint counts and conditional data counts, without signed weights."""

    raw_counts: dict[str, int]
    accepted_counts: dict[str, int]
    rejected_counts: dict[str, int]
    metadata: dict[str, Any]


def _controlled_pauli(circuit: Any, label: str, control: int) -> None:
    """Implement a signed tensor Pauli controlled on one check ancilla."""
    sign, bare = _label(label)
    if sign == -1:
        circuit.z(control)
    for qubit, letter in enumerate(reversed(bare)):
        if letter != "I":
            getattr(circuit, "c" + letter.lower())(control, qubit)


def build_spacetime_circuit(payload: Any, options: SpacetimeOptions) -> CheckedCircuit:
    """Sandwich a bound Clifford unitary without projecting its arbitrary input.

    If input checks do not commute, their conjugated output checks must be
    applied in reverse order. The signs in U P U† remain controlled phases.
    This endpoint construction does not perform distributed-check optimization.
    """
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.exceptions import QiskitError
    from qiskit.quantum_info import Clifford, Pauli

    modes, count = payload.num_qubits, len(options.checks)
    if modes < 1 or modes != len(_label(options.checks[0])[1]):
        raise ConfigError("spacetime check width must equal the nonempty payload width")
    if modes + count > options.max_qubits:
        raise ConfigError("spacetime data and check qubits exceed max_qubits")
    if payload.num_clbits or payload.num_parameters:
        raise ConfigError("spacetime payload must be fully bound without classical bits")
    if any(
        instruction.operation.name in {"reset", "measure", "initialize", "delay"}
        or hasattr(instruction.operation, "blocks")
        for instruction in payload.data
    ):
        raise ConfigError("spacetime payload must be a Clifford unitary without control flow")
    estimated = len(payload.data) + 2 * count * (modes + 2) + 2
    if estimated > options.max_circuit_operations:
        raise ConfigError("spacetime checked circuit exceeds max_circuit_operations")
    try:
        clifford = Clifford(payload)
    except QiskitError as exc:
        raise ConfigError("spacetime payload must contain supported Clifford operations") from exc
    outgoing = tuple(
        Pauli(label).evolve(clifford, frame="s").to_label() for label in options.checks
    )
    circuit = QuantumCircuit(QuantumRegister(modes, "data"), QuantumRegister(count, "checks"))
    circuit.h(range(modes, modes + count))
    for index, label in enumerate(options.checks):
        _controlled_pauli(circuit, label, modes + index)
    circuit.barrier(label="spacetime_payload_start")
    circuit.compose(payload, qubits=range(modes), inplace=True)
    circuit.barrier(label="spacetime_payload_end")
    for index in reversed(range(count)):
        _controlled_pauli(circuit, outgoing[index], modes + index)
    circuit.h(range(modes, modes + count))
    return CheckedCircuit(circuit, modes, options.checks, outgoing)


def _acceptance(accepted: int, shots: int, confidence: float) -> dict[str, Any]:
    """Return binomial rate, plug-in standard error and Wilson score interval."""
    rate = accepted / shots
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    divisor = 1 + z * z / shots
    midpoint = (rate + z * z / (2 * shots)) / divisor
    half = z * (rate * (1 - rate) / shots + z * z / (4 * shots * shots)) ** 0.5 / divisor
    return {
        "accepted_shots": accepted,
        "raw_shots": shots,
        "acceptance_rate": rate,
        "acceptance_standard_error": (rate * (1 - rate) / shots) ** 0.5,
        "acceptance_confidence": confidence,
        "acceptance_wilson_interval": [max(0.0, midpoint - half), min(1.0, midpoint + half)],
        "sampling_overhead": shots / accepted if accepted else None,
    }


def postselect_spacetime_counts(
    counts: dict[str, int], checked: CheckedCircuit, options: SpacetimeOptions
) -> SpacetimeResult:
    """Condition physical joint counts on zero syndrome, preserving rejected shots.

    Keys have the explicit form ``syndrome data``, each in Qiskit display order.
    The function never interprets quasiprobabilities as physical counts.
    """
    accepted: Counter[str] = Counter()
    rejected: dict[str, int] = {}
    width, checks = checked.num_data_qubits, len(checked.input_checks)
    if options.checks != checked.input_checks:
        raise ConfigError("spacetime options must match the circuit checks")
    for bits, frequency in counts.items():
        parts = bits.split(" ") if isinstance(bits, str) else []
        if (
            len(parts) != 2
            or len(parts[0]) != checks
            or len(parts[1]) != width
            or set("".join(parts)) - {"0", "1"}
            or isinstance(frequency, bool)
            or not isinstance(frequency, int)
            or frequency < 1
        ):
            raise ConfigError(
                "spacetime requires positive integer physical counts with syndrome data keys"
            )
        if "1" in parts[0]:
            rejected[bits] = frequency
        else:
            accepted[parts[1]] += frequency
    shots = sum(counts.values())
    if shots != options.shots:
        raise ConfigError("spacetime physical counts do not sum to the requested shots")
    kept = sum(accepted.values())
    estimates = {}
    for label in options.diagonal_observables:
        sign, bare = _label(label, diagonal=True)
        total = sum(
            sign
            * (-1)
            ** sum(bit == "1" and pauli == "Z" for bit, pauli in zip(bits, bare, strict=True))
            * frequency
            for bits, frequency in accepted.items()
        )
        mean = total / kept if kept else None
        estimates[label] = {
            "conditional_mean": mean,
            "conditional_standard_error": (
                (max(0.0, 1 - mean * mean) / (kept - 1)) ** 0.5
                if mean is not None and kept > 1
                else None
            ),
            "accepted_shots": kept,
        }
    return SpacetimeResult(
        dict(counts),
        dict(accepted),
        rejected,
        {
            **_acceptance(kept, shots, options.confidence),
            "conditional_observables": estimates,
            "input_checks": list(checked.input_checks),
            "output_checks": list(checked.output_checks),
            "output_check_order": "reverse input order",
            "num_data_qubits": width,
            "num_check_qubits": checks,
            "joint_count_order": "syndrome data; qubit zero right within each register",
            "method": "experimental endpoint coherent Pauli checks",
            "estimate_semantics": "conditional on zero measured syndrome; may be biased",
            "uncertainty_semantics": (
                "independent-shot binomial sampling only; excludes noise-model error"
            ),
        },
    )


def _noise_applies(noise: dict[str, Any], circuit: Any) -> bool:
    """Match global/local serialized channels against the final physical instructions."""
    operations: dict[str, set[tuple[int, ...]]] = {}
    for item in circuit.data:
        positions = tuple(circuit.find_bit(bit).index for bit in item.qubits)
        operations.setdefault(item.operation.name, set()).add(positions)
    measured = {position[0] for position in operations.get("measure", ())}
    for error in noise["errors"]:
        for name in error["operations"]:
            if name not in operations:
                continue
            targets = error.get("gate_qubits")
            if targets is None:
                return True
            if error["type"] == "roerror":
                if any(set(group) <= measured for group in targets):
                    return True
            elif any(tuple(group) in operations[name] for group in targets):
                return True
    return False


def collect_spacetime_counts(
    payload: Any,
    selection: ComponentSelection,
    options: SpacetimeOptions,
    *,
    preparation: Any | None = None,
    device: Literal["cpu", "cuda"] = "cpu",
    cores: int = 1,
) -> SpacetimeResult:
    """Compile and sample the complete checked circuit with explicit nonideal Aer noise."""
    from qiskit import ClassicalRegister
    from qiskit_aer import AerError
    from qiskit_aer.noise import NoiseModel

    sampler_options = SAMPLERS.options_for(selection)
    noise = getattr(sampler_options, "noise_model", None)
    if selection.name != "aer" or not isinstance(noise, dict) or not noise:
        raise ConfigError(
            "spacetime postselection requires sampler aer and a serialized nonideal noise_model"
        )
    try:
        # Aer <0.18 retains this established serialized-model boundary.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r".*from_dict.*deprecated.*", category=DeprecationWarning
            )
            model = NoiseModel.from_dict(noise)
    except (AerError, ValueError, TypeError, KeyError) as exc:
        raise ConfigError("spacetime noise_model is not a serialized Aer NoiseModel") from exc
    if model.is_ideal():
        raise ConfigError("spacetime postselection requires a nonideal noise_model")
    checked = build_spacetime_circuit(payload, options)
    circuit = checked.circuit
    width, count = checked.num_data_qubits, len(checked.input_checks)
    if preparation is not None:
        if preparation.num_qubits != width or preparation.num_clbits or preparation.num_parameters:
            raise ConfigError(
                "spacetime preparation must be bound, width-matched and without classical bits"
            )
        circuit = circuit.copy_empty_like()
        circuit.compose(preparation, qubits=range(width), inplace=True)
        circuit.compose(checked.circuit, inplace=True)
    if len(circuit.data) + width + count > options.max_circuit_operations:
        raise ConfigError("spacetime prepared circuit exceeds max_circuit_operations")
    if options.shots * (width + count + 128) > options.max_memory_mb * 1024**2:
        raise ConfigError("spacetime acquisition exceeds max_memory_mb")
    circuit.add_register(
        ClassicalRegister(width, "data_bits"), ClassicalRegister(count, "check_bits")
    )
    circuit.measure(range(width), circuit.cregs[0])
    circuit.measure(range(width, width + count), circuit.cregs[1])
    resource = SAMPLERS.build(selection, cores=cores, device=device)
    with resource as sampler:
        if resource.transpiler is not None:
            circuit = resource.transpiler.run(circuit)
        if not _noise_applies(model.to_dict(serializable=True), circuit):
            raise ConfigError(
                "spacetime noise_model has no channel applicable to the compiled circuit"
            )
        result = sampler.run([circuit], shots=options.shots).result()
        try:
            data = result[0].data
            words = data.data_bits.get_bitstrings()
            syndromes = data.check_bits.get_bitstrings()
            if len(words) != len(syndromes):
                raise ValueError("unequal register shot counts")
            counts = Counter(
                syndrome + " " + word for syndrome, word in zip(syndromes, words, strict=True)
            )
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            raise ConfigError(
                "spacetime sampler returned invalid data/check measurement registers"
            ) from exc
    outcome = postselect_spacetime_counts(dict(counts), checked, options)
    outcome.metadata.update(
        {
            "sampler": selection.name,
            "noise_model": noise,
            "sampler_options": sampler_options.model_dump(mode="json"),
            "compiled_qubits": circuit.num_qubits,
            "compiled_depth": circuit.depth(),
            "compiled_operations": dict(circuit.count_ops()),
            "preparation_is_checked": False,
        }
    )
    return outcome

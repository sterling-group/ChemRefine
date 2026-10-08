"""Bounded circuit cutting with isolated QPD randomness and physical shot records."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from chemrefine.engines.qiskit.bundles import MAX_DESCRIPTOR_BYTES
from chemrefine.engines.qiskit.options import ComponentSelection
from chemrefine.engines.qiskit.registry import SAMPLERS
from chemrefine.errors import ConfigError


class WireCut(BaseModel):
    """Cut one original qubit immediately before an original instruction index."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    qubit: StrictInt = Field(ge=0)
    before_instruction: StrictInt = Field(ge=0)


class CuttingOptions(BaseModel):
    """Separate QPD draws, per-experiment shots, and hard generation budgets."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    mode: Literal["manual", "partition", "automatic"] = "automatic"
    gate_cuts: tuple[StrictInt, ...] = ()
    wire_cuts: tuple[WireCut, ...] = ()
    partition_labels: tuple[str, ...] | None = None
    max_subcircuit_qubits: StrictInt = Field(12, ge=1, le=256)
    num_samples: Annotated[StrictInt, Field(ge=1)] | Literal["exact"] = 256
    shots: StrictInt = Field(4096, ge=1)
    seed: StrictInt = Field(0, ge=0, le=2**32 - 1)
    allow_gate_cuts: bool = True
    allow_wire_cuts: bool = True
    max_backjumps: StrictInt = Field(10000, ge=1)
    max_sampling_overhead: float = Field(1000000.0, ge=1)
    max_cuts: StrictInt = Field(20, ge=0)
    max_input_qubits: StrictInt = Field(256, ge=1)
    max_input_instructions: StrictInt = Field(10000, ge=1)
    max_observables: StrictInt = Field(256, ge=1)
    max_exact_terms: StrictInt = Field(1000000, ge=1)
    max_subexperiments: StrictInt = Field(1024, ge=1)
    max_generated_instructions: StrictInt = Field(10000000, ge=1)
    max_total_shots: StrictInt = Field(10000000, ge=1)
    max_working_bytes: StrictInt = Field(268435456, ge=1)
    max_qpy_bytes: StrictInt = Field(33554432, ge=1)
    worker_timeout_seconds: float = Field(120.0, gt=0, le=3600)
    publications_per_job: StrictInt = Field(64, ge=1)

    @model_validator(mode="after")
    def _mode_inputs(self) -> Self:
        """Reject duplicate locations and controls ignored by the selected planning mode."""
        wires = {(cut.qubit, cut.before_instruction) for cut in self.wire_cuts}
        if len(wires) != len(self.wire_cuts) or len(set(self.gate_cuts)) != len(self.gate_cuts):
            raise ValueError("cut locations must be unique")
        if any(index < 0 for index in self.gate_cuts):
            raise ValueError("gate indices must be nonnegative")
        if self.mode == "manual":
            if not self.gate_cuts and not self.wire_cuts:
                raise ValueError("manual cutting requires a gate or wire cut")
        elif self.gate_cuts or self.wire_cuts:
            raise ValueError("explicit gate/wire locations require mode=manual")
        if (self.mode == "partition") != (self.partition_labels is not None):
            raise ValueError("partition_labels are required exactly for mode=partition")
        if self.partition_labels is not None and any(not label for label in self.partition_labels):
            raise ValueError("partition labels must be nonempty strings")
        if self.mode != "automatic" and (
            not self.allow_gate_cuts or not self.allow_wire_cuts or self.max_backjumps != 10000
        ):
            raise ValueError("automatic search controls require mode=automatic")
        if self.mode == "automatic" and not (self.allow_gate_cuts or self.allow_wire_cuts):
            raise ValueError("automatic cutting must allow gate or wire cuts")
        return self


@dataclass(frozen=True)
class CuttingPlan:
    """Generated measured circuits and signed SDK coefficients, ready for a V2 sampler."""

    circuits: dict[str, list[Any]]
    subobservables: dict[str, list[str]]
    coefficients: tuple[tuple[float, str], ...]
    observable: dict[str, float]
    options: CuttingOptions
    metadata: dict[str, Any]
    qpy_data: bytes = b""


@dataclass(frozen=True)
class CuttingResult:
    """Unclipped reconstruction plus raw BitArray buffers and their register descriptors."""

    expectation: float
    term_expectations: NDArray[np.float64]
    arrays: dict[str, NDArray[Any]]
    metadata: dict[str, Any]


def validate_cutting_input(
    circuit: Any, observable: Mapping[str, float], options: CuttingOptions
) -> None:
    """Validate index conventions and bounded inputs without importing the cutting addon."""
    if circuit.num_clbits or circuit.num_parameters or circuit.num_qubits < 1:
        raise ConfigError("cutting requires a bound circuit without classical bits")
    if (
        circuit.num_qubits > options.max_input_qubits
        or len(circuit.data) > options.max_input_instructions
    ):
        raise ConfigError("cutting input exceeds qubit or instruction budget")
    if not observable or len(observable) > options.max_observables:
        raise ConfigError("cutting observable count is empty or exceeds max_observables")
    for label, value in observable.items():
        if (
            not isinstance(label, str)
            or len(label) != circuit.num_qubits
            or set(label) - set("IXYZ")
            or np.iscomplexobj(value)
            or not np.isfinite(value)
        ):
            raise ConfigError(
                "cutting requires finite real coefficients and matching I/X/Y/Z labels"
            )
    for index in options.gate_cuts:
        if index >= len(circuit.data) or len(circuit.data[index].qubits) != 2:
            raise ConfigError("gate_cuts must index original two-qubit instructions")
    for cut in options.wire_cuts:
        if cut.qubit >= circuit.num_qubits or cut.before_instruction > len(circuit.data):
            raise ConfigError("wire cuts must reference original qubits and instruction boundaries")
    if options.partition_labels is not None and len(options.partition_labels) != circuit.num_qubits:
        raise ConfigError("partition_labels must contain one label per original qubit")
    if any(inst.operation.name in {"cut_wire", "qpd_1q", "qpd_2q"} for inst in circuit.data):
        raise ConfigError(
            "supply cut locations through CuttingOptions, not embedded QPD instructions"
        )
    if options.mode == "automatic" and any(
        len(inst.qubits) > 2 and inst.operation.name != "barrier" for inst in circuit.data
    ):
        raise ConfigError("automatic cutting requires gates on at most two qubits")


def plan_cutting(
    circuit: Any, observable: Mapping[str, float], options: CuttingOptions | None = None
) -> CuttingPlan:
    """Generate cuts in a fresh interpreter so addon NumPy randomness cannot leak.

    Only trusted QPY circuits and JSON cross the subprocess boundary. The child
    never receives sampler settings and cannot submit a quantum execution job.
    """
    from qiskit import qpy

    options = options or CuttingOptions()
    validate_cutting_input(circuit, observable, options)
    with tempfile.TemporaryDirectory(prefix="chemrefine-cutting-") as scratch:
        directory = Path(scratch)
        with (directory / "input.qpy").open("wb") as stream:
            qpy.dump(circuit, stream)
        if (directory / "input.qpy").stat().st_size > options.max_qpy_bytes:
            raise ConfigError("cutting input exceeds max_qpy_bytes")
        (directory / "request.json").write_text(
            json.dumps(
                {
                    "options": options.model_dump(mode="json"),
                    "observable": dict(observable),
                },
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        with (directory / "worker.log").open("wb") as log:
            try:
                completed = subprocess.run(  # noqa: S603 - fixed module, interpreter and owned temp path
                    [
                        sys.executable,
                        "-m",
                        "chemrefine.engines.qiskit.cutting_worker",
                        str(directory),
                    ],
                    stdout=log,
                    stderr=log,
                    check=False,
                    timeout=options.worker_timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                raise ConfigError("cutting planning exceeded worker_timeout_seconds") from exc
        manifest = directory / "plan.json"
        if completed.returncode != 0 or not manifest.is_file():
            with (directory / "worker.log").open("rb") as log:
                log.seek(max(0, log.seek(0, 2) - 4096))
                message = log.read().decode("utf-8", errors="replace")
            raise ConfigError(f"isolated cutting planner failed: {message[-4096:]}")
        if manifest.stat().st_size > MAX_DESCRIPTOR_BYTES:
            raise ConfigError("cutting plan descriptor exceeds its byte limit")
        description = json.loads(manifest.read_text(encoding="utf-8"))
        generated = directory / "experiments.qpy"
        if generated.stat().st_size > options.max_qpy_bytes:
            raise ConfigError("generated cutting circuits exceed max_qpy_bytes")
        with generated.open("rb") as stream:
            circuits = qpy.load(stream)
        qpy_data = generated.read_bytes()
        counts = description["circuit_counts"]
        if sum(counts.values()) != len(circuits) or len(circuits) > options.max_subexperiments:
            raise ConfigError("cutting plan circuit counts are inconsistent")
        by_partition = {}
        offset = 0
        for label, count in counts.items():
            by_partition[label] = circuits[offset : offset + count]
            offset += count
    return CuttingPlan(
        by_partition,
        description["subobservables"],
        tuple((float(value), str(kind)) for value, kind in description["coefficients"]),
        dict(observable),
        options,
        description["metadata"],
        qpy_data,
    )


def collect_cutting_samples(
    plan: CuttingPlan,
    selection: ComponentSelection,
    *,
    device: Literal["cpu", "cuda"] = "cpu",
    cores: int = 1,
) -> tuple[dict[str, Any], dict[str, NDArray[Any]], list[dict[str, Any]]]:
    """Execute existing measurement registers unchanged and close the sampler reliably."""
    from qiskit.primitives import BitArray, PrimitiveResult

    options = plan.options
    total = sum(map(len, plan.circuits.values()))
    if (
        total < 1
        or total > options.max_subexperiments
        or total * options.shots > options.max_total_shots
    ):
        raise ConfigError("cutting execution exceeds experiment or total-shot budget")
    record_bytes = sum(
        options.shots * sum((len(register) + 7) // 8 for register in circuit.cregs)
        for circuits in plan.circuits.values()
        for circuit in circuits
    )
    if 2 * record_bytes > options.max_working_bytes or total * 1024 > MAX_DESCRIPTOR_BYTES:
        raise ConfigError("cutting physical records exceed memory or descriptor budget")
    if selection.name in {"statevector", "basic_backend", "aer"}:
        from chemrefine.engines.qiskit.components.samplers import AerSamplerOptions

        sampler_options = SAMPLERS.options_for(selection)
        method = (
            sampler_options.method
            if isinstance(sampler_options, AerSamplerOptions)
            else "statevector"
        )
        exponent = 2 if method == "density_matrix" else 1
        width = max(
            circuit.num_qubits for circuits in plan.circuits.values() for circuit in circuits
        )
        if method in {"statevector", "automatic", "density_matrix"} and (
            16 * (1 << (exponent * width)) + 2 * record_bytes > options.max_working_bytes
        ):
            raise ConfigError("cutting simulator allocation exceeds max_working_bytes")
    resource = SAMPLERS.build(selection, device=device, cores=cores)
    results: dict[str, Any] = {}
    arrays: dict[str, NDArray[Any]] = {}
    records = []
    with resource as sampler:
        for label, circuits in plan.circuits.items():
            publications = []
            for start in range(0, len(circuits), options.publications_per_job):
                batch = []
                for circuit in circuits[start : start + options.publications_per_job]:
                    compiled = (
                        circuit
                        if resource.transpiler is None
                        else resource.transpiler.run(circuit, **(resource.transpiler_options or {}))
                    )
                    if compiled.num_qubits > options.max_subcircuit_qubits:
                        raise ConfigError("compiled cutting circuit exceeds max_subcircuit_qubits")
                    expected = {register.name: len(register) for register in circuit.cregs}
                    actual = {register.name: len(register) for register in compiled.cregs}
                    if actual != expected or set(actual) != {
                        "observable_measurements",
                        "qpd_measurements",
                    }:
                        raise ConfigError("transpilation changed cutting measurement registers")
                    batch.append(compiled)
                output = sampler.run(batch, shots=options.shots).result()
                if len(output) != len(batch):
                    raise ConfigError("cutting sampler returned the wrong publication count")
                for local, (pub, circuit) in enumerate(zip(output, batch, strict=True)):
                    index = start + local
                    register_records = {}
                    for register in circuit.cregs:
                        bits = getattr(pub.data, register.name, None)
                        if not isinstance(bits, BitArray) or bits.num_bits != len(register):
                            raise ConfigError(
                                "cutting sampler returned a missing or wrong-width BitArray"
                            )
                        if bits.num_shots != options.shots or bits.array.ndim != 2:
                            raise ConfigError(
                                "cutting sampler returned an invalid shot count or shape"
                            )
                        key = f"{label}_e{index}_{register.name}"
                        values = np.array(bits.array, dtype=np.uint8, copy=True)
                        arrays[key] = values
                        register_records[register.name] = {"array": key, "num_bits": bits.num_bits}
                    records.append(
                        {
                            "partition": label,
                            "experiment": index,
                            "shots": options.shots,
                            "registers": register_records,
                        }
                    )
                    publications.append(pub)
            results[label] = PrimitiveResult(publications)
    return results, arrays, records


def execute_cutting(
    plan: CuttingPlan,
    sampler: ComponentSelection | str = "basic_backend",
    *,
    device: Literal["cpu", "cuda"] = "cpu",
    cores: int = 1,
) -> CuttingResult:
    """Reconstruct signed expectations with the SDK from unmodified physical records."""
    from qiskit.quantum_info import PauliList
    from qiskit_addon_cutting import reconstruct_expectation_values
    from qiskit_addon_cutting.qpd import WeightType

    selected = ComponentSelection.named(sampler) if isinstance(sampler, str) else sampler
    results, arrays, records = collect_cutting_samples(plan, selected, device=device, cores=cores)
    coefficients = [(value, WeightType[kind]) for value, kind in plan.coefficients]
    subobservables = {label: PauliList(terms) for label, terms in plan.subobservables.items()}
    values = np.asarray(
        reconstruct_expectation_values(results, coefficients, subobservables), dtype=float
    )
    if values.shape != (len(plan.observable),) or not np.isfinite(values).all():
        raise ConfigError("cutting reconstruction returned invalid observable expectations")
    expectation = float(np.dot(list(plan.observable.values()), values))
    arrays["term_expectations"] = values
    arrays["observable_coefficients"] = np.asarray(list(plan.observable.values()), dtype=float)
    arrays["qpd_coefficients"] = np.asarray([value for value, _kind in plan.coefficients])
    from qiskit import qpy

    buffer = io.BytesIO(plan.qpy_data)
    if not plan.qpy_data:
        qpy.dump([circuit for group in plan.circuits.values() for circuit in group], buffer)
    arrays["logical_experiments_qpy"] = np.frombuffer(buffer.getvalue(), dtype=np.uint8).copy()
    return CuttingResult(
        expectation,
        values,
        arrays,
        {
            **plan.metadata,
            "expectation": expectation,
            "standard_error": None,
            "uncertainty_note": "QPD and shot noise both contribute; no error bar inferred",
            "observable": plan.observable,
            "observable_labels": list(plan.observable),
            "subobservables": plan.subobservables,
            "logical_qpy_partition_counts": {
                label: len(group) for label, group in plan.circuits.items()
            },
            "options": plan.options.model_dump(mode="json"),
            "sampler": {
                "name": selected.name,
                "options": SAMPLERS.options_for(selected).model_dump(mode="json"),
            },
            "weight_types": [kind for _value, kind in plan.coefficients],
            "records": records,
            "physical_record_format": (
                "raw Qiskit BitArray bytes; big-endian, little-indexed register bits"
            ),
            "signed_reconstruction": True,
            "clipped": False,
        },
    )


def reconstruct_cutting_records(
    arrays: Mapping[str, NDArray[Any]], metadata: Mapping[str, Any]
) -> NDArray[np.float64]:
    """Rebuild SDK V2 results from durable physical bytes and repeat signed reconstruction."""
    from qiskit.primitives import BitArray, DataBin, PrimitiveResult, SamplerPubResult
    from qiskit.quantum_info import PauliList
    from qiskit_addon_cutting import reconstruct_expectation_values
    from qiskit_addon_cutting.qpd import WeightType

    try:
        results: dict[str, list[Any]] = {label: [] for label in metadata["subobservables"]}
        for record in metadata["records"]:
            partition = record["partition"]
            if record["experiment"] != len(results[partition]):
                raise ValueError("nonconsecutive cutting experiment records")
            registers = {}
            if set(record["registers"]) != {"observable_measurements", "qpd_measurements"}:
                raise ValueError("missing cutting measurement register")
            for name, descriptor in record["registers"].items():
                data = arrays[descriptor["array"]]
                if data.dtype != np.uint8 or data.ndim != 2 or data.shape[0] != record["shots"]:
                    raise ValueError("invalid physical BitArray buffer")
                registers[name] = BitArray(data, descriptor["num_bits"])
            results[partition].append(SamplerPubResult(DataBin(**registers)))
        weights = np.asarray(arrays["qpd_coefficients"])
        kinds = metadata["weight_types"]
        if weights.shape != (len(kinds),) or not np.isfinite(weights).all():
            raise ValueError("invalid QPD coefficient array")
        coefficients = [
            (float(value), WeightType[kind]) for value, kind in zip(weights, kinds, strict=True)
        ]
        reconstructed = reconstruct_expectation_values(
            {label: PrimitiveResult(group) for label, group in results.items()},
            coefficients,
            {label: PauliList(terms) for label, terms in metadata["subobservables"].items()},
        )
        values = np.asarray(reconstructed, dtype=float)
        if values.shape != (len(metadata["observable"]),) or not np.isfinite(values).all():
            raise ValueError("invalid reconstructed cutting expectations")
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"invalid physical cutting records: {exc}") from exc
    return values

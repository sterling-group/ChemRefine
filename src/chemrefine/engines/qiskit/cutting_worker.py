"""Private QPY/JSON planner process isolating the addon's process-global QPD RNG."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from chemrefine.engines.qiskit.bundles import MAX_DESCRIPTOR_BYTES
from chemrefine.engines.qiskit.cutting import CuttingOptions, validate_cutting_input
from chemrefine.errors import ConfigError


def without_barriers(circuit: Any) -> Any:
    """Remove physical no-ops after original-index cut placement, before connectivity analysis."""
    result = circuit.copy_empty_like()
    for instruction in circuit.data:
        if instruction.operation.name != "barrier":
            result.append(instruction)
    return result


def expand_wire_cuts(circuit: Any) -> Any:
    """Apply the public wire transform one marker at a time, retaining multiple cuts.

    Sequential application avoids assumptions about consecutive same-wire markers
    in the addon's wire-allocation implementation. Original qubit objects remain
    the final wire segments, which is the public expand_observables convention.
    """
    from qiskit.circuit.library import IGate
    from qiskit_addon_cutting import cut_wires
    from qiskit_addon_cutting.instructions import CutWire

    current = circuit.copy()
    marker = "chemrefine_pending_wire_" + uuid4().hex
    while locations := [
        i for i, inst in enumerate(current.data) if inst.operation.name == "cut_wire"
    ]:
        for index in locations[1:]:
            current.data[index] = current.data[index].replace(operation=IGate(label=marker))
        current = cut_wires(current)
        for index, inst in enumerate(current.data):
            if inst.operation.label == marker:
                current.data[index] = inst.replace(operation=CutWire())
    return current


def generation_budget(
    subcircuits: dict[str, Any], bases: list[Any], groups: dict[str, int], options: CuttingOptions
) -> dict[str, int | float]:
    """Bound enumeration, generated circuits and shot storage before QPD sampling."""
    if len(bases) > options.max_cuts:
        raise ConfigError("cutting decomposition exceeds max_cuts")
    widths = [circuit.num_qubits for circuit in subcircuits.values()]
    if not widths or max(widths) > options.max_subcircuit_qubits:
        raise ConfigError("actual partition width exceeds max_subcircuit_qubits")
    log_overhead = 2 * sum(math.log(float(basis.kappa)) for basis in bases)
    if log_overhead > math.log(options.max_sampling_overhead) + 1e-12:
        raise ConfigError("actual QPD sampling overhead exceeds max_sampling_overhead")
    cardinality = math.prod(len(basis.maps) for basis in bases)
    smallest = math.prod(min(p for p in basis.probabilities if p > 0) for basis in bases)
    enumerate_all = options.num_samples == "exact" or smallest >= 1 / options.num_samples
    if enumerate_all and cardinality > options.max_exact_terms:
        raise ConfigError("exact QPD enumeration exceeds max_exact_terms")
    sample_bound = (
        cardinality if options.num_samples == "exact" else min(cardinality, options.num_samples)
    )
    experiment_bound = sample_bound * sum(groups.values())
    if experiment_bound > options.max_subexperiments:
        raise ConfigError("prospective cutting circuits exceed max_subexperiments")
    if experiment_bound * options.shots > options.max_total_shots:
        raise ConfigError("prospective cutting shots exceed max_total_shots")
    if experiment_bound * 1024 > MAX_DESCRIPTOR_BYTES:
        raise ConfigError("prospective cutting record descriptor exceeds its byte limit")
    # Bound the longest local map for every cut, including both halves; this
    # deliberately overestimates work when a partition contains only one half.
    map_instructions = sum(max(sum(map(len, mapping)) for mapping in basis.maps) for basis in bases)
    qpd_bits = sum(
        max(
            sum(inst.name == "qpd_measure" for local in mapping for inst in local)
            for mapping in basis.maps
        )
        for basis in bases
    )
    instructions = sample_bound * sum(
        groups[label] * (len(circuit.data) + map_instructions + 3 * circuit.num_qubits + 1)
        for label, circuit in subcircuits.items()
    )
    if instructions > options.max_generated_instructions:
        raise ConfigError(
            "prospective cutting instruction count exceeds max_generated_instructions"
        )
    record_bytes = (
        sample_bound
        * options.shots
        * sum(
            groups[label] * ((circuit.num_qubits + 7) // 8 + (qpd_bits + 7) // 8)
            for label, circuit in subcircuits.items()
        )
    )
    if 2 * record_bytes + 512 * instructions > options.max_working_bytes:
        raise ConfigError("prospective cutting allocation exceeds max_working_bytes")
    return {
        "sampling_overhead": math.exp(log_overhead),
        "qpd_cartesian_terms": cardinality,
        "qpd_unique_sample_bound": sample_bound,
        "subexperiment_bound": experiment_bound,
        "shot_record_bytes_bound": record_bytes,
        "generated_instruction_bound": instructions,
    }


def generate_plan(directory: Path) -> None:
    """Read a trusted local request, use public addon APIs, and publish a bounded plan."""
    import importlib.metadata

    from qiskit import qpy
    from qiskit.quantum_info import PauliList
    from qiskit_addon_cutting import (
        DeviceConstraints,
        OptimizationParameters,
        cut_gates,
        expand_observables,
        find_cuts,
        generate_cutting_experiments,
        partition_problem,
    )
    from qiskit_addon_cutting.instructions import CutWire
    from qiskit_addon_cutting.utils.observable_grouping import ObservableCollection

    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    options = CuttingOptions.model_validate(request["options"])
    with (directory / "input.qpy").open("rb") as stream:
        circuits = qpy.load(stream)
    if len(circuits) != 1:
        raise ConfigError("isolated cutting request requires exactly one circuit")
    original = circuits[0]
    validate_cutting_input(original, request["observable"], options)
    # This mutates only this short-lived subprocess, never the host's NumPy RNG.
    np.random.seed(options.seed)
    labels = options.partition_labels
    search: dict[str, Any] | None = None
    if options.mode == "automatic":
        marked, found = find_cuts(
            without_barriers(original),
            OptimizationParameters(
                seed=options.seed,
                max_gamma=math.sqrt(options.max_sampling_overhead),
                max_backjumps=options.max_backjumps,
                gate_lo=options.allow_gate_cuts,
                wire_lo=options.allow_wire_cuts,
            ),
            DeviceConstraints(qubits_per_subcircuit=options.max_subcircuit_qubits),
        )
        search = {
            "cuts": [[str(kind), int(index)] for kind, index in found["cuts"]],
            "sampling_overhead": float(found["sampling_overhead"]),
            "minimum_reached": bool(found["minimum_reached"]),
        }
    else:
        marked, _bases = cut_gates(original, options.gate_cuts)
        if options.wire_cuts:
            inserted = marked.copy_empty_like()
            for index in range(len(marked.data) + 1):
                for cut in options.wire_cuts:
                    if cut.before_instruction == index:
                        inserted.append(CutWire(), [marked.qubits[cut.qubit]])
                if index < len(marked.data):
                    inserted.append(marked.data[index])
            marked = inserted
    expanded = expand_wire_cuts(without_barriers(marked))
    observables = expand_observables(PauliList(list(request["observable"])), original, expanded)
    # Keep idle |0> wires as explicit one-qubit partitions so X/Y observables on
    # them contribute zero instead of disappearing from the tensor product.
    for qubit in expanded.qubits:
        expanded.id(qubit)
    problem = partition_problem(expanded, partition_labels=labels, observables=observables)
    subcircuits = {
        f"p{index}": circuit for index, circuit in enumerate(problem.subcircuits.values())
    }
    subobservables = {
        f"p{index}": problem.subobservables[label]
        for index, label in enumerate(problem.subcircuits)
    }
    groups = {
        label: len(ObservableCollection(terms).groups) for label, terms in subobservables.items()
    }
    budget = generation_budget(subcircuits, problem.bases, groups, options)
    experiments, coefficients = generate_cutting_experiments(
        subcircuits,
        subobservables,
        num_samples=math.inf if options.num_samples == "exact" else options.num_samples,
    )
    count = sum(map(len, experiments.values()))
    if (
        count > budget["subexperiment_bound"]
        or sum(len(circuit.data) for group in experiments.values() for circuit in group)
        > options.max_generated_instructions
    ):
        raise ConfigError("generated cutting workload exceeded its prospective bounds")
    with (directory / "experiments.qpy").open("wb") as stream:
        qpy.dump([circuit for group in experiments.values() for circuit in group], stream)
    if (directory / "experiments.qpy").stat().st_size > options.max_qpy_bytes:
        raise ConfigError("generated cutting circuits exceed max_qpy_bytes")
    metadata = {
        "provider": "qiskit-addon-cutting",
        "version": importlib.metadata.version("qiskit-addon-cutting"),
        "original_qubits": original.num_qubits,
        "expanded_qubits": expanded.num_qubits,
        "partition_widths": {label: circuit.num_qubits for label, circuit in subcircuits.items()},
        "partition_labels": [str(label) for label in problem.subcircuits],
        "search": search,
        "cut_count": len(problem.bases),
        "qpd_unique_samples": len(coefficients),
        "subexperiments": count,
        "total_shots": count * options.shots,
        "generated_qpy_bytes": (directory / "experiments.qpy").stat().st_size,
        "qpd_seed": options.seed,
        "rng_isolation": "dedicated QPY/JSON planner subprocess",
        **budget,
    }
    description = {
        "circuit_counts": {label: len(group) for label, group in experiments.items()},
        "subobservables": {label: terms.to_labels() for label, terms in subobservables.items()},
        "coefficients": [[float(value), kind.name] for value, kind in coefficients],
        "metadata": metadata,
    }
    encoded = json.dumps(description, allow_nan=False)
    if len(encoded.encode("utf-8")) > MAX_DESCRIPTOR_BYTES:
        raise ConfigError("cutting plan descriptor exceeds its byte limit")
    (directory / "plan.json").write_text(encoded, encoding="utf-8")


def main() -> None:
    """Execute only when explicitly launched as the private planner module."""
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m chemrefine.engines.qiskit.cutting_worker DIRECTORY")
    generate_plan(Path(sys.argv[1]))


if __name__ == "__main__":
    main()

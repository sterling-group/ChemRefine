"""Cutting preflight and register-preserving execution without the optional addon."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from qiskit import ClassicalRegister, QuantumCircuit, qpy

from chemrefine.engines.qiskit.context import SamplerResource
from chemrefine.engines.qiskit.cutting import (
    CuttingOptions,
    CuttingPlan,
    WireCut,
    collect_cutting_samples,
    plan_cutting,
    validate_cutting_input,
)
from chemrefine.engines.qiskit.cutting_worker import generation_budget, without_barriers
from chemrefine.engines.qiskit.options import ComponentSelection
from chemrefine.engines.qiskit.registry import SAMPLERS
from chemrefine.errors import ConfigError


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"mode": "manual"}, "requires a gate"),
        ({"gate_cuts": [1]}, "mode=manual"),
        ({"mode": "manual", "gate_cuts": [1, 1]}, "unique"),
        ({"mode": "manual", "gate_cuts": [-1]}, "nonnegative"),
        ({"mode": "manual", "wire_cuts": [{"qubit": 0, "before_instruction": 0}] * 2}, "unique"),
        ({"mode": "partition"}, "required exactly"),
        ({"partition_labels": ["A"]}, "required exactly"),
        ({"mode": "partition", "partition_labels": [""]}, "nonempty"),
        ({"mode": "manual", "gate_cuts": [0], "max_backjumps": 10}, "automatic search"),
        ({"allow_gate_cuts": False, "allow_wire_cuts": False}, "must allow"),
    ],
)
def test_cutting_controls_have_no_ignored_or_ambiguous_inputs(changes, match):
    """Each planning mode consumes exactly its supported controls."""
    with pytest.raises(ValidationError, match=match):
        CuttingOptions(**changes)


@pytest.mark.parametrize("observable", [{}, {"Z": 1}, {"AA": 1}, {"ZZ": 1j}, {"ZZ": np.nan}])
def test_cutting_rejects_invalid_observables(observable):
    """Width and Hermitian coefficient checks precede optional imports."""
    with pytest.raises(ConfigError, match=r"observable|coefficients"):
        validate_cutting_input(QuantumCircuit(2), observable, CuttingOptions())


def test_original_index_and_domain_guards():
    """Original locations remain unambiguous even when cuts expand the circuit."""
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    validate_cutting_input(circuit, {"ZZ": 1}, CuttingOptions(mode="manual", gate_cuts=(1,)))
    for index in (0, 2):
        with pytest.raises(ConfigError, match="two-qubit"):
            validate_cutting_input(
                circuit, {"ZZ": 1}, CuttingOptions(mode="manual", gate_cuts=(index,))
            )
    for wire in (WireCut(qubit=2, before_instruction=0), WireCut(qubit=0, before_instruction=3)):
        with pytest.raises(ConfigError, match="instruction boundaries"):
            validate_cutting_input(
                circuit, {"ZZ": 1}, CuttingOptions(mode="manual", wire_cuts=(wire,))
            )
    with pytest.raises(ConfigError, match="one label"):
        validate_cutting_input(
            circuit, {"ZZ": 1}, CuttingOptions(mode="partition", partition_labels=("A",))
        )
    with pytest.raises(ConfigError, match="qubit or instruction"):
        validate_cutting_input(circuit, {"ZZ": 1}, CuttingOptions(max_input_qubits=1))
    measured = circuit.copy()
    measured.measure_all()
    with pytest.raises(ConfigError, match="without classical"):
        validate_cutting_input(measured, {"ZZ": 1}, CuttingOptions())
    three = QuantumCircuit(3)
    three.ccx(0, 1, 2)
    with pytest.raises(ConfigError, match="at most two"):
        validate_cutting_input(three, {"ZZZ": 1}, CuttingOptions())


def test_barriers_do_not_force_artificial_connectivity():
    """Removing a no-op retains all original qubits and physical operations."""
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.barrier()
    circuit.x(1)
    clean = without_barriers(circuit)
    assert clean.qubits == circuit.qubits
    assert [instruction.operation.name for instruction in clean] == ["h", "x"]


def budget_inputs() -> tuple[dict[str, Any], list[Any], dict[str, int]]:
    """Small algebraic QPD data tests allocation math independently of the provider."""
    measurement = SimpleNamespace(name="qpd_measure")
    identity = SimpleNamespace(name="id")
    basis = SimpleNamespace(
        kappa=3,
        probabilities=[0.5, 0.5],
        maps=[([measurement], [identity]), ([identity], [measurement])],
    )
    circuit = QuantumCircuit(1)
    circuit.x(0)
    return {"p0": circuit}, [basis], {"p0": 1}


def test_prospective_cutting_budget_counts_cartesian_samples_and_shots():
    """Exact and sampled QPD paths impose limits before allocating experiment circuits."""
    result = generation_budget(*budget_inputs(), CuttingOptions(num_samples="exact", shots=10))
    assert result["sampling_overhead"] == pytest.approx(9)
    assert result["qpd_cartesian_terms"] == 2
    assert result["subexperiment_bound"] == 2
    assert result["shot_record_bytes_bound"] == 40
    sampled = generation_budget(*budget_inputs(), CuttingOptions(num_samples=1, shots=10))
    assert sampled["subexperiment_bound"] == 1


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"max_cuts": 0}, "max_cuts"),
        ({"max_sampling_overhead": 8}, "sampling overhead"),
        ({"num_samples": "exact", "max_exact_terms": 1}, "exact QPD"),
        ({"num_samples": "exact", "max_subexperiments": 1}, "max_subexperiments"),
        ({"max_total_shots": 1}, "total_shots"),
        ({"max_generated_instructions": 1}, "max_generated_instructions"),
        ({"max_working_bytes": 1}, "allocation"),
    ],
)
def test_prospective_cutting_budget_rejections(changes, match):
    """No enumerated expansion or quantum execution occurs after a predicted budget failure."""
    with pytest.raises(ConfigError, match=match):
        generation_budget(*budget_inputs(), CuttingOptions(**changes))


def physical_plan(**changes):
    """One real reset-containing publication with both required measurement registers."""
    circuit = QuantumCircuit(1)
    observable = ClassicalRegister(1, "observable_measurements")
    qpd = ClassicalRegister(1, "qpd_measurements")
    circuit.add_register(observable, qpd)
    circuit.x(0)
    circuit.measure(0, qpd[0])
    circuit.reset(0)
    circuit.h(0)
    circuit.measure(0, observable[0])
    return CuttingPlan(
        {"p0": [circuit]},
        {"p0": ["Z"]},
        ((1.0, "EXACT"),),
        {"Z": 1.0},
        CuttingOptions(shots=100, **changes),
        {},
    )


def test_real_sampler_preserves_qpd_mid_measurements_and_closes(monkeypatch):
    """Raw physical QPD outcomes survive sampling, reset and host-side publication grouping."""
    build = SAMPLERS.build
    closed = []

    def resource(*args, **kwargs):
        """Attach lifecycle observation to the real released sampler resource."""
        result = build(*args, **kwargs)
        result.close = lambda: closed.append(True)
        return result

    monkeypatch.setattr(SAMPLERS, "build", resource)
    selected = ComponentSelection(name="basic_backend", options={"seed_simulator": 7})
    results, arrays, records = collect_cutting_samples(physical_plan(), selected)
    assert len(results["p0"]) == 1
    assert np.all(arrays["p0_e0_qpd_measurements"] == 1)
    assert 20 < np.sum(arrays["p0_e0_observable_measurements"]) < 80
    assert records[0]["registers"]["qpd_measurements"]["num_bits"] == 1
    assert closed == [True]


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"max_total_shots": 1}, "total-shot"),
        ({"max_working_bytes": 1}, "physical records"),
        ({"max_working_bytes": 425}, "simulator allocation"),
    ],
)
def test_sampler_budget_failures_precede_resource_creation(monkeypatch, changes, match):
    """A prospective failure does not even create a provider resource."""

    def forbidden(*args, **kwargs):
        """Record an unexpected side effect as an assertion failure."""
        raise AssertionError("sampler must not be built")

    monkeypatch.setattr(SAMPLERS, "build", forbidden)
    with pytest.raises(ConfigError, match=match):
        collect_cutting_samples(physical_plan(**changes), ComponentSelection.named("basic_backend"))


def test_planning_transport_is_explicit_qpy_json_and_does_not_include_sampler(monkeypatch):
    """A fresh fixed module receives circuits and scientific planning options only."""

    def child(command, **kwargs):
        """Emulate the file protocol to isolate its serialization and lifecycle contract."""
        assert command[1:3] == ["-m", "chemrefine.engines.qiskit.cutting_worker"]
        directory = Path(command[-1])
        request = json.loads((directory / "request.json").read_text())
        assert set(request) == {"options", "observable"}
        with (directory / "input.qpy").open("rb") as stream:
            incoming = qpy.load(stream)
        with (directory / "experiments.qpy").open("wb") as stream:
            qpy.dump(incoming, stream)
        (directory / "plan.json").write_text(
            json.dumps(
                {
                    "circuit_counts": {"p0": 1},
                    "subobservables": {"p0": ["Z"]},
                    "coefficients": [[1, "EXACT"]],
                    "metadata": {},
                }
            )
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", child)
    plan = plan_cutting(QuantumCircuit(1), {"Z": 1})
    assert len(plan.circuits["p0"]) == 1
    assert plan.coefficients == ((1.0, "EXACT"),)
    assert plan.qpy_data.startswith(b"QISKIT")


def test_planning_timeout_and_failure_are_config_errors(monkeypatch):
    """Terminated child execution cannot publish a partial plan."""

    def timed_out(command, **kwargs):
        """Inject a timeout at the process boundary."""
        assert kwargs["timeout"] == 1.25
        raise subprocess.TimeoutExpired(command, 0.01)

    monkeypatch.setattr(subprocess, "run", timed_out)
    with pytest.raises(ConfigError, match="worker_timeout"):
        plan_cutting(QuantumCircuit(1), {"Z": 1}, CuttingOptions(worker_timeout_seconds=1.25))
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1))
    with pytest.raises(ConfigError, match="planner failed"):
        plan_cutting(QuantumCircuit(1), {"Z": 1})
    with pytest.raises(ConfigError, match="input exceeds max_qpy"):
        plan_cutting(QuantumCircuit(1), {"Z": 1}, CuttingOptions(max_qpy_bytes=1))


@pytest.mark.parametrize("failure", ["width", "registers", "publication"])
def test_execution_failure_still_closes_the_sampler(monkeypatch, failure):
    """Compilation and malformed results cannot leave a provider session open."""
    closed = []
    transpiler = None
    if failure != "publication":
        wrong = QuantumCircuit(2 if failure == "width" else 1)
        transpiler = SimpleNamespace(run=lambda *args, **kwargs: wrong)
    sampler = SimpleNamespace(run=lambda *args, **kwargs: SimpleNamespace(result=lambda: []))
    resource = SamplerResource(sampler, close=lambda: closed.append(True), transpiler=transpiler)
    monkeypatch.setattr(SAMPLERS, "build", lambda *args, **kwargs: resource)
    with pytest.raises(ConfigError, match=r"qubits|registers|publication"):
        collect_cutting_samples(
            physical_plan(max_subcircuit_qubits=1), ComponentSelection.named("test")
        )
    assert closed == [True]


def test_cutting_rejects_embedded_markers_and_remaining_input_limits():
    """External cuts and every input allocation limit have one preflight authority."""
    from qiskit.circuit import Gate, Parameter

    embedded = QuantumCircuit(1)
    embedded.append(Gate("cut_wire", 1, []), [0])
    with pytest.raises(ConfigError, match="embedded QPD"):
        validate_cutting_input(embedded, {"Z": 1}, CuttingOptions())
    parameterized = QuantumCircuit(1)
    parameterized.rx(Parameter("theta"), 0)
    for circuit in (QuantumCircuit(), parameterized):
        with pytest.raises(ConfigError, match="bound circuit"):
            validate_cutting_input(circuit, {"Z": 1}, CuttingOptions())
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.x(0)
    with pytest.raises(ConfigError, match="instruction budget"):
        validate_cutting_input(circuit, {"Z": 1}, CuttingOptions(max_input_instructions=1))
    with pytest.raises(ConfigError, match="max_observables"):
        validate_cutting_input(circuit, {"Z": 1, "X": 1}, CuttingOptions(max_observables=1))
    for changed in ({"allow_gate_cuts": False}, {"allow_wire_cuts": False}):
        with pytest.raises(ValidationError, match="automatic search"):
            CuttingOptions(mode="manual", gate_cuts=(0,), **changed)


def test_partition_and_record_descriptor_budgets_precede_generation():
    """Small quantum width cannot bypass empty-partition or manifest-size budgets."""
    circuits, bases, groups = budget_inputs()
    with pytest.raises(ConfigError, match="partition width"):
        generation_budget({}, bases, {}, CuttingOptions())
    with pytest.raises(ConfigError, match="partition width"):
        generation_budget(
            {"p0": QuantumCircuit(2)}, bases, groups, CuttingOptions(max_subcircuit_qubits=1)
        )
    with pytest.raises(ConfigError, match="record descriptor"):
        generation_budget(
            circuits,
            bases,
            {"p0": 1025},
            CuttingOptions(num_samples=1, shots=1, max_subexperiments=2000),
        )


@pytest.mark.parametrize("fault", ["manifest", "qpy", "counts", "max_circuits", "missing"])
def test_parent_rejects_oversized_or_inconsistent_child_publication(monkeypatch, fault):
    """Parent-side bounds remain authoritative even if a planner publishes bad output."""
    from chemrefine.engines.qiskit import cutting

    options = CuttingOptions(max_qpy_bytes=1000, max_subexperiments=1)

    def child(command, **kwargs):
        """Publish malformed bounded transport cases without performing provider work."""
        directory = Path(command[-1])
        if fault == "missing":
            return SimpleNamespace(returncode=0)
        description = {
            "circuit_counts": {"p0": 1},
            "subobservables": {"p0": ["Z"]},
            "coefficients": [[1, "EXACT"]],
            "metadata": {},
        }
        if fault == "manifest":
            monkeypatch.setattr(cutting, "MAX_DESCRIPTOR_BYTES", 1)
        if fault == "counts":
            description["circuit_counts"] = {"p0": 0}
        count = 2 if fault == "max_circuits" else 1
        if fault == "max_circuits":
            description["circuit_counts"] = {"p0": 2}
        with (directory / "experiments.qpy").open("wb") as stream:
            qpy.dump([QuantumCircuit(1)] * count, stream)
        if fault == "qpy":
            (directory / "experiments.qpy").write_bytes(b"x" * 1001)
        (directory / "plan.json").write_text(json.dumps(description))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", child)
    with pytest.raises(ConfigError, match=r"descriptor|circuits|counts|planner failed"):
        plan_cutting(QuantumCircuit(1), {"Z": 1}, options)


@pytest.mark.parametrize("shape", ["empty", "many", "descriptor"])
def test_public_plan_execution_cannot_bypass_count_limits(monkeypatch, shape):
    """Hand-created plans receive the same shot and record-count checks as SDK plans."""
    from dataclasses import replace

    original = physical_plan()
    circuits: dict[str, Any]
    changes = {"shots": 1, "max_subexperiments": 2000}
    if shape == "empty":
        circuits = {"p0": []}
    elif shape == "many":
        circuits = {"p0": original.circuits["p0"] * 2}
        changes["max_subexperiments"] = 1
    else:
        circuits = {"p0": original.circuits["p0"] * 1025}
    plan = replace(original, circuits=circuits, options=CuttingOptions(**changes))
    with pytest.raises(ConfigError, match=r"experiment|descriptor"):
        collect_cutting_samples(plan, ComponentSelection.named("unused"))


@pytest.mark.parametrize("fault", ["missing", "width", "shots", "shape"])
def test_malformed_physical_bitarrays_are_rejected_and_resource_closed(monkeypatch, fault):
    """A V2 publication must contain two width-correct scalar shot buffers."""
    from qiskit.primitives import BitArray, DataBin, SamplerPubResult

    closed = []
    data = np.zeros((100, 1), dtype=np.uint8)
    width = 2 if fault == "width" else 1
    if fault == "shots":
        data = data[:1]
    if fault == "shape":
        data = data[None, ...]
    bits = BitArray(data, width)
    registers = (
        {} if fault == "missing" else {"observable_measurements": bits, "qpd_measurements": bits}
    )
    pub = SamplerPubResult(DataBin(**registers))
    sampler = SimpleNamespace(run=lambda *a, **kw: SimpleNamespace(result=lambda: [pub]))
    resource = SamplerResource(sampler, close=lambda: closed.append(True))
    monkeypatch.setattr(SAMPLERS, "build", lambda *a, **kw: resource)
    with pytest.raises(ConfigError, match=r"BitArray|shot count"):
        collect_cutting_samples(physical_plan(), ComponentSelection.named("test"))
    assert closed == [True]


def test_typed_aer_memory_guard_and_publication_batches(monkeypatch):
    """Dense density matrices are budgeted separately; non-dense Aer methods are delegated."""
    from dataclasses import replace

    plan = physical_plan(max_working_bytes=440)
    with pytest.raises(ConfigError, match="simulator allocation"):
        collect_cutting_samples(
            plan, ComponentSelection(name="aer", options={"method": "density_matrix"})
        )
    plan = replace(
        plan,
        circuits={"p0": plan.circuits["p0"] * 2},
        options=CuttingOptions(shots=100, publications_per_job=1),
    )
    build = SAMPLERS.build
    seen = []

    def resource(selection, **kwargs):
        """Use the actual basic provider behind a typed non-dense planning selection."""
        result = build(ComponentSelection.named("basic_backend"), **kwargs)
        result.transpiler = SimpleNamespace(run=lambda circuit, **kw: circuit)
        seen.append(selection.name)
        return result

    monkeypatch.setattr(SAMPLERS, "build", resource)
    results, arrays, records = collect_cutting_samples(
        plan, ComponentSelection(name="aer", options={"method": "matrix_product_state"})
    )
    assert seen == ["aer"] and len(results["p0"]) == 2
    assert len(arrays) == 4 and [record["experiment"] for record in records] == [0, 1]

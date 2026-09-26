"""Real Qiskit addon cutting contracts, including subprocess planning and signed samples."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qiskit_addon_cutting")

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from chemrefine.engines.qiskit.cutting import (
    CuttingOptions,
    WireCut,
    execute_cutting,
    plan_cutting,
    reconstruct_cutting_records,
)
from chemrefine.engines.qiskit.options import ComponentSelection
from chemrefine.errors import ConfigError


@pytest.mark.parametrize("mode", ["manual", "partition"])
def test_bell_gate_cuts_reconstruct_three_noncommuting_terms(mode):
    """Independent local shots plus signed QPD coefficients reproduce entangled correlators."""
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    arguments = {"gate_cuts": (1,)} if mode == "manual" else {"partition_labels": ("A", "B")}
    controls = CuttingOptions(
        mode=mode, num_samples="exact", max_subcircuit_qubits=1, shots=20000, **arguments
    )
    plan = plan_cutting(circuit, {"XX": 1, "ZZ": 0.5, "YY": 0.25}, controls)
    result = execute_cutting(
        plan,
        ComponentSelection(
            name="basic_backend",
            options={
                "seed_simulator": 47,
                "seed_transpiler": 47,
            },
        ),
    )
    np.testing.assert_allclose(result.term_expectations, [1, 1, -1], atol=0.08)
    assert result.expectation == pytest.approx(1.25, abs=0.1)
    assert plan.metadata["sampling_overhead"] == pytest.approx(9)
    assert all(width == 1 for width in plan.metadata["partition_widths"].values())
    assert np.any(result.arrays["qpd_coefficients"] < 0)
    assert all(record["shots"] == 20000 for record in result.metadata["records"])
    assert result.metadata["clipped"] is False
    assert result.metadata["observable_labels"] == ["XX", "ZZ", "YY"]
    assert (
        np.dot(result.arrays["observable_coefficients"], result.term_expectations)
        == result.expectation
    )
    np.testing.assert_array_equal(
        reconstruct_cutting_records(result.arrays, result.metadata), result.term_expectations
    )
    assert len(result.arrays["logical_experiments_qpy"]) == plan.metadata["generated_qpy_bytes"]


def test_wire_cut_maps_observables_to_final_segment():
    """The original wire's final segment carries the observable after an intervening cut."""
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.ry(0.3, 0)
    circuit.rz(-0.2, 0)
    labels = {"Z": 0.7, "X": 0.2}
    reference = Statevector(circuit).expectation_value(
        SparsePauliOp.from_list(list(labels.items()))
    )
    plan = plan_cutting(
        circuit,
        labels,
        CuttingOptions(
            mode="manual",
            wire_cuts=(WireCut(qubit=0, before_instruction=1),),
            max_subcircuit_qubits=1,
            num_samples="exact",
            shots=20000,
        ),
    )
    result = execute_cutting(
        plan, ComponentSelection(name="basic_backend", options={"seed_simulator": 19})
    )
    assert plan.metadata["expanded_qubits"] == 2
    assert plan.metadata["cut_count"] == 1
    assert result.expectation == pytest.approx(reference.real, abs=0.08)


def test_automatic_cut_search_and_postcheck():
    """Search budgets are advisory; actual resulting widths and overhead remain mandatory."""
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.barrier()
    circuit.cx(1, 2)
    plan = plan_cutting(
        circuit,
        {"ZZI": 1},
        CuttingOptions(
            max_subcircuit_qubits=2,
            allow_wire_cuts=False,
            num_samples="exact",
            shots=100,
        ),
    )
    assert max(plan.metadata["partition_widths"].values()) <= 2
    assert isinstance(plan.metadata["search"]["minimum_reached"], bool)
    with pytest.raises(ConfigError, match="planner failed"):
        plan_cutting(
            circuit,
            {"ZZI": 1},
            CuttingOptions(
                max_subcircuit_qubits=1,
                max_sampling_overhead=1,
            ),
        )


def test_sampled_qpd_is_reproducible_and_parent_rng_is_untouched():
    """The addon's global RNG is confined to a child interpreter, including repeated wire cuts."""
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.h(1)
    circuit.rz(0.4, 0)
    circuit.ry(0.2, 1)
    circuit.rx(0.1, 0)
    controls = CuttingOptions(
        mode="manual",
        num_samples=32,
        seed=41,
        shots=10,
        max_subcircuit_qubits=1,
        wire_cuts=(
            WireCut(qubit=0, before_instruction=2),
            WireCut(qubit=1, before_instruction=3),
            WireCut(qubit=0, before_instruction=4),
        ),
    )
    np.random.seed(37)
    expected = np.random.RandomState(37).random_sample(2)
    assert np.random.random() == expected[0]
    first = plan_cutting(circuit, {"ZZ": 1}, controls)
    assert np.random.random() == expected[1]
    second = plan_cutting(circuit, {"ZZ": 1}, controls)
    assert first.coefficients == second.coefficients
    assert first.metadata["expanded_qubits"] == 5
    assert first.metadata["cut_count"] == 3
    assert all(width == 1 for width in first.metadata["partition_widths"].values())


def test_idle_wires_retain_their_zero_state_observable_factor():
    """An unacted-on X factor must contribute zero instead of disappearing from the product."""
    circuit = QuantumCircuit(2)
    circuit.x(0)
    plan = plan_cutting(
        circuit, {"XI": 1, "ZI": 1}, CuttingOptions(num_samples="exact", shots=10000)
    )
    result = execute_cutting(
        plan, ComponentSelection(name="basic_backend", options={"seed_simulator": 13})
    )
    np.testing.assert_allclose(result.term_expectations, [0, 1], atol=0.04)


def test_artifact_cutting_publishes_replayable_ordered_observations(tmp_path):
    """The artifact workflow preserves signed records and observable order in a saved bundle."""
    from qiskit import qpy

    from chemrefine.engines.qiskit.bundles import read_bundle
    from chemrefine.engines.qiskit.experiment import run_experiment

    circuit = QuantumCircuit(1)
    circuit.x(0)
    source = tmp_path / "input.qpy"
    with source.open("wb") as stream:
        qpy.dump(circuit, stream)
    path = tmp_path / "cutting.json"
    run_experiment(
        {
            "experiment": {
                "name": "circuit_cutting",
                "options": {
                    "circuit_path": str(source),
                    "observable": {"Z": 1, "I": 0.5},
                    "cutting": {"num_samples": "exact", "shots": 100},
                    "sampler": "basic_backend",
                },
            }
        },
        output_path=path,
    )
    bundle = read_bundle(path)
    assert bundle.description.kind == "circuit_cutting"
    assert bundle.metadata["observable_labels"] == ["Z", "I"]
    assert bundle.metadata["expectation"] == pytest.approx(-0.5)
    np.testing.assert_allclose(
        reconstruct_cutting_records(bundle.arrays, bundle.metadata), [-1, 1], atol=1e-14
    )


def test_cutting_output_budget_rejects_before_sampler_submission(monkeypatch):
    """Planning may proceed but an oversized durable result never submits quantum shots."""
    from types import SimpleNamespace

    from chemrefine.engines.qiskit import experiment_cutting as adapter

    options = adapter.CuttingExperimentOptions(circuit_path="not-read.qpy", observable={"Z": 1})
    monkeypatch.setattr(adapter, "read_circuit", lambda *args, **kwargs: QuantumCircuit(1))
    monkeypatch.setattr(
        adapter,
        "plan_cutting",
        lambda *args, **kwargs: SimpleNamespace(
            metadata={"shot_record_bytes_bound": 100, "generated_qpy_bytes": 10},
            coefficients=[(1, "EXACT")],
            observable={"Z": 1},
        ),
    )
    with pytest.raises(ConfigError, match="max_output_bytes"):
        adapter.cutting_experiment(options=options, max_output_bytes=1, device="cpu", cores=1)


@pytest.fixture
def reconstruction_case():
    """An actual one-term measured result exercises durable reconstruction validation."""
    from qiskit import ClassicalRegister

    from chemrefine.engines.qiskit.cutting import CuttingPlan

    circuit = QuantumCircuit(1)
    observable = ClassicalRegister(1, "observable_measurements")
    qpd = ClassicalRegister(0, "qpd_measurements")
    circuit.add_register(observable, qpd)
    circuit.x(0)
    circuit.measure(0, observable[0])
    plan = CuttingPlan(
        {"p0": [circuit]}, {"p0": ["Z"]}, ((1, "EXACT"),), {"Z": 1}, CuttingOptions(shots=100), {}
    )
    return plan, execute_cutting(plan)


@pytest.mark.parametrize(
    "fault", ["index", "register", "dtype", "shape", "shots", "weights", "nonfinite", "kind"]
)
def test_durable_reconstruction_rejects_malformed_records(reconstruction_case, fault):
    """Stored physical bytes and signed weights are checked before SDK reconstruction."""
    from copy import deepcopy

    _plan, result = reconstruction_case
    arrays = dict(result.arrays)
    metadata = deepcopy(result.metadata)
    record = metadata["records"][0]
    key = record["registers"]["observable_measurements"]["array"]
    if fault == "index":
        record["experiment"] = 1
    elif fault == "register":
        record["registers"].pop("qpd_measurements")
    elif fault == "dtype":
        arrays[key] = arrays[key].astype(float)
    elif fault == "shape":
        arrays[key] = arrays[key][None, ...]
    elif fault == "shots":
        record["shots"] = 1
    elif fault == "weights":
        arrays["qpd_coefficients"] = np.array([1, 2])
    elif fault == "nonfinite":
        arrays["qpd_coefficients"] = np.array([np.nan])
    else:
        metadata["weight_types"] = ["invalid"]
    with pytest.raises(ConfigError, match="invalid physical cutting records"):
        reconstruct_cutting_records(arrays, metadata)


@pytest.mark.parametrize("values", [[np.nan], [0, 1]])
def test_invalid_sdk_reconstruction_is_rejected(monkeypatch, reconstruction_case, values):
    """Neither live nor replayed signed output can silently publish nonfinite or missing terms."""
    import qiskit_addon_cutting

    plan, result = reconstruction_case
    monkeypatch.setattr(
        qiskit_addon_cutting, "reconstruct_expectation_values", lambda *a, **kw: values
    )
    with pytest.raises(ConfigError, match="invalid observable"):
        execute_cutting(plan)
    with pytest.raises(ConfigError, match="invalid reconstructed"):
        reconstruct_cutting_records(result.arrays, result.metadata)


def write_planner_request(directory, options, circuits):
    """Publish the same trusted local input transport used by the public parent planner."""
    import json

    from qiskit import qpy

    (directory / "request.json").write_text(
        json.dumps({"options": options.model_dump(mode="json"), "observable": {"Z": 1}})
    )
    with (directory / "input.qpy").open("wb") as stream:
        qpy.dump(circuits, stream)


@pytest.mark.parametrize(
    "fault", ["count", "generated_count", "generated_instructions", "qpy", "descriptor"]
)
def test_worker_validates_generated_workload_before_publication(tmp_path, monkeypatch, fault):
    """A faulty or oversized generated workload cannot publish a success descriptor."""
    import qiskit_addon_cutting

    from chemrefine.engines.qiskit import cutting_worker as worker

    options = CuttingOptions(num_samples="exact", shots=1)
    if fault == "qpy":
        options = CuttingOptions(num_samples="exact", shots=1, max_qpy_bytes=1)
    write_planner_request(tmp_path, options, [QuantumCircuit(1)] * (2 if fault == "count" else 1))
    if fault in {"generated_count", "generated_instructions"}:
        generate = qiskit_addon_cutting.generate_cutting_experiments

        def faulty(*args, **kwargs):
            """Inject an unexpected provider expansion after the prospective checks."""
            circuits, coefficients = generate(*args, **kwargs)
            if fault == "generated_count":
                circuits["p0"] *= 2
            else:
                for _ in range(3):
                    circuits["p0"][0].id(0)
            return circuits, coefficients

        monkeypatch.setattr(qiskit_addon_cutting, "generate_cutting_experiments", faulty)
        if fault == "generated_instructions":
            options = CuttingOptions(num_samples="exact", shots=1, max_generated_instructions=1)
            write_planner_request(tmp_path, options, [QuantumCircuit(1)])
            budget = worker.generation_budget

            def allow_prospective(circuits, bases, groups, controls):
                """Allow a provider to exceed the final cap only after generation."""
                return budget(circuits, bases, groups, CuttingOptions(num_samples="exact", shots=1))

            monkeypatch.setattr(worker, "generation_budget", allow_prospective)
    if fault == "descriptor":
        monkeypatch.setattr(worker, "MAX_DESCRIPTOR_BYTES", 1100)
        # This valid width-one publication has a descriptor larger than 1.1 KiB.
        options = CuttingOptions(
            num_samples="exact", shots=1, mode="partition", partition_labels=("A" * 500,)
        )
        write_planner_request(tmp_path, options, [QuantumCircuit(1)])
    with pytest.raises(ConfigError, match=r"exactly one|prospective bounds|max_qpy|descriptor"):
        worker.generate_plan(tmp_path)
    assert not (tmp_path / "plan.json").exists()


def test_worker_command_line_requires_one_owned_directory(monkeypatch):
    """The private entry point has no ambiguous defaults or import-time execution."""
    from chemrefine.engines.qiskit import cutting_worker

    monkeypatch.setattr("sys.argv", ["cutting_worker"])
    with pytest.raises(SystemExit, match="DIRECTORY"):
        cutting_worker.main()

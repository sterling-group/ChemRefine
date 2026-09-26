"""Offline numerical tests against the actual pinned Runtime SDK and fake backend targets."""

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest

from chemrefine.engines.qiskit.journal import read_journal
from chemrefine.engines.qiskit.runtime import build_runtime_resource
from chemrefine.engines.qiskit.runtime_options import RuntimeEstimatorOptions, RuntimeSamplerOptions

pytest.importorskip("qiskit_ibm_runtime")
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import BaseEstimatorV2, BaseSamplerV2
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.state_fidelities import ComputeUncompute

pytestmark = [
    pytest.mark.filterwarnings("ignore:.*:DeprecationWarning:qiskit.*"),
    pytest.mark.filterwarnings("ignore:.*:DeprecationWarning:qiskit_ibm_runtime.*"),
    pytest.mark.filterwarnings(
        "ignore:The (Estimator|Sampler)V2 class is deprecated:DeprecationWarning"
    ),
]


@pytest.mark.parametrize("implementation", ["executor", "legacy_v2"])
def test_real_runtime_estimator_and_sampler_use_fake_backend_offline(tmp_path, implementation):
    assert version("qiskit-ibm-runtime").startswith("0.50.")
    common = {
        "fake_backend": "FakeManilaV2",
        "implementation": implementation,
        "journal_dir": str(tmp_path),
        "seed_simulator": 4,
        "seed_transpiler": 7,
        "initial_layout": [3, 4],
    }
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    observable = SparsePauliOp("ZZ")
    estimator = build_runtime_resource(
        RuntimeEstimatorOptions(**common, default_shots=4096), kind="estimator", device="cpu"
    )
    with estimator as primitive:
        assert isinstance(primitive, BaseEstimatorV2)
        physical = estimator.transpiler.run(circuit)
        value = (
            primitive.run([(physical, observable.apply_layout(physical.layout))])
            .result()[0]
            .data.evs
        )
        assert float(value) > 0.8
    sampler = build_runtime_resource(
        RuntimeSamplerOptions(**common, default_shots=4096), kind="sampler", device="cpu"
    )
    with sampler as primitive:
        assert isinstance(primitive, BaseSamplerV2)
        measured = circuit.copy()
        measured.measure_all()
        physical = sampler.transpiler.run(measured)
        counts = primitive.run([physical]).result()[0].data.meas.get_counts()
        assert sum(counts.values()) == 4096
        assert (counts.get("00", 0) + counts.get("11", 0)) / 4096 > 0.85
    records = read_journal(tmp_path)
    assert len(records) == 2 and all(record.state == "completed" for record in records)
    assert {record.summary.kind for record in records} == {"estimator", "sampler"}


def test_executor_trex_and_zne_return_numeric_estimate_with_explicit_local_sampling(tmp_path):
    options = RuntimeEstimatorOptions(
        fake_backend="FakeManilaV2",
        journal_dir=str(tmp_path),
        default_shots=4096,
        seed_simulator=5,
        seed_transpiler=5,
        trex=True,
        zne={"enable": True, "noise_factors": [1, 3, 5], "extrapolator": ["linear"]},
        twirling={
            "enable_gates": True,
            "enable_measure": True,
            "num_randomizations": 8,
            "shots_per_randomization": 512,
        },
    )
    resource = build_runtime_resource(options, kind="estimator", device="cpu")
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    with resource as primitive:
        physical = resource.transpiler.run(circuit)
        result = primitive.run(
            [(physical, SparsePauliOp("ZZ").apply_layout(physical.layout))]
        ).result()[0]
    assert np.isfinite(result.data.evs).all()
    assert float(result.data.evs) > 0.8
    assert np.isfinite(result.data.stds).all()
    assert np.asarray(result.data.evs_noise_factors).shape[-1] == 3
    assert np.asarray(result.data.evs_extrapolated).shape[-2:] == (1, 1)
    assert read_journal(tmp_path)[0].state == "completed"


def test_runtime_sampler_fulfils_compute_uncompute_metric_contract(tmp_path):
    resource = build_runtime_resource(
        RuntimeSamplerOptions(
            fake_backend="FakeManilaV2", journal_dir=str(tmp_path), seed_simulator=8
        ),
        kind="sampler",
        device="cpu",
    )
    circuit = QuantumCircuit(1)
    circuit.ry(Parameter("theta"), 0)
    with resource as sampler:
        fidelity = ComputeUncompute(sampler, shots=2048, transpiler=resource.transpiler)
        result = fidelity.run([circuit], [circuit], [[0.2]], [[0.2]]).result()
    assert result.fidelities[0] > 0.85
    assert read_journal(tmp_path)[0].state == "completed"


def test_executor_calibration_layers_roundtrip_through_public_schema_without_network(tmp_path):
    from ibm_quantum_schemas.common import QpyModelV13ToV17

    resource = build_runtime_resource(
        RuntimeEstimatorOptions(fake_backend="FakeManilaV2", journal_dir=str(tmp_path)),
        kind="estimator",
        device="cpu",
    )
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    with resource as primitive:
        physical = resource.transpiler.run(circuit)
        primitive.primitive.options.resilience.pec_mitigation = True
        layers = primitive.primitive.find_unique_layers(
            [(physical, SparsePauliOp("ZZ").apply_layout(physical.layout))], types="gates"
        )
        assert layers
        for layer in layers:
            boxed = QuantumCircuit(list(layer.qubits), list(layer.clbits), name="noise_layer")
            boxed.append(layer)
            serialized = QpyModelV13ToV17.from_quantum_circuit(boxed, qpy_version=17)
            assert serialized.to_quantum_circuit() == boxed
    # Layer discovery and serialization do not submit calibration or primitive jobs.
    assert read_journal(tmp_path) == ()


def test_offline_molecular_vqe_uses_registered_runtime_resource_and_layout(tmp_path):
    from chemrefine.engines.qiskit.data import ElectronicStructureData
    from chemrefine.engines.qiskit.problem import prepare_problem
    from chemrefine.engines.qiskit.workflow import run_problem

    fixture = Path(__file__).parent / "data/engines/qiskit/h2_integrals.json"
    prepared = prepare_problem(ElectronicStructureData(**json.loads(fixture.read_text())))
    result = run_problem(
        prepared,
        options={
            "algorithm": "vqe",
            "mapper": {"name": "parity", "options": {"two_qubit_reduction": True}},
            "estimator": {
                "name": "ibm_runtime",
                "options": {
                    "fake_backend": "FakeManilaV2",
                    "journal_dir": str(tmp_path),
                    "initial_layout": [3, 4],
                    "seed_transpiler": 4,
                    "seed_simulator": 4,
                    "default_shots": 4096,
                    "max_jobs": 10,
                },
            },
            "optimizer": {
                "name": "spsa",
                "options": {"maxiter": 2, "learning_rate": 0.1, "perturbation": 0.1, "seed": 4},
            },
        },
    )
    assert -1.3 < result.energy_hartree < -0.5
    assert read_journal(tmp_path)
    assert all(record.state == "completed" for record in read_journal(tmp_path))


@pytest.mark.parametrize("protocol", ["pec", "pea"])
def test_real_executor_known_noise_correction_preserves_signed_normalization(
    tmp_path, monkeypatch, protocol
):
    """Substitute known calibration only; execute real samplex, simulator and postprocessor."""
    from types import SimpleNamespace

    from qiskit.primitives.primitive_job import PrimitiveJob
    from qiskit.quantum_info import PauliLindbladMap
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from qiskit_ibm_runtime import noise_learner_v3
    from qiskit_ibm_runtime.executor_estimator import Estimator
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2

    from chemrefine.engines.qiskit.journal import RequestJournal
    from chemrefine.engines.qiskit.runtime import RuntimePrimitive, sdk_options

    # Keep the real fake-device target and replace its snapshot noise with a known
    # layer channel exp(lambda * (X rho X - rho)). Bell ZZ decays as exp(-2 lambda).
    backend = AerSimulator.from_backend(FakeManilaV2(), noise_model=NoiseModel())
    backend.name = "synthetic_pauli_noise"
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    physical = generate_preset_pass_manager(
        backend=backend, initial_layout=[3, 4], optimization_level=1, seed_transpiler=7
    ).run(circuit)
    pub = (physical, SparsePauliOp("ZZ").apply_layout(physical.layout))
    settings = (
        {"pec": {"enable": True, "noise_gain": 0}}
        if protocol == "pec"
        else {
            "zne": {
                "enable": True,
                "amplifier": "pea",
                "noise_factors": [1, 2, 3],
                "extrapolator": ["exponential"],
            }
        }
    )
    randomizations = 1024 if protocol == "pec" else 4096
    nominal_shots = randomizations * 16
    options = RuntimeEstimatorOptions(
        backend_name="synthetic_calibration",
        trex=False,
        noise_learning={"enabled": True},
        default_shots=nominal_shots,
        twirling={
            "enable_gates": True,
            "enable_measure": True,
            "num_randomizations": randomizations,
            "shots_per_randomization": 16,
        },
        **settings,
    )
    # This test-only SDK construction bypasses the production prohibition on local
    # NoiseLearnerV3. Its sole substitute supplies the analytically known channel.
    estimator = Estimator(
        mode=backend,
        options=sdk_options(options) | {"simulator": {"seed_simulator": 12, "warn_absent": False}},
    )
    layers = estimator.find_unique_layers([pub], types="gates")
    assert len(layers) == 1
    noise = PauliLindbladMap.from_sparse_list([("X", [0], 0.2)], num_qubits=2)
    estimator.options.simulator.layer_noise_model = [(layers[0], noise)]

    class KnownCalibration:
        def __init__(self, **kwargs):
            assert kwargs["mode"] is backend

        def run(self, actual_layers):
            assert actual_layers == layers
            assert read_journal(tmp_path)[0].state == "intent"
            return SimpleNamespace(
                job_id=lambda: f"known_{protocol}_calibration",
                result=lambda: SimpleNamespace(to_pauli_lindblad_maps=lambda: [noise]),
            )

    monkeypatch.setattr(noise_learner_v3, "NoiseLearnerV3", KnownCalibration)
    adapter = RuntimePrimitive(
        estimator, options, RequestJournal(tmp_path), kind="estimator", backend=backend
    )
    job = adapter.run([pub])
    result = job.result()[0]
    # The public PrimitiveJob base returns the same completed local execution's
    # physical data before LocalRuntimeJob applies its estimator result decoder.
    raw = PrimitiveJob.result(job.jobs[0].job)
    bits = raw[0]["_meas"]
    assert bits.dtype == np.bool_
    decoded = bits ^ raw[0]["measurement_flips._meas"]
    parity = 1 - 2 * (decoded[..., 3] ^ decoded[..., 4]).astype(np.int8)
    labels = bits[..., 3:].reshape(-1, 2)
    _, counts = np.unique(labels, axis=0, return_counts=True)
    assert np.issubdtype(counts.dtype, np.integer) and np.all(counts > 0)
    assert counts.sum() == parity.size
    assert abs(float(result.data.evs) - 1) < 0.08
    assert abs(float(result.data.evs) - 1) < 1 - np.exp(-0.4)

    if protocol == "pec":
        signs = 1 - 2 * raw[0]["pauli_signs"].astype(np.int8)
        assert set(np.unique(signs)) == {-1, 1}
        gamma = raw.passthrough_data["qiskit_mitigation"][0]["pec_gamma"]
        assert gamma == pytest.approx(np.exp(0.4))
        # Normalize signed contributions by the number of physical samples, then
        # multiply by gamma. Positive count normalization cannot replace this.
        manual = gamma * np.mean(parity * signs)
        assert float(result.data.evs) == pytest.approx(manual, abs=1e-12)
        assert abs(float(result.data.evs) - parity.mean()) > 0.3
        assert counts.sum() == int(np.ceil(1024 * gamma**2)) * 16
    else:
        assert not np.any(raw[0]["pauli_signs"])
        physical_means = parity.mean(axis=(1, 2, 3))
        np.testing.assert_allclose(result.data.evs_noise_factors, physical_means, atol=1e-12)
        np.testing.assert_allclose(physical_means, np.exp(-0.4 * np.array([1, 2, 3])), atol=0.05)
        assert counts.sum() == 3 * nominal_shots
    records = read_journal(tmp_path)
    assert len(records) == 2 and all(record.state == "completed" for record in records)
    energy_record = next(record for record in records if record.summary.kind == "estimator")
    assert energy_record.summary.nominal_shots == nominal_shots < counts.sum()

"""Provider-independent Runtime contracts tested with actual Qiskit PUB containers."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from chemrefine.engines.qiskit.journal import RequestJournal, read_journal
from chemrefine.engines.qiskit.runtime import RuntimePrimitive, fingerprint_pubs, sdk_options
from chemrefine.engines.qiskit.runtime_options import (
    NoiseLearningOptions,
    PECOptions,
    RuntimeEstimatorOptions,
    RuntimeSamplerOptions,
    TwirlingOptions,
    ZNEOptions,
)
from chemrefine.errors import ConfigError

pytest.importorskip("qiskit")
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.primitives.containers import EstimatorPub
from qiskit.providers.basic_provider import BasicSimulator


def _adapter(tmp_path, *, kind="estimator", settings=None):
    """Use real ideal primitives behind the journal adapter, without a Runtime SDK import."""
    options: RuntimeEstimatorOptions | RuntimeSamplerOptions
    if kind == "estimator":
        options = RuntimeEstimatorOptions(fake_backend="FakeManilaV2", **(settings or {}))
        primitive = StatevectorEstimator(seed=2)
    else:
        options = RuntimeSamplerOptions(fake_backend="FakeManilaV2", **(settings or {}))
        primitive = StatevectorSampler(seed=2)
    return RuntimePrimitive(
        primitive, options, RequestJournal(tmp_path), kind=kind, backend=BasicSimulator()
    )


def test_request_fingerprint_reproducible_but_detects_physical_input_changes():
    def make(angle):
        circuit = QuantumCircuit(2)
        circuit.ry(Parameter("angle"), 0)
        return EstimatorPub.coerce((circuit, {"IZ": 1}, [angle]), 0.01)

    first, second = make(0.2), make(0.2)
    kwargs = {"options": {"backend": "backend_a"}, "maximum": 100000}
    assert fingerprint_pubs([first], **kwargs) == fingerprint_pubs([second], **kwargs)
    assert fingerprint_pubs([first], **kwargs) != fingerprint_pubs([make(0.3)], **kwargs)
    assert fingerprint_pubs([first], **kwargs) != fingerprint_pubs(
        [first], options={"backend": "backend_b"}, maximum=100000
    )
    with pytest.raises(ConfigError, match="max_request_bytes"):
        fingerprint_pubs([first], options={}, maximum=10)


def test_runtime_estimator_splits_precision_groups_and_preserves_pub_order(tmp_path):
    adapter = _adapter(tmp_path, settings={"max_pubs_per_job": 2})
    circuit = QuantumCircuit(1)
    circuit.ry(0.4, 0)
    pubs = [(circuit, {"Z": 1}, None, precision) for precision in (0.01, 0.02, 0.02, 0.02)]
    job = adapter.run(pubs)
    assert len(job.job_ids) == 3
    result = job.result()
    assert len(result) == 4
    assert result.metadata["provider_job_ids"] == job.job_ids
    assert all(record.state == "completed" for record in read_journal(tmp_path))
    assert sum(record.summary.nominal_shots for record in read_journal(tmp_path)) == 17500
    # The controlled artificial precision produces approximately the ideal observable.
    np.testing.assert_allclose([item.data.evs for item in result], np.cos(0.4), atol=0.07)


def test_runtime_sampler_counts_shots_and_binding_shapes(tmp_path):
    adapter = _adapter(tmp_path, kind="sampler")
    circuit = QuantumCircuit(1)
    circuit.x(0)
    circuit.measure_all()
    job = adapter.run([(circuit, None, 12), (circuit, None, 20)])
    assert len(job.job_ids) == 2
    assert [item.data.meas.get_counts() for item in job.result()] == [{"1": 12}, {"1": 20}]
    assert adapter.nominal_shots == 32


def test_explicit_retrieval_uses_matching_journal_and_never_submits(tmp_path):
    adapter = _adapter(tmp_path)
    circuit = QuantumCircuit(1)
    submitted = adapter.run([(circuit, {"Z": 1}, None, 0.1)])
    provider_job = submitted.jobs[0].job
    identifier = submitted.job_ids[0]
    retrieved = []

    def retrieve(job_id):
        retrieved.append(job_id)
        return provider_job

    settings = adapter.options.model_copy(
        update={"fake_backend": None, "backend_name": "backend", "retrieve_job_ids": (identifier,)}
    )
    replay = RuntimePrimitive(
        None,
        settings,
        adapter.journal,
        kind="estimator",
        backend=adapter.backend,
        service=SimpleNamespace(job=retrieve),
    )
    # Backend target and effective execution options must match the original request.
    result = replay.run([(circuit, {"Z": 1}, None, 0.1)]).result()
    assert len(result) == 1 and retrieved == [identifier]
    assert len(read_journal(tmp_path)) == 1
    with pytest.raises(ConfigError, match="exhausted"):
        replay.run([(circuit, {"Z": 1}, None, 0.1)])


def test_runtime_restart_refuses_ambiguous_submission_and_allows_explicit_policy(tmp_path):
    """Identical PUB fingerprints are guarded across adapter restarts, before provider execution."""
    first = _adapter(tmp_path)
    calls = []

    def fail(_pubs):
        calls.append("lost response")
        raise TimeoutError("simulated accepted request without returned ID")

    first.primitive = SimpleNamespace(run=fail)
    circuit = QuantumCircuit(1)
    pubs = [(circuit, {"Z": 1}, None, 0.1)]
    with pytest.raises(TimeoutError):
        first.run(pubs)
    restarted = _adapter(tmp_path)
    restarted.primitive = SimpleNamespace(run=lambda _pubs: calls.append("unexpected"))
    with pytest.raises(ConfigError, match="submission_unknown"):
        restarted.run(pubs)
    assert calls == ["lost response"]
    acknowledged = _adapter(tmp_path, settings={"resubmission_policy": "allow_unresolved"})
    assert len(acknowledged.run(pubs).result()) == 1
    assert sorted(record.state for record in read_journal(tmp_path)) == [
        "completed",
        "submission_unknown",
    ]
    for model in (RuntimeEstimatorOptions, RuntimeSamplerOptions):
        with pytest.raises(ValidationError, match="resubmission_policy"):
            model(fake_backend="FakeManilaV2", resubmission_policy="retry")
    assert "resubmission_policy" not in sdk_options(acknowledged.options)


@pytest.mark.parametrize(
    "settings,pubs,run,match",
    [
        ({"max_jobs": 1, "max_pubs_per_job": 1}, 2, {}, "max_jobs"),
        ({"max_jobs": 1}, "different_precision", {}, "max_jobs"),
        ({"max_nominal_shots": 1}, 1, {}, "max_nominal_shots"),
        ({"max_parameter_sets": 1}, "two_bindings", {}, "max_parameter_sets"),
        ({}, 0, {}, "at least one"),
        ({}, 1, {"shots": 10}, "precision, not"),
        ({}, 1, {"precision": 0}, "strictly positive"),
        ({}, "zero_precision", {}, "strictly positive"),
    ],
)
def test_preflight_budget_errors_do_not_create_provider_jobs(tmp_path, settings, pubs, run, match):
    adapter = _adapter(tmp_path, settings=settings)
    circuit = QuantumCircuit(1)
    publications: list[Any]
    if pubs == "two_bindings":
        circuit.ry(Parameter("x"), 0)
        publications = [(circuit, {"Z": 1}, [[0], [0.1]])]
    elif pubs == "different_precision":
        publications = [(circuit, {"Z": 1}, None, value) for value in (0.1, 0.2)]
    elif pubs == "zero_precision":
        publications = [(circuit, {"Z": 1}, None, 0)]
    else:
        publications = [(circuit, {"Z": 1})] * pubs
    with pytest.raises(ConfigError, match=match):
        adapter.run(publications, **run)
    assert read_journal(tmp_path) == ()


def test_typed_mitigation_translation_and_sdk_specific_restrictions():
    options = RuntimeEstimatorOptions(
        backend_name="ibm_example",
        resilience_level=1,
        zne={"enable": True},
        dynamical_decoupling={"enable": True},
    )
    translated = sdk_options(options)
    assert translated["resilience"]["measure_mitigation"]
    assert translated["resilience"]["zne_mitigation"]
    assert translated["dynamical_decoupling"]["enable"]
    assert "token" not in translated
    assert (
        "group"
        not in sdk_options(options.model_copy(update={"implementation": "legacy_v2"}))["twirling"]
    )
    for changes, match in [
        ({"pec": {"enable": True}}, "noise_learning"),
        ({"pec": {"enable": True}, "resilience_level": 2}, "PEC and ZNE"),
        ({"trex": True, "twirling": {"enable_measure": False}}, "measurement twirling"),
        ({"pec": {"enable": True}, "twirling": {"enable_gates": False}}, "gate twirling"),
        ({"noise_learning": {"enabled": True}}, "PEC/PEA only"),
        ({"implementation": "legacy_v2", "twirling": {"group": "local_c1"}}, "executor"),
    ]:
        with pytest.raises(ValidationError, match=match):
            RuntimeEstimatorOptions(backend_name="ibm_example", **changes)
    calibrated = RuntimeEstimatorOptions(
        backend_name="ibm_example", pec={"enable": True}, noise_learning={"enabled": True}
    )
    assert sdk_options(calibrated)["resilience"]["pec_mitigation"]


@pytest.mark.parametrize(
    "model,kwargs",
    [
        (ZNEOptions, {"noise_factors": [1, 1]}),
        (ZNEOptions, {"noise_factors": [1, 2], "extrapolator": ["polynomial_degree_3"]}),
        (ZNEOptions, {"extrapolator": []}),
        (ZNEOptions, {"extrapolated_noise_factors": [-1]}),
        (ZNEOptions, {"noise_factors": [1, float("nan")]}),
        (TwirlingOptions, {"num_randomizations": 0}),
        (PECOptions, {"noise_gain": -1}),
        (NoiseLearningOptions, {"layer_pair_depths": [0, 0]}),
    ],
)
def test_mitigation_math_options_reject_invalid_fits_and_costs(model, kwargs):
    with pytest.raises(ValidationError):
        model(**kwargs)


@pytest.mark.parametrize(
    "model,kwargs,match",
    [
        (RuntimeEstimatorOptions, {}, "exactly one"),
        (
            RuntimeEstimatorOptions,
            {"backend_name": "ibm_a", "fake_backend": "FakeManilaV2"},
            "exactly one",
        ),
        (RuntimeEstimatorOptions, {"backend_name": "ibm_a", "journal_dir": "relative"}, "absolute"),
        (
            RuntimeEstimatorOptions,
            {"backend_name": "ibm_a", "journal_history_dirs": ["relative"]},
            "absolute",
        ),
        (RuntimeEstimatorOptions, {"backend_name": "ibm_a", "mode_id": "session"}, "mode_id"),
        (
            RuntimeEstimatorOptions,
            {"backend_name": "ibm_a", "mode": "session", "mode_id": "session"},
            "caller-owned",
        ),
        (
            RuntimeEstimatorOptions,
            {"backend_name": "ibm_a", "mode": "batch", "retrieve_job_ids": ["j"]},
            "retrieval uses job",
        ),
        (
            RuntimeEstimatorOptions,
            {"backend_name": "ibm_a", "initial_layout": [0, 0]},
            "initial_layout",
        ),
        (
            RuntimeEstimatorOptions,
            {"fake_backend": "FakeManilaV2", "account_name": "saved"},
            "remote credentials",
        ),
        (RuntimeEstimatorOptions, {"backend_name": "ibm_a", "seed_simulator": 2}, "fake backend"),
        (
            RuntimeEstimatorOptions,
            {"fake_backend": "FakeManilaV2", "implementation": "legacy_v2", "seed_simulator": 0},
            "positive seed",
        ),
        (
            RuntimeEstimatorOptions,
            {"backend_name": "ibm_a", "retrieve_job_ids": ["j", "j"]},
            "distinct",
        ),
        (
            RuntimeEstimatorOptions,
            {
                "fake_backend": "FakeManilaV2",
                "pec": {"enable": True},
                "noise_learning": {"enabled": True},
            },
            "does not support local",
        ),
        (
            RuntimeEstimatorOptions,
            {"fake_backend": "FakeManilaV2", "implementation": "legacy_v2", "trex": True},
            "ignores mitigation",
        ),
        (
            RuntimeSamplerOptions,
            {
                "fake_backend": "FakeManilaV2",
                "implementation": "legacy_v2",
                "dynamical_decoupling": {"enable": True},
            },
            "ignores DD",
        ),
    ],
)
def test_execution_options_reject_ambiguous_or_ignored_choices(model, kwargs, match):
    with pytest.raises(ValidationError, match=match):
        model(**kwargs)


def test_finite_default_shots_overrides_only_missing_precision(tmp_path):
    adapter = _adapter(tmp_path, settings={"default_shots": 100})
    circuit = QuantumCircuit(1)
    adapter.run([(circuit, {"Z": 1}), (circuit, {"Z": 1}, None, 0.05)])
    assert adapter.nominal_shots == 500
    translated = sdk_options(adapter.options)
    assert translated["default_shots"] == 100


def test_invalid_primitive_kinds_and_sampler_controls(tmp_path):
    with pytest.raises(ConfigError, match="typed options"):
        RuntimePrimitive(
            None,
            RuntimeSamplerOptions(fake_backend="FakeManilaV2"),
            RequestJournal(tmp_path),
            kind="estimator",
            backend=BasicSimulator(),
        )
    adapter = _adapter(tmp_path, kind="sampler")
    circuit = QuantumCircuit(1)
    circuit.measure_all()
    for kwargs, match in (
        ({"precision": 0.1}, "sampler accepts shots"),
        ({"shots": 0}, "positive integer"),
    ):
        with pytest.raises(ConfigError, match=match):
            adapter.run([circuit], **kwargs)
    adapter.submissions = adapter.options.max_jobs
    with pytest.raises(ConfigError, match="max_jobs"):
        adapter._submit("unused", None, None)


def test_target_fingerprint_captures_instruction_connectivity(tmp_path):
    from qiskit.circuit.library import XGate
    from qiskit.transpiler import Target

    from chemrefine.engines.qiskit.runtime import _backend_fingerprint

    target = Target(num_qubits=2)
    target.add_instruction(XGate(), {(0,): None, (1,): None})
    metadata = _backend_fingerprint(SimpleNamespace(name="device", num_qubits=2, target=target))
    assert metadata["instructions"] == [
        {"name": "x", "qubits": (0,)},
        {"name": "x", "qubits": (1,)},
    ]
    assert _backend_fingerprint(SimpleNamespace(name="empty", num_qubits=1))["instructions"] == []
    options = RuntimeSamplerOptions(fake_backend="FakeManilaV2", seed_simulator=7)
    assert sdk_options(options)["simulator"] == {"seed_simulator": 7}
    assert "twirling" not in sdk_options(options.model_copy(update={"implementation": "legacy_v2"}))


@pytest.fixture
def mocked_runtime_sdk(monkeypatch):
    """Test adapter construction/cleanup boundaries without pretending to emulate SDK numerics."""
    import sys
    from types import ModuleType

    from chemrefine.engines.qiskit import runtime

    events = []
    backend = BasicSimulator()

    class Mode:
        def __init__(self, backend, max_time):
            self.session_id = "mode_id"
            events.append(("mode", backend, max_time))

        def close(self):
            events.append("close")

        @classmethod
        def from_id(cls, identifier, service):
            events.append(("existing", identifier, service))
            instance = object.__new__(cls)
            instance.session_id = identifier
            return instance

    class Service:
        def __init__(self, **kwargs):
            events.append(("service", kwargs))

        def backend(self, name):
            events.append(("backend", name))
            return backend

    class Estimator:
        def __init__(self, **kwargs):
            events.append(("estimator", kwargs))

    class Sampler:
        def __init__(self, **kwargs):
            events.append(("sampler", kwargs))

    root = ModuleType("qiskit_ibm_runtime")
    root.Session = Mode
    root.Batch = Mode
    root.QiskitRuntimeService = Service
    root.fake_provider = SimpleNamespace(FakeManilaV2=BasicSimulator)
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", root)
    for suffix, attribute, cls in [
        ("executor_estimator", "Estimator", Estimator),
        ("executor_sampler", "Sampler", Sampler),
        ("estimator", "EstimatorV2", Estimator),
        ("sampler", "SamplerV2", Sampler),
    ]:
        module = ModuleType(f"qiskit_ibm_runtime.{suffix}")
        setattr(module, attribute, cls)
        monkeypatch.setitem(sys.modules, f"qiskit_ibm_runtime.{suffix}", module)
    monkeypatch.setattr(runtime, "version", lambda _: "0.50.0")
    return events, root


@pytest.mark.parametrize(
    "kind,implementation",
    [
        (kind, implementation)
        for kind in ("estimator", "sampler")
        for implementation in ("executor", "legacy_v2")
    ],
)
def test_factory_selects_explicit_sdk_implementation_and_real_v2_type(
    tmp_path, mocked_runtime_sdk, kind, implementation
):
    from qiskit.primitives import BaseEstimatorV2, BaseSamplerV2

    from chemrefine.engines.qiskit.runtime import build_runtime_resource

    model = RuntimeEstimatorOptions if kind == "estimator" else RuntimeSamplerOptions
    resource = build_runtime_resource(
        model(
            fake_backend="FakeManilaV2", journal_dir=str(tmp_path), implementation=implementation
        ),
        kind=kind,
        device="cpu",
    )
    with resource as primitive:
        assert isinstance(primitive, BaseEstimatorV2 if kind == "estimator" else BaseSamplerV2)
        assert primitive.options.implementation == implementation
    assert mocked_runtime_sdk[0][0][0] == kind


@pytest.mark.parametrize("mode", ["session", "batch"])
def test_remote_mode_creation_records_id_and_closes_owned_resource(
    tmp_path, mocked_runtime_sdk, mode
):
    from chemrefine.engines.qiskit.runtime import build_runtime_resource

    settings = RuntimeEstimatorOptions(
        backend_name="ibm_a",
        account_name="saved_name",
        channel="ibm_quantum_platform",
        instance="instance_name",
        mode=mode,
        journal_dir=str(tmp_path),
    )
    resource = build_runtime_resource(settings, kind="estimator", device="cpu")
    (record,) = read_journal(tmp_path)
    assert record.summary.kind == mode and record.job_id == "mode_id"
    assert "saved_name" not in next(tmp_path.glob("*.json")).read_text()
    with resource:
        pass
    assert mocked_runtime_sdk[0][-1] == "close"


def test_existing_mode_remains_caller_owned_and_factory_failure_closes_new_mode(
    tmp_path, mocked_runtime_sdk, monkeypatch
):
    from chemrefine.engines.qiskit import runtime

    events, _ = mocked_runtime_sdk
    settings = RuntimeEstimatorOptions(
        backend_name="ibm_a",
        mode="session",
        mode_id="existing",
        close_mode=False,
        journal_dir=str(tmp_path),
    )
    with runtime.build_runtime_resource(settings, kind="estimator", device="cpu"):
        pass
    assert "close" not in events and read_journal(tmp_path) == ()

    def fail(_options):
        raise ConfigError("cannot construct primitive")

    monkeypatch.setattr(runtime, "sdk_options", fail)
    with pytest.raises(ConfigError, match="cannot construct"):
        runtime.build_runtime_resource(
            settings.model_copy(update={"mode_id": None, "close_mode": True}),
            kind="estimator",
            device="cpu",
        )
    assert events[-1] == "close"


def test_factory_requires_durable_journal_and_supported_release_before_provider(
    tmp_path, monkeypatch, mocked_runtime_sdk
):
    from chemrefine.engines.qiskit import runtime

    monkeypatch.delenv("CHEMREFINE_PROVIDER_JOURNAL_DIR", raising=False)
    settings = RuntimeEstimatorOptions(backend_name="ibm_a")
    with pytest.raises(ConfigError, match="journal_dir"):
        runtime.build_runtime_resource(settings, kind="estimator", device="cpu")
    with pytest.raises(ConfigError, match="device"):
        runtime.build_runtime_resource(settings, kind="estimator", device="cuda")
    with pytest.raises(ConfigError, match="typed options"):
        runtime.build_runtime_resource(settings, kind="sampler", device="cpu")
    monkeypatch.setenv("CHEMREFINE_PROVIDER_JOURNAL_DIR", str(tmp_path / "provider_jobs"))
    old = tmp_path / "attempt0" / "provider_jobs"
    old.mkdir(parents=True)
    resource = runtime.build_runtime_resource(settings, kind="estimator", device="cpu")
    with resource as primitive:
        assert primitive.journal.history_directories == (old,)
    monkeypatch.setattr(runtime, "version", lambda _: "0.51.0")
    with pytest.raises(ConfigError, match=r"0\.50"):
        runtime.build_runtime_resource(settings, kind="estimator", device="cpu")


def test_fake_session_has_no_remote_creation_record_and_bad_fake_name_fails(
    tmp_path, mocked_runtime_sdk
):
    from chemrefine.engines.qiskit.runtime import build_runtime_resource

    settings = RuntimeSamplerOptions(
        fake_backend="FakeManilaV2", mode="batch", journal_dir=str(tmp_path)
    )
    with build_runtime_resource(settings, kind="sampler", device="cpu"):
        pass
    assert read_journal(tmp_path) == ()
    with pytest.raises(ConfigError, match="unknown public"):
        build_runtime_resource(
            settings.model_copy(update={"fake_backend": "FakeMissing"}),
            kind="sampler",
            device="cpu",
        )


@pytest.fixture
def calibration_boundary(tmp_path, monkeypatch):
    """Exercise journal/model association around calibration without claiming SDK emulation."""
    import sys
    from types import ModuleType

    from qiskit.quantum_info import PauliLindbladMap

    circuit = QuantumCircuit(3)
    circuit.cx(1, 2)
    layers = list(circuit.data)
    model = PauliLindbladMap.from_sparse_list([("XX", [0, 1], 0.01)], num_qubits=2)
    state = SimpleNamespace(layers=layers, models=[model], submissions=[], serialized=[])

    class Serializer:
        @staticmethod
        def from_quantum_circuit(value, *, qpy_version):
            assert qpy_version == 17
            state.serialized.append(value)
            return SimpleNamespace(model_dump_json=lambda: value.name + str(value.data))

    class Learner:
        def __init__(self, **kwargs):
            state.settings = kwargs

        def run(self, actual_layers):
            # The intent is durable before the provider create call executes.
            assert read_journal(tmp_path)[-1].state == "intent"
            state.submissions.append(actual_layers)
            return SimpleNamespace(
                job_id=lambda: "calibration_id",
                result=lambda: SimpleNamespace(to_pauli_lindblad_maps=lambda: state.models),
            )

    for name, attr, value in (
        ("ibm_quantum_schemas.common", "QpyModelV13ToV17", Serializer),
        ("qiskit_ibm_runtime.noise_learner_v3", "NoiseLearnerV3", Learner),
    ):
        module = ModuleType(name)
        setattr(module, attr, value)
        monkeypatch.setitem(sys.modules, name, module)

    settings = RuntimeEstimatorOptions(
        backend_name="ibm_test", pec={"enable": True}, noise_learning={"enabled": True}
    )
    ideal = StatevectorEstimator(seed=10)
    primitive = SimpleNamespace(
        options=SimpleNamespace(resilience=SimpleNamespace(layer_noise_model=None)),
        find_unique_layers=lambda pubs, types: state.layers,
        run=ideal.run,
    )
    adapter = RuntimePrimitive(
        primitive, settings, RequestJournal(tmp_path), kind="estimator", backend=BasicSimulator()
    )
    return adapter, state


def test_calibration_journal_precedes_energy_and_models_match_actual_layers(
    tmp_path, calibration_boundary
):
    adapter, state = calibration_boundary
    circuit = QuantumCircuit(1)
    outcome = adapter.run([(circuit, {"Z": 1}, None, 0.1)]).result()
    assert len(outcome) == 1
    records = read_journal(tmp_path)
    assert {record.summary.kind for record in records} == {"noise_learner", "estimator"}
    assert all(record.state == "completed" for record in records)
    assert state.submissions == [state.layers]
    assert adapter.primitive.options.resilience.layer_noise_model == list(
        zip(state.layers, state.models, strict=True)
    )
    assert tuple(state.serialized[0].qubits) == state.layers[0].qubits
    assert state.serialized[0].name == "chemrefine_runtime_noise_layer"
    assert state.settings["options"]["max_execution_time"] == 300


@pytest.mark.parametrize("failure", ["too_many", "missing", "wrong_width", "nonfinite"])
def test_bad_calibration_never_submits_energy(tmp_path, calibration_boundary, failure):
    from qiskit.quantum_info import PauliLindbladMap

    adapter, state = calibration_boundary
    if failure == "too_many":
        state.layers *= adapter.options.noise_learning.max_layers + 1
        match = "max_layers"
    elif failure == "missing":
        state.models = []
        match = "incompatible"
    elif failure == "wrong_width":
        state.models = [PauliLindbladMap.from_sparse_list([("X", [0], 0.1)], num_qubits=1)]
        match = "incompatible"
    else:
        state.models = [
            PauliLindbladMap.from_sparse_list([("XX", [0, 1], float("nan"))], num_qubits=2)
        ]
        match = "nonfinite"
    with pytest.raises(ConfigError, match=match):
        adapter.run([(QuantumCircuit(1), {"Z": 1})])
    assert all(record.summary.kind == "noise_learner" for record in read_journal(tmp_path))
    assert adapter.primitive.options.resilience.layer_noise_model is None


def test_empty_noise_layers_do_not_create_calibration_job(tmp_path, calibration_boundary):
    adapter, state = calibration_boundary
    state.layers = []
    adapter.run([(QuantumCircuit(1), {"Z": 1})]).result()
    assert adapter.primitive.options.resilience.layer_noise_model == []
    assert len(read_journal(tmp_path)) == 1
    assert state.submissions == []


def test_calibration_and_energy_reserve_job_budget_before_any_submission(
    tmp_path, calibration_boundary
):
    adapter, _ = calibration_boundary
    adapter.options = adapter.options.model_copy(update={"max_jobs": 1})
    with pytest.raises(ConfigError, match="calibration exceed"):
        adapter.run([(QuantumCircuit(1), {"Z": 1})])
    assert read_journal(tmp_path) == ()


@pytest.mark.parametrize("failure", ["create", "missing_id", "journal"])
def test_remote_mode_failure_retains_recovery_state(
    tmp_path, mocked_runtime_sdk, monkeypatch, failure
):
    from chemrefine.engines.qiskit import runtime

    events, root = mocked_runtime_sdk
    error: type[Exception]

    if failure == "create":

        def fail_create(**kwargs):
            assert read_journal(tmp_path)[0].state == "intent"
            raise ConnectionError("private provider response")

        root.Session = fail_create
        error, match = ConnectionError, "private provider"
    elif failure == "missing_id":
        original = root.Session.__init__

        def missing_id(self, **kwargs):
            original(self, **kwargs)
            self.session_id = None

        monkeypatch.setattr(root.Session, "__init__", missing_id)
        error, match = ConfigError, "did not return an identifier"
    else:

        def fail_update(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(RequestJournal, "update", fail_update)
        error, match = ConfigError, "mode_id.*journal update failed"
    with pytest.raises(error, match=match):
        runtime.build_runtime_resource(
            RuntimeEstimatorOptions(
                backend_name="ibm_a", mode="session", journal_dir=str(tmp_path)
            ),
            kind="estimator",
            device="cpu",
        )
    (record,) = read_journal(tmp_path)
    assert record.state == ("intent" if failure == "journal" else "submission_unknown")
    assert "private provider response" not in next(tmp_path.glob("*.json")).read_text()
    assert ("close" in events) == (failure != "create")


def test_retrieval_requires_service_and_finite_pub_precision(tmp_path):
    adapter = _adapter(tmp_path)
    circuit = QuantumCircuit(1)
    submitted = adapter.run([(circuit, {"Z": 1}, None, 0.1)])
    adapter.options = adapter.options.model_copy(update={"retrieve_job_ids": submitted.job_ids})
    with pytest.raises(ConfigError, match="requires a remote service"):
        adapter.run([(circuit, {"Z": 1}, None, 0.1)])
    with pytest.raises(ConfigError, match="finite"):
        adapter.run([(circuit, {"Z": 1}, None, float("nan"))])


def test_runtime_registry_builders_return_typed_resources(tmp_path, mocked_runtime_sdk):
    from chemrefine.engines.qiskit.components.runtime import (
        build_runtime_estimator,
        build_runtime_sampler,
    )
    from chemrefine.engines.qiskit.context import EstimatorResource, SamplerResource

    for builder, model, resource_type in (
        (build_runtime_estimator, RuntimeEstimatorOptions, EstimatorResource),
        (build_runtime_sampler, RuntimeSamplerOptions, SamplerResource),
    ):
        resource = builder(options=model(fake_backend="FakeManilaV2", journal_dir=str(tmp_path)))
        assert isinstance(resource, resource_type)

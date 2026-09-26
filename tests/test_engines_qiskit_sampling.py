"""Logical count conventions and sampler provider lifetimes."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from chemrefine.engines.qiskit.context import SamplerResource
from chemrefine.engines.qiskit.options import ComponentSelection, QiskitOptions
from chemrefine.engines.qiskit.registry import SAMPLERS, ComponentSpec, NoComponentOptions
from chemrefine.engines.qiskit.sampling import SampleBatch, SamplingSession, sample_circuit
from chemrefine.engines.qiskit.workflow import validate_options
from chemrefine.errors import ConfigError

pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning:qiskit"),
    pytest.mark.filterwarnings("ignore::PendingDeprecationWarning:qiskit"),
]


@pytest.mark.parametrize(
    "counts,width,shots",
    [
        ({}, 2, 1),
        ({"01": 1}, 0, 1),
        ({"01": 1}, True, 1),
        ({"01": 1}, 2, False),
        ({"01": 1}, 2, 0),
        ({"01": 1}, 2.0, 1),
        ({"01": 1}, 2, 1.0),
        ({"0x": 1}, 2, 1),
        ({"1": 1}, 2, 1),
        ({1: 1}, 2, 1),
        ({"01": True}, 2, 1),
        ({"01": -1}, 2, 1),
        ({"01": 1.0}, 2, 1),
        ({"01": 2}, 2, 1),
    ],
)
def test_sample_batch_rejects_ambiguous_or_nonphysical_counts(counts, width, shots):
    """Mitigation quasiprobabilities and malformed registers are not raw shots."""
    with pytest.raises(ConfigError):
        SampleBatch(counts, width, shots)


def test_sample_batch_copies_input():
    """Editing a caller's dictionary does not mutate a completed batch."""
    counts = {"01": 2}
    metadata = {"seed": 4}
    result = SampleBatch(counts, 2, 2, metadata)
    counts["01"] = 3
    metadata["seed"] = 5
    assert result.counts == {"01": 2}
    assert result.metadata == {"seed": 4}


@pytest.mark.parametrize("provider", ["statevector", "basic_backend", "aer"])
def test_sampler_preserves_logical_bit_order(provider):
    """Compilation keeps logical q0 at the right of every returned bitstring."""
    qiskit = pytest.importorskip("qiskit")
    if provider == "aer":
        pytest.importorskip("qiskit_aer")
    circuit = qiskit.QuantumCircuit(3)
    circuit.x(0)
    circuit.x(2)
    result = sample_circuit(circuit, provider, shots=64)
    assert result.counts == {"101": 64}
    assert result.metadata["transpiled"] is (provider != "statevector")
    assert circuit.num_clbits == 0


def test_sampler_parameter_binding_and_seed():
    """Sampling binds a copy and a fixed seed repeats the same shot sequence."""
    qiskit = pytest.importorskip("qiskit")
    from qiskit.circuit import Parameter

    circuit = qiskit.QuantumCircuit(2)
    circuit.ry(Parameter("theta"), 0)
    circuit.cx(0, 1)
    selection = ComponentSelection(name="statevector", options={"seed": 71})
    first = sample_circuit(circuit, selection, shots=100, parameter_values=[1.2])
    second = sample_circuit(circuit, selection, shots=100, parameter_values=[1.2])
    assert first == second
    assert set(first.counts) == {"00", "11"}
    assert circuit.num_parameters == 1


@pytest.mark.parametrize("parameters", [[float("nan")], [1j], ["bad"], [], [[1]]])
def test_sampler_rejects_invalid_parameters(parameters):
    """Parameter conversion never drops imaginary parts or silently broadcasts."""
    qiskit = pytest.importorskip("qiskit")
    from qiskit.circuit import Parameter

    circuit = qiskit.QuantumCircuit(1)
    circuit.ry(Parameter("theta"), 0)
    with pytest.raises(ConfigError, match=r"parameters|finite value"):
        sample_circuit(circuit, shots=10, parameter_values=parameters)


def test_sampler_rejects_unbound_measured_and_empty_circuits():
    """Terminal occupation sampling has a deliberately unambiguous input contract."""
    qiskit = pytest.importorskip("qiskit")
    from qiskit.circuit import Parameter

    circuit = qiskit.QuantumCircuit(1)
    circuit.ry(Parameter("theta"), 0)
    with pytest.raises(ConfigError, match="fully bound"):
        sample_circuit(circuit, shots=10)
    with pytest.raises(ConfigError, match="positive integer"):
        sample_circuit(circuit, shots=True)
    circuit.measure_all()
    with pytest.raises(ConfigError, match="classical registers"):
        sample_circuit(circuit, shots=10)
    with pytest.raises(ConfigError, match="at least one qubit"):
        sample_circuit(qiskit.QuantumCircuit(), shots=10)


@pytest.mark.parametrize("broken", [False, True])
def test_sampler_closes_provider_on_success_and_invalid_result(monkeypatch, broken):
    """Provider cleanup runs even when result decoding fails."""
    qiskit = pytest.importorskip("qiskit")
    closed = []
    item = SimpleNamespace(data=SimpleNamespace(meas=SimpleNamespace(get_counts=lambda: {"1": 4})))
    sampler = SimpleNamespace(
        run=lambda *a, **k: SimpleNamespace(result=lambda: [] if broken else [item])
    )
    spec = ComponentSpec(
        NoComponentOptions, lambda **k: SamplerResource(sampler, lambda: closed.append(True))
    )
    monkeypatch.setitem(SAMPLERS._specs, "managed_test", spec)
    circuit = qiskit.QuantumCircuit(1)
    if broken:
        with pytest.raises(ConfigError, match="terminal measurement counts"):
            sample_circuit(circuit, "managed_test", shots=4)
    else:
        assert sample_circuit(circuit, "managed_test", shots=4).counts == {"1": 4}
    assert closed == [True]


@pytest.mark.parametrize("provider", ["statevector", "basic_backend"])
def test_cpu_sampler_rejects_cuda(provider):
    """A CPU provider cannot silently discard a GPU request."""
    with pytest.raises(ConfigError, match="cpu only"):
        SAMPLERS.build(ComponentSelection.named(provider), device="cuda")


def test_sampler_graph_checks_cuda_methods():
    """Preflight rejects unsupported execution before constructing providers."""
    with pytest.raises(ConfigError, match="device: cuda"):
        validate_options(QiskitOptions(algorithm="sqd", device="cuda"))
    with pytest.raises(ConfigError, match="not GPU-compatible"):
        validate_options(
            QiskitOptions(
                algorithm="sqd",
                device="cuda",
                sampler={"name": "aer", "options": {"method": "matrix_product_state"}},
            )
        )
    with pytest.raises(ConfigError, match="requires device: cuda"):
        validate_options(
            QiskitOptions(
                algorithm="sqd", sampler={"name": "aer", "options": {"method": "tensor_network"}}
            )
        )


@pytest.mark.parametrize(
    "provider,key",
    [("statevector", "seed"), ("basic_backend", "seed_simulator"), ("aer", "seed_simulator")],
)
def test_sampling_session_advances_reproducible_local_streams(provider, key):
    """Independent requests differ but repeat exactly when the whole experiment is replayed."""
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(1)
    circuit.h(0)
    selection = ComponentSelection(name=provider, options={key: 17})
    results = []
    for _ in range(2):
        with SamplingSession(selection, seed=53) as session:
            results.append([session.sample(circuit, shots=127) for _ in range(8)])
    assert results[0] == results[1]
    assert len({tuple(sorted(batch.counts.items())) for batch in results[0]}) > 1
    assert len({batch.metadata["sampling_seed"] for batch in results[0]}) == 8
    assert [batch.metadata["request_index"] for batch in results[0]] == list(range(8))


def test_session_lifetime_validation_cleanup_and_compiler_options(monkeypatch):
    """One provider receives all requests, closes on failure, and cannot be reused afterward."""
    from qiskit import QuantumCircuit
    from qiskit.primitives import StatevectorSampler

    built, closed, compiled = [], [], []

    def compile(circuit, **options):
        """Record provider-specific compilation controls and inject a second-request failure."""
        compiled.append(options)
        if len(compiled) == 2:
            raise ValueError("compiler failed")
        return circuit

    def build(**kwargs):
        """Own one numerical provider even when a later compilation fails."""
        built.append(kwargs)
        return SamplerResource(
            StatevectorSampler(seed=np.random.default_rng(1)),
            close=lambda: closed.append(True),
            transpiler=SimpleNamespace(run=compile),
            transpiler_options={"example": 2},
        )

    monkeypatch.setitem(SAMPLERS._specs, "session_test", ComponentSpec(NoComponentOptions, build))
    circuit = QuantumCircuit(1)
    session = SamplingSession("session_test", cores=3, device="cuda")
    session.__exit__()
    with pytest.raises(ConfigError, match="open SamplingSession"):
        session.sample(circuit, shots=2)
    with pytest.raises(ValueError, match="compiler failed"), session:
        session.sample(circuit, shots=2)
        session.sample(circuit, shots=2)
    assert len(built) == 1 and built[0]["cores"] == 3 and built[0]["device"] == "cuda"
    assert closed == [True] and compiled == [{"example": 2}, {"example": 2}]
    with pytest.raises(ConfigError, match="entered once"), session:
        pass
    with pytest.raises(ConfigError, match="nonnegative"):
        SamplingSession(seed=-1)
    with pytest.raises(ConfigError, match="nonnegative"):
        SamplingSession(seed=True)


def test_seeded_custom_provider_requires_explicit_stream_support(monkeypatch):
    """A provider-specific integer seed cannot silently restart on every request."""
    from chemrefine.engines.qiskit.components.samplers import StatevectorSamplerOptions

    closed = []
    spec = ComponentSpec(
        StatevectorSamplerOptions,
        lambda **kwargs: SamplerResource(None, close=lambda: closed.append(True)),
    )
    monkeypatch.setitem(SAMPLERS._specs, "seeded_custom", spec)
    with (
        pytest.raises(ConfigError, match="set_sampling_seed"),
        SamplingSession(ComponentSelection(name="seeded_custom", options={"seed": 1})),
    ):
        pass
    assert closed == [True]

"""Logical count conventions and sampler provider lifetimes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from chemrefine.engines.qiskit.context import SamplerResource
from chemrefine.engines.qiskit.options import ComponentSelection
from chemrefine.engines.qiskit.registry import SAMPLERS, ComponentSpec, NoComponentOptions
from chemrefine.engines.qiskit.sampling import SampleBatch, sample_circuit
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

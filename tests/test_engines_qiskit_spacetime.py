"""Independent conjugation, fault syndromes and physical postselection contracts."""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from chemrefine.engines.qiskit.options import ComponentSelection
from chemrefine.engines.qiskit.spacetime import (
    SpacetimeOptions,
    build_spacetime_circuit,
    collect_spacetime_counts,
    postselect_spacetime_counts,
)
from chemrefine.errors import ConfigError


def _pauli(label):
    """Tensor independent dense Pauli matrices in display order."""
    matrices = {
        "I": np.eye(2),
        "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.diag([1, -1]),
    }
    result = np.array([[1]]) * (-1 if label.startswith("-") else 1)
    for letter in label.lstrip("+-"):
        result = np.kron(result, matrices[letter])
    return result


def _noise():
    """Declare actual stochastic Aer noise on data and check entangling gates."""
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    model = NoiseModel()
    model.add_all_qubit_quantum_error(depolarizing_error(0.12, 2), ["cx", "cy", "cz"])
    return model.to_dict(serializable=True)


def test_all_single_qubit_cliffords_signed_noncommuting_checks():
    """All 24 Clifford actions and signed pairs preserve the complete unitary."""
    from qiskit.quantum_info import Clifford, Operator
    from qiskit.quantum_info.random import random_clifford

    cliffords = {}
    for seed in range(256):
        clifford = random_clifford(1, seed=seed)
        cliffords[str(clifford.tableau)] = clifford
    assert len(cliffords) == 24
    for clifford in cliffords.values():
        circuit = clifford.to_circuit()
        unitary = Operator(circuit).data
        for first, second in itertools.product(("X", "-Y", "+Z"), repeat=2):
            checked = build_spacetime_circuit(circuit, SpacetimeOptions(checks=(first, second)))
            assert np.allclose(Operator(checked.circuit).data, np.kron(np.eye(4), unitary))
            for before, after in zip(checked.input_checks, checked.output_checks, strict=True):
                assert np.allclose(_pauli(after), unitary @ _pauli(before) @ unitary.conj().T)
    assert isinstance(clifford, Clifford)


def test_complex_arbitrary_inputs_and_every_two_qubit_pauli_fault():
    """Independent matrices predict syndromes and data amplitudes for every fault."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator, Statevector

    payload = QuantumCircuit(2)
    payload.h(0)
    payload.s(1)
    payload.cx(0, 1)
    payload.global_phase = 0.37
    controls = SpacetimeOptions(checks=("XY", "-ZI", "+YY"))
    checked = build_spacetime_circuit(payload, controls)
    rng = np.random.default_rng(17)
    initial = rng.normal(size=4) + 1j * rng.normal(size=4)
    initial /= np.linalg.norm(initial)
    prepared = np.kron(np.eye(8)[0], initial)
    evolved = Operator(payload).data @ initial
    for letters in itertools.product("IXYZ", repeat=2):
        label = "".join(letters)
        fault = _pauli(label)
        circuit = checked.circuit.copy_empty_like()
        for item in checked.circuit.data:
            circuit.append(item.operation, item.qubits, item.clbits)
            if item.operation.label == "spacetime_payload_end":
                circuit.unitary(fault, [0, 1])
        syndrome = sum(
            (not np.allclose(fault @ _pauli(check), _pauli(check) @ fault)) << i
            for i, check in enumerate(checked.output_checks)
        )
        expected = np.kron(np.eye(8)[syndrome], fault @ evolved)
        assert np.allclose(Statevector(prepared).evolve(circuit).data, expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"checks": ()},
        {"checks": ("",)},
        {"checks": ("iX",)},
        {"checks": ("II",)},
        {"checks": ("X", "ZZ")},
        {"checks": ("X",), "diagonal_observables": ("X",)},
        {"checks": ("X",), "diagonal_observables": ("ZZ",)},
        {"checks": ("X",), "shots": True},
        {"checks": ("X",), "confidence": 1},
        {"checks": ("X",), "unknown": 2},
    ],
)
def test_options_reject_invalid_paulis_and_limits(kwargs):
    """Catalog validation remains strict before SDK-dependent construction."""
    with pytest.raises(ValidationError):
        SpacetimeOptions(**kwargs)


def test_builder_rejects_unsupported_payloads_and_allocations():
    """Non-Clifford, dynamic and mismatched inputs fail before any provider calls."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter

    options = SpacetimeOptions(checks=("X",))
    for circuit, message in [
        (QuantumCircuit(0), "width"),
        (QuantumCircuit(2), "width"),
        (QuantumCircuit(1, 1), "classical"),
    ]:
        with pytest.raises(ConfigError, match=message):
            build_spacetime_circuit(circuit, options)
    for gate in ("t", "reset", "delay"):
        circuit = QuantumCircuit(1)
        getattr(circuit, gate)(1, 0) if gate == "delay" else getattr(circuit, gate)(0)
        with pytest.raises(ConfigError, match="Clifford"):
            build_spacetime_circuit(circuit, options)
    circuit = QuantumCircuit(1)
    circuit.rx(Parameter("t"), 0)
    with pytest.raises(ConfigError, match="bound"):
        build_spacetime_circuit(circuit, options)
    with pytest.raises(ConfigError, match="max_qubits"):
        build_spacetime_circuit(
            QuantumCircuit(1), SpacetimeOptions(checks=("X", "Z"), max_qubits=2)
        )
    with pytest.raises(ConfigError, match="max_circuit_operations"):
        build_spacetime_circuit(
            QuantumCircuit(1), SpacetimeOptions(checks=("X",), max_circuit_operations=1)
        )


def test_physical_counts_wilson_intervals_and_conditional_estimates():
    """Acceptance and parity use actual accepted shots, including zero/one limits."""
    checked = SimpleNamespace(num_data_qubits=2, input_checks=("XI",), output_checks=("ZI",))
    options = SpacetimeOptions(checks=("XI",), diagonal_observables=("ZZ", "-IZ"), shots=10)
    result = postselect_spacetime_counts({"0 00": 3, "0 01": 2, "1 10": 5}, checked, options)
    assert result.raw_counts == {"0 00": 3, "0 01": 2, "1 10": 5}
    assert result.accepted_counts == {"00": 3, "01": 2}
    assert result.rejected_counts == {"1 10": 5}
    assert result.metadata["acceptance_rate"] == 0.5
    assert result.metadata["acceptance_wilson_interval"] == pytest.approx(
        [0.2365930905, 0.7634069095]
    )
    estimate = result.metadata["conditional_observables"]
    assert estimate["ZZ"]["conditional_mean"] == 0.2
    assert estimate["-IZ"]["conditional_mean"] == -0.2
    assert estimate["ZZ"]["conditional_standard_error"] == pytest.approx(np.sqrt(0.24))
    for syndrome in ("0", "1"):
        single = postselect_spacetime_counts(
            {syndrome + " 00": 1}, checked, options.model_copy(update={"shots": 1})
        )
        assert (
            single.metadata["conditional_observables"]["ZZ"]["conditional_standard_error"] is None
        )
        if syndrome == "1":
            assert single.accepted_counts == {}
            assert single.metadata["conditional_observables"]["ZZ"]["conditional_mean"] is None
            assert single.metadata["sampling_overhead"] is None


@pytest.mark.parametrize(
    "counts",
    [
        {"0 00": -1},
        {"0 00": 1.0},
        {"0 00": True},
        {"00": 1},
        {"0 0x": 1},
        {"00 00": 1},
        {3: 1},
        {},
        {"0 00": 2},
    ],
)
def test_postselection_rejects_quasiprobabilities_and_malformed_counts(counts):
    """Weighted mitigation outputs cannot silently become physical samples."""
    checked = SimpleNamespace(num_data_qubits=2, input_checks=("XI",))
    with pytest.raises(ConfigError, match="physical counts"):
        postselect_spacetime_counts(counts, checked, SpacetimeOptions(checks=("XI",), shots=1))


def test_actual_aer_noise_postselection_and_preparation():
    """Compile all checks and retain correlated register shots from real Aer."""
    from qiskit import QuantumCircuit

    payload = QuantumCircuit(2)
    payload.h(0)
    payload.cx(0, 1)
    preparation = QuantumCircuit(2)
    preparation.ry(0.32, 1)
    selection = ComponentSelection(
        name="aer",
        options={
            "noise_model": _noise(),
            "optimization_level": 0,
            "seed_simulator": 43,
            "seed_transpiler": 5,
        },
    )
    options = SpacetimeOptions(checks=("ZI", "IX"), diagonal_observables=("ZZ",), shots=1024)
    result = collect_spacetime_counts(payload, selection, options, preparation=preparation, cores=1)
    assert sum(result.raw_counts.values()) == 1024
    assert 0 < sum(result.accepted_counts.values()) < 1024
    assert sum(result.rejected_counts.values()) + sum(result.accepted_counts.values()) == 1024
    assert result.metadata["compiled_qubits"] == 4
    assert result.metadata["preparation_is_checked"] is False


def test_acquisition_requires_nonideal_noise_and_valid_preparation():
    """Ideal sampling cannot be accidentally presented as mitigation."""
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(1)
    options = SpacetimeOptions(checks=("X",), shots=1)
    for selection in (
        ComponentSelection.named("statevector"),
        ComponentSelection.named("aer"),
        ComponentSelection(name="aer", options={"noise_model": {"errors": []}}),
    ):
        with pytest.raises(ConfigError, match="noise_model"):
            collect_spacetime_counts(circuit, selection, options)
    selection = ComponentSelection(name="aer", options={"noise_model": _noise()})
    for preparation in (QuantumCircuit(2), QuantumCircuit(1, 1)):
        with pytest.raises(ConfigError, match="preparation"):
            collect_spacetime_counts(circuit, selection, options, preparation=preparation)
    with pytest.raises(ConfigError, match="max_memory_mb"):
        collect_spacetime_counts(
            circuit, selection, options.model_copy(update={"shots": 10000000, "max_memory_mb": 1})
        )


def test_register_failures_close_resources_and_preserve_compilation(monkeypatch):
    """Malformed register alignment fails instead of independently pairing counts."""
    from qiskit import QuantumCircuit

    from chemrefine.engines.qiskit.context import SamplerResource
    from chemrefine.engines.qiskit.registry import SAMPLERS

    selection = ComponentSelection(name="aer", options={"noise_model": _noise()})
    options = SpacetimeOptions(checks=("X",), shots=1)
    closed = []
    for data in (
        None,
        SimpleNamespace(
            data_bits=SimpleNamespace(get_bitstrings=lambda: ["0"]),
            check_bits=SimpleNamespace(get_bitstrings=lambda: []),
        ),
    ):
        resource = SamplerResource(
            SimpleNamespace(
                run=lambda *a, data=data, **k: SimpleNamespace(
                    result=lambda: [SimpleNamespace(data=data)]
                )
            ),
            close=lambda: closed.append(True),
        )
        monkeypatch.setattr(SAMPLERS, "build", lambda *a, resource=resource, **k: resource)
        with pytest.raises(ConfigError, match="registers"):
            collect_spacetime_counts(QuantumCircuit(1), selection, options)
    assert closed == [True, True]


def test_acquisition_failure_boundaries_and_uncompiled_resource(monkeypatch):
    """Decode missing/invalid providers and retain the sampler cleanup contract."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter

    from chemrefine.engines.qiskit.context import SamplerResource
    from chemrefine.engines.qiskit.registry import SAMPLERS

    selection = ComponentSelection(name="aer", options={"noise_model": _noise()})
    options = SpacetimeOptions(checks=("X",), shots=1)
    bad = ComponentSelection(name="aer", options={"noise_model": {"errors": [{"type": "qerror"}]}})
    with pytest.raises(ConfigError, match="serialized"):
        collect_spacetime_counts(QuantumCircuit(1), bad, options)
    prep = QuantumCircuit(1)
    prep.rx(Parameter("a"), 0)
    with pytest.raises(ConfigError, match="preparation"):
        collect_spacetime_counts(QuantumCircuit(1), selection, options, preparation=prep)
    prep = QuantumCircuit(1)
    for _ in range(100):
        prep.x(0)
    with pytest.raises(ConfigError, match="max_circuit_operations"):
        collect_spacetime_counts(
            QuantumCircuit(1),
            selection,
            options.model_copy(update={"max_circuit_operations": 50}),
            preparation=prep,
        )
    resource = SamplerResource(
        SimpleNamespace(
            run=lambda *a, **k: SimpleNamespace(
                result=lambda: [
                    SimpleNamespace(
                        data=SimpleNamespace(
                            data_bits=SimpleNamespace(get_bitstrings=lambda: ["0"]),
                            check_bits=SimpleNamespace(get_bitstrings=lambda: ["0"]),
                        )
                    )
                ]
            )
        )
    )
    monkeypatch.setattr(SAMPLERS, "build", lambda *a, resource=resource, **k: resource)
    result = collect_spacetime_counts(QuantumCircuit(1), selection, options)
    assert result.accepted_counts == {"0": 1}
    checked = SimpleNamespace(input_checks=("Z",), num_data_qubits=1)
    with pytest.raises(ConfigError, match="match"):
        postselect_spacetime_counts({"0 0": 1}, checked, options)


def test_control_flow_without_classical_bits_is_explicitly_rejected():
    """Loop-based quantum control flow does not slip through Clifford conversion."""
    from qiskit import QuantumCircuit

    payload = QuantumCircuit(1)
    with payload.for_loop(range(2)):
        payload.x(0)
    with pytest.raises(ConfigError, match="control flow"):
        build_spacetime_circuit(payload, SpacetimeOptions(checks=("Z",)))


def test_noise_channels_must_apply_to_final_physical_gates_and_qubits():
    """Nonideal models on unused operations or physical wires cannot label an ideal run."""
    from qiskit import QuantumCircuit
    from qiskit_aer.noise import NoiseModel, ReadoutError, pauli_error

    payload = QuantumCircuit(1)
    options = SpacetimeOptions(checks=("X",), shots=4)
    unused_gate = NoiseModel()
    unused_gate.add_all_qubit_quantum_error(pauli_error([("XXX", 1.0)]), ["ccx"])
    unused_qubit = NoiseModel()
    unused_qubit.add_quantum_error(pauli_error([("X", 1.0)]), ["h"], [7])
    unused_readout = NoiseModel()
    unused_readout.add_readout_error(ReadoutError([[0, 1], [1, 0]]), [7])
    for noise in (unused_gate, unused_qubit, unused_readout):
        selection = ComponentSelection(
            name="aer",
            options={
                "noise_model": noise.to_dict(serializable=True),
                "optimization_level": 0,
            },
        )
        with pytest.raises(ConfigError, match="no channel applicable"):
            collect_spacetime_counts(payload, selection, options)
    for readout in (False, True):
        noise = NoiseModel()
        if readout:
            noise.add_readout_error(ReadoutError([[0.9, 0.1], [0.2, 0.8]]), [1])
        else:
            noise.add_quantum_error(pauli_error([("X", 0.5), ("I", 0.5)]), ["h"], [1])
        selection = ComponentSelection(
            name="aer",
            options={
                "noise_model": noise.to_dict(serializable=True),
                "optimization_level": 0,
            },
        )
        assert sum(collect_spacetime_counts(payload, selection, options).raw_counts.values()) == 4

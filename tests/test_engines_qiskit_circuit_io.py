"""Bound molecular circuits retain physical interpretation through validated QPY bundles."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from chemrefine.engines.qiskit.bundles import read_bundle, write_bundle
from chemrefine.engines.qiskit.circuit_io import (
    BoundCircuit,
    CircuitDescription,
    bound_circuit,
    circuit_input_dependencies,
    load_circuit,
    save_circuit,
    validate_circuit_bundle,
)
from chemrefine.engines.qiskit.components.initial_states import build_explicit_reference
from chemrefine.engines.qiskit.data import ElectronicStructureData
from chemrefine.engines.qiskit.mapping import map_problem
from chemrefine.engines.qiskit.problem import prepare_problem
from chemrefine.errors import ConfigError, OutputParseError

pytest.importorskip("qiskit_nature")
from qiskit import QuantumCircuit, qpy
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector


@pytest.fixture
def molecular_context():
    """Prepare real stored H2 integrals without invoking a classical driver."""
    source = Path(__file__).parent / "data/engines/qiskit/h2_integrals.json"
    prepared = prepare_problem(ElectronicStructureData(**json.loads(source.read_text())))
    prepared.problem.hamiltonian.constants["supplied_inactive_energy"] = -0.25
    return map_problem(prepared, "parity")


@pytest.fixture
def retained(molecular_context):
    """Retain vector parameters whose natural order differs from alphabetical order."""
    circuit = build_explicit_reference(molecular_context, (True, False, True, False))
    parameters = ParameterVector("angle", 12)
    circuit.ry(parameters[10], 0)
    circuit.ry(parameters[2], 1)
    circuit.metadata = {"unrelated": "not part of the circuit interpretation"}
    bindings = {parameters[10]: 0.7, parameters[2]: 0.2}
    return bound_circuit(molecular_context, circuit, bindings, root=1), circuit, bindings


def _raw_qpy_bundle(path, circuits, description):
    """Bypass writer guards to test independent reader validation with real QPY."""
    stream = io.BytesIO()
    qpy.dump(circuits or [QuantumCircuit(1)], stream, version=13)
    payload = stream.getvalue()
    if not circuits:
        # QPY v13 stores an eight-byte program count after the ten-byte prefix.
        # Keep the complete file header and circuit type key, with no programs.
        payload = payload[:10] + bytes(8) + payload[18:20]
    return write_bundle(
        path,
        kind="bound_circuit",
        arrays={"qpy": np.frombuffer(payload, dtype=np.uint8)},
        metadata=description.model_dump(mode="json"),
    )


def test_bound_h2_roundtrip_preserves_parameter_order_mapping_offsets_and_state(
    tmp_path, molecular_context, retained
):
    """The exported state and Hamiltonian reproduce the same molecular energy."""
    value, circuit, bindings = retained
    assert value.description.parameter_order == ("angle[2]", "angle[10]")
    assert value.description.parameter_order != tuple(sorted(value.description.parameter_order))
    assert value.description.parameter_values == (0.2, 0.7)
    assert value.description.root == 1
    assert value.description.mapping == molecular_context.mapping_metadata
    assert value.description.active_space == molecular_context.active_space_metadata
    assert value.description.provenance == molecular_context.provenance
    assert value.description.energy_offsets == molecular_context.problem.hamiltonian.constants
    assert value.description.energy_offsets["supplied_inactive_energy"] == -0.25
    assert value.circuit.metadata == {} and circuit.metadata
    path = save_circuit(tmp_path / "state.circuit.json", value)
    assert read_bundle(path).arrays["qpy"][6] == 13
    restored = load_circuit(path)
    assert restored.description == value.description
    expected = Statevector(circuit.assign_parameters(bindings))
    np.testing.assert_allclose(Statevector(restored.circuit).data, expected.data, atol=1e-13)
    saved_hamiltonian = SparsePauliOp.from_list(
        list(restored.description.active_hamiltonian.items())
    )
    np.testing.assert_allclose(
        saved_hamiltonian.to_matrix(), molecular_context.qubit_hamiltonian.to_matrix()
    )
    offsets = sum(restored.description.energy_offsets.values())
    assert (
        Statevector(restored.circuit).expectation_value(saved_hamiltonian) + offsets
    ) == pytest.approx(expected.expectation_value(molecular_context.qubit_hamiltonian) + offsets)
    sequence = bound_circuit(molecular_context, circuit, [0.2, 0.7], root=1)
    np.testing.assert_allclose(Statevector(sequence.circuit).data, expected.data, atol=1e-13)
    assert validate_circuit_bundle(read_bundle(path)) == value.description


def test_complex_hopping_hamiltonian_retains_y_terms_and_matrix_elements(tmp_path):
    """Complex Hermitian integrals retain phases despite having real Pauli coefficients."""
    source = Path(__file__).parent / "data/engines/qiskit/h2_integrals.json"
    data = ElectronicStructureData(**json.loads(source.read_text()))
    one_body = np.asarray(data.one_body_integrals, dtype=complex)
    one_body[0, 1], one_body[1, 0] = 0.3j, -0.3j
    context = map_problem(prepare_problem(replace(data, one_body_integrals=one_body)))
    circuit = QuantumCircuit(context.num_qubits)
    circuit.h(0)
    circuit.s(0)
    circuit.h(1)
    value = load_circuit(
        save_circuit(tmp_path / "complex.json", bound_circuit(context, circuit, []))
    )
    saved = SparsePauliOp.from_list(list(value.description.active_hamiltonian.items()))
    assert any("Y" in label for label in value.description.active_hamiltonian)
    assert np.any(abs(context.qubit_hamiltonian.to_matrix().imag) > 0.1)
    np.testing.assert_allclose(saved.to_matrix(), context.qubit_hamiltonian.to_matrix(), atol=1e-13)
    assert Statevector(value.circuit).expectation_value(saved) == pytest.approx(
        Statevector(circuit).expectation_value(context.qubit_hamiltonian), abs=1e-12
    )


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"parameter_values": []}, "differ in length"),
        ({"mapping": {}}, "mapping and active Hamiltonian"),
        ({"active_hamiltonian": {}}, "mapping and active Hamiltonian"),
        ({"active_hamiltonian": {"ZZZ": 1}}, "logical register"),
        ({"active_hamiltonian": {"AQ": 1}}, "logical register"),
        ({"num_spin_orbitals": 3}, "spin orbitals"),
        ({"num_particles": [-1, 1]}, "spin orbitals"),
        ({"num_particles": [3, 1]}, "spin orbitals"),
        ({"parameter_values": [float("nan"), 1]}, "finite"),
        ({"energy_offsets": {"bad": float("inf")}}, "finite"),
        ({"root": True}, "integer"),
        ({"unknown": 1}, "Extra inputs"),
    ],
)
def test_circuit_description_rejects_incoherent_or_unknown_fields(retained, changes, match):
    """The descriptor validates conventions independently of binary integrity."""
    value, _circuit, _bindings = retained
    with pytest.raises(ValidationError, match=match):
        CircuitDescription.model_validate(value.description.model_dump() | changes)


@pytest.mark.parametrize("mode", ["width", "measurements", "nonhermitian"])
def test_binding_rejects_wrong_logical_register_and_nonhermitian_metadata(molecular_context, mode):
    """Unsafe interpretation must fail before serialization."""
    context = molecular_context
    circuit = QuantumCircuit(context.num_qubits)
    if mode == "width":
        circuit = QuantumCircuit(context.num_qubits + 1)
    elif mode == "measurements":
        circuit.measure_all()
    else:
        context = replace(context, qubit_hamiltonian=SparsePauliOp.from_list([("ZZ", 1j)]))
    with pytest.raises(ConfigError, match=r"logical preparation|Hermitian Pauli"):
        bound_circuit(context, circuit, [])


@pytest.mark.parametrize("mode", ["parameters", "measurements", "width"])
def test_writer_refuses_invalid_bound_circuit_objects(tmp_path, retained, mode):
    """The writer revalidates transient SDK objects supplied directly by callers."""
    value, original, _bindings = retained
    circuit = original if mode == "parameters" else value.circuit.copy()
    if mode == "measurements":
        circuit.measure_all()
    elif mode == "width":
        circuit = QuantumCircuit(value.description.num_qubits + 1)
    with pytest.raises(ConfigError, match="bound, unmeasured logical preparation"):
        save_circuit(tmp_path / "bad.json", BoundCircuit(circuit, value.description))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("mode", ["kind", "arrays", "dtype", "shape", "size", "header", "metadata"])
def test_validation_rejects_incoherent_bundle_payloads(tmp_path, retained, mode):
    """Generic bundle integrity alone cannot establish the circuit artifact contract."""
    value, _circuit, _bindings = retained
    arrays: dict[str, Any] = {"qpy": np.frombuffer(b"QISKIT\x0d", dtype=np.uint8)}
    metadata = value.description.model_dump(mode="json")
    if mode == "arrays":
        arrays["extra"] = np.zeros(1)
    elif mode == "dtype":
        arrays["qpy"] = arrays["qpy"].astype(float)
    elif mode == "shape":
        arrays["qpy"] = arrays["qpy"].reshape(1, -1)
    elif mode == "size":
        arrays["qpy"] = arrays["qpy"][:6]
    elif mode == "header":
        arrays["qpy"] = np.zeros(7, dtype=np.uint8)
    elif mode == "metadata":
        metadata["root"] = -1
    path = write_bundle(
        tmp_path / "bad.json",
        kind="other" if mode == "kind" else "bound_circuit",
        arrays=arrays,
        metadata=metadata,
    )
    with pytest.raises(OutputParseError, match="invalid bound circuit bundle"):
        validate_circuit_bundle(read_bundle(path))


@pytest.mark.parametrize("mode", ["empty", "multiple", "unbound", "measured", "width"])
def test_reader_independently_rejects_qpy_content_mismatches(tmp_path, retained, mode):
    """Actual QPY must contain exactly the declared bound preparation."""
    value, _original, _bindings = retained
    circuits = [value.circuit]
    if mode == "empty":
        circuits = []
    elif mode == "multiple":
        circuits *= 2
    elif mode == "unbound":
        circuit = QuantumCircuit(value.description.num_qubits)
        circuit.ry(Parameter("unbound"), 0)
        circuits = [circuit]
    elif mode == "measured":
        circuits[0] = value.circuit.copy()
        circuits[0].measure_all()
    else:
        circuits = [QuantumCircuit(value.description.num_qubits + 1)]
    path = _raw_qpy_bundle(tmp_path / "invalid.json", circuits, value.description)
    with pytest.raises(OutputParseError, match=r"exactly one|differ from"):
        load_circuit(path)


@pytest.mark.parametrize("payload", [b"QISKIT\x0d", b"QISKIT\x0d" + b"\0" * 100])
def test_reader_normalizes_malformed_qpy_exceptions(tmp_path, retained, payload):
    """Corrupt binary data is a parse failure, including provider struct/type decoding errors."""
    value, _circuit, _bindings = retained
    path = write_bundle(
        tmp_path / "truncated.json",
        kind="bound_circuit",
        arrays={"qpy": np.frombuffer(payload, dtype=np.uint8)},
        metadata=value.description.model_dump(mode="json"),
    )
    with pytest.raises(OutputParseError, match="invalid circuit QPY"):
        load_circuit(path)


def test_export_and_import_obey_byte_limits_and_payload_digest(tmp_path, retained):
    """Capacity checks happen during QPY writes; corruption invalidates existing artifacts."""
    value, _circuit, _bindings = retained
    path = tmp_path / "state.json"
    with pytest.raises(ConfigError, match="exceeds max_bytes"):
        save_circuit(path, value, max_bytes=6)
    assert not path.exists()
    save_circuit(path, value)
    with pytest.raises(OutputParseError, match="exceed max_bytes"):
        load_circuit(path, max_bytes=6)
    dependencies = circuit_input_dependencies(path)
    assert set(dependencies) == {"payload"}
    dependencies["payload"].write_bytes(b"damaged")
    with pytest.raises(OutputParseError, match="digest mismatch"):
        load_circuit(path)
    with pytest.raises(ConfigError, match="cannot read circuit input"):
        circuit_input_dependencies(tmp_path / "absent.qpy")
    raw = tmp_path / "raw.qpy"
    with raw.open("wb") as stream:
        qpy.dump(value.circuit, stream)
    assert circuit_input_dependencies(raw) == {}


def test_output_validation_and_dependency_discovery_do_not_import_optional_sdks(tmp_path, retained):
    """Cache/rebuild validation works in a plain orchestrator without Qiskit or providers."""
    value, _circuit, _bindings = retained
    path = save_circuit(tmp_path / "state.json", value)
    script = """
import importlib.abc
import sys
from pathlib import Path
class RejectSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'qiskit', 'qiskit_nature', 'qiskit_ibm_runtime', 'ffsim'}:
            raise AssertionError('SDK import during local validation: ' + fullname)
sys.meta_path.insert(0, RejectSDK())
from chemrefine.engines.qiskit.bundles import read_bundle
from chemrefine.engines.qiskit.circuit_io import validate_circuit_bundle, circuit_input_dependencies
path = Path(sys.argv[1])
assert validate_circuit_bundle(read_bundle(path)).root == 1
assert set(circuit_input_dependencies(path)) == {'payload'}
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(path)], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr

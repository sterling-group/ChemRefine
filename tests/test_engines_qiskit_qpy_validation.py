"""Check QPY structure independently of optional SDK deserialization."""

from __future__ import annotations

import base64
import io
import struct
import subprocess
import sys

import numpy as np
import pytest

from chemrefine.engines.qiskit.qpy_validation import validate_qpy_payload

# Actual Qiskit 1.4.0 (v10/12) and 2.5.2 (v13/17) output for
# QuantumCircuit(1, name="qpy-validation").
_V10 = (
    "UUlTS0lUCgEEAAAAAAAAAAABZXEADmYACAAAAAEAAAAAAAAAAAAAAAIAAAABAAAAAAAAAABxcHkt"
    "dmFsaWRhdGlvbgAAAAAAAAAAe31xAQAAAAEAAQFxAAAAAAAAAAAAAAAAAAAAAAAAAP//////////"
    "/////wAAAAAAAAAA"
)
_V12 = (
    "UUlTS0lUDAEEAAAAAAAAAAABZXEADmYACAAAAAEAAAAAAAAAAAAAAAIAAAABAAAAAAAAAAAAAAAA"
    "cXB5LXZhbGlkYXRpb24AAAAAAAAAAHt9cQEAAAABAAEBcQAAAAAAAAAAAAAAAAAAAAAAAAD/////"
    "//////////8AAAAAAAAAAA=="
)
_V13 = (
    "UUlTS0lUDQIFAgAAAAAAAAABcHEADmYACAAAAAEAAAAAAAAAAAAAAAIAAAABAAAAAAAAAAAAAAAA"
    "cXB5LXZhbGlkYXRpb24AAAAAAAAAAHt9cQEAAAABAAEBcQAAAAAAAAAAAAAAAAAAAAAAAAD/////"
    "//////////8AAAAAAAAAAA=="
)
_V17 = (
    "UUlTS0lUEQIFAgAAAAAAAAABcHEAAAAAAAAAHAAOZgAIAAAAAQAAAAAAAAAAAAAAAgAAAAEAAAAA"
    "AAAAAAAAAABxcHktdmFsaWRhdGlvbgAAAAAAAAAAe31xAQAAAAEAAQFxAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAD///////////////8AAAAAAAAAAA=="
)


def _array(payload: bytes | bytearray) -> np.ndarray:
    """Use immutable byte storage to exercise the actual NPZ payload shape."""
    return np.frombuffer(payload, dtype=np.uint8)


def _payload(version: int = 13) -> bytearray:
    """Return a mutable copy of one independently generated reference stream."""
    return bytearray(base64.b64decode({10: _V10, 12: _V12, 13: _V13, 17: _V17}[version]))


def _edit(payload: bytearray, offset: int, format_: str, value) -> bytearray:
    """Modify a declared field without removing the surrounding byte stream."""
    struct.pack_into(format_, payload, offset, value)
    return payload


@pytest.mark.parametrize("version", [10, 12, 13, 17])
def test_actual_serialized_reference_has_correct_dimensions(version):
    """Both sequential and indexed fixtures accept matching structural constraints."""
    payload = _array(_payload(version))
    validate_qpy_payload(payload, circuits=1, num_qubits=1, num_clbits=0)
    validate_qpy_payload(payload, circuits=1, versions=(10, 12, 13, 17))
    # A strided uint8 array must not depend on memoryview contiguity.
    strided = np.repeat(payload, 2)[::2]
    validate_qpy_payload(strided, circuits=1)


@pytest.mark.parametrize("length", [0, 6, 7, 18, 19, 20, 56])
def test_file_and_header_only_truncations_fail(length):
    """Magic alone and even a complete file header cannot establish a circuit."""
    with pytest.raises(ValueError, match="QPY structure"):
        validate_qpy_payload(_array(_payload()[:length]), circuits=1)


@pytest.mark.parametrize("payload", [[], np.ones(5, dtype=np.int8), np.ones((2, 2), np.uint8)])
def test_byte_array_contract(payload):
    """Object types, signed bytes and non-vector payloads are rejected consistently."""
    with pytest.raises(ValueError, match="uint8"):
        validate_qpy_payload(payload, circuits=1)


@pytest.mark.parametrize("circuits", [0, -1, True, 1.0])
def test_invalid_requested_count(circuits):
    """Validation cannot accidentally authorize empty streams through caller mistakes."""
    with pytest.raises(ValueError, match="positive"):
        validate_qpy_payload(_array(_payload()), circuits=circuits)


@pytest.mark.parametrize(
    ("offset", "format_", "value", "message"),
    [
        (0, "c", b"X", "magic"),
        (6, "B", 9, "unsupported format"),
        (6, "B", 255, "unsupported format"),
        (10, "Q", 0, "count disagrees"),
        (10, "Q", 2, "count disagrees"),
        (19, "c", b"s", "program kind"),
        (22, "c", b"?", "global-phase type"),
        (23, "H", 1, "scalar global-phase length"),
        (20, "H", 65535, "sections exceed"),
        (33, "Q", 2**63, "sections exceed"),
        (41, "I", 2**31, "sections exceed"),
        (45, "Q", 2**63, "sections exceed"),
        (53, "I", 2**31, "sections exceed"),
        (81, "c", b"?", "register kind"),
        (83, "I", 2**31, "truncated register"),
        (87, "H", 65535, "truncated register"),
        (91, "q", 1, "bit index"),
        (91, "q", -2, "bit index"),
    ],
)
def test_corrupt_declared_fields_fail(offset, format_, value, message):
    """Valid outer bytes cannot excuse unsupported headers and impossible body lengths."""
    with pytest.raises(ValueError, match=message):
        validate_qpy_payload(_array(_edit(_payload(), offset, ">" + format_, value)), circuits=1)


def test_symbolic_global_phase_requires_nonempty_payload():
    """Parameter phases have variable lengths and cannot use scalar-only assumptions."""
    payload = _edit(_payload(), 22, "c", b"e")
    validate_qpy_payload(_array(payload), circuits=1)
    _edit(payload, 23, ">H", 0)
    with pytest.raises(ValueError, match="empty symbolic"):
        validate_qpy_payload(_array(payload), circuits=1)


@pytest.mark.parametrize("field", ["num_qubits", "num_clbits"])
def test_circuit_dimensions_checked(field):
    """Molecular metadata cannot silently change logical or measurement widths."""
    with pytest.raises(ValueError, match="count disagrees"):
        validate_qpy_payload(_array(_payload()), circuits=1, **{field: 2})


def test_explicit_version_cap_and_unused_symbolic_encoding():
    """Version 13 no longer gives the old symbolic encoding field meaning."""
    with pytest.raises(ValueError, match="unsupported"):
        validate_qpy_payload(_array(_payload()), circuits=1, versions=(17,))
    payload = _edit(_payload(), 18, "c", b"?")
    validate_qpy_payload(_array(payload), circuits=1)


@pytest.mark.parametrize("version", [10, 12])
def test_legacy_symbolic_encoding_is_meaningful(version):
    """Legacy symbolic payloads require one of the two published serialization tags."""
    payload = _edit(_payload(version), 18, "c", b"?")
    with pytest.raises(ValueError, match="symbolic encoding"):
        validate_qpy_payload(_array(payload), circuits=1)


def _indexed_pair() -> bytearray:
    """Combine two real circuit bodies with a correctly relocated v17 start table."""
    source = _payload(17)
    body = source[28:]
    header = _edit(source[:20], 10, ">Q", 2)
    return header + struct.pack(">QQ", 36, 36 + len(body)) + body + body


def test_indexed_multiple_circuits_and_later_dimensions():
    """Every addressable circuit is checked, including non-first program metadata."""
    payload = _indexed_pair()
    validate_qpy_payload(_array(payload), circuits=2, num_qubits=1, num_clbits=0)
    second = struct.unpack_from(">Q", payload, 28)[0]
    _edit(payload, second + 5, ">I", 3)
    with pytest.raises(ValueError, match="qubit count"):
        validate_qpy_payload(_array(payload), circuits=2, num_qubits=1)


@pytest.mark.parametrize(
    ("offset", "value", "message"),
    [
        (20, 20, "offsets must start"),
        (20, 37, "offsets must start"),
        (28, 36, "offset exceeds"),
        (28, 2**63, "offset exceeds"),
        (28, 40, "truncated circuit header"),
        (28, 76, "sections exceed"),
    ],
)
def test_invalid_offset_table(offset, value, message):
    """Offsets cannot alias the table, overlap prefixes or point outside the payload."""
    with pytest.raises(ValueError, match=message):
        validate_qpy_payload(_array(_edit(_indexed_pair(), offset, ">Q", value)), circuits=2)


def test_sequential_multiple_circuits_have_bounded_available_prefix():
    """Pre-v16 streams keep their supported sequential format without fake random access."""
    source = _payload()
    payload = _edit(source[:20], 10, ">Q", 2) + source[20:] + source[20:]
    validate_qpy_payload(_array(payload), circuits=2, num_qubits=1)
    with pytest.raises(ValueError, match="insufficient bytes"):
        validate_qpy_payload(_array(payload[:100]), circuits=2)


def test_validation_never_imports_qiskit():
    """The recovery path remains usable before provider environments are provisioned."""
    code = f"""
import base64, builtins, numpy as np
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'qiskit' or name.startswith('qiskit.'):
        raise AssertionError('Qiskit imported during structural validation')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from chemrefine.engines.qiskit.qpy_validation import validate_qpy_payload
validate_qpy_payload(np.frombuffer(base64.b64decode('{_V13}'), dtype=np.uint8), circuits=1)
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_all_locally_supported_real_writer_versions():
    """Run unchanged against both floor and current SDKs, including multi-program streams."""
    qiskit = pytest.importorskip("qiskit")
    from qiskit import qpy
    from qiskit.circuit import Parameter

    for version in range(qpy.QPY_COMPATIBILITY_VERSION, qpy.QPY_VERSION + 1):
        circuit = qiskit.QuantumCircuit(2, 1, name="nontrivial", metadata={"fixture": version})
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.rz(Parameter("angle"), 1)
        circuit.measure(0, 0)
        for count in (1, 2):
            stream = io.BytesIO()
            qpy.dump([circuit] * count, stream, version=version)
            payload = _array(stream.getvalue())
            validate_qpy_payload(payload, circuits=count, num_qubits=2, num_clbits=1)
            assert len(qpy.load(io.BytesIO(payload.tobytes()))) == count
            if version < 13:
                malformed = _edit(bytearray(stream.getvalue()), 18, "c", b"?")
                with pytest.raises(ValueError, match="symbolic encoding"):
                    validate_qpy_payload(_array(malformed), circuits=count)

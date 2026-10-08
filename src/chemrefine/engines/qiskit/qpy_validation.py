"""Bounded, SDK-free structural checks for supported QPY circuit containers.

The layouts follow the published Qiskit QPY format, versions 10 through 17.
These checks do not decode instruction parameters, custom operations, annotations,
or circuit semantics. Worker-side ``qpy.load`` remains necessary before execution.
"""

from __future__ import annotations

import struct
from typing import Any

import numpy as np
from numpy.typing import NDArray

_SUPPORTED_VERSIONS = tuple(range(10, 18))
_FILE = struct.Struct("!6sBBBBQc")
_HEADER = struct.Struct("!HcHIIQIQ")
_HEADER_V12 = struct.Struct("!HcHIIQIQI")
_REGISTER = struct.Struct("!c?IH?")
_OFFSET = struct.Struct("!Q")


def _require(condition: bool, message: str) -> None:
    """Normalize malformed container and caller constraints to one exception type."""
    if not condition:
        raise ValueError(f"invalid QPY structure: {message}")


def _minimum_body(version: int) -> int:
    """Count mandatory custom-operation, calibration, layout and annotation headers."""
    return 8 + 2 + 21 + (4 if version >= 15 else 0)


def _circuit_header(
    data: memoryview,
    start: int,
    stop: int,
    version: int,
    num_qubits: int | None,
    num_clbits: int | None,
) -> None:
    """Check one addressable circuit prefix without following instruction definitions."""
    header = _HEADER_V12 if version >= 12 else _HEADER
    _require(stop - start >= header.size, "truncated circuit header")
    fields = header.unpack_from(data, start)
    name_size, phase_type, phase_size, qubits, clbits, metadata_size, registers, instructions = (
        fields[:8]
    )
    variables = fields[8] if version >= 12 else 0
    _require(num_qubits is None or qubits == num_qubits, "circuit qubit count disagrees")
    _require(num_clbits is None or clbits == num_clbits, "circuit classical-bit count disagrees")
    _require(phase_type in (b"i", b"f", b"p", b"v", b"e", b"n"), "invalid global-phase type")
    if phase_type in (b"i", b"f"):
        _require(phase_size == 8, "invalid scalar global-phase length")
    else:
        _require(phase_size > 0, "empty symbolic global phase")
    cursor = start + header.size + name_size + phase_size + metadata_size
    # A declaration has a 19-byte header and at least a one-byte type. Every
    # instruction has a 33-byte fixed header in all supported versions.
    tail_minimum = variables * 20 + instructions * 33 + _minimum_body(version)
    _require(
        cursor + registers * _REGISTER.size + tail_minimum <= stop,
        "declared circuit sections exceed payload bounds",
    )
    for register_index in range(registers):
        kind, _, size, register_name_size, _ = _REGISTER.unpack_from(data, cursor)
        _require(kind in (b"q", b"c"), "invalid register kind")
        cursor += _REGISTER.size + register_name_size
        _require(
            cursor + size * 8 + (registers - register_index - 1) * _REGISTER.size + tail_minimum
            <= stop,
            "truncated register payload",
        )
        width = qubits if kind == b"q" else clbits
        for index in range(size):
            bit = struct.unpack_from("!q", data, cursor + index * 8)[0]
            _require(-1 <= bit < width, "register bit index exceeds circuit dimensions")
        cursor += size * 8


def validate_qpy_payload(
    payload: NDArray[Any],
    *,
    circuits: int,
    num_qubits: int | None = None,
    num_clbits: int | None = None,
    versions: tuple[int, ...] | None = None,
) -> None:
    """Validate container headers, addressable dimensions and declared byte bounds.

    ``versions`` can restrict the known format versions, for example to version
    13 for portable molecular exports. Formats 16 and 17 have an offset table,
    allowing every circuit prefix to be checked independently. Earlier formats
    expose only the first circuit prefix without decoding variable-length
    instructions; the remaining declared circuits receive a total-size lower
    bound. This function makes no claim to full QPY deserialization or semantic
    validation and never imports Qiskit or calls a provider.
    """
    _require(
        isinstance(payload, np.ndarray) and payload.dtype == np.uint8 and payload.ndim == 1,
        "payload must be a one-dimensional uint8 array",
    )
    _require(
        isinstance(circuits, int) and not isinstance(circuits, bool) and circuits > 0,
        "expected circuit count must be positive",
    )
    _require(payload.size >= _FILE.size + 1, "truncated file header or program type")
    data = memoryview(np.ascontiguousarray(payload))
    magic, version, _, _, _, count, encoding = _FILE.unpack_from(data)
    _require(magic == b"QISKIT", "invalid magic")
    _require(
        version in _SUPPORTED_VERSIONS and (versions is None or version in versions),
        "unsupported format version",
    )
    _require(count == circuits, "circuit count disagrees")
    if version < 13:
        _require(encoding in (b"p", b"e"), "invalid symbolic encoding")
    _require(data[_FILE.size] == ord("q"), "program kind must be circuit")
    start = _FILE.size + 1
    header_size = _HEADER_V12.size if version >= 12 else _HEADER.size
    circuit_minimum = header_size + 1 + _minimum_body(version)
    table_size = count * _OFFSET.size if version >= 16 else 0
    _require(
        start + table_size + count * circuit_minimum <= len(data),
        "insufficient bytes for declared circuit count",
    )
    if version >= 16:
        table_start = start
        start += table_size
        for index in range(count):
            offset = _OFFSET.unpack_from(data, table_start + index * _OFFSET.size)[0]
            _require(offset == start, "circuit offsets must start after the table and increase")
            stop = (
                _OFFSET.unpack_from(data, table_start + (index + 1) * _OFFSET.size)[0]
                if index + 1 < count
                else len(data)
            )
            _require(start < stop <= len(data), "circuit offset exceeds payload bounds")
            _circuit_header(data, start, stop, version, num_qubits, num_clbits)
            start = stop
    else:
        _circuit_header(
            data,
            start,
            len(data) - (count - 1) * circuit_minimum,
            version,
            num_qubits,
            num_clbits,
        )

"""SDK-free reconstruction of certified Pauli measurement artifacts.

The certificate proves an algebraically valid measurement frame, not that a
provider physically executed it. Counts remain empirical acquisition records.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from chemrefine.engines.qiskit.bundles import QuantumBundle

BIT_ORDER = "Qiskit display order, packed big-endian, trailing zero padding"
TABLEAU_CONVENTION = (
    "signed_C_P_Cdagger; rows=X0..Xn-1,Z0..Zn-1; columns=x0..xn-1,z0..zn-1,negative"
)


def _require(condition: Any, message: str) -> None:
    """Reject inconsistent scientific records through the outer artifact contract."""
    if not condition:
        raise ValueError(message)


def _number(value: Any) -> float:
    """Require finite real metadata, excluding booleans and string coercion."""
    _require(
        type(value) in (float, int) and -sys.float_info.max <= value <= sys.float_info.max,
        "invalid measurement number",
    )
    return float(value)


def _equal(actual: Any, expected: Any, name: str) -> None:
    """Allow only floating summation noise when comparing reconstructed statistics."""
    _require(
        np.allclose(actual, expected, rtol=1e-10, atol=1e-12),
        f"measurement {name} disagrees with production counts",
    )


def _array(
    bundle: QuantumBundle, name: str, shape: tuple[int | None, ...], kinds: str
) -> NDArray[Any]:
    """Require an exact rank and numerical storage class before allocating workspace."""
    value = bundle.arrays[name]
    _require(
        value.dtype.kind in kinds
        and value.ndim == len(shape)
        and all(
            expected is None or actual == expected
            for actual, expected in zip(value.shape, shape, strict=True)
        ),
        "invalid measurement array dimensions or dtype",
    )
    return value


def _pauli(label: str) -> tuple[int, int]:
    """Encode a display-order Pauli as little-endian X and Z binary integers."""
    x = z = 0
    for index, axis in enumerate(reversed(label)):
        x |= (axis in "XY") << index
        z |= (axis in "ZY") << index
    return x, z


def _frame(tableau: NDArray[Any], width: int) -> list[tuple[int, int, int]]:
    """Validate a signed symplectic frame and return canonical i-phase Pauli rows."""
    _require(tableau.dtype == np.dtype("uint8") and np.all(tableau <= 1), "invalid binary tableau")
    rows = []
    for row in tableau:
        x = sum(int(bit) << index for index, bit in enumerate(row[:width]))
        z = sum(int(bit) << index for index, bit in enumerate(row[width:-1]))
        rows.append((x, z, ((x & z).bit_count() + 2 * int(row[-1])) % 4))
    for i, (x, z, _) in enumerate(rows):
        for j in range(i + 1, len(rows)):
            other_x, other_z, _ = rows[j]
            parity = ((x & other_z).bit_count() + (z & other_x).bit_count()) % 2
            _require(parity == (j == i + width), "measurement tableau is not symplectic")
    return rows


def _image(label: str, rows: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    """Conjugate a Pauli, retaining phases from ordered X/Z generator products."""
    input_x, input_z = _pauli(label)
    width = len(label)
    selected = input_x | (input_z << width)
    x = z = 0
    phase = (input_x & input_z).bit_count() % 4
    for index, (other_x, other_z, other_phase) in enumerate(rows):
        if selected >> index & 1:
            phase = (phase + other_phase + 2 * (z & other_x).bit_count()) % 4
            x ^= other_x
            z ^= other_z
    return x, z, phase


def _partition(labels: list[str], grouping: str) -> None:
    """Check the configured partition domain without requiring one SDK's coloring."""
    _require(grouping != "none" or len(labels) == 1, "ungrouped measurement has multiple terms")
    if grouping == "qwc":
        _require(
            all(len(set(column) - {"I"}) <= 1 for column in zip(*labels, strict=True)),
            "measurement group is not qubit-wise commuting",
        )


def _statistics(
    bits: NDArray[Any], counts: NDArray[Any], masks: list[int], signs: list[int], width: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Reconstruct first and second moments using one outcome vector at a time."""
    number = len(masks)
    first = np.zeros(number)
    second = np.zeros((number, number))
    shots = sum(map(int, counts))
    for packed, count in zip(bits, counts, strict=True):
        value = int.from_bytes(packed.tobytes(), "big") >> ((-width) % 8)
        outcomes = np.asarray(
            [
                sign * (1 - 2 * ((value & mask).bit_count() % 2))
                for mask, sign in zip(masks, signs, strict=True)
            ],
            dtype=float,
        )
        first += int(count) * outcomes
        second += int(count) * np.outer(outcomes, outcomes)
    means = first / shots
    covariance = (second - shots * np.outer(means, means)) / (shots - 1)
    return means, covariance


def validate_measurement(bundle: QuantumBundle, options: Mapping[str, Any]) -> None:
    """Certify grouped Pauli images and independently reconstruct their joint statistics."""
    metadata, controls = bundle.metadata, options["measurement"]
    _require(
        type(metadata.get("measurement_format_version")) is int
        and metadata["measurement_format_version"] == 2,
        "measurement format requires version 2 certificates; regenerate legacy output",
    )
    for name, expected_type in (("num_qubits", int), ("groups", list), ("shots", int)):
        _require(type(metadata[name]) is expected_type, f"invalid measurement {name}")
    width = len(next(iter(options["observable"])))
    _require(metadata["num_qubits"] == width, "observable register width disagrees")
    _require(metadata["bit_order"] == BIT_ORDER, "invalid measurement bit order")
    _require(
        metadata["clifford_tableau_convention"] == TABLEAU_CONVENTION, "invalid tableau convention"
    )
    _require(metadata["units"] == "observable_coefficient_units", "invalid measurement units")
    grouping = controls["grouping"]
    _require(metadata["grouping"] == grouping, "configured measurement grouping disagrees")
    _require(metadata["pilot_policy"] == "independent_allocation_only", "invalid pilot policy")
    expected = {
        key: value for key, value in options["observable"].items() if value and key != "I" * width
    }
    _require(
        max(1, sum(value != 0 for value in options["observable"].values()))
        <= controls["max_terms"],
        "observable exceeds measurement max_terms",
    )
    resident = sum(value.nbytes for value in bundle.arrays.values())
    largest = max((len(group["paulis"]) for group in metadata["groups"]), default=0)
    workspace = 32 * largest**2 + 64 * largest + 4 * width * (width + 64)
    _require(
        resident + workspace <= controls["max_memory_mb"] * 1024**2,
        "measurement reconstruction exceeds max_memory_mb",
    )
    total = 0
    labels = []
    expectation = float(options["observable"].get("I" * width, 0))
    uncertainty = 0.0
    for group in metadata["groups"]:
        _require(type(group["paulis"]) is list and group["paulis"], "invalid measurement labels")
        number = len(group["paulis"])
        _require(
            all(label in expected for label in group["paulis"]), "unconfigured measurement label"
        )
        for name in ("coefficients", "z_masks", "signs"):
            _require(
                type(group[name]) is list and len(group[name]) == number,
                "invalid measurement group",
            )
        coefficients = np.asarray([_number(value) for value in group["coefficients"]])
        _require(
            all(
                value == expected[label]
                for value, label in zip(coefficients, group["paulis"], strict=True)
            ),
            "measurement coefficients disagree with observable",
        )
        masks, signs = group["z_masks"], group["signs"]
        _require(
            all(type(mask) is int and 0 < mask < (1 << width) for mask in masks), "invalid Z masks"
        )
        _require(
            all(type(sign) is int and sign in (-1, 1) for sign in signs), "invalid Pauli signs"
        )
        _partition(group["paulis"], grouping)
        tableau = _array(bundle, group["clifford_tableau_array"], (2 * width, 2 * width + 1), "u")
        rows = _frame(tableau, width)
        for label, mask, sign in zip(group["paulis"], masks, signs, strict=True):
            _require(
                _image(label, rows) == (0, mask, 0 if sign == 1 else 2),
                "measurement signed Z image disagrees with Clifford certificate",
            )
        covariance = _array(bundle, group["covariance_array"], (number, number), "f")
        counts = _array(bundle, group["counts_array"], (None,), "iu")
        bits = _array(bundle, group["bitstrings_array"], (len(counts), (width + 7) // 8), "u")
        _require(
            bits.dtype == np.dtype("uint8") and np.all(counts > 0), "invalid physical count storage"
        )
        _require(np.all((bits[:, -1] & ((1 << ((-width) % 8)) - 1)) == 0), "nonzero bit padding")
        shots = group["shots"]
        _require(
            type(shots) is int and shots >= 2 and sum(map(int, counts)) == shots,
            "invalid production shots",
        )
        _require(
            type(group["pilot_shots"]) is int and group["pilot_shots"] == controls["pilot_shots"],
            "measurement pilot allocation disagrees",
        )
        _require(_number(group["pilot_variance"]) >= 0, "negative pilot variance")
        means, reconstructed = _statistics(bits, counts, masks, signs, width)
        _equal(covariance, reconstructed, "covariance")
        mean = float(coefficients @ means)
        _equal(_number(group["expectation"]), mean, "group expectation")
        expectation += mean
        uncertainty += max(0.0, float(coefficients @ reconstructed @ coefficients)) / shots
        total += shots + group["pilot_shots"]
        labels.extend(group["paulis"])
    _require(sorted(labels) == sorted(expected), "measurement groups do not cover the observable")
    _require(
        total == metadata["shots"] == (controls["shots"] if expected else 0),
        "measurement budget disagrees",
    )
    _equal(_number(metadata["expectation"]), expectation, "expectation")
    _require(_number(metadata["standard_error"]) >= 0, "negative measurement uncertainty")
    _equal(metadata["standard_error"], math.sqrt(uncertainty), "standard error")

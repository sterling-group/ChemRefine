"""Measurement recovery verifies scientific contents independently of SDK acquisition."""

from __future__ import annotations

import copy
import json
import subprocess
import sys

import numpy as np
import pytest

from chemrefine.engines.qiskit.bundles import read_bundle, write_bundle
from chemrefine.engines.qiskit.experiment_outputs import validate_experiment_output
from chemrefine.errors import OutputParseError


def _case():
    """Use a hand-derived Z measurement, not a provider-produced expected answer."""
    from test_engines_qiskit_experiment_outputs import _case as fixture

    return fixture("pauli_measurement")


def _validate(tmp_path, case):
    """Publish fresh integrity digests so failures specifically test scientific consistency."""
    kind, arrays, metadata, options = case
    path = write_bundle(tmp_path / "measurement.json", kind=kind, arrays=arrays, metadata=metadata)
    bundle = read_bundle(path)
    validate_experiment_output("pauli_measurement", bundle, options)
    return path


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("measurement_format_version", 1),
        ("measurement_format_version", True),
        ("measurement_format_version", None),
        ("num_qubits", True),
        ("num_qubits", 2),
        ("groups", {}),
        ("shots", True),
        ("shots", 1),
        ("bit_order", "little endian"),
        ("clifford_tableau_convention", "unsigned"),
        ("units", "hartree"),
        ("grouping", "none"),
        ("pilot_policy", "reuse_pilots"),
        ("expectation", 99.0),
        ("expectation", True),
        ("expectation", 10**400),
        ("standard_error", -1.0),
        ("standard_error", 0.5),
    ],
)
def test_reject_inconsistent_top_level_metadata(tmp_path, key, value):
    """Critical configuration, conventions and totals cannot be changed behind valid hashes."""
    case = _case()
    case[2][key] = value
    with pytest.raises(OutputParseError):
        _validate(tmp_path, case)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("paulis", []),
        ("paulis", ["X"]),
        ("coefficients", [2.0]),
        ("coefficients", [True]),
        ("coefficients", "1"),
        ("coefficients", []),
        ("z_masks", [0]),
        ("z_masks", [-1]),
        ("z_masks", [2]),
        ("z_masks", [True]),
        ("signs", [0]),
        ("signs", [-1]),
        ("signs", [1.0]),
        ("signs", [True]),
        ("expectation", 99.0),
        ("expectation", "1"),
        ("shots", 1),
        ("shots", 3),
        ("shots", True),
        ("pilot_shots", 3),
        ("pilot_shots", True),
        ("pilot_variance", -1.0),
    ],
)
def test_reject_inconsistent_group_metadata(tmp_path, key, value):
    """Signs, masks, coefficients and production denominators are scientific contracts."""
    case = _case()
    case[2]["groups"][0][key] = value
    with pytest.raises(OutputParseError):
        _validate(tmp_path, case)


def test_self_consistent_wrong_coefficients_do_not_replace_configured_observable(tmp_path):
    """Coordinated result edits cannot change which scientific observable was requested."""
    case = _case()
    case[2]["groups"][0].update(coefficients=[2.0], expectation=2.0)
    case[2]["expectation"] = 2.0
    with pytest.raises(OutputParseError, match="coefficients"):
        _validate(tmp_path, case)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("covariance", np.array([[-1.0]])),
        ("covariance", np.array([[0.1]])),
        ("covariance", np.array([0.0])),
        ("covariance", np.array([[0]])),
        ("counts", np.array([1, 1], np.uint64)),
        ("counts", np.array([0], np.uint64)),
        ("bitstrings", np.array([[128]], np.uint8)),
        ("bitstrings", np.array([[1]], np.uint8)),
        ("bitstrings", np.array([[0]], np.uint16)),
        ("tableau", np.array([[1, 0, 0], [0, 1, 0]], np.uint16)),
        ("tableau", np.array([[2, 0, 0], [0, 1, 0]], np.uint8)),
        ("tableau", np.array([[1, 0, 0], [1, 0, 0]], np.uint8)),
        ("tableau", np.array([[0, 1, 0], [1, 0, 0]], np.uint8)),
        ("tableau", np.array([[1, 0, 0], [0, 1, 1]], np.uint8)),
    ],
)
def test_reject_regenerated_payloads_with_inconsistent_science(tmp_path, name, value):
    """Binary/symplectic certificates and statistics are checked beyond integrity digests."""
    case = _case()
    case[1][name] = value
    with pytest.raises(OutputParseError):
        _validate(tmp_path, case)


def test_limits_and_duplicate_coverage_precede_reconstruction(tmp_path):
    """Configured limits guard workspace and duplicated groups cannot inflate the observable."""
    case = _case()
    case[3]["measurement"]["max_terms"] = 1
    case[3]["observable"]["I"] = 1.0
    with pytest.raises(OutputParseError, match="max_terms"):
        _validate(tmp_path, case)
    case = _case()
    case[3]["measurement"]["max_memory_mb"] = 1
    case[1]["bounded_padding"] = np.zeros(1024**2, dtype=np.uint8)
    with pytest.raises(OutputParseError, match="max_memory_mb"):
        _validate(tmp_path, case)
    case = _case()
    case[3]["measurement"]["max_memory_mb"] = 1
    case[1]["covariance"] = np.zeros((256, 256))
    case[2]["groups"][0]["paulis"] = ["Z"] * 256
    with pytest.raises(OutputParseError, match="max_memory_mb"):
        _validate(tmp_path, case)
    case = _case()
    case[2]["groups"].append(copy.deepcopy(case[2]["groups"][0]))
    case[2]["shots"] = case[3]["measurement"]["shots"] = 8
    with pytest.raises(OutputParseError, match="cover the observable"):
        _validate(tmp_path, case)


def test_alternative_valid_clifford_is_accepted_without_optional_sdks(tmp_path):
    """X changes the Z sign and readout bit; that valid frame must not be canonicalized away."""
    case = _case()
    case[2]["groups"][0]["signs"] = [-1]
    case[1]["tableau"][1, -1] = 1
    case[1]["bitstrings"][0, 0] = 128
    path = _validate(tmp_path, case)
    code = """
import importlib.abc, json, sys
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'qiskit','qiskit_nature','qiskit_algorithms','qiskit_aer'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from chemrefine.engines.qiskit.bundles import read_bundle
from chemrefine.engines.qiskit.experiment_outputs import validate_experiment_output
validate_experiment_output(
    'pauli_measurement', read_bundle(Path(sys.argv[1])), json.loads(sys.argv[2])
)
"""
    subprocess.run([sys.executable, "-c", code, str(path), json.dumps(case[3])], check=True)


@pytest.mark.parametrize("grouping", ["none", "qwc", "commuting"])
def test_actual_complex_state_counts_certify_and_reconstruct(tmp_path, grouping):
    """Actual acquisition agrees with explicit outcome products and independent exact means."""
    from qiskit import QuantumCircuit, qpy
    from qiskit.quantum_info import SparsePauliOp, Statevector

    from chemrefine.engines.qiskit.experiment_measurement import (
        MeasurementExperimentOptions,
        measurement_experiment,
    )
    from chemrefine.engines.qiskit.measurement import MeasurementOptions

    circuit = QuantumCircuit(2)
    circuit.ry(0.8, 0)
    circuit.rz(0.6, 0)
    circuit.rx(0.3, 1)
    circuit.cx(0, 1)
    path = tmp_path / "state.qpy"
    with path.open("wb") as stream:
        qpy.dump(circuit, stream, version=13)
    observable = {"XX": 0.7, "YY": -0.3, "ZZ": 0.4, "ZI": 0.6, "IZ": -0.2, "II": 0.8}
    options = MeasurementExperimentOptions(
        circuit_path=str(path),
        observable=observable,
        measurement=MeasurementOptions(grouping=grouping, shots=4096, pilot_shots=8, seed=23),
    )
    result = measurement_experiment(options=options, cores=1, device="cpu")
    _validate(
        tmp_path, (result.kind, result.arrays, result.metadata, options.model_dump(mode="json"))
    )
    total, variance = 0.8, 0.0
    for group in result.metadata["groups"]:
        bits = np.unpackbits(result.arrays[group["bitstrings_array"]], axis=1)[:, :2]
        counts = result.arrays[group["counts_array"]]
        coefficients = np.asarray(group["coefficients"])
        per_shot = []
        for bitstring, count in zip(bits, counts, strict=True):
            integer = int("".join(map(str, bitstring)), 2)
            values = [
                sign * (-1) ** (integer & mask).bit_count()
                for sign, mask in zip(group["signs"], group["z_masks"], strict=True)
            ]
            per_shot.extend([values] * int(count))
        samples = np.asarray(per_shot, dtype=float)
        covariance = np.atleast_2d(np.cov(samples, rowvar=False, ddof=1))
        np.testing.assert_allclose(result.arrays[group["covariance_array"]], covariance, atol=1e-12)
        total += float(coefficients @ samples.mean(axis=0))
        variance += float(coefficients @ covariance @ coefficients) / len(samples)
    assert result.metadata["expectation"] == pytest.approx(total)
    assert result.metadata["standard_error"] == pytest.approx(np.sqrt(variance))
    exact = Statevector(circuit).expectation_value(
        SparsePauliOp.from_list(list(observable.items()))
    )
    assert abs(total - exact.real) < 5 * np.sqrt(variance)


def test_group_partition_and_signed_relations_are_independent_of_statistics(tmp_path):
    """A self-consistent covariance cannot make noncommuting labels a valid joint group."""
    from qiskit.quantum_info import Clifford, SparsePauliOp

    from chemrefine.engines.qiskit.measurement import MeasurementOptions, measurement_groups

    case = _case()
    case[2]["num_qubits"] = 2
    group = case[2]["groups"][0]
    group.update(paulis=["ZI", "IZ"], coefficients=[1.0, 1.0], z_masks=[2, 1], signs=[1, 1])
    case[3]["observable"] = {"ZI": 1.0, "IZ": 1.0}
    case[1]["covariance"] = np.zeros((2, 2))
    case[1]["tableau"] = np.column_stack((np.eye(4, dtype=np.uint8), np.zeros(4, np.uint8)))
    case[2]["grouping"] = case[3]["measurement"]["grouping"] = "none"
    with pytest.raises(OutputParseError, match="ungrouped"):
        _validate(tmp_path, case)
    _, groups = measurement_groups(
        SparsePauliOp.from_list([("XX", 1.0), ("ZZ", 1.0)]),
        MeasurementOptions(grouping="commuting"),
    )
    group.update(
        paulis=list(groups[0].labels), z_masks=list(groups[0].z_masks), signs=list(groups[0].signs)
    )
    case[1]["tableau"] = Clifford(groups[0].circuit).tableau.astype(np.uint8)
    case[3]["observable"] = {"XX": 1.0, "ZZ": 1.0}
    case[2]["grouping"] = case[3]["measurement"]["grouping"] = "qwc"
    with pytest.raises(OutputParseError, match="qubit-wise"):
        _validate(tmp_path, case)


def test_constant_only_reports_keep_exact_constant_and_zero_error(tmp_path):
    """No pilots or imaginary production data are needed for identity-only observables."""
    case = _case()
    case[1].clear()
    case[2].update(groups=[], shots=0, expectation=-0.7, standard_error=0.0)
    case[3]["observable"] = {"I": -0.7, "X": 0.0}
    _validate(tmp_path, case)
    case[2]["expectation"] = -0.6
    with pytest.raises(OutputParseError, match="expectation"):
        _validate(tmp_path, case)


def test_signed_tableau_phase_algebra_matches_dense_conjugation():
    """Independent dense products check Y phases and entangling Clifford generator order."""
    from itertools import product

    from qiskit.quantum_info import Operator, Pauli, random_clifford

    from chemrefine.engines.qiskit.measurement_validation import _frame, _image

    for width in (1, 2, 3):
        clifford = random_clifford(width, seed=29 + width)
        matrix = Operator(clifford).data
        rows = _frame(clifford.tableau.astype(np.uint8), width)
        for axes in product("IXYZ", repeat=width):
            label = "".join(axes)
            x, z, phase = _image(label, rows)
            x_operator = Pauli("".join("X" if x >> i & 1 else "I" for i in reversed(range(width))))
            z_operator = Pauli("".join("Z" if z >> i & 1 else "I" for i in reversed(range(width))))
            np.testing.assert_allclose(
                matrix @ Pauli(label).to_matrix() @ matrix.conj().T,
                1j**phase * x_operator.to_matrix() @ z_operator.to_matrix(),
                atol=1e-12,
            )

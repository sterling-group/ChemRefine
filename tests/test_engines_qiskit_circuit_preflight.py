"""Predictable preparation-export failures precede solver and provider acquisition."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from chemrefine.engines.qiskit import assembly, circuit_io
from chemrefine.engines.qiskit.bundles import MAX_DESCRIPTOR_BYTES
from chemrefine.engines.qiskit.context import AnsatzArtifacts, EstimatorResource
from chemrefine.engines.qiskit.data import ElectronicStructureData
from chemrefine.engines.qiskit.mapping import map_problem
from chemrefine.engines.qiskit.options import QiskitOptions
from chemrefine.engines.qiskit.problem import prepare_problem
from chemrefine.engines.qiskit.workflow import run_problem
from chemrefine.errors import ConfigError

pytestmark = [
    pytest.mark.filterwarnings("ignore:.*:DeprecationWarning:qiskit.*"),
    pytest.mark.filterwarnings("ignore:.*:scipy.sparse.SparseEfficiencyWarning"),
]


@pytest.fixture
def h2():
    """Prepare an independent stored integral fixture without running an SCF engine."""
    raw = json.loads((Path(__file__).parent / "data/engines/qiskit/h2_integrals.json").read_text())
    return prepare_problem(ElectronicStructureData(**raw))


def _forbid_resources(monkeypatch):
    """Make any primitive or optimizer acquisition fail independently of export checks."""

    def forbidden(*_args, **_kwargs):
        """Expose acquisition before predictable input failure as a test failure."""
        pytest.fail("acquired a primitive or optimizer before circuit-export validation")

    for registry in (assembly.ESTIMATORS, assembly.SAMPLERS, assembly.OPTIMIZERS):
        monkeypatch.setattr(registry, "build", forbidden)


@pytest.mark.parametrize("algorithm", ["vqe", "adapt_vqe", "tetris_adapt", "ceo_adapt", "vqd"])
@pytest.mark.parametrize("invalid", ["provenance", "budget"])
def test_export_common_limits_fail_before_every_solver_provider(
    h2, monkeypatch, algorithm, invalid
):
    """All advertised export paths reject real oversized metadata and tiny payload budgets."""
    _forbid_resources(monkeypatch)
    if invalid == "provenance":
        h2 = replace(h2, provenance={"oversized": "x" * MAX_DESCRIPTOR_BYTES})
    options = {
        "algorithm": {"name": algorithm, "options": {"k": 1} if algorithm == "vqd" else {}},
        "ansatz": "ceo" if algorithm == "ceo_adapt" else "uccsd",
        "circuit_export": {"max_bytes": 1} if invalid == "budget" else {},
    }
    with pytest.raises(ConfigError, match=r"size limit|max_bytes"):
        run_problem(h2, options=options)


@pytest.mark.parametrize("algorithm", ["vqe", "vqd"])
def test_known_parameter_storage_is_checked_before_initialization(h2, monkeypatch, algorithm):
    """A fixed ansatz's actual parameter names are bounded before optimization resources."""
    _forbid_resources(monkeypatch)
    circuit = QuantumCircuit(4)
    circuit.ry(Parameter("p" * 4096), 0)
    monkeypatch.setattr(
        assembly.ANSATZE, "build", lambda *_args, **_kwargs: AnsatzArtifacts(circuit=circuit)
    )

    def forbidden(*_args, **_kwargs):
        """Require known parameter storage to fail before constructing initial vectors."""
        pytest.fail("initialized parameters before checking their export footprint")

    monkeypatch.setattr(assembly.INITIAL_POINTS, "build", forbidden)
    with pytest.raises(ConfigError, match="max_bytes"):
        run_problem(
            h2,
            options={
                "algorithm": {
                    "name": algorithm,
                    "options": {"k": 1} if algorithm == "vqd" else {},
                },
                "circuit_export": {"max_bytes": 2048},
            },
        )


@pytest.mark.parametrize("algorithm", ["adapt_vqe", "tetris_adapt", "ceo_adapt"])
def test_adaptive_preflight_checks_reference_instead_of_unused_fixed_ansatz(
    h2, monkeypatch, algorithm
):
    """A pool's optional full circuit does not falsely reject a bounded adaptive reference."""
    circuit = QuantumCircuit(4)
    circuit.ry(Parameter("unused" * 1024), 0)
    monkeypatch.setattr(
        assembly.ANSATZE,
        "build",
        lambda *_args, **_kwargs: AnsatzArtifacts(circuit=circuit, operator_pool=()),
    )
    monkeypatch.setattr(
        assembly.ESTIMATORS, "build", lambda *_args, **_kwargs: EstimatorResource(object())
    )
    original = circuit_io.preflight_circuit_export
    checked = []

    def capture(context, *, max_bytes, circuit=None):
        """Retain the exact logical preparation passed through the shared guard."""
        checked.append(circuit)
        original(context, max_bytes=max_bytes, circuit=circuit)

    monkeypatch.setattr(circuit_io, "preflight_circuit_export", capture)
    options = QiskitOptions.from_raw(
        {"algorithm": algorithm, "circuit_export": {"max_bytes": 2048}}
    )
    with assembly.assemble_components(map_problem(h2), options, defer_optimizer=True) as built:
        assert checked == [None, built.initial_state]
        assert built.initial_state.num_parameters == 0


def test_disabled_export_leaves_exact_solver_independent_of_export_limits(h2, monkeypatch):
    """Legacy exact calculations never call preparation-export checks or build primitives."""
    _forbid_resources(monkeypatch)

    def forbidden(*_args, **_kwargs):
        """Fail if an unrequested export imposes new restrictions on an exact calculation."""
        pytest.fail("export preflight ran without circuit_export")

    monkeypatch.setattr(circuit_io, "preflight_circuit_export", forbidden)
    result = run_problem(h2, options={"algorithm": "exact"})
    assert result.energy_hartree == pytest.approx(-1.1373060357534, abs=1e-10)
    assert not result.circuits

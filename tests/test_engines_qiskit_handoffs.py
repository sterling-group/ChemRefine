"""Molecular preparations hand off to scheduled experiments without losing conventions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from qiskit.quantum_info import SparsePauliOp, Statevector

from chemrefine.config import load_config
from chemrefine.engines.qiskit.bundles import read_bundle
from chemrefine.engines.qiskit.circuit_io import load_circuit
from chemrefine.engines.qiskit.data import ElectronicStructureData
from chemrefine.engines.qiskit.options import QiskitOptions
from chemrefine.engines.qiskit.problem import prepare_problem
from chemrefine.engines.qiskit.state_io import validate_state_references
from chemrefine.engines.qiskit.workflow import run_problem, validate_options
from chemrefine.errors import ConfigError
from chemrefine.scaffold import scaffold_templates
from chemrefine.state import PipelineState, Structure
from chemrefine.step import run_step

pytestmark = [
    pytest.mark.filterwarnings("ignore:.*:DeprecationWarning:qiskit.*"),
    pytest.mark.filterwarnings("ignore:.*:scipy.sparse.SparseEfficiencyWarning"),
]


@pytest.fixture
def h2():
    """Prepare the stored molecular reference without invoking an SCF driver."""
    raw = json.loads((Path(__file__).parent / "data/engines/qiskit/h2_integrals.json").read_text())
    return prepare_problem(ElectronicStructureData(**raw))


@pytest.mark.parametrize(
    "algorithm,ansatz",
    [
        ("vqe", "uccsd"),
        ("adapt_vqe", "uccsd"),
        ("tetris_adapt", "uccsd"),
        ("ceo_adapt", "ceo"),
        ("vqd", "uccsd"),
    ],
)
def test_exported_preparation_reproduces_molecular_energy(h2, algorithm, ansatz):
    """Each advertised export returns its retained logical state and physical energy."""
    options = {
        "algorithm": {"name": algorithm, "options": {"k": 1} if algorithm == "vqd" else {}},
        "ansatz": ansatz,
        "circuit_export": {"max_bytes": 1048576},
        "optimizer": {"name": "slsqp", "options": {"maxiter": 200, "ftol": 1e-12}},
    }
    result = run_problem(h2, options=options)
    assert len(result.circuits) == 1
    exported = result.circuits[0]
    physical = Statevector(exported.circuit).expectation_value(
        SparsePauliOp.from_list(list(exported.description.active_hamiltonian.items()))
    ).real + sum(exported.description.energy_offsets.values())
    assert physical == pytest.approx(result.energy_hartree, abs=1e-10)
    assert exported.description.parameter_values
    assert exported.description.mapping["name"] == "jordan_wigner"
    assert exported.description.active_space["active_orbitals"] == [0, 1]
    assert "circuits" not in result.as_dict()


@pytest.mark.parametrize("mapper", ["parity", {"name": "z2_tapered", "options": {}}])
def test_export_retains_reduced_register_interpretation(h2, mapper):
    """A saved reduced circuit carries enough mapping facts to reject occupation misuse."""
    result = run_problem(h2, options={"algorithm": "vqe", "mapper": mapper, "circuit_export": {}})
    exported = result.circuits[0]
    assert exported.description.num_qubits < exported.description.num_spin_orbitals
    assert exported.description.mapping["num_qubits_after_reduction"] == exported.circuit.num_qubits
    assert exported.circuit.num_parameters == 0


def test_unsupported_exports_fail_before_provider_construction():
    """Exact and response roots do not advertise executable variational preparations."""
    for name in ("exact", "qeom", "sqd"):
        with pytest.raises(ConfigError, match="cannot export"):
            validate_options(QiskitOptions.from_raw({"algorithm": name, "circuit_export": {}}))


def test_generated_molecular_worker_feeds_measurement_bundle(tmp_path):
    """Run both engines locally, preserving canonical molecular energy and artifact replay."""
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "cpu.slurm.header").write_text("#!/bin/bash\n")
    integrals = Path(__file__).resolve().parents[1] / "examples/tutorials/qiskit_sp/h2_mo.json"
    target = tmp_path / "outputs/step1/0/step1_0_inp.root0.circuit.json"
    config_path = tmp_path / "input.yaml"
    config_path.write_text(
        json.dumps(
            {
                "template_dir": "templates",
                "output_dir": "outputs",
                "dispatch": "local",
                "charge": 0,
                "multiplicity": 1,
                "max_cores": 1,
                "steps": [
                    {
                        "step": 1,
                        "engine": "qiskit",
                        "options": {
                            "backend_python": sys.executable,
                            "algorithm": "vqe",
                            "integral_source": {"bundle_path": str(integrals)},
                            "circuit_export": {},
                        },
                    },
                    {
                        "step": 2,
                        "engine": "qiskit-experiment",
                        "options": {
                            "backend_python": sys.executable,
                            "experiment": {
                                "name": "pauli_measurement",
                                "options": {
                                    "circuit_path": str(target),
                                    "observable": {"IIZZ": 1.0},
                                    "measurement": {"shots": 128, "pilot_shots": 16, "seed": 7},
                                    "sampler": {"name": "statevector", "options": {"seed": 7}},
                                },
                            },
                        },
                    },
                ],
            }
        )
    )
    config = load_config(config_path)
    scaffold_templates(config)
    seed = PipelineState(
        structures=(Structure(id="0", atoms=Atoms("HH", positions=[[0, 0, 0], [0, 0, 0.735]])),)
    )
    first = run_step(config, config.steps[0], seed)
    assert target.is_file()
    assert load_circuit(target).description.root == 0
    output = target.parent / "step1_0.json"
    validate_state_references(output)
    second = run_step(config, config.steps[1], first.state)
    assert second.state.structures == first.state.structures
    assert (
        read_bundle(tmp_path / "outputs/step2/experiment/artifact.json").metadata["expectation"]
        == -1
    )
    assert run_step(config, config.steps[1], first.state).cache_hit
    # A changed NPZ invalidates integrity at either end of this native handoff.
    bundle = read_bundle(target)
    payload = target.parent / bundle.description.payload
    payload.write_bytes(payload.read_bytes() + b"changed")
    from chemrefine.errors import OutputParseError

    with pytest.raises(OutputParseError, match="digest"):
        validate_state_references(output)


def test_vqd_multi_root_exports_track_physical_sorting_and_persisted_roots(tmp_path, monkeypatch):
    """Every exported root reproduces its physical energy, including the selected excited root."""
    from chemrefine.engines.qiskit import workflow

    one_body = np.array([[-1, 0.2 + 0.3j], [0.2 - 0.3j, 0.5]])
    prepared = prepare_problem(
        ElectronicStructureData(1, 0, 2, one_body, np.zeros((2,) * 4), nuclear_repulsion_energy=0.7)
    )
    monkeypatch.setattr(workflow, "prepare_pyscf_problem", lambda *_args, **_kwargs: prepared)
    result = workflow.run_job(
        tmp_path / "molecule.xyz",
        charge=0,
        multiplicity=2,
        artifact_dir=tmp_path,
        options={
            "algorithm": {
                "name": "vqd",
                "options": {
                    "k": 2,
                    "betas": [3],
                    "initial_points": [[0.2, 0.1], [0.9, 0.2]],
                    "target_root": 1,
                },
            },
            "ansatz": {"name": "uccsd", "options": {"include_imaginary": True}},
            "sampler": {"name": "statevector", "options": {"seed": 3}},
            "optimizer": {"name": "cobyla", "options": {"maxiter": 500}},
            "circuit_export": {},
        },
    )
    assert result.root_energies_hartree is not None
    np.testing.assert_allclose(
        result.root_energies_hartree, np.linalg.eigvalsh(one_body) + 0.7, atol=2e-3
    )
    assert result.energy_hartree == result.root_energies_hartree[1]
    assert result.metadata["quantum_artifacts"] == [
        "molecule.root0.circuit.json",
        "molecule.root1.circuit.json",
    ]
    vectors = []
    for root, filename in enumerate(result.metadata["quantum_artifacts"]):
        exported = load_circuit(tmp_path / filename)
        assert exported.description.root == root
        assert exported.description.num_particles == (1, 0)
        state = Statevector(exported.circuit)
        vectors.append(state.data)
        physical = state.expectation_value(
            SparsePauliOp.from_list(list(exported.description.active_hamiltonian.items()))
        ).real + sum(exported.description.energy_offsets.values())
        assert physical == pytest.approx(result.root_energies_hartree[root], abs=1e-12)
    assert abs(np.vdot(*vectors)) ** 2 < 0.01
    output = tmp_path / "molecule.json"
    output.write_text(json.dumps({"engine_metadata": result.as_metadata()}))
    validate_state_references(output)


@pytest.mark.parametrize("routed", [False, True])
def test_adapt_export_rebuilds_retained_logical_state_after_energy_rollback(monkeypatch, routed):
    """A rolled-back candidate and physical layout cannot leak into the retained preparation."""
    from dataclasses import replace

    from qiskit.primitives import StatevectorEstimator
    from qiskit.transpiler import CouplingMap
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    from chemrefine.engines.qiskit.context import EstimatorResource
    from chemrefine.engines.qiskit.registry import ESTIMATORS

    if routed:
        manager = generate_preset_pass_manager(
            coupling_map=CouplingMap.from_line(5),
            initial_layout=[4, 2, 0, 1],
            basis_gates=["rz", "sx", "x", "cx"],
            optimization_level=1,
            seed_transpiler=7,
        )

        def provider(**_options):
            return EstimatorResource(StatevectorEstimator(), transpiler=manager)

        monkeypatch.setitem(
            ESTIMATORS._specs,
            "statevector",
            replace(ESTIMATORS.spec("statevector"), builder=provider),
        )
    one_body = np.array([[-1, 0.2 + 0.3j], [0.2 - 0.3j, 0.5]])
    prepared = prepare_problem(
        ElectronicStructureData(1, 0, 2, one_body, np.zeros((2,) * 4), nuclear_repulsion_energy=0.7)
    )
    result = run_problem(
        prepared,
        options={
            "algorithm": {
                "name": "adapt_vqe",
                "options": {
                    "max_iterations": 4,
                    "eigenvalue_threshold": 100,
                },
            },
            "ansatz": {"name": "uccsd", "options": {"include_imaginary": True}},
            "optimizer": {"name": "slsqp", "options": {"maxiter": 150, "ftol": 1e-12}},
            "circuit_export": {},
        },
    )
    assert result.adapt_iterations == 2
    assert [entry["retained"] for entry in result.adapt_gradient_history] == [True, False]
    exported = result.circuits[0]
    assert exported.circuit.num_qubits == exported.description.num_qubits == 4
    assert exported.circuit.layout is None
    assert len(exported.description.parameter_values) == 1
    np.testing.assert_allclose(
        exported.description.parameter_values, result.metadata["solver"]["optimal_point"]
    )
    energy = Statevector(exported.circuit).expectation_value(
        SparsePauliOp.from_list(list(exported.description.active_hamiltonian.items()))
    ).real + sum(exported.description.energy_offsets.values())
    assert energy == pytest.approx(result.energy_hartree, abs=1e-10)
    if routed:
        assert result.transpiled_circuit_metrics is not None


def test_circuit_recovery_enforces_configured_limit_and_allows_larger_declared_caps(
    h2, tmp_path, monkeypatch
):
    """Recovery uses the same configured QPY budget as publication, without huge allocations."""
    from chemrefine.engines.qiskit import state_io
    from chemrefine.engines.qiskit.circuit_io import save_circuit
    from chemrefine.errors import OutputParseError

    result = run_problem(h2, options={"algorithm": "vqe", "circuit_export": {}})
    path = save_circuit(tmp_path / "root.json", result.circuits[0])
    size = read_bundle(path).arrays["qpy"].nbytes
    output = tmp_path / "output.json"
    output.write_text(json.dumps({"engine_metadata": {"quantum_artifacts": [path.name]}}))
    with pytest.raises(OutputParseError, match="circuit_max_bytes"):
        validate_state_references(output, circuit_max_bytes=1)
    monkeypatch.setattr(state_io, "DEFAULT_MAX_BYTES", 64)
    validate_state_references(output, circuit_max_bytes=size)


@pytest.mark.parametrize("limit", [None, 567])
def test_engine_forwards_configured_circuit_budget_to_local_validation(
    tmp_path, monkeypatch, limit
):
    """The scheduler-facing recovery hook passes the active option cap to native validation."""
    from types import SimpleNamespace

    from chemrefine.config import StepConfig
    from chemrefine.engines.qiskit import state_io
    from chemrefine.engines.qiskit.engine import QiskitEngine
    from chemrefine.state import StepInputs

    calls = []
    monkeypatch.setattr(
        state_io,
        "validate_state_references",
        lambda path, *, circuit_max_bytes: calls.append((path, circuit_max_bytes)),
    )
    options = {} if limit is None else {"circuit_export": {"max_bytes": limit}}
    context = SimpleNamespace(step_cfg=StepConfig(step=1, engine="qiskit", options=options))
    output = tmp_path / "result.json"
    QiskitEngine().validate_outputs(
        StepInputs(files=((tmp_path / "input.py", output, "0"),)), context
    )
    assert calls == [(output, limit if limit is not None else 33554432)]


def test_raw_qpy_decode_type_error_is_an_actionable_configuration_failure(tmp_path):
    """Malformed raw QPY has the same normalized failure contract as a circuit bundle."""
    from chemrefine.engines.qiskit.experiment_measurement import read_circuit

    path = tmp_path / "malformed.qpy"
    path.write_bytes(b"QISKIT\x0d" + b"\0" * 100)
    with pytest.raises(ConfigError, match="cannot load circuit"):
        read_circuit(path)


@pytest.mark.parametrize("encoding", ["jordan_wigner", "parity", "inconsistent_width"])
def test_shadow_handoff_requires_the_full_jordan_wigner_occupation_register(h2, tmp_path, encoding):
    """Real exported JW preparations run; reduced or contradictory occupations are rejected."""
    from dataclasses import replace

    from chemrefine.engines.qiskit.circuit_io import save_circuit
    from chemrefine.engines.qiskit.experiment_shadows import (
        ShadowExperimentOptions,
        shadow_experiment,
    )

    mapper = "parity" if encoding == "parity" else "jordan_wigner"
    result = run_problem(h2, options={"algorithm": "vqe", "mapper": mapper, "circuit_export": {}})
    exported = result.circuits[0]
    if encoding == "inconsistent_width":
        # The mapper name alone cannot establish occupation-register compatibility.
        exported = replace(
            exported,
            description=exported.description.model_copy(update={"num_spin_orbitals": 6}),
        )
    path = save_circuit(tmp_path / "preparation.json", exported)
    options = ShadowExperimentOptions.model_validate(
        {
            "circuit_path": str(path),
            "shadows": {
                "ensemble": "majorana_clifford",
                "num_settings": 3,
                "shots_per_setting": 16,
                "max_order": 1,
                "seed": 4,
            },
            "sampler": {"name": "statevector", "options": {"seed": 7}},
        }
    )
    if encoding != "jordan_wigner":
        with pytest.raises(ConfigError, match="unreduced Jordan-Wigner"):
            shadow_experiment(options=options, device="cpu", cores=1, max_output_bytes=1048576)
        return
    acquired = shadow_experiment(options=options, device="cpu", cores=1, max_output_bytes=1048576)
    assert acquired.kind == "fermionic_shadows"
    assert acquired.metadata["num_modes"] == 4
    assert acquired.metadata["ensemble"] == "majorana_clifford"
    assert acquired.arrays["settings"].shape == (3, 8, 8)
    assert acquired.arrays["one_body"].shape == (4, 4)
    assert np.isfinite(acquired.arrays["one_body"]).all()
    assert np.sum(acquired.arrays["counts"]) == 48
    np.testing.assert_allclose(
        acquired.arrays["one_body"], acquired.arrays["one_body"].conj().T, atol=1e-12
    )


def test_experiment_dependencies_hash_bound_payload_and_ignore_unrelated_pointers(h2, tmp_path):
    """Only declared circuit fields contribute bundle payloads to generic dependency hashing."""
    from chemrefine.engines.qiskit.circuit_io import save_circuit
    from chemrefine.engines.qiskit.experiment import QiskitExperimentEngine

    result = run_problem(h2, options={"algorithm": "vqe", "circuit_export": {}})
    path = save_circuit(tmp_path / "preparation.json", result.circuits[0])
    payload = path.parent / read_bundle(path).description.payload
    options = {
        "experiment": {
            "name": "pauli_measurement",
            "options": {"circuit_path": str(path), "observable": {"IIII": 1.0}},
        }
    }
    dependencies = QiskitExperimentEngine().input_file_dependencies(
        options,
        {
            "/experiment/options/circuit_path": path,
            "/experiment/options/undeclared_path": tmp_path / "unrelated-missing-file.json",
        },
    )
    assert dependencies == {"experiment/options/circuit_path/payload": payload}

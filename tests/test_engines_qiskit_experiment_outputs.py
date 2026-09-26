"""Semantic recovery contracts reject intact bundles containing the wrong science."""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from chemrefine.engines.qiskit.bundles import read_bundle, write_bundle
from chemrefine.engines.qiskit.experiment import EXPERIMENTS, LatticeExperimentOptions
from chemrefine.engines.qiskit.experiment_outputs import (
    register_experiment_output,
    require_experiment_output,
    validate_experiment_output,
)
from chemrefine.errors import ConfigError, OutputParseError

NAMES = (
    "lattice_dynamics",
    "variational_dynamics",
    "double_factorized_evolution",
    "pauli_measurement",
    "fermionic_shadows",
    "rdm_reconstruction",
    "spacetime_postselection",
    "circuit_cutting",
    "pauli_resources",
    "factorized_resources",
    "surface_code_resources",
)

# Complete Qiskit 2.5.2 version-13 streams, generated from empty one-/two-qubit
# circuits. Keep fixtures SDK-free so local recovery can run without providers.
_QPY_FIXTURES = {
    "double_factorized_evolution": (
        "UUlTS0lUDQIFAgAAAAAAAAABcHEAEGYACAAAAAIAAAAAAAAAAAAAAAIAAAABAAAAAAAAAAAAAAAA"
        "Y29udHJhY3QtZml4dHVyZQAAAAAAAAAAe31xAQAAAAIAAQFxAAAAAAAAAAAAAAAAAAAAAQAAAAAA"
        "AAAAAAAA////////////////AAAAAAAAAAA="
    ),
    "spacetime_postselection": (
        "UUlTS0lUDQIFAgAAAAAAAAABcHEAEGYACAAAAAIAAAAAAAAAAAAAAAIAAAABAAAAAAAAAAAAAAAA"
        "Y29udHJhY3QtZml4dHVyZQAAAAAAAAAAe31xAQAAAAIAAQFxAAAAAAAAAAAAAAAAAAAAAQAAAAAA"
        "AAAAAAAA////////////////AAAAAAAAAAA="
    ),
    "circuit_cutting": (
        "UUlTS0lUDQIFAgAAAAAAAAABcHEAEGYACAAAAAEAAAAAAAAAAAAAAAIAAAABAAAAAAAAAAAAAAAA"
        "Y29udHJhY3QtZml4dHVyZQAAAAAAAAAAe31xAQAAAAEAAQFxAAAAAAAAAAAAAAAAAAAAAAAAAP//"
        "/////////////wAAAAAAAAAA"
    ),
}


def _case(name):
    """Construct compact, explicit scientific products independently of provider builders."""
    zero = np.zeros
    qpy = np.frombuffer(
        base64.b64decode(_QPY_FIXTURES.get(name, _QPY_FIXTURES["circuit_cutting"])), dtype=np.uint8
    )
    if name == "lattice_dynamics":
        options = LatticeExperimentOptions().model_dump(mode="json")
        arrays = {
            "times": np.array([0.0, 1.0]),
            "statevectors": zero((2, 2), complex),
            "occupations": zero((2, 1)),
            "energies": zero(2),
        }
        metadata = {
            "model": options["model"],
            "dynamics": options["dynamics"],
            "observations": [{}, {}],
            "mode_order": "site",
            "units": {"energy": "model_energy", "time": "hbar/model_energy", "hbar": 1},
        }
        return "lattice_trajectory", arrays, metadata, options
    if name == "variational_dynamics":
        options = {
            "dynamics": {"time": 1.0, "steps": 1, "method": "varqite", "integrator": "euler"},
            "initial_parameters": [0.0],
            "observables": [],
        }
        arrays = {
            "times": np.array([0.0, 1.0]),
            "parameters": zero((2, 1)),
            "expectations": zero((2, 1)),
            "metric_diagnostics": zero((1, 5)),
        }
        metadata = {
            "parameter_order": ["theta"],
            "publications": 2,
            "units": {},
            "method": "varqite",
            "metric_columns": [
                "retained_rank",
                "min_eigenvalue",
                "max_eigenvalue",
                "residual_norm",
                "velocity_norm",
            ],
        }
        return "variational_trajectory", arrays, metadata, options
    if name == "double_factorized_evolution":
        options = {"times": [0.0]}
        arrays = {
            "times": zero(1),
            "statevectors": zero((1, 4), complex),
            "occupations": zero((1, 2)),
            "energies": zero(1),
            "factor_one_body": zero((1, 1), complex),
            "diagonal_coulomb": zero((1, 1, 1), complex),
            "orbital_rotations": zero((1, 1, 1), complex),
            "circuits_qpy": qpy,
        }
        metadata = {
            "observations": [{}],
            "reference_occupations": {"alpha": [1], "beta": [0]},
            "circuits_format": "QPY version 13",
            "orbital_order": "alpha_then_beta",
            "units": {"energy": "hartree", "time": "hbar/hartree", "hbar": 1},
        }
        return "double_factorized_trajectory", arrays, metadata, options
    if name == "pauli_measurement":
        options = {"observable": {"Z": 1.0}, "measurement": {"shots": 4, "pilot_shots": 2}}
        arrays = {
            "covariance": zero((1, 1)),
            "bitstrings": zero((1, 1), np.uint8),
            "counts": np.array([2], np.uint64),
        }
        metadata = {
            "num_qubits": 1,
            "groups": [
                {
                    "paulis": ["Z"],
                    "shots": 2,
                    "pilot_shots": 2,
                    "coefficients": [1.0],
                    "z_masks": [1],
                    "signs": [1],
                    "covariance_array": "covariance",
                    "bitstrings_array": "bitstrings",
                    "counts_array": "counts",
                }
            ],
            "shots": 4,
            "bit_order": "big endian",
            "expectation": 1.0,
            "standard_error": 0.0,
        }
        return name, arrays, metadata, options
    if name == "fermionic_shadows":
        options = {
            "shadows": {
                "ensemble": "orbital_haar",
                "num_settings": 2,
                "max_order": 2,
                "shots_per_setting": 2,
            }
        }
        arrays = {
            "settings": np.ones((2, 1, 1), complex),
            "one_body": np.ones((1, 1), complex),
            "two_body": zero((1,) * 4, complex),
            "setting_one_body": np.ones((2, 1, 1), complex),
            "setting_two_body": zero((2, 1, 1, 1, 1), complex),
            "bitstrings": zero((2, 1), np.uint8),
            "counts": np.array([2, 2], np.uint64),
            "setting_offsets": np.array([0, 1, 2], np.uint64),
        }
        for name_ in ("one_body", "two_body"):
            for part in ("real", "imag"):
                arrays[f"standard_error_{part}_{name_}"] = zero(arrays[name_].shape)
        metadata = {
            "num_modes": 1,
            "num_settings": 2,
            "total_shots": 4,
            "ensemble": "orbital_haar",
            "acceptance": [{}, {}],
            "rdm_convention": "creation before annihilation",
        }
        return name, arrays, metadata, options
    if name == "rdm_reconstruction":
        options = {
            "reconstruction": {
                "accept_inaccurate": False,
                "constraints": "DQG",
                "energy_weight": 0.0,
                "loss": "frobenius",
                "num_particles": 1,
            }
        }
        arrays = {
            "one_body": np.ones((1, 1), complex),
            "raw_one_body": np.ones((1, 1), complex),
            "two_body": zero((1,) * 4, complex),
            "raw_two_body": zero((1,) * 4, complex),
        }
        metadata = {
            "constraint_checks": {},
            "rdm_convention": "creation before annihilation",
            "units": {},
            "status": "optimal",
            "solver": "SCS",
            "constraints": "DQG",
            "energy_weight": 0.0,
            "loss": "frobenius",
            "necessary_not_sufficient": True,
            "variational_bound": False,
        }
        return "reconstructed_rdms", arrays, metadata, options
    if name == "spacetime_postselection":
        options = {"spacetime": {"checks": ["Z"], "shots": 2}}
        arrays = {"checked_circuit_qpy": qpy}
        for prefix in ("raw", "accepted", "rejected"):
            arrays[prefix + "_counts"] = (
                np.array([2], np.uint64) if prefix != "rejected" else np.array([], np.uint64)
            )
            arrays[prefix + "_bitstrings"] = zero((len(arrays[prefix + "_counts"]), 1), np.uint8)
        metadata = {
            "num_data_qubits": 1,
            "num_check_qubits": 1,
            "raw_shots": 2,
            "accepted_shots": 2,
            "input_checks": ["Z"],
            "output_checks": ["Z"],
            "acceptance_rate": 1.0,
        }
        return name, arrays, metadata, options
    if name == "circuit_cutting":
        options = {"observable": {"Z": 1.0}, "cutting": {"shots": 2}}
        arrays = {
            "term_expectations": np.array([1.0]),
            "observable_coefficients": np.array([1.0]),
            "qpd_coefficients": np.array([1.0]),
            "logical_experiments_qpy": qpy,
            "register": zero((2, 1), np.uint8),
        }
        metadata = {
            "observable_labels": ["Z"],
            "weight_types": ["EXACT"],
            "records": [{"shots": 2, "registers": {"obs": {"array": "register", "num_bits": 1}}}],
            "subobservables": {"A": ["Z"]},
            "logical_qpy_partition_counts": {"A": 1},
            "observable": {"Z": 1.0},
            "signed_reconstruction": True,
            "clipped": False,
        }
        return name, arrays, metadata, options
    metadata = {"executable_circuit": False, "exclusions": ["routing"]}
    if name == "pauli_resources":
        options = {"hamiltonian": {"Z": 1.0}, "budget": {}}
        metadata.update(
            estimate_kind="analytical_query_bound",
            system_qubits=1,
            controlled_walk_queries=2,
            phase_bits=1,
            budget={},
            normalization_hartree=1.0,
        )
    elif name == "factorized_resources":
        options = {"method": "df"}
        metadata.update(
            estimate_kind="provider_analytical_cost_model",
            method="df",
            system_qubits=2,
            provider="openfermion",
            version="1.8.1",
            factorization_parameters={},
            conservative_standard_qpe={},
            provider_toffoli_total_single_run=2,
            provider_logical_qubits_including_system_and_phase=3,
            normalization_hartree=1.0,
        )
    else:
        options = {"logical_qubits": 2}
        metadata.update(
            estimate_kind="phenomenological_surface_code_model",
            physical_qubits=10,
            assumptions=options.copy(),
            meets_failure_budget_at_runtime_lower_bound=True,
            runtime_cycles_lower_bound=10,
            runtime_seconds_lower_bound=0.1,
            failure_union_bound_at_runtime_lower_bound=0.001,
        )
    return "resource_estimate", {}, metadata, options


def _bundle(tmp_path, case):
    """Publish fresh integrity-valid bytes so failures concern semantics, not digests."""
    kind, arrays, metadata, _options = case
    return read_bundle(
        write_bundle(tmp_path / "artifact.json", kind=kind, arrays=arrays, metadata=metadata)
    )


@pytest.mark.parametrize("name", NAMES)
def test_each_registered_builtin_requires_its_scientific_product(tmp_path, name):
    """Every built-in accepts its contract and rejects unrelated or empty numerical content."""
    case = _case(name)
    validate_experiment_output(name, _bundle(tmp_path, case), case[3])
    assert set(NAMES) == set(EXPERIMENTS.names())
    broken: tuple[str, dict, dict, dict]
    for broken in (("unrelated", case[1], case[2], case[3]), (case[0], {}, {}, case[3])):
        with pytest.raises(OutputParseError, match="experiment product"):
            validate_experiment_output(name, _bundle(tmp_path, broken), case[3])
    for key in case[1]:
        arrays = {field: value for field, value in case[1].items() if field != key}
        with pytest.raises(OutputParseError):
            validate_experiment_output(
                name, _bundle(tmp_path, (case[0], arrays, case[2], case[3])), case[3]
            )


@pytest.mark.parametrize(
    ("name", "array"),
    [
        ("double_factorized_evolution", "circuits_qpy"),
        ("spacetime_postselection", "checked_circuit_qpy"),
        ("circuit_cutting", "logical_experiments_qpy"),
    ],
)
@pytest.mark.parametrize("corruption", ["stub", "header_only", "version", "kind", "count", "dtype"])
def test_qpy_products_reject_incomplete_or_incompatible_streams(tmp_path, name, array, corruption):
    """Checksummed retained circuits still need a complete supported QPY structure."""
    kind, arrays, metadata, options = _case(name)
    payload = arrays[array].copy()
    if corruption == "stub":
        payload = payload[:7]
    elif corruption == "header_only":
        payload = payload[:20]
    elif corruption == "version":
        payload[6] = 255
    elif corruption == "kind":
        payload[19] = ord("s")
    elif corruption == "count":
        payload[17] = 2
    else:
        payload = payload.astype(np.uint16)
    arrays[array] = payload
    with pytest.raises(OutputParseError, match="QPY structure"):
        validate_experiment_output(
            name, _bundle(tmp_path, (kind, arrays, metadata, options)), options
        )


@pytest.mark.parametrize(
    ("name", "array"),
    [
        ("double_factorized_evolution", "circuits_qpy"),
        ("spacetime_postselection", "checked_circuit_qpy"),
    ],
)
@pytest.mark.parametrize("field_offset", [28, 32])
def test_unitary_qpy_products_match_declared_logical_dimensions(
    tmp_path, name, array, field_offset
):
    """Evolution and check circuits retain unmeasured widths without hardware layout expansion."""
    kind, arrays, metadata, options = _case(name)
    payload = arrays[array].copy()
    # In v13 the first header starts at byte 20; qubit/clbit fields end at 28/32.
    payload[field_offset] += 1
    arrays[array] = payload
    with pytest.raises(OutputParseError, match="count disagrees"):
        validate_experiment_output(
            name, _bundle(tmp_path, (kind, arrays, metadata, options)), options
        )


@pytest.mark.parametrize(
    ("name", "array", "replacement"),
    [
        ("lattice_dynamics", "energies", np.zeros(2, dtype=np.uint8)),
        ("lattice_dynamics", "energies", np.zeros((2, 1))),
        ("lattice_dynamics", "statevectors", np.zeros((2, 4))),
        ("lattice_dynamics", "times", np.array([0.0, 2.0])),
        ("double_factorized_evolution", "circuits_qpy", np.array([1], np.uint8)),
        ("double_factorized_evolution", "circuits_qpy", np.zeros(20, np.uint8)),
        ("pauli_measurement", "bitstrings", np.array([[1]], np.uint8)),
        ("pauli_measurement", "counts", np.array([0], np.uint64)),
        ("pauli_measurement", "counts", np.array([1], np.uint64)),
        ("pauli_measurement", "bitstrings", np.zeros((1, 1), np.uint16)),
        ("fermionic_shadows", "setting_offsets", np.array([1, 1, 2], np.uint64)),
        ("fermionic_shadows", "counts", np.array([3, 1], np.uint64)),
        ("fermionic_shadows", "standard_error_real_one_body", np.array([[-1.0]])),
        ("circuit_cutting", "observable_coefficients", np.array([-1.0])),
        ("circuit_cutting", "register", np.zeros((2, 1), np.uint16)),
    ],
)
def test_wrong_array_semantics_fail_even_with_matching_descriptor(
    tmp_path, name, array, replacement
):
    """Edited products cannot be legitimized merely by regenerating integrity digests."""
    kind, arrays, metadata, options = _case(name)
    arrays[array] = replacement
    with pytest.raises(OutputParseError):
        validate_experiment_output(
            name, _bundle(tmp_path, (kind, arrays, metadata, options)), options
        )


@pytest.mark.parametrize(
    ("name", "key", "value"),
    [
        ("lattice_dynamics", "observations", []),
        ("lattice_dynamics", "units", {}),
        ("lattice_dynamics", "model", {}),
        ("variational_dynamics", "publications", True),
        ("variational_dynamics", "metric_columns", []),
        ("double_factorized_evolution", "orbital_order", "interleaved"),
        ("pauli_measurement", "groups", []),
        ("pauli_measurement", "shots", 1),
        ("fermionic_shadows", "ensemble", "majorana_clifford"),
        ("fermionic_shadows", "num_modes", 0),
        ("rdm_reconstruction", "status", "infeasible"),
        ("rdm_reconstruction", "variational_bound", True),
        ("spacetime_postselection", "raw_shots", 1),
        ("spacetime_postselection", "accepted_shots", 0),
        ("spacetime_postselection", "acceptance_rate", 0.5),
        ("circuit_cutting", "records", []),
        ("circuit_cutting", "signed_reconstruction", False),
        ("pauli_resources", "estimate_kind", "phenomenological_surface_code_model"),
        ("factorized_resources", "method", "thc"),
        ("surface_code_resources", "physical_qubits", -1),
    ],
)
def test_wrong_critical_metadata_is_not_a_completed_product(tmp_path, name, key, value):
    """Units, scientific conventions, root reports and physical counters are required."""
    kind, arrays, metadata, options = _case(name)
    metadata[key] = value
    with pytest.raises(OutputParseError):
        validate_experiment_output(
            name, _bundle(tmp_path, (kind, arrays, metadata, options)), options
        )


def test_legitimate_empty_and_optional_measurement_products(tmp_path):
    """Constant observables, one-setting shadows and accepted inaccurate solves remain valid."""
    kind, _arrays, metadata, options = _case("pauli_measurement")
    metadata.update(groups=[], shots=0)
    options["observable"] = {"I": 1.0}
    validate_experiment_output(
        "pauli_measurement", _bundle(tmp_path, (kind, {}, metadata, options)), options
    )
    kind, arrays, metadata, options = _case("fermionic_shadows")
    options["shadows"].update(ensemble="majorana_clifford", max_order=1, num_settings=1)
    metadata.update(ensemble="majorana_clifford", num_settings=1, total_shots=2, acceptance=[{}])
    arrays.update(
        settings=np.eye(2, dtype=complex)[None],
        setting_one_body=arrays["setting_one_body"][:1],
        counts=np.array([2], np.uint64),
        bitstrings=np.zeros((1, 1), np.uint8),
        setting_offsets=np.array([0, 1], np.uint64),
    )
    validate_experiment_output(
        "fermionic_shadows", _bundle(tmp_path, (kind, arrays, metadata, options)), options
    )
    kind, arrays, metadata, options = _case("rdm_reconstruction")
    metadata["status"] = "optimal_inaccurate"
    options["reconstruction"]["accept_inaccurate"] = True
    validate_experiment_output(
        "rdm_reconstruction", _bundle(tmp_path, (kind, arrays, metadata, options)), options
    )


def test_byte_aligned_packing_and_registered_extensions(tmp_path, monkeypatch):
    """A separate local contract supports extensions without built-in name exemptions."""
    from chemrefine.engines.qiskit import experiment_outputs

    monkeypatch.setattr(experiment_outputs, "_CONTRACTS", dict(experiment_outputs._CONTRACTS))
    case = _case("pauli_measurement")
    case[2]["num_qubits"] = 8
    case[2]["groups"][0]["paulis"] = ["ZIIIIIII"]
    case[3]["observable"] = {"ZIIIIIII": 1.0}
    validate_experiment_output("pauli_measurement", _bundle(tmp_path, case), case[3])

    def validate_report(bundle, options):
        """A custom callback owns its metadata semantics independently of providers."""
        if bundle.metadata.get("answer") != options["answer"]:
            raise ValueError("answer disagrees")

    register_experiment_output(" Custom-Report ", "report", validate_report)
    require_experiment_output("custom_report")
    custom: tuple[str, dict, dict, dict] = ("report", {}, {"answer": 42}, {"answer": 42})
    validate_experiment_output("custom_report", _bundle(tmp_path, custom), custom[3])
    with pytest.raises(OutputParseError, match="must declare"):
        validate_experiment_output("unregistered", _bundle(tmp_path, custom), custom[3])
    for name, kind in (("custom_report", "report"), (" ", "report"), ("other", "")):
        with pytest.raises(ValueError, match="unique"):
            register_experiment_output(name, kind, validate_report)


def test_undeclared_output_contract_is_refused_before_builder_or_submission(monkeypatch, tmp_path):
    """Provider-free preflight refuses an extension whose completion contract is absent."""
    from chemrefine.engines.qiskit.experiment import QiskitExperimentEngine, run_experiment

    monkeypatch.setattr(EXPERIMENTS, "_specs", dict(EXPERIMENTS._specs))

    @EXPERIMENTS.register("no_output_contract")
    def never_build(**context):
        """A missing local contract must be discovered before executing scientific code."""
        pytest.fail("preflight invoked experiment builder")

    raw = {"experiment": "no_output_contract"}
    with pytest.raises(ConfigError, match="must declare"):
        QiskitExperimentEngine().backend_requirement(raw)
    with pytest.raises(ConfigError, match="must declare"):
        run_experiment(raw, tmp_path / "artifact.json")
    assert not (tmp_path / "artifact.json").exists()


def test_archived_lattice_metadata_accepts_omitted_compatible_defaults():
    """The recorded pre-complex-hopping fixture remains semantically valid after expansion."""
    from chemrefine.engines.qiskit.experiment import QiskitExperimentOptions

    path = Path(__file__).parent / "data/engines/qiskit-experiment/lattice/artifact.json"
    bundle = read_bundle(path)
    selection = QiskitExperimentOptions.from_raw(bundle.metadata["resolved_options"]).experiment
    validate_experiment_output(
        selection.name, bundle, EXPERIMENTS.options_for(selection).model_dump(mode="json")
    )


def test_contract_discovery_and_validation_need_no_optional_sdk(tmp_path):
    """Archived products remain independently valid after source files disappear."""
    case = _case("double_factorized_evolution")
    _bundle(tmp_path, case)
    code = """
import importlib.abc, sys, json
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {
            'qiskit', 'qiskit_nature', 'qiskit_algorithms', 'qiskit_aer', 'ffsim',
            'qiskit_fermions', 'cvxpy', 'scs', 'openfermion', 'qualtran',
        }:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from chemrefine.engines.qiskit.bundles import read_bundle
from chemrefine.engines.qiskit.experiment_outputs import validate_experiment_output
validate_experiment_output(
    'double_factorized_evolution', read_bundle(Path(sys.argv[1])), json.loads(sys.argv[2])
)
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(tmp_path / "artifact.json"),
            json.dumps({**case[3], "integral_bundle_path": "/nonexistent/input.json"}),
        ],
        check=True,
    )


def test_engine_rejects_semantically_empty_but_integrity_valid_bundle(tmp_path):
    """The public completion/cache/rebuild capability executes the scientific contract."""
    from test_engines_qiskit_experiment import _ctx

    from chemrefine.engines.qiskit.experiment import QiskitExperimentEngine

    engine, ctx = QiskitExperimentEngine(), _ctx(tmp_path)
    inputs = engine.prepare(ctx)
    write_bundle(engine.artifact(ctx), kind="lattice_trajectory", arrays={}, metadata={})
    with pytest.raises(OutputParseError, match="lattice_dynamics experiment product"):
        engine.validate_outputs(inputs, ctx)
    with pytest.raises(OutputParseError):
        engine.parse(inputs, ctx)


@pytest.mark.parametrize(
    "example",
    [
        "input",
        "measurement",
        "variational",
        "double_factorized",
        "orbital_shadows",
        "majorana_shadows",
        "spacetime_postselection",
    ],
)
def test_real_example_builders_satisfy_local_semantic_contract(tmp_path, example):
    """Contract dimensions and counters agree with real provider products, including pilots."""
    import yaml

    from chemrefine.engines.qiskit.experiment import QiskitExperimentOptions

    pytest.importorskip("qiskit")
    if example in {"input", "double_factorized", "orbital_shadows", "majorana_shadows"}:
        pytest.importorskip("ffsim")
    root = Path(__file__).resolve().parents[1] / "examples/tutorials/qiskit_experiment"
    data = yaml.safe_load((root / f"{example}.yaml").read_text())["steps"][0]["options"]
    raw = data["experiment"]["options"]
    for key in ("circuit_path", "integral_bundle_path", "preparation_path"):
        if raw.get(key):
            raw[key] = str(root / raw[key])
    resolved = QiskitExperimentOptions.from_raw(data)
    result = EXPERIMENTS.build(
        resolved.experiment, cores=1, device="cpu", max_output_bytes=resolved.max_output_bytes
    )
    options = EXPERIMENTS.options_for(resolved.experiment).model_dump(mode="json")
    bundle = _bundle(tmp_path, (result.kind, result.arrays, result.metadata, options))
    validate_experiment_output(resolved.experiment.name, bundle, options)


def test_semantic_corruption_invalidates_cache_and_rebuild_without_submission(
    tmp_path, monkeypatch
):
    """An intact wrong-kind bundle cannot authorize cache reuse or passive rebuilding."""
    from test_engines_qiskit_experiment import _ctx

    from chemrefine.config import Config
    from chemrefine.engines.qiskit.experiment import QiskitExperimentEngine
    from chemrefine.errors import NoUsableCacheError
    from chemrefine.step import StepMode, rebuild_cache_step, run_step

    engine, ctx = QiskitExperimentEngine(), _ctx(tmp_path)
    config = Config(
        template_dir=ctx.template_dir,
        output_dir=tmp_path,
        charge=0,
        multiplicity=1,
        max_cores=2,
        steps=[ctx.step_cfg],
        dispatch="local",
    )
    run_step(config, ctx.step_cfg, ctx.prev_state, engine=engine)
    assert run_step(config, ctx.step_cfg, ctx.prev_state, engine=engine).cache_hit
    write_bundle(engine.artifact(ctx), kind="unrelated", arrays={}, metadata={})

    def refuse_submit(*args, **kwargs):
        """Passive recovery must not execute or retrieve a quantum workload."""
        pytest.fail("passive recovery attempted submission")

    monkeypatch.setattr(engine, "submit", refuse_submit)
    monkeypatch.setattr("chemrefine.step.get_engine", lambda name: engine)
    with pytest.raises(NoUsableCacheError):
        run_step(config, ctx.step_cfg, ctx.prev_state, engine=engine, mode=StepMode.CACHE_ONLY)
    with pytest.raises(OutputParseError):
        rebuild_cache_step(config, ctx.step_cfg, ctx.prev_state)

"""Reusable sampled states retain arbitrary-width determinants and complex phase."""

from __future__ import annotations

import json

import numpy as np
import pytest

from chemrefine.engines.qiskit.bundles import read_bundle, write_bundle
from chemrefine.engines.qiskit.determinants import DeterminantState
from chemrefine.engines.qiskit.engine import QiskitEngine
from chemrefine.engines.qiskit.state_io import load_states, save_states, validate_state_references
from chemrefine.engines.qiskit.workflow import run_job
from chemrefine.errors import ConfigError, OutputParseError
from chemrefine.state import StepInputs


def _state():
    """One electron in a coherent superposition spanning three machine words."""
    return DeterminantState(130, (1, 1 << 129), np.array([1, 1j]) / np.sqrt(2))


def test_arbitrary_width_state_and_transition_phase_survive_storage(tmp_path):
    """Determinants are never narrowed to signed or unsigned machine integers."""
    state = _state()
    path = save_states(tmp_path / "states.json", [state])
    restored = load_states(path)[0]
    assert restored.determinants == state.determinants
    np.testing.assert_array_equal(restored.amplitudes, state.amplitudes)
    assert not restored.amplitudes.flags.writeable
    assert restored.rdms(max_order=1).one_body[0, 129] == pytest.approx(0.5j)
    with pytest.raises(ConfigError, match="max_bytes"):
        save_states(tmp_path / "huge.json", [state], max_bytes=1)
    with pytest.raises(ConfigError, match="at least one"):
        save_states(tmp_path / "empty.json", [])


@pytest.mark.parametrize("fault", ["kind", "empty", "schema", "limbs", "normalization", "missing"])
def test_state_semantics_are_validated_beyond_bundle_checks(tmp_path, fault):
    """Integrity-correct arrays may still contain invalid scientific state data."""
    path = save_states(tmp_path / "states.json", [_state()])
    bundle = read_bundle(path)
    metadata, arrays = bundle.metadata, dict(bundle.arrays)
    kind = bundle.description.kind
    if fault == "kind":
        kind = "unrelated"
    elif fault == "empty":
        metadata["states"] = []
    elif fault == "schema":
        metadata["states"][0]["num_modes"] = 0
    elif fault == "limbs":
        arrays["root_0_determinants"] = np.zeros((2, 1), dtype="<u8")
    elif fault == "normalization":
        arrays["root_0_amplitudes"] = np.ones(2, dtype=complex)
    else:
        del arrays["root_0_amplitudes"]
    write_bundle(path, kind=kind, arrays=arrays, metadata=metadata)
    with pytest.raises(OutputParseError):
        load_states(path)


@pytest.mark.parametrize(
    "document",
    [
        [],
        {"engine_metadata": []},
        {"engine_metadata": {"quantum_artifacts": "wrong"}},
        {"engine_metadata": {"quantum_artifacts": ["../outside.json"]}},
        {"engine_metadata": {"quantum_artifacts": [3]}},
    ],
)
def test_invalid_native_references_are_parse_failures(tmp_path, document):
    """Malformed references do not escape the normal output-failure contract."""
    path = tmp_path / "result.json"
    path.write_text(json.dumps(document))
    with pytest.raises(OutputParseError):
        validate_state_references(path)


def test_external_state_descriptor_symlink_is_refused(tmp_path):
    """Relative descriptor names must still resolve within the output directory."""
    output = tmp_path / "job"
    output.mkdir()
    external = save_states(tmp_path / "outside.json", [_state()])
    (output / "states.json").symlink_to(external)
    path = output / "result.json"
    path.write_text(json.dumps({"engine_metadata": {"quantum_artifacts": ["states.json"]}}))
    with pytest.raises(OutputParseError, match="escapes"):
        validate_state_references(path)


def test_old_sidecar_without_retained_states_remains_valid(tmp_path):
    """Existing molecular results need no new artifact files to be parsed."""
    path = tmp_path / "result.json"
    path.write_text('{"energy_hartree": -1.0}')
    validate_state_references(path)
    path.unlink()
    with pytest.raises(OutputParseError):
        validate_state_references(path)


def test_pipeline_solver_persists_all_roots_and_engine_validates_references(tmp_path):
    """Actual H2 SQD states travel from solver through the molecular sidecar boundary."""
    xyz = tmp_path / "h2.xyz"
    xyz.write_text("2\nH2\nH 0 0 0\nH 0 0 0.735\n")
    result = run_job(
        xyz,
        charge=0,
        multiplicity=1,
        options={
            "algorithm": {
                "name": "sqd",
                "options": {
                    "projection": "explicit",
                    "counts": {"0101": 10, "0110": 10, "1001": 10, "1010": 10},
                    "configuration_recovery": False,
                    "samples_per_batch": 4,
                    "num_batches": 1,
                    "num_roots": 4,
                    "target_root": 2,
                    "spin_constraint": "report",
                },
            },
        },
        artifact_dir=tmp_path,
    )
    output = tmp_path / "result.json"
    output.write_text(json.dumps({"engine_metadata": result.as_metadata()}))
    restored = load_states(tmp_path / result.metadata["quantum_artifacts"][0])
    assert len(restored) == 4
    assert result.root_energies_hartree is not None
    assert result.energy_hartree == result.root_energies_hartree[2]
    for original, loaded in zip(result.states, restored, strict=True):
        np.testing.assert_array_equal(original.amplitudes, loaded.amplitudes)
    from types import SimpleNamespace

    from chemrefine.config import StepConfig

    context = SimpleNamespace(step_cfg=StepConfig(step=1, engine="qiskit"))
    QiskitEngine().validate_outputs(StepInputs(files=((xyz, output, "0"),)), context)
    assert "*.npz" in QiskitEngine.output_globs
    assert "provider_jobs" in QiskitEngine().output_dirs(None)


def test_rotated_orbital_basis_survives_state_storage(tmp_path):
    """A retained rotation defines how observables return to the original basis."""
    state = DeterminantState(
        2,
        (1, 2),
        np.ones(2) / np.sqrt(2),
        orbital_rotation=np.array([[1, 0], [0, 1j]]),
    )
    restored = load_states(save_states(tmp_path / "rotated.json", [state]))[0]
    np.testing.assert_array_equal(restored.orbital_rotation, state.orbital_rotation)
    np.testing.assert_allclose(restored.rdms().one_body, state.rdms().one_body)

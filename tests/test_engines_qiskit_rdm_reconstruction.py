"""Independent complex CAR and bounded convex reconstruction regressions."""

import builtins
from itertools import combinations, product

import numpy as np
import pytest
from pydantic import ValidationError

from chemrefine.engines.qiskit.determinants import (
    DeterminantState,
    FermionicHamiltonian,
    ReducedDensityMatrices,
)
from chemrefine.engines.qiskit.rdm_reconstruction import (
    RDMReconstructionOptions,
    dqg_matrices,
    rdm_constraint_diagnostics,
    reconstruct_rdms,
)
from chemrefine.errors import ConfigError


def _complex_state():
    """A correlated state with genuinely complex coherences and unrestricted spins."""
    vector = np.array([1, 2j, -1j, 2, 1j, 1], dtype=complex)
    vector /= np.linalg.norm(vector)
    return DeterminantState(4, (3, 5, 6, 9, 10, 12), vector)


def _annihilator(mode, modes):
    """Construct Fock annihilation independently with explicit Pauli tensor products."""
    identity = np.eye(2)
    z = np.diag([1, -1])
    local = np.array([[0, 1], [0, 0]])
    factors = [
        z if index < mode else local if index == mode else identity for index in range(modes)
    ]
    matrix = np.ones((1, 1))
    for factor in reversed(factors):
        matrix = np.kron(matrix, factor)
    return matrix


def test_complex_dqg_equal_independent_operator_gram_matrices():
    state = _complex_state()
    vector = np.zeros(16, dtype=complex)
    vector[list(state.determinants)] = state.amplitudes
    annihilators = [_annihilator(mode, 4) for mode in range(4)]
    pairs = list(combinations(range(4), 2))
    ordered = list(product(range(4), repeat=2))
    actions = {
        "D": [annihilators[q] @ annihilators[p] @ vector for p, q in pairs],
        "Q": [annihilators[p].conj().T @ annihilators[q].conj().T @ vector for p, q in pairs],
        "G": [annihilators[p].conj().T @ annihilators[q] @ vector for p, q in ordered],
    }
    actual = dqg_matrices(state.rdms())
    for name, values in actions.items():
        stacked = np.array(values).T
        expected = stacked.conj().T @ stacked
        np.testing.assert_allclose(actual[name], expected, atol=1e-13)
        assert np.linalg.eigvalsh(actual[name]).min() > -1e-12
    checks = rdm_constraint_diagnostics(state.rdms(), 2)
    assert max(value for key, value in checks.items() if key.endswith("residual")) < 1e-12
    assert max(abs(value) for value in checks["minimum_eigenvalues"].values()) < 1e-12


def test_single_mode_and_empty_pair_spaces_have_valid_diagnostics():
    rdms = DeterminantState(1, (1,), [1]).rdms()
    matrices = dqg_matrices(rdms)
    assert matrices["D"].shape == matrices["Q"].shape == (0, 0)
    np.testing.assert_allclose(matrices["G"], [[1]])
    checks = rdm_constraint_diagnostics(rdms, 1)
    assert checks["minimum_eigenvalues"] == {"D": 0, "Q": 0, "G": 1}


@pytest.mark.parametrize(
    ("one", "two"),
    [
        ([["invalid"]], np.zeros((1,) * 4)),
        (np.eye(2), None),
        (np.ones((1, 2)), np.zeros((1,) * 4)),
        (np.array([[np.nan]]), np.zeros((1,) * 4)),
        (np.eye(1), np.full((1,) * 4, np.inf)),
        (np.zeros((0, 0)), np.zeros((0,) * 4)),
    ],
)
def test_invalid_rdm_arrays_fail_before_solver_import(one, two):
    with pytest.raises(ConfigError, match="finite"):
        reconstruct_rdms(
            ReducedDensityMatrices(one, two), RDMReconstructionOptions(num_particles=1)
        )


def test_reconstruction_sector_budget_and_regularization_preflight():
    raw = _complex_state().rdms()
    with pytest.raises(ConfigError, match="exceeds the number"):
        reconstruct_rdms(raw, RDMReconstructionOptions(num_particles=5))
    with pytest.raises(ConfigError, match="storage estimate"):
        reconstruct_rdms(
            ReducedDensityMatrices(np.eye(5), np.zeros((5,) * 4)),
            RDMReconstructionOptions(num_particles=2, max_memory_mb=1),
        )
    with pytest.raises(ConfigError, match="explicit Hamiltonian"):
        reconstruct_rdms(raw, RDMReconstructionOptions(num_particles=2, energy_weight=0.1))
    with pytest.raises(ConfigError, match="mode counts"):
        reconstruct_rdms(
            raw, RDMReconstructionOptions(num_particles=2), hamiltonian=FermionicHamiltonian(2, ())
        )
    with pytest.raises(ConfigError, match="at least one"):
        reconstruct_rdms(
            raw,
            RDMReconstructionOptions(num_particles=2),
            one_body_weights=np.zeros((4, 4)),
            two_body_weights=np.zeros((4,) * 4),
        )


@pytest.mark.parametrize("weights", [[[1j]], [["bad"]], [[-1]], [[np.nan]], [[1]]])
def test_invalid_entry_weights_fail_before_solver_import(weights):
    with pytest.raises(ConfigError, match="weights"):
        reconstruct_rdms(
            _complex_state().rdms(),
            RDMReconstructionOptions(num_particles=2),
            one_body_weights=weights,
        )


def test_missing_convex_solver_provider_has_actionable_error(monkeypatch):
    original = builtins.__import__

    def without_cvxpy(name, *args, **kwargs):
        """Simulate the lean worker rather than importing any optional solver."""
        if name == "cvxpy":
            raise ImportError("optional provider absent")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_cvxpy)
    with pytest.raises(ConfigError, match="optional cvxpy and SCS"):
        reconstruct_rdms(_complex_state().rdms(), RDMReconstructionOptions(num_particles=2))


def test_reconstruction_options_are_frozen_strict_and_finite():
    options = RDMReconstructionOptions(num_particles=2)
    for patch in ({"num_particles": True}, {"energy_weight": np.nan}, {"unknown": 1}):
        with pytest.raises(ValidationError):
            RDMReconstructionOptions.model_validate({"num_particles": 2, **patch})
    with pytest.raises(ValidationError):
        options.num_particles = 1


def test_reconstruction_artifact_preflight_is_sdk_free_and_never_solves_bad_inputs(tmp_path):
    from chemrefine.engines.qiskit.bundles import write_bundle
    from chemrefine.engines.qiskit.experiment import QiskitExperimentEngine
    from chemrefine.engines.qiskit.experiment_rdm import RDMExperimentOptions, rdm_experiment

    path = write_bundle(tmp_path / "wrong.json", kind="determinant_states", arrays={}, metadata={})
    options = RDMExperimentOptions(rdm_bundle_path=str(path), reconstruction={"num_particles": 1})
    with pytest.raises(ConfigError, match="measured_rdms"):
        rdm_experiment(options=options, max_output_bytes=10000)
    path = write_bundle(tmp_path / "missing.json", kind="measured_rdms", arrays={}, metadata={})
    with pytest.raises(ConfigError, match="missing a required"):
        rdm_experiment(
            options=options.model_copy(update={"rdm_bundle_path": str(path)}),
            max_output_bytes=10000,
        )
    raw = DeterminantState(2, (1,), [1]).rdms()
    path = write_bundle(
        tmp_path / "valid.json",
        kind="measured_rdms",
        arrays={"one_body": raw.one_body, "two_body": raw.two_body},
        metadata={},
    )
    with pytest.raises(ConfigError, match="max_output_bytes"):
        rdm_experiment(
            options=options.model_copy(update={"rdm_bundle_path": str(path)}), max_output_bytes=1
        )
    with pytest.raises(ValidationError, match="hamiltonian_bundle_path"):
        RDMExperimentOptions(
            rdm_bundle_path=str(path), reconstruction={"num_particles": 1, "energy_weight": 0.1}
        )
    engine = QiskitExperimentEngine()
    raw_options = {
        "experiment": {
            "name": "rdm_reconstruction",
            "options": options.model_dump(exclude_defaults=True),
        }
    }
    assert engine.input_file_options(raw_options) == (("experiment", "options", "rdm_bundle_path"),)
    assert engine.backend_requirement(raw_options).extra == "qiskit-rdm"


def test_reconstruction_example_filing_covers_every_public_nested_control():
    from pathlib import Path

    import yaml

    from chemrefine.engines.qiskit.experiment_rdm import RDMExperimentOptions

    root = Path(__file__).resolve().parents[1] / "examples/tutorials/qiskit_experiment"
    options = yaml.safe_load((root / "rdm_reconstruction.yaml").read_text())["steps"][1]["options"][
        "experiment"
    ]["options"]
    RDMExperimentOptions.model_validate(options)
    assert set(options) == set(RDMExperimentOptions.model_fields)
    assert set(options["reconstruction"]) == set(RDMReconstructionOptions.model_fields)

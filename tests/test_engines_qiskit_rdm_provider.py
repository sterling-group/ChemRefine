"""Real CVXPY/SCS numerical checks, collected only by an installed RDM provider stack."""

import numpy as np
import pytest

cp = pytest.importorskip(
    "cvxpy", reason="RDM numerical provider requires approved CVXPY/SCS installation"
)
pytest.importorskip("scs", reason="RDM numerical provider requires approved SCS installation")

from chemrefine.engines.qiskit.bundles import read_bundle, write_bundle  # noqa: E402
from chemrefine.engines.qiskit.data import ElectronicStructureData  # noqa: E402
from chemrefine.engines.qiskit.determinants import (  # noqa: E402
    DeterminantState,
    FermionicHamiltonian,
    FermionTerm,
    ReducedDensityMatrices,
)
from chemrefine.engines.qiskit.experiment import run_experiment  # noqa: E402
from chemrefine.engines.qiskit.experiment_rdm import (  # noqa: E402
    RDMExperimentOptions,
    rdm_experiment,
)
from chemrefine.engines.qiskit.integral_io import save_integrals  # noqa: E402
from chemrefine.engines.qiskit.rdm_reconstruction import (  # noqa: E402
    RDMReconstructionOptions,
    rdm_constraint_diagnostics,
    reconstruct_rdms,
)
from chemrefine.errors import ConfigError  # noqa: E402


def _state():
    """Independent fixed-N complex state with two-body coherence."""
    values = np.array([1, 2j, -1j, 2, 1j, 1])
    return DeterminantState(4, (3, 5, 6, 9, 10, 12), values / np.linalg.norm(values))


@pytest.mark.parametrize("constraints", ["D", "DQ", "DQG"])
@pytest.mark.parametrize("loss", ["frobenius", "nuclear"])
def test_real_solver_preserves_complex_physical_rdms(constraints, loss):
    raw = _state().rdms()
    result = reconstruct_rdms(
        raw,
        RDMReconstructionOptions(
            num_particles=2,
            constraints=constraints,
            loss=loss,
            solver_tolerance=1e-6,
            feasibility_tolerance=1e-5,
        ),
    )
    np.testing.assert_allclose(result.reconstructed.one_body, raw.one_body, atol=1e-4)
    np.testing.assert_allclose(result.reconstructed.two_body, raw.two_body, atol=1e-4)
    assert result.diagnostics["status"] == cp.OPTIMAL
    assert result.diagnostics["iterations"] > 0
    assert result.diagnostics["necessary_not_sufficient"]
    assert not result.diagnostics["variational_bound"]
    assert not result.reconstructed.one_body.flags.writeable


def test_real_projection_removes_unphysical_noise_without_asserting_representability():
    physical = _state().rdms()
    rng = np.random.default_rng(13)
    raw = ReducedDensityMatrices(
        physical.one_body + 0.03 * (rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))),
        physical.two_body + 0.03 * (rng.normal(size=(4,) * 4) + 1j * rng.normal(size=(4,) * 4)),
    )
    result = reconstruct_rdms(
        raw,
        RDMReconstructionOptions(
            num_particles=2,
            solver_tolerance=1e-7,
            feasibility_tolerance=1e-4,
        ),
    )
    projected = result.reconstructed
    checks = rdm_constraint_diagnostics(projected, 2)
    assert max(value for name, value in checks.items() if name.endswith("residual")) < 1e-4
    assert min(checks["minimum_eigenvalues"].values()) > -1e-4
    before = (
        np.linalg.norm(raw.one_body - physical.one_body) ** 2
        + np.linalg.norm(raw.two_body - physical.two_body) ** 2
    )
    after = (
        np.linalg.norm(projected.one_body - physical.one_body) ** 2
        + np.linalg.norm(projected.two_body - physical.two_body) ** 2
    )
    assert after < before


@pytest.mark.parametrize("particles", [0, 1])
def test_single_mode_and_empty_pair_variable_are_valid_real_solver_problems(particles):
    raw = ReducedDensityMatrices(np.array([[0.4 + 0.1j]]), np.ones((1,) * 4) * 0.1)
    result = reconstruct_rdms(raw, RDMReconstructionOptions(num_particles=particles))
    np.testing.assert_allclose(result.reconstructed.one_body, [[particles]], atol=1e-5)
    np.testing.assert_allclose(result.reconstructed.two_body, 0)


def test_energy_regularization_is_explicit_and_preserves_complex_raw_diagnostics():
    one = np.eye(2, dtype=complex) / 2
    one[0, 0] += 0.1j
    raw = ReducedDensityMatrices(one, np.zeros((2,) * 4))
    hamiltonian = FermionicHamiltonian(
        2,
        (
            FermionTerm((), (), 0.7),
            FermionTerm((0,), (0,), -1),
            FermionTerm((1,), (1,), 1),
            FermionTerm((0, 1), (1, 0), 0.2),
        ),
    )
    ordinary = reconstruct_rdms(
        raw, RDMReconstructionOptions(num_particles=1), hamiltonian=hamiltonian
    )
    biased = reconstruct_rdms(
        raw, RDMReconstructionOptions(num_particles=1, energy_weight=0.5), hamiltonian=hamiltonian
    )
    assert ordinary.diagnostics["reconstructed_energy"] == pytest.approx(0.7, abs=1e-5)
    assert biased.diagnostics["reconstructed_energy"] == pytest.approx(0.2, abs=1e-4)
    assert ordinary.diagnostics["raw_energy"] == pytest.approx({"real": 0.7, "imag": -0.1})
    assert biased.diagnostics["energy_weight"] == 0.5


def test_zero_weights_mask_unobserved_entries_in_a_real_fit():
    raw = _state().rdms()
    weights = np.ones((4, 4))
    weights[0, 1] = 0
    result = reconstruct_rdms(
        raw, RDMReconstructionOptions(num_particles=2), one_body_weights=weights
    )
    assert result.diagnostics["data_fit_objective"] < 1e-6


def test_absent_scs_and_solver_failures_are_distinct(monkeypatch):
    raw, options = _state().rdms(), RDMReconstructionOptions(num_particles=2)
    original = cp.installed_solvers
    monkeypatch.setattr(cp, "installed_solvers", lambda: [])
    with pytest.raises(ConfigError, match="requires the SCS"):
        reconstruct_rdms(raw, options)
    monkeypatch.setattr(cp, "installed_solvers", original)

    def fail(self, **kwargs):
        raise cp.error.SolverError("backend failure")

    monkeypatch.setattr(cp.Problem, "solve", fail)
    with pytest.raises(ConfigError, match="semidefinite solver failed"):
        reconstruct_rdms(raw, options)


def test_reject_failed_or_missing_primal_and_explicitly_accept_inaccurate_status(monkeypatch):
    raw, options = _state().rdms(), RDMReconstructionOptions(num_particles=2)
    original = cp.Problem.solve

    def unavailable(self, **kwargs):
        self._status = cp.INFEASIBLE

    monkeypatch.setattr(cp.Problem, "solve", unavailable)
    with pytest.raises(ConfigError, match="accepted solution"):
        reconstruct_rdms(raw, options)

    def missing(self, **kwargs):
        self._status = cp.OPTIMAL

    monkeypatch.setattr(cp.Problem, "solve", missing)
    with pytest.raises(ConfigError, match="accepted solution"):
        reconstruct_rdms(raw, options)

    def inaccurate(self, **kwargs):
        original(self, **kwargs)
        self._status = cp.OPTIMAL_INACCURATE

    monkeypatch.setattr(cp.Problem, "solve", inaccurate)
    with pytest.raises(ConfigError, match="accepted solution"):
        reconstruct_rdms(raw, options)
    result = reconstruct_rdms(raw, options.model_copy(update={"accept_inaccurate": True}))
    assert result.diagnostics["status"] == cp.OPTIMAL_INACCURATE


@pytest.mark.parametrize(
    "violation",
    ["contraction_residual", "one_body_minimum_eigenvalue", "one_body_maximum_eigenvalue", "D"],
)
def test_solver_status_does_not_override_measured_feasibility(monkeypatch, violation):
    import chemrefine.engines.qiskit.rdm_reconstruction as module

    original = module.rdm_constraint_diagnostics

    def invalid(rdms, particles):
        checks = original(rdms, particles)
        if violation == "D":
            checks["minimum_eigenvalues"]["D"] = -1
        else:
            checks[violation] = -1 if "minimum" in violation else 2
        return checks

    monkeypatch.setattr(module, "rdm_constraint_diagnostics", invalid)
    with pytest.raises(ConfigError, match="feasibility tolerance"):
        reconstruct_rdms(_state().rdms(), RDMReconstructionOptions(num_particles=2))


def test_reconstruction_worker_preserves_raw_arrays_weights_and_energy_offsets(tmp_path):
    data = ElectronicStructureData(
        1, 0, 1, [[-1]], np.zeros((1,) * 4), nuclear_repulsion_energy=0.7
    )
    integrals = save_integrals(tmp_path / "integrals.json", data)
    raw = DeterminantState(2, (1,), [1]).rdms()
    source = write_bundle(
        tmp_path / "measured.json",
        kind="measured_rdms",
        arrays={
            "one_body": raw.one_body,
            "two_body": raw.two_body,
            "one_weights": np.ones((2, 2)),
            "two_weights": np.ones((2,) * 4),
        },
        metadata={"property_source": "measured test data", "mode_order": "alpha_then_beta"},
    )
    output = run_experiment(
        {
            "experiment": {
                "name": "rdm_reconstruction",
                "options": {
                    "rdm_bundle_path": str(source),
                    "hamiltonian_bundle_path": str(integrals),
                    "one_body_weights_array": "one_weights",
                    "two_body_weights_array": "two_weights",
                    "reconstruction": {"num_particles": 1},
                },
            }
        },
        tmp_path / "result.json",
    )
    restored = read_bundle(output)
    assert restored.description.kind == "reconstructed_rdms"
    np.testing.assert_array_equal(restored.arrays["raw_one_body"], raw.one_body)
    assert restored.metadata["reconstructed_energy"] == pytest.approx(-0.3, abs=1e-5)
    assert restored.metadata["units"]["energy"] == "hartree"


def test_reconstruction_worker_without_hamiltonian_has_no_energy_claim(tmp_path):
    raw = _state().rdms()
    source = write_bundle(
        tmp_path / "measured.json",
        kind="fermionic_shadows",
        arrays={"one_body": raw.one_body, "two_body": raw.two_body},
        metadata={"ensemble": "orbital_haar"},
    )
    result = rdm_experiment(
        options=RDMExperimentOptions(
            rdm_bundle_path=str(source), reconstruction={"num_particles": 2}
        ),
        max_output_bytes=1_000_000,
        cores=1,
        device="cpu",
    )
    assert result.metadata["reconstructed_energy"] is None
    assert result.metadata["source_ensemble"] == "orbital_haar"
    assert result.metadata["units"]["energy"] is None


def test_actual_solver_iteration_cap_is_not_reported_as_convergence():
    """An exhausted SCS run fails explicitly before any reconstructed artifact is returned."""
    with pytest.raises(ConfigError, match="accepted solution"):
        reconstruct_rdms(
            _state().rdms(), RDMReconstructionOptions(num_particles=2, max_iterations=1)
        )

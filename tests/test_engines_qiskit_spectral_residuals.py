"""Physical spectral residuals use configured expectations and independent dense references."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from chemrefine.engines.qiskit.components.spectra import QEOMOptions, VQDOptions
from chemrefine.engines.qiskit.context import EstimatorResource
from chemrefine.engines.qiskit.data import ElectronicStructureData
from chemrefine.engines.qiskit.mapping import map_problem
from chemrefine.engines.qiskit.problem import prepare_problem
from chemrefine.engines.qiskit.registry import (
    ESTIMATORS,
    OPTIMIZERS,
    ComponentSpec,
    NoComponentOptions,
)
from chemrefine.engines.qiskit.spectra import (
    energy_residual,
    optimizer_termination,
    residual_hamiltonian,
)
from chemrefine.engines.qiskit.workflow import run_problem
from chemrefine.errors import ConfigError

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning:qiskit")


@pytest.fixture
def h2():
    """Prepare stored H₂ integrals without classical electronic-structure execution."""
    return prepare_problem(
        ElectronicStructureData(
            **json.loads(
                (Path(__file__).parent / "data/engines/qiskit/h2_integrals.json").read_text()
            )
        )
    )


@pytest.fixture
def stopped_optimizer(monkeypatch):
    """Return a nonstationary requested point with an explicit failed stopping verdict."""

    class Optimizer:
        """Expose provider termination facts while evaluating the configured objective once."""

        def minimize(self, fun, x0, **kwargs):
            """Keep the HF trial unchanged so its residual remains independently nonzero."""
            values = np.asarray(x0)
            return SimpleNamespace(
                x=values,
                fun=fun(values),
                nfev=1,
                nit=0,
                success=False,
                status=9,
                message="iteration budget exhausted",
            )

    monkeypatch.setattr(OPTIMIZERS, "build", lambda *args, **kwargs: Optimizer())


def test_vqd_residual_detects_nonstationary_sector_valid_trial(h2, stopped_optimizer):
    """Orthogonality/sector checks alone accept HF, whose independent H residual is nonzero."""
    plain = run_problem(
        h2, options={"algorithm": {"name": "vqd", "options": {"k": 1}}}, initial_point=[0, 0, 0]
    )
    measured = run_problem(
        h2,
        options={"algorithm": {"name": "vqd", "options": {"k": 1, "measure_residuals": True}}},
        initial_point=[0, 0, 0],
    )
    matrix = map_problem(h2).qubit_hamiltonian.to_matrix()
    reference = np.eye(16)[:, 5]
    energy = float(np.vdot(reference, matrix @ reference).real)
    expected = np.linalg.norm(matrix @ reference - energy * reference)
    diagnostics = measured.metadata["solver"]
    residual = diagnostics["root_hamiltonian_residuals"][0]
    assert residual["hamiltonian_residual_hartree"] == pytest.approx(expected, abs=1e-12)
    assert expected > 0.1
    assert residual["active_rayleigh_energy_hartree"] == pytest.approx(energy)
    assert measured.energy_hartree == plain.energy_hartree
    assert plain.metadata["solver"]["root_hamiltonian_residuals"] is None
    assert (
        diagnostics["post_optimization_measurements"]
        == plain.metadata["solver"]["post_optimization_measurements"] + 1
    )
    assert measured.converged is False
    termination = diagnostics["optimizer_termination"][0]
    assert termination["details"] == {
        "success": False,
        "status": 9,
        "message": "iteration budget exhausted",
        "nit": 0,
        "nfev": 1,
    }


def _annihilator(mode, width):
    """Build an independent dense fermion annihilator from occupation-basis CAR signs."""
    matrix = np.zeros((2**width, 2**width), dtype=complex)
    for bits in range(2**width):
        if bits & (1 << mode):
            matrix[bits ^ (1 << mode), bits] = (-1) ** (bits & ((1 << mode) - 1)).bit_count()
    return matrix


def test_qeom_physical_residual_is_separate_from_response_pencil(h2, stopped_optimizer):
    """Reconstruct every reported response state independently in the full Fock matrix."""
    result = run_problem(
        h2,
        options={"algorithm": {"name": "qeom", "options": {"measure_residuals": True}}},
        initial_point=[0, 0, 0],
    )
    diagnostics = result.metadata["solver"]
    matrix = map_problem(h2).qubit_hamiltonian.to_matrix()
    reference = np.eye(16)[:, 5]
    operators = []
    annihilators = [_annihilator(mode, 4) for mode in range(4)]
    for occupied, virtual in diagnostics["excitation_indices"]:
        operator = np.eye(16, dtype=complex)
        for mode in virtual:
            operator = operator @ annihilators[mode].conj().T
        for mode in occupied:
            operator = operator @ annihilators[mode]
        operators.append(operator)
    basis = operators + [operator.conj().T for operator in operators]
    coefficients = np.asarray(diagnostics["expansion_coefficients_real"]) + 1j * np.asarray(
        diagnostics["expansion_coefficients_imag"]
    )
    states = [reference]
    for vector in coefficients.T:
        state = sum(
            coefficient * operator @ reference
            for coefficient, operator in zip(vector, basis, strict=True)
        )
        state -= np.vdot(reference, state) * reference
        states.append(state / np.linalg.norm(state))
    residuals = diagnostics["reconstructed_root_hamiltonian_residuals"]
    for state, residual in zip(states, residuals, strict=True):
        energy = float(np.vdot(state, matrix @ state).real)
        expected = np.linalg.norm(matrix @ state - energy * state) ** 2
        assert residual["variance_hartree_squared"] == pytest.approx(expected, abs=2e-12)
        assert residual["energy_reference"] == "rayleigh_energy"
    assert max(diagnostics["generalized_eigenpair_residuals"]) < 1e-12
    assert residuals[0]["hamiltonian_residual_hartree"] > 0.1
    assert result.converged is False
    assert diagnostics["reference_optimizer_termination"]["details"]["status"] == 9


def test_optimized_h2_residuals_vanish_without_fabricating_optimizer_success(h2):
    """Accurate roots have tiny physical variances; unavailable optimizer status stays unknown."""
    result = run_problem(
        h2,
        options={
            "algorithm": {"name": "qeom", "options": {"measure_residuals": True}},
            "optimizer": {"name": "slsqp", "options": {"maxiter": 300, "ftol": 1e-12}},
        },
    )
    residuals = result.metadata["solver"]["reconstructed_root_hamiltonian_residuals"]
    assert max(abs(item["variance_hartree_squared"]) for item in residuals) < 1e-10
    assert result.converged is None
    assert result.metadata["solver"]["reference_optimizer_termination"]["converged"] is None


@pytest.mark.parametrize("algorithm", ["vqd", "qeom"])
def test_residual_product_budget_fails_before_provider_build(h2, monkeypatch, algorithm):
    """H² is bounded before optimization or any optional measurement is submitted."""

    def unexpected(*args, **kwargs):
        """A preflight product limit must prevent provider construction."""
        pytest.fail("provider built before residual product check")

    monkeypatch.setattr(ESTIMATORS, "build", unexpected)
    with pytest.raises(ConfigError, match="max_residual_product_terms before"):
        run_problem(
            h2,
            options={
                "algorithm": {
                    "name": algorithm,
                    "options": {"measure_residuals": True, "max_residual_product_terms": 1},
                }
            },
        )


def test_residual_normal_ordering_expansion_and_measurement_budget(h2, stopped_optimizer):
    """Post-normal-ordering growth and extra publications use explicit existing guards."""
    from qiskit_nature.second_q.operators import FermionicOp

    with pytest.raises(ConfigError, match="after normal ordering"):
        residual_hamiltonian(FermionicOp({"-_0 +_0": 1}), max_product_terms=1)
    plain = run_problem(
        h2, options={"algorithm": {"name": "vqd", "options": {"k": 1}}}, initial_point=[0, 0, 0]
    )
    count = plain.metadata["solver"]["post_optimization_measurements"]
    with pytest.raises(ConfigError, match="max_measurements"):
        run_problem(
            h2,
            options={
                "algorithm": {
                    "name": "vqd",
                    "options": {"k": 1, "measure_residuals": True, "max_measurements": count},
                }
            },
            initial_point=[0, 0, 0],
        )


def test_residual_uses_selected_estimator_result_including_negative_variance(
    h2, stopped_optimizer, monkeypatch
):
    """A provider's noisy H² estimate is retained instead of substituted by exact simulation."""
    from qiskit.primitives import StatevectorEstimator
    from qiskit.quantum_info import SparsePauliOp

    matrix = map_problem(h2).qubit_hamiltonian.to_matrix()
    reference = np.eye(16)[:, 5]
    energy = float(np.vdot(reference, matrix @ reference).real)
    square = matrix @ matrix
    real = StatevectorEstimator()
    moments, closed = [], []

    def run(pubs):
        """Inject a possible finite-shot fluctuation only for the requested second moment."""
        pubs = list(pubs)
        if (
            len(pubs) == 1
            and isinstance(pubs[0][1], SparsePauliOp)
            and np.allclose(pubs[0][1].to_matrix(), square, atol=1e-12)
        ):
            moments.append(True)
            return SimpleNamespace(
                result=lambda: [SimpleNamespace(data=SimpleNamespace(evs=energy**2 - 0.02))]
            )
        return real.run(pubs)

    def build(**kwargs):
        """Verify the numerical worker grant reaches the selected estimator."""
        assert kwargs["cores"] == 3
        return EstimatorResource(SimpleNamespace(run=run), close=lambda: closed.append(True))

    monkeypatch.setitem(
        ESTIMATORS._specs, "residual_probe", ComponentSpec(NoComponentOptions, build)
    )
    result = run_problem(
        h2,
        options={
            "algorithm": {"name": "vqd", "options": {"k": 1, "measure_residuals": True}},
            "estimator": "residual_probe",
            "cores": 3,
        },
        initial_point=[0, 0, 0],
    )
    residual = result.metadata["solver"]["root_hamiltonian_residuals"][0]
    assert residual["variance_hartree_squared"] == pytest.approx(-0.02)
    assert residual["variance_status"] == "negative_beyond_tolerance"
    assert residual["hamiltonian_residual_hartree"] is None
    assert moments == closed == [True]


@pytest.mark.parametrize(
    "second,status,residual",
    [
        (1 - 1e-10, "negative_within_tolerance", None),
        (0.5, "negative_beyond_tolerance", None),
        (1, "nonnegative_estimate", 0),
        (2, "nonnegative_estimate", 1),
    ],
)
def test_negative_measured_variances_are_never_silently_good_residuals(second, status, residual):
    """Negative estimates retain signed data and a null residual even within tolerance."""
    report = energy_residual(1, second, variance_tolerance=1e-8)
    assert report["variance_status"] == status
    assert report["hamiltonian_residual_hartree"] == residual
    assert report["variance_hartree_squared"] == second - 1
    with pytest.raises(ConfigError, match="finite and real"):
        energy_residual(1, complex("nan"), variance_tolerance=1e-8)


def test_missing_optimizer_termination_is_unknown_and_success_is_preserved():
    """Absent or nonboolean success fields do not become inferred convergence."""
    assert optimizer_termination(None)["converged"] is None
    assert optimizer_termination(SimpleNamespace(success="unknown"))["converged"] is None
    assert optimizer_termination(SimpleNamespace(success=np.bool_(True)))["converged"] is True


@pytest.mark.parametrize("algorithm,model", [("vqd", VQDOptions), ("qeom", QEOMOptions)])
def test_residual_knob_verdicts_match_runnable_examples(algorithm, model):
    """Every added public residual control has a declared example and a rejecting guard."""
    root = Path(__file__).parents[1]
    path = root / f"examples/tutorials/qiskit_sp/{algorithm}.yaml"
    options = yaml.safe_load(path.read_text())["steps"][0]["options"]["algorithm"]["options"]
    fields = {"measure_residuals", "max_residual_product_terms", "residual_variance_tolerance"}
    assert fields <= set(options)
    assert model(**options).measure_residuals
    assert model().measure_residuals is False
    for value in (
        {"max_residual_product_terms": 0},
        {"residual_variance_tolerance": 0},
        {"residual_variance_tolerance": float("nan")},
        {"unknown_residual_control": True},
    ):
        with pytest.raises(ValidationError):
            model(**value)

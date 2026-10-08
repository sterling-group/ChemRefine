"""Real primitive validation of molecular deflation, complex response and shared resources."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from chemrefine.engines.qiskit.assembly import assemble_components
from chemrefine.engines.qiskit.components.qnspsa import QNSPSAOptions, build_qnspsa
from chemrefine.engines.qiskit.components.spectra import (
    QEOMOptions,
    VQDOptions,
    _check_circuit,
    _product,
    _response_basis,
)
from chemrefine.engines.qiskit.context import (
    AnsatzArtifacts,
    EstimatorResource,
    SamplerResource,
    SolverComponents,
)
from chemrefine.engines.qiskit.data import ElectronicStructureData
from chemrefine.engines.qiskit.mapping import map_problem
from chemrefine.engines.qiskit.options import QiskitOptions
from chemrefine.engines.qiskit.problem import prepare_problem
from chemrefine.engines.qiskit.registry import (
    ESTIMATORS,
    OPTIMIZERS,
    SAMPLERS,
    consumed_component_categories,
    validate_component_graph,
)
from chemrefine.engines.qiskit.spectra import (
    ExpectationSession,
    real_value,
    sector_diagnostics,
    solve_response_problem,
)
from chemrefine.engines.qiskit.workflow import run_problem
from chemrefine.errors import ConfigError

pytest.importorskip("qiskit_algorithms")
pytest.importorskip("qiskit_nature")
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp, Statevector

pytestmark = [
    pytest.mark.filterwarnings("ignore:.*:DeprecationWarning:qiskit.*"),
    pytest.mark.filterwarnings("ignore:.*:scipy.sparse.SparseEfficiencyWarning"),
]


@pytest.fixture
def h2():
    """Stored integrals isolate quantum execution from classical SCF workflows."""
    path = Path(__file__).parent / "data/engines/qiskit/h2_integrals.json"
    return prepare_problem(ElectronicStructureData(**json.loads(path.read_text())))


@pytest.fixture
def complex_problem():
    """One electron with a complex Hermitian hopping amplitude and an analytic spectrum."""
    h = np.array([[-1, 0.2 + 0.3j], [0.2 - 0.3j, 0.5]])
    return prepare_problem(ElectronicStructureData(1, 0, 2, h, np.zeros((2,) * 4))), h


def _options(algorithm, **extra):
    """Supply deterministic optimizer and sampler controls for ideal primitive tests."""
    return {
        "algorithm": algorithm,
        "initial_point": {"name": "random", "options": {"seed": 31}},
        "optimizer": {"name": "slsqp", "options": {"maxiter": 300, "ftol": 1e-12}},
        **extra,
    }


def test_qeom_h2_all_roots_match_exact_sector_with_offsets_and_triplet(h2):
    """The response pencil recovers the full four-state H2 sector, including S=1."""
    records: list[dict[str, Any]] = []
    result = run_problem(
        h2,
        options=_options({"name": "qeom", "options": {"target_root": 2}}),
        callback=records.append,
    )
    matrix = map_problem(h2).qubit_hamiltonian.to_matrix()
    indices = [5, 6, 9, 10]  # one alpha and one beta in alpha-then-beta JW ordering
    exact = np.linalg.eigvalsh(matrix[np.ix_(indices, indices)]) + sum(h2.energy_offsets.values())
    np.testing.assert_allclose(result.root_energies_hartree, exact, atol=1e-8)
    assert result.root_energies_hartree is not None
    assert result.energy_hartree == result.root_energies_hartree[2]
    solver = result.metadata["solver"]
    np.testing.assert_allclose(
        [root["AngularMomentum"] for root in solver["root_sectors"]], [0, 2, 0, 0], atol=1e-8
    )
    assert max(solver["generalized_eigenpair_residuals"]) < 1e-12
    assert records == result.metadata["evaluations"]
    records[0]["metadata"]["changed"] = True
    assert "changed" not in result.metadata["evaluations"][0]["metadata"]
    result.as_dict()


def test_complex_qeom_retains_imaginary_response_matrix_and_uses_both_observable_parts(
    complex_problem,
):
    """A complex-orbital model exercises matrix entries that a real cast would change."""
    prepared, h = complex_problem
    result = run_problem(
        prepared,
        options=_options("qeom", ansatz={"name": "uccsd", "options": {"include_imaginary": True}}),
    )
    np.testing.assert_allclose(result.root_energies_hartree, np.linalg.eigvalsh(h), atol=2e-7)
    assert np.max(np.abs(result.metadata["solver"]["hessian_imag"])) > 0.05
    assert result.metadata["solver"]["root_sectors"][1]["AngularMomentum"] == pytest.approx(0.75)


@pytest.mark.parametrize("seed", [3, 17])
def test_vqd_complex_spectrum_uses_selected_sampler_and_physical_energies(complex_problem, seed):
    """Finite-shot deflation reaches both analytic roots with derivative-free search."""
    prepared, h = complex_problem
    options = _options(
        {
            "name": "vqd",
            "options": {
                "k": 2,
                "betas": [3],
                "initial_points": [[0.2, 0.1], [0.9, 0.2]],
                "target_root": 1,
                # Resolve the excited root to 2 mEh despite sampled deflation noise.
                "fidelity_shots": 65536,
            },
        },
        ansatz={"name": "uccsd", "options": {"include_imaginary": True}},
        sampler={"name": "statevector", "options": {"seed": seed}},
        optimizer={"name": "cobyla", "options": {"maxiter": 500}},
    )
    result = run_problem(prepared, options=options)
    np.testing.assert_allclose(result.root_energies_hartree, np.linalg.eigvalsh(h), atol=2e-3)
    assert result.root_energies_hartree is not None
    assert result.energy_hartree == result.root_energies_hartree[1]
    assert result.metadata["solver"]["root_overlap_matrix"][0][1] < 0.01
    assert result.metadata["solver"]["root_sectors"][1]["ParticleNumber"] == pytest.approx(1)
    assert result.converged is None
    assert any(record["objective_includes_deflation"] for record in result.metadata["evaluations"])


def test_qeom_actual_reference_order_and_spin_enforcement(h2):
    """Excitation labels follow the configured determinant, including its orbital order."""
    context = map_problem(h2)
    state = QuantumCircuit(4)
    state.metadata = {"chemrefine_reference_occupations": {"alpha": [0, 1], "beta": [0, 1]}}
    _, excitations = _response_basis(context, state, QEOMOptions())
    assert excitations == [((1,), (0,)), ((3,), (2,)), ((1, 3), (0, 2))]
    with pytest.raises(ConfigError, match="AngularMomentum"):
        run_problem(h2, options=_options({"name": "qeom", "options": {"target_s2": 0}}))


@pytest.mark.parametrize(
    "algorithm,controls,message",
    [
        ("qeom", {"max_excitations": 1}, "max_excitations"),
        ("qeom", {"target_root": 4}, "target_root"),
        ("qeom", {"max_evaluations": 1}, "max_evaluations"),
        ("qeom", {"max_measurements": 1}, "max_measurements"),
        ("qeom", {"max_product_terms": 1}, "max_product_terms"),
        ("vqd", {"k": 5}, "sector dimension"),
    ],
)
def test_spectrum_budgets_fail_explicitly(h2, algorithm, controls, message):
    with pytest.raises(ConfigError, match=message):
        run_problem(h2, options=_options({"name": algorithm, "options": controls}))


@pytest.mark.parametrize(
    "cls,options",
    [
        (VQDOptions, {"k": 1, "target_root": 1}),
        (VQDOptions, {"betas": []}),
        (VQDOptions, {"betas": [-1]}),
        (VQDOptions, {"initial_points": [[0]]}),
        (VQDOptions, {"betas": [float("nan")]}),
        (QEOMOptions, {"excitation_ranks": []}),
        (QEOMOptions, {"excitation_ranks": [0]}),
        (QEOMOptions, {"excitation_ranks": [1, 1]}),
        (QNSPSAOptions, {"learning_rate": 0.1}),
        (QNSPSAOptions, {"perturbation": 0.1}),
        (QNSPSAOptions, {"fidelity_shots": 0}),
    ],
)
def test_spectrum_and_metric_options_are_strict(cls, options):
    with pytest.raises(ValidationError):
        cls(**options)


def test_qnspsa_real_sampler_improves_objective_and_restores_global_rng():
    """The metric comes from the declared sampler and private perturbations are reproducible."""
    from qiskit_algorithms.utils import algorithm_globals

    circuit = QuantumCircuit(1)
    circuit.ry(Parameter("theta"), 0)
    options = QNSPSAOptions(
        maxiter=20,
        blocking=False,
        learning_rate=0.15,
        perturbation=0.1,
        fidelity_shots=2048,
        seed=9,
    )
    previous = deepcopy(algorithm_globals.random.bit_generator.state)

    def optimize():
        components = SolverComponents(
            ansatz=AnsatzArtifacts(circuit=circuit), sampler=StatevectorSampler(seed=5)
        )
        optimizer = build_qnspsa(options=options, components=components)
        return optimizer.minimize(lambda x: float(np.cos(x[0])), [1.0])

    first, second = optimize(), optimize()
    assert first.fun < -0.95
    np.testing.assert_allclose(first.x, second.x, atol=0)
    assert algorithm_globals.random.bit_generator.state == previous
    optimizer = build_qnspsa(
        options=options,
        components=SolverComponents(
            ansatz=AnsatzArtifacts(circuit=circuit), sampler=StatevectorSampler(seed=5)
        ),
    )
    with pytest.raises(RuntimeError, match="objective failed"):
        optimizer.minimize(lambda x: (_ for _ in ()).throw(RuntimeError("objective failed")), [1.0])
    assert algorithm_globals.random.bit_generator.state == previous


def test_qnspsa_graph_consumes_sampler_and_rejects_adaptive_circuit():
    options = QiskitOptions(algorithm="vqe", optimizer="qnspsa")
    assert {"sampler", "estimator", "ansatz", "optimizer"} <= consumed_component_categories(options)
    validate_component_graph(options)
    for algorithm in ("adapt_vqe", "ffsim_vqe"):
        with pytest.raises(ConfigError, match="fixed-circuit"):
            validate_component_graph(QiskitOptions(algorithm=algorithm, optimizer="qnspsa"))
    with pytest.raises(ConfigError, match=r"sampler.*cuda"):
        validate_component_graph(
            QiskitOptions(
                algorithm="vqe", optimizer="qnspsa", estimator="aer_statevector", device="cuda"
            )
        )
    with pytest.raises(ConfigError, match="fixed circuit"):
        build_qnspsa(options=QNSPSAOptions(), components=SolverComponents())


def test_joint_resource_lifecycle_closes_estimator_when_sampler_or_optimizer_fails(h2, monkeypatch):
    """A partial graph failure closes already acquired providers in reverse order."""
    closed = []
    monkeypatch.setattr(
        ESTIMATORS,
        "build",
        lambda *a, **kw: EstimatorResource(object(), close=lambda: closed.append("estimator")),
    )
    options = QiskitOptions(algorithm="vqd", optimizer="qnspsa")
    context = map_problem(h2)

    def fail(*a, **kw):
        raise RuntimeError("build failed")

    monkeypatch.setattr(SAMPLERS, "build", fail)
    with pytest.raises(RuntimeError, match="build failed"), assemble_components(context, options):
        pytest.fail("partial graph must not be yielded")
    assert closed == ["estimator"]
    monkeypatch.setattr(
        SAMPLERS,
        "build",
        lambda *a, **kw: SamplerResource(object(), close=lambda: closed.append("sampler")),
    )
    monkeypatch.setattr(OPTIMIZERS, "build", fail)
    with pytest.raises(RuntimeError, match="build failed"), assemble_components(context, options):
        pytest.fail("partial graph must not be yielded")
    assert closed == ["estimator", "sampler", "estimator"]


def test_expectation_session_measures_complex_operator_with_layout_and_budget(h2):
    """Complex splitting reproduces a direct matrix element after a nontrivial circuit layout."""
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    context = map_problem(h2)
    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.s(0)
    manager = generate_preset_pass_manager(
        optimization_level=0, basis_gates=["u", "cx"], initial_layout=[3, 2, 1, 0]
    )
    components = SolverComponents(estimator=StatevectorEstimator(), transpiler=manager)
    session = ExpectationSession(context, components, circuit, 10, 10)
    operator = SparsePauliOp.from_list([("IIIY", 1 + 2j), ("IIIZ", 0.3j)])
    assert session.mapped(operator) == pytest.approx(
        Statevector(circuit).expectation_value(operator)
    )
    assert session.measurements == 2
    assert session.mapped(SparsePauliOp("IIII", 0)) == 0
    with pytest.raises(ConfigError, match="max_pauli_terms"):
        ExpectationSession(context, components, circuit, 10, 1).mapped(operator)
    with pytest.raises(ConfigError, match="max_measurements"):
        ExpectationSession(context, components, circuit, 1, 10).mapped(operator)


@pytest.mark.parametrize(
    "h,s,message",
    [
        (np.eye(3), np.eye(3), "finite square"),
        (np.array([[1, 1j], [1j, 1]]), np.eye(2), "not Hermitian"),
        (np.eye(2), np.zeros((2, 2)), "singular"),
        (np.ones((2, 2)), np.diag([1, -1]), "zero/unstable"),
        (np.diag([1, 2]), np.diag([1, -1]), "± pairs"),
        (np.diag([-1, -1]), np.diag([1, -1]), "nonpositive metric"),
        (np.array([[0, 1], [1, 0]]), np.diag([1, -1]), "complex excitation"),
    ],
)
def test_response_rejects_nonphysical_or_singular_pencils(h, s, message):
    with pytest.raises(ConfigError, match=message):
        solve_response_problem(
            h, s, conditioning_tolerance=1e-10, residual_tolerance=1e-7, frequency_tolerance=1e-6
        )


def test_response_complex_pencil_and_validation_of_auxiliary_values():
    """Unitary changes of response coordinates preserve real gaps and metric normalization."""
    u = np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
    h = u @ np.diag([2, 2]) @ u.conj().T
    s = u @ np.diag([1, -1]) @ u.conj().T
    gaps, vectors, _ = solve_response_problem(
        h, s, conditioning_tolerance=1e-10, residual_tolerance=1e-7, frequency_tolerance=1e-6
    )
    np.testing.assert_allclose(gaps, [2])
    np.testing.assert_allclose(vectors.conj().T @ s @ vectors, np.eye(1))
    for value in [complex("nan"), 1j]:
        with pytest.raises(ConfigError, match="finite and real"):
            real_value(value, "test")
    with pytest.raises(ConfigError, match="parameterized"):
        _check_circuit(QuantumCircuit(1))


@pytest.fixture
def stationary_optimizer(monkeypatch):
    """Return the requested starting point to expose failed deflation explicitly."""

    class Optimizer:
        def minimize(self, fun, x0, **kwargs):
            values = np.asarray(x0)
            return SimpleNamespace(x=values, fun=fun(values), nfev=1)

    monkeypatch.setattr(OPTIMIZERS, "build", lambda *args, **kwargs: Optimizer())


def test_vqd_single_root_supplied_initial_point_and_duplicate_root_failure(
    h2, stationary_optimizer
):
    result = run_problem(
        h2, options=_options({"name": "vqd", "options": {"k": 1}}), initial_point=[0, 0, 0]
    )
    assert result.metadata["solver"]["optimal_points"] == [[0, 0, 0]]
    assert result.metadata["solver"]["root_overlap_matrix"] == [[1]]
    with pytest.raises(ConfigError, match="overlap_tolerance"):
        run_problem(h2, options=_options("vqd"), initial_point=[0, 0, 0])
    with pytest.raises(ConfigError, match="not both"):
        run_problem(
            h2,
            options=_options(
                {"name": "vqd", "options": {"initial_points": [[0, 0, 0], [0, 0, 0]]}}
            ),
            initial_point=[0, 0, 0],
        )


def test_qeom_detects_invalid_upstream_basis_and_zero_reconstruction(
    h2, monkeypatch, stationary_optimizer
):
    import qiskit_nature.second_q.circuit.library.ansatzes.utils as utils

    import chemrefine.engines.qiskit.components.spectra as implementation

    context = map_problem(h2)
    with monkeypatch.context() as local:
        local.setattr(utils, "generate_fermionic_excitations", lambda *a, **kw: [])
        with pytest.raises(ConfigError, match="max_excitations"):
            _response_basis(context, None, QEOMOptions())
    monkeypatch.setattr(
        implementation,
        "solve_response_problem",
        lambda *a, **kw: (np.array([1]), np.zeros((6, 1)), {}),
    )
    with pytest.raises(ConfigError, match="zero norm"):
        run_problem(h2, options=_options("qeom"))
    from qiskit_nature.second_q.operators import FermionicOp

    with pytest.raises(ConfigError, match="after normal ordering"):
        _product(FermionicOp({"-_0": 1}), FermionicOp({"+_0": 1}), QEOMOptions(max_product_terms=1))


def test_qnspsa_nonfinite_metric_fails_before_using_bad_hessian(monkeypatch):
    import qiskit_algorithms.state_fidelities as fidelities

    class Metric:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            return SimpleNamespace(result=lambda: SimpleNamespace(fidelities=[float("nan")]))

    monkeypatch.setattr(fidelities, "ComputeUncompute", Metric)
    circuit = QuantumCircuit(1)
    circuit.ry(Parameter("x"), 0)
    optimizer = build_qnspsa(
        options=QNSPSAOptions(),
        components=SolverComponents(ansatz=AnsatzArtifacts(circuit=circuit), sampler=object()),
    )
    with pytest.raises(ConfigError, match="nonfinite"):
        optimizer.fidelity([0], [0.1])


@pytest.mark.parametrize(
    "payload,message",
    [
        ([], "result count"),
        ([float("nan")], "nonfinite/nonreal"),
        ([1j], "nonfinite/nonreal"),
    ],
)
def test_expectation_rejects_invalid_provider_payload(h2, payload, message):
    values = [SimpleNamespace(data=SimpleNamespace(evs=np.asarray(value))) for value in payload]
    estimator = SimpleNamespace(run=lambda pubs: SimpleNamespace(result=lambda: values))
    session = ExpectationSession(
        map_problem(h2), SolverComponents(estimator=estimator), QuantumCircuit(4), 10, 10
    )
    with pytest.raises(ConfigError, match=message):
        session.mapped(SparsePauliOp("IIIZ"))


def test_missing_estimator_observable_or_tapered_sector_is_not_silently_ignored(h2):
    context = map_problem(h2)
    session = ExpectationSession(context, SolverComponents(), QuantumCircuit(4), 10, 10)
    with pytest.raises(ConfigError, match="requires an estimator"):
        session.mapped(SparsePauliOp("IIIZ"))
    session.context = replace(context, mapper=SimpleNamespace(map=lambda op: None))
    from qiskit_nature.second_q.operators import FermionicOp

    with pytest.raises(ConfigError, match="tapering sector"):
        session.fermionic(FermionicOp({"+_0 -_1": 1}))
    with pytest.raises(ConfigError, match="ParticleNumber observable"):
        sector_diagnostics(
            replace(context, problem=SimpleNamespace(second_q_ops=lambda: (None, {}))),
            lambda op: 0,
            tolerance=1e-3,
            target_s2=None,
            spin_tolerance=1e-3,
        )


@pytest.mark.parametrize(
    "frequencies,vectors,message",
    [
        (np.array([complex("nan"), 1]), np.eye(2, dtype=complex), "nonfinite"),
        (
            np.array([-1, 1], dtype=complex),
            np.array([[0, 1], [1, 0.5]], dtype=complex),
            "residual_tolerance",
        ),
    ],
)
def test_generalized_eigenpair_outputs_are_validated(monkeypatch, frequencies, vectors, message):
    import scipy.linalg

    monkeypatch.setattr(scipy.linalg, "eig", lambda *a: (frequencies, vectors))
    with pytest.raises(ConfigError, match=message):
        solve_response_problem(
            np.eye(2),
            np.diag([1, -1]),
            conditioning_tolerance=1e-10,
            residual_tolerance=1e-7,
            frequency_tolerance=1e-6,
        )


def test_qnspsa_runs_through_standard_vqe_component_graph(complex_problem):
    prepared, _ = complex_problem
    result = run_problem(
        prepared,
        initial_point=[0.2, 0.1],
        options={
            "algorithm": "vqe",
            "ansatz": {"name": "uccsd", "options": {"include_imaginary": True}},
            "optimizer": {
                "name": "qnspsa",
                "options": {
                    "maxiter": 20,
                    "learning_rate": 0.1,
                    "perturbation": 0.1,
                    "blocking": False,
                    "seed": 9,
                },
            },
            "sampler": {"name": "statevector", "options": {"seed": 5}},
        },
    )
    assert result.energy_hartree < -1.07
    assert result.optimizer == "qnspsa"
    assert result.energy_evaluation_count is not None
    assert result.energy_evaluation_count >= 40

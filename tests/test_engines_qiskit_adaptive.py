"""Real primitive adaptive growth, optimizer rebuilding and retained-state diagnostics."""

import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from chemrefine.engines.qiskit import adaptive
from chemrefine.engines.qiskit.assembly import assemble_components
from chemrefine.engines.qiskit.components.adaptive import AdaptiveOptions, CEOOptions
from chemrefine.engines.qiskit.context import AnsatzArtifacts, EstimatorResource, SamplerResource
from chemrefine.engines.qiskit.data import ElectronicStructureData
from chemrefine.engines.qiskit.mapping import map_problem
from chemrefine.engines.qiskit.native import NativeSolveRequest
from chemrefine.engines.qiskit.operators import OperatorPool
from chemrefine.engines.qiskit.options import QiskitOptions
from chemrefine.engines.qiskit.problem import prepare_problem
from chemrefine.engines.qiskit.registry import (
    ESTIMATORS,
    SAMPLERS,
    consumed_component_categories,
    validate_component_graph,
)
from chemrefine.engines.qiskit.workflow import run_problem
from chemrefine.errors import ConfigError

pytest.importorskip("qiskit_nature")
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorEstimator, StatevectorSampler

pytestmark = [
    pytest.mark.filterwarnings("ignore:.*:DeprecationWarning:qiskit.*"),
    pytest.mark.filterwarnings("ignore:.*:scipy.sparse.SparseEfficiencyWarning"),
]


@pytest.fixture
def h2():
    """Use stored integrals rather than introducing an SCF reference calculation."""
    return prepare_problem(
        ElectronicStructureData(
            **json.loads(
                (Path(__file__).parent / "data/engines/qiskit/h2_integrals.json").read_text()
            )
        )
    )


def options(algorithm="tetris_adapt", ansatz="uccsd", **extra):
    """Set a tight deterministic inner solve for convergence comparisons."""
    return {
        "algorithm": algorithm,
        "ansatz": ansatz,
        "optimizer": {"name": "slsqp", "options": {"maxiter": 200, "ftol": 1e-12}},
        **extra,
    }


@pytest.mark.parametrize(
    "algorithm,ansatz,cx",
    [
        ("tetris_adapt", "uccsd", 48),
        ("tetris_adapt", "qe", 13),
        ("ceo_adapt", "ceo", 9),
    ],
)
def test_h2_adaptive_energy_offsets_convergence_and_circuit_cost(h2, algorithm, ansatz, cx):
    records: list[dict[str, Any]] = []
    result = run_problem(h2, options=options(algorithm, ansatz), callback=records.append)
    assert result.energy_hartree == pytest.approx(-1.1373060357534, abs=2e-10)
    assert result.converged
    diagnostics = result.metadata["solver"]
    assert diagnostics["gradient_history"][-1]["gradient_norm"] < 1e-5
    assert diagnostics["logical_circuit_metrics"]["cx_count"] == cx
    assert diagnostics["final_sector"]["ParticleNumber"] == pytest.approx(2)
    assert diagnostics["final_sector"]["AngularMomentum_squared_residual"] < 1e-12
    assert diagnostics["optimizer_rebuilt_each_inner_solve"]
    assert records and records[0]["inner_run"] == 1
    records[0]["metadata"]["changed"] = True
    assert "changed" not in result.metadata["evaluations"][0]["metadata"]


@pytest.mark.parametrize("variant", ["ovp", "mvp"])
def test_h2_declared_ceo_variants(h2, variant):
    result = run_problem(
        h2, options=options({"name": "ceo_adapt", "options": {"variant": variant}}, "ceo")
    )
    assert result.energy_hartree == pytest.approx(-1.1373060357534, abs=2e-10)
    assert result.metadata["solver"]["selected_blocks"][0]["kind"] == variant


@pytest.mark.parametrize("ansatz", ["uccsd", "ceo"])
def test_tapered_actual_reference_pool_and_observables(h2, ansatz):
    algorithm = "tetris_adapt" if ansatz == "uccsd" else "ceo_adapt"
    result = run_problem(h2, options=options(algorithm, ansatz, mapper="z2_tapered"))
    assert result.energy_hartree == pytest.approx(-1.1373060357534, abs=2e-10)
    assert result.num_qubits is not None
    assert result.num_qubits < 4
    assert result.metadata["solver"]["final_sector"]["ParticleNumber"] == pytest.approx(2)


def test_complex_quadratures_reach_one_electron_exact_spectrum():
    h = np.array([[-1, 0.2 + 0.3j], [0.2 - 0.3j, 0.5]])
    prepared = prepare_problem(ElectronicStructureData(1, 0, 2, h, np.zeros((2,) * 4)))
    result = run_problem(
        prepared,
        options=options(
            {"name": "tetris_adapt", "options": {"max_iterations": 15, "eigenvalue_threshold": 0}},
            {"name": "qe", "options": {"include_imaginary": True}},
        ),
    )
    assert result.energy_hartree == pytest.approx(np.linalg.eigvalsh(h)[0], abs=2e-10)
    assert result.metadata["solver"]["final_sector"]["AngularMomentum"] == pytest.approx(0.75)


def test_qnspsa_uses_new_fidelity_circuit_after_each_growth(monkeypatch):
    h = np.array([[-1.0, 0.3, 0], [0.3, 0.1, 0.4], [0, 0.4, 0.8]])
    prepared = prepare_problem(ElectronicStructureData(1, 0, 3, h, np.zeros((3,) * 4)))
    actual = adaptive.build_optimizer
    parameter_counts, optimizer_ids = [], []

    def record(*args):
        parameter_counts.append(args[2].ansatz.circuit.num_parameters)
        optimizer = actual(*args)
        optimizer_ids.append(optimizer)
        return optimizer

    monkeypatch.setattr(adaptive, "build_optimizer", record)
    result = run_problem(
        prepared,
        options=options(
            {
                "name": "ceo_adapt",
                "options": {"max_iterations": 3, "tetris": False, "eigenvalue_threshold": 0},
            },
            "ceo",
            optimizer={
                "name": "qnspsa",
                "options": {
                    "maxiter": 35,
                    "learning_rate": 0.15,
                    "perturbation": 0.1,
                    "blocking": True,
                    "allowed_increase": 0.001,
                    "seed": 12,
                    "fidelity_shots": 1024,
                },
            },
            sampler={"name": "statevector", "options": {"seed": 3}},
        ),
    )
    assert parameter_counts[:2] == [1, 2]
    assert len({id(item) for item in optimizer_ids}) == len(optimizer_ids)
    assert result.energy_hartree < -1.08
    assert {item["inner_run"] for item in result.metadata["evaluations"]} >= {1, 2}


@pytest.mark.parametrize(
    "algorithm,extra,match",
    [
        ("ceo_adapt", {}, "ceo_pool"),
        ("tetris_adapt", {"ansatz": "efficient_su2"}, "operator_pool"),
        ("tetris_adapt", {"ansatz": {"name": "uccsd", "options": {"reps": 2}}}, "reps to 1"),
    ],
)
def test_invalid_graph_rejects_unsupported_pools(algorithm, extra, match):
    with pytest.raises(ConfigError, match=match):
        validate_component_graph(QiskitOptions(**{"algorithm": algorithm, **extra}))


def test_owned_adaptive_qnspsa_closure_and_reference_aware_exact():
    resolved = QiskitOptions(algorithm="tetris_adapt", optimizer="qnspsa", ansatz="qe")
    validate_component_graph(resolved)
    assert {
        "sampler",
        "ansatz",
        "initial_state",
        "estimator",
        "optimizer",
    } <= consumed_component_categories(resolved)
    assert "initial_state" in consumed_component_categories(
        QiskitOptions(algorithm="exact", mapper="z2_tapered")
    )
    with pytest.raises(ValidationError):
        AdaptiveOptions(gradient_threshold=float("nan"))
    with pytest.raises(ValidationError):
        CEOOptions(variant="not_ceo")


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("max_pool_size", 1, "max_pool_size"),
        ("max_product_terms", 1, "max_product_terms"),
        ("max_measurements", 1, "max_measurements"),
        ("max_pauli_terms", 1, "max_pauli_terms"),
        ("max_evaluations", 1, "max_evaluations"),
    ],
)
def test_adaptive_resource_budgets_fail_explicitly(h2, field, value, match):
    with pytest.raises(ConfigError, match=match):
        run_problem(h2, options=options({"name": "tetris_adapt", "options": {field: value}}))


def test_no_candidate_and_max_iteration_are_not_convergence(h2):
    no_candidate = run_problem(
        h2, options=options({"name": "tetris_adapt", "options": {"selection_threshold": 10}})
    )
    assert not no_candidate.converged
    assert no_candidate.termination_reason == "no_candidate_above_selection_threshold"
    limited = run_problem(
        h2, options=options({"name": "tetris_adapt", "options": {"max_iterations": 1}})
    )
    assert not limited.converged
    assert limited.termination_reason == "maximum_iterations"


def test_owned_initial_point_rejected_before_provider(h2):
    for extra in ({"initial_point": [0]}, {"options": options(initial_point="random")}):
        with pytest.raises(ConfigError, match="no parameters"):
            run_problem(h2, **({"options": options()} | extra))


def test_external_pool_preserves_operator_not_potentially_misleading_metadata(h2):
    context = map_problem(h2)
    resolved = QiskitOptions(**options())
    with assemble_components(context, resolved, defer_optimizer=True) as components:
        pool = OperatorPool(
            tuple(components.ansatz.operator_pool),
            tuple(
                {"quadrature": "antisymmetric", "source": [0], "target": [1], "role": "qe"}
                for _ in components.ansatz.operator_pool
            ),
        )
    result = run_problem(h2, options=options(ansatz="qe"), operator_pool=pool)
    assert result.energy_hartree == pytest.approx(-1.1373060357534, abs=2e-10)
    assert result.metadata["solver"]["logical_circuit_metrics"]["cx_count"] == 48


def test_provider_cleanup_on_callback_failure(h2, monkeypatch):
    closed = []
    monkeypatch.setattr(
        ESTIMATORS,
        "build",
        lambda *_args, **_kwargs: EstimatorResource(
            StatevectorEstimator(), close=lambda: closed.append("estimator")
        ),
    )
    monkeypatch.setattr(
        SAMPLERS,
        "build",
        lambda *_args, **_kwargs: SamplerResource(
            StatevectorSampler(seed=2), close=lambda: closed.append("sampler")
        ),
    )

    def fail(_record):
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        run_problem(
            h2,
            options=options(optimizer={"name": "qnspsa", "options": {"maxiter": 1}}),
            callback=fail,
        )
    assert closed == ["sampler", "estimator"]


@pytest.mark.parametrize(
    "bad,match",
    [
        ("pool", "operator pool"),
        ("reference", "prepared initial state"),
        ("parameters", "fully bound"),
        ("family", "coupled-exchange pool"),
    ],
)
def test_malformed_plugin_artifacts_fail_with_resource_cleanup(h2, monkeypatch, bad, match):
    closed = []

    @contextmanager
    def malformed(*args, **kwargs):
        with assemble_components(*args, **kwargs) as built:
            if bad == "pool":
                built = replace(built, ansatz=AnsatzArtifacts())
            elif bad == "reference":
                built = replace(built, initial_state=None)
            elif bad == "parameters":
                circuit = QuantumCircuit(4)
                circuit.ry(Parameter("unfinished_reference"), 0)
                built = replace(built, initial_state=circuit)
            try:
                yield built
            finally:
                closed.append(True)

    monkeypatch.setattr(adaptive, "assemble_components", malformed)
    request = NativeSolveRequest(h2, QiskitOptions(**options()))
    with pytest.raises(ConfigError, match=match):
        adaptive.adaptive_solve(request, AdaptiveOptions(), ceo=bad == "family")
    assert closed == [True]


def test_parameter_budget_is_checked_before_adding_mvp(h2):
    with pytest.raises(ConfigError, match="max_parameters"):
        run_problem(
            h2,
            options=options(
                {"name": "ceo_adapt", "options": {"variant": "mvp", "max_parameters": 1}}, "ceo"
            ),
        )


def test_repeated_selection_and_energy_rollback_retain_correct_state(h2, monkeypatch):
    import qiskit_algorithms

    value = 0.0

    class FixedInnerSolve:
        def __init__(self, _estimator, circuit, _optimizer, **_kwargs):
            self.circuit = circuit

        def compute_minimum_eigenvalue(self, _hamiltonian):
            return SimpleNamespace(optimal_parameters=dict.fromkeys(self.circuit.parameters, value))

    monkeypatch.setattr(qiskit_algorithms, "VQE", FixedInnerSolve)
    selected = options({"name": "tetris_adapt", "options": {"eigenvalue_threshold": 0}})
    repeated = run_problem(h2, options=selected)
    assert repeated.termination_reason == "repeated_selection"
    assert repeated.parameter_count == 1
    assert not repeated.converged
    value = np.pi / 2
    rolled_back = run_problem(h2, options=selected)
    assert rolled_back.termination_reason == "energy_increase_rollback"
    assert rolled_back.parameter_count == 0
    assert rolled_back.energy_hartree == pytest.approx(repeated.energy_hartree)
    assert rolled_back.metadata["solver"]["selected_blocks"] == []
    assert not rolled_back.metadata["solver"]["gradient_history"][0]["retained"]


def test_energy_criterion_distinct_from_gradient_convergence(h2):
    result = run_problem(
        h2,
        options=options(
            {
                "name": "tetris_adapt",
                "options": {
                    "eigenvalue_threshold": 1.0,
                    "gradient_norm": "max",
                },
            }
        ),
    )
    assert result.converged
    assert result.termination_reason == "energy_converged"
    assert len(result.metadata["solver"]["gradient_history"]) == 1


def test_tetris_batches_two_disjoint_spin_channels_in_one_iteration():
    h = np.array([[-1, 0.2], [0.2, 0.5]])
    prepared = prepare_problem(ElectronicStructureData(1, 1, 2, h, np.zeros((2,) * 4)))
    result = run_problem(prepared, options=options(ansatz="qe"))
    blocks = result.metadata["solver"]["gradient_history"][0]["blocks"]
    assert [block["support"] for block in blocks] == [[0, 1], [2, 3]]
    assert result.parameter_count == 2
    assert result.energy_hartree == pytest.approx(2 * np.linalg.eigvalsh(h)[0], abs=1e-10)
    assert result.metadata["solver"]["logical_circuit_metrics"]["cx_count"] == 4


def test_explicit_ceo_product_formula_is_not_replaced_with_optimized_network(h2):
    result = run_problem(
        h2,
        options=options(
            {"name": "ceo_adapt", "options": {"evolution": "suzuki", "repetitions": 2}}, "ceo"
        ),
    )
    assert result.energy_hartree == pytest.approx(-1.1373060357534, abs=2e-10)
    assert result.metadata["solver"]["logical_circuit_metrics"]["cx_count"] > 9

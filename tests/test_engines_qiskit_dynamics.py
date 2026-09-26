"""Variational time evolution agrees with independent complex matrix propagation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit, qpy
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.linalg import expm

from chemrefine.engines.qiskit.bundles import read_bundle
from chemrefine.engines.qiskit.context import EstimatorResource
from chemrefine.engines.qiskit.dynamics import (
    VariationalDynamicsOptions,
    metric_velocity,
    variational_dynamics,
)
from chemrefine.engines.qiskit.execution import logical_estimator
from chemrefine.engines.qiskit.experiment import run_experiment
from chemrefine.engines.qiskit.options import ComponentSelection
from chemrefine.engines.qiskit.registry import ESTIMATORS
from chemrefine.errors import ConfigError

# Qiskit 1.4's own converter touches deprecated DAG timing fields while compiling
# LCU circuits. These two upstream warnings do not occur on the current provider.
pytestmark = [
    pytest.mark.filterwarnings(
        r"ignore:The property ``qiskit.dagcircuit.dagcircuit.DAGCircuit.(duration|unit)`` "
        r"is deprecated:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:The property ``qiskit.circuit.instruction.Instruction.condition`` "
        r"is deprecated:DeprecationWarning"
    ),
]


def _ry():
    """Return the complete real one-qubit state manifold."""
    circuit = QuantumCircuit(1)
    circuit.ry(Parameter("theta"), 0)
    return circuit


def test_imaginary_time_tracks_normalized_matrix_exponential():
    """RK4 refinement reduces error for the nonlinear normalized imaginary-time flow."""
    errors = []
    for steps in (2, 8):
        result = variational_dynamics(
            _ry(),
            SparsePauliOp("Z"),
            [np.pi / 2],
            options=VariationalDynamicsOptions(time=0.5, steps=steps),
        )
        errors.append(abs(result.expectations[-1, 0] + np.tanh(1)))
        assert np.all(np.diff(result.expectations[:, 0]) < 0)
        assert result.times[0] == 0
        assert not result.parameters.flags.writeable
    assert errors[1] < errors[0] / 100
    assert errors[1] < 1e-6


@pytest.mark.parametrize("method", ["varqite", "varqrte"])
def test_complex_hopping_dynamics_in_one_particle_sector(method):
    """Complex current terms and unrestricted phases retain their physical signs."""
    theta, phi = Parameter("theta"), Parameter("phi")
    circuit = QuantumCircuit(2)
    circuit.x(1)
    circuit.ry(theta, 0)
    circuit.cx(0, 1)
    circuit.rz(phi, 0)
    hamiltonian = SparsePauliOp.from_list(
        [("XX", 0.3), ("YY", 0.3), ("XY", 0.2), ("YX", -0.2), ("ZI", 0.1)]
    )
    point = [0.2, 0.8]  # Lexical parameter order is phi, theta.
    result = variational_dynamics(
        circuit,
        hamiltonian,
        point,
        options=VariationalDynamicsOptions(method=method, time=0.2, steps=8),
        observables=(SparsePauliOp("ZI"),),
    )
    initial = Statevector(circuit.assign_parameters(point)).data
    exact = expm((-1 if method == "varqite" else -1j) * 0.2 * hamiltonian.to_matrix()) @ initial
    exact /= np.linalg.norm(exact)
    evolved = Statevector(circuit.assign_parameters(result.parameters[-1])).data
    assert abs(np.vdot(exact, evolved)) ** 2 > 1 - 1e-9
    assert abs(evolved[0]) + abs(evolved[3]) < 1e-12
    assert result.expectations[-1, 1] == pytest.approx(
        float(np.vdot(exact, SparsePauliOp("ZI").to_matrix() @ exact).real), abs=1e-7
    )


def test_selected_estimator_receives_geometry_gradients_and_grants(monkeypatch):
    """All derivative circuits reach the requested resource and are closed on failure."""
    calls, closed, grants = [], [], []
    real = StatevectorEstimator()

    def run(pubs):
        """Record actual logical publications while evaluating numerically."""
        pubs = list(pubs)
        calls.extend(pubs)
        return real.run(pubs)

    monkeypatch.setattr(ESTIMATORS, "_specs", dict(ESTIMATORS._specs))

    @ESTIMATORS.register("observed")
    def build(**kwargs):
        """Return an instrumented actual provider with explicit cleanup."""
        grants.append((kwargs["device"], kwargs["cores"]))
        return EstimatorResource(SimpleNamespace(run=run), close=lambda: closed.append(True))

    selection = ComponentSelection.named("observed")
    result = variational_dynamics(
        _ry(),
        SparsePauliOp("Z"),
        [1.0],
        estimator=selection,
        cores=3,
        options=VariationalDynamicsOptions(steps=1, time=0.01, integrator="euler"),
    )
    assert len(calls) > 2 and any(pub.circuit.num_qubits == 2 for pub in calls)
    assert grants == [("cpu", 3)] and closed == [True]
    assert result.metadata["publications"] >= len(calls)
    with pytest.raises(ConfigError, match="max_publications"):
        variational_dynamics(
            _ry(),
            SparsePauliOp("Z"),
            [1.0],
            estimator=selection,
            observables=(SparsePauliOp("X"),),
            options=VariationalDynamicsOptions(max_publications=1),
        )
    assert closed == [True, True]


def test_real_aer_estimator_compiles_derivative_ancillas():
    """Provider compilation reaches both base and LCU-ancilla circuits with exact values."""
    result = variational_dynamics(
        _ry(),
        SparsePauliOp("Z"),
        [1.0],
        estimator=ComponentSelection.named("aer_statevector"),
        options=VariationalDynamicsOptions(time=0.01, steps=1, integrator="euler"),
    )
    assert result.parameters[-1, 0] == pytest.approx(1 + 0.02 * np.sin(1), abs=1e-10)


@pytest.mark.parametrize(
    "matrix,force",
    [
        ([[1j]], [0]),
        ([[1]], [1j]),
        ([[np.nan]], [0]),
        ([[1, 0]], [0]),
        ([[1]], []),
        ([[1, 1], [0, 1]], [1, 1]),
    ],
)
def test_invalid_metric_or_force_is_rejected(matrix, force):
    """A malformed or nonreal geometry cannot silently generate a parameter update."""
    with pytest.raises(ConfigError, match="finite real"):
        metric_velocity(matrix, force, VariationalDynamicsOptions())


def test_metric_truncation_regularization_and_velocity_guards():
    """Null directions are explicit, ridge bias is visible, and negative geometry fails."""
    velocity, diagnostics = metric_velocity(
        np.diag([0.0, 0.25]), [0.0, 0.5], VariationalDynamicsOptions(regularization=0.25)
    )
    assert np.array_equal(velocity, [0, 1])
    assert diagnostics == [1, 0, 0.25, 0.25, 1]
    with pytest.raises(ConfigError, match="negative eigenvalue"):
        metric_velocity([[-0.1]], [0], VariationalDynamicsOptions())
    with pytest.raises(ConfigError, match="max_velocity"):
        metric_velocity([[0.25]], [2], VariationalDynamicsOptions(max_velocity=1))


def test_dynamics_rejects_bad_inputs_and_memory_before_provider():
    """No allocation or execution follows an invalid circuit, observable or initial state."""
    circuit, hamiltonian = _ry(), SparsePauliOp("Z")
    for parameters in ([1j], [np.nan], []):
        with pytest.raises(ConfigError):
            variational_dynamics(circuit, hamiltonian, parameters)
    with pytest.raises(ConfigError, match="Hermitian Pauli"):
        variational_dynamics(circuit, SparsePauliOp("X", coeffs=[1j]), [0])
    with pytest.raises(ConfigError, match="Hermitian Pauli"):
        variational_dynamics(circuit, SparsePauliOp("XX"), [0])
    large = QuantumCircuit(20)
    large.ry(Parameter("theta"), 0)
    with pytest.raises(ConfigError, match="max_memory_mb"):
        variational_dynamics(
            large, SparsePauliOp("Z" * 20), [0], options=VariationalDynamicsOptions(max_memory_mb=1)
        )


def test_invalid_estimator_values_fail_and_close(monkeypatch):
    """Invalid observed quantities cannot be published as a scientific trajectory."""
    closed = []
    resource = EstimatorResource(
        SimpleNamespace(
            run=lambda pubs: SimpleNamespace(
                result=lambda: [SimpleNamespace(data=SimpleNamespace(evs=[complex("nan")]))]
            )
        ),
        close=lambda: closed.append(True),
    )
    monkeypatch.setattr(ESTIMATORS, "build", lambda *args, **kwargs: resource)
    with pytest.raises(ConfigError, match="invalid expectation"):
        variational_dynamics(_ry(), SparsePauliOp("Z"), [0])
    assert closed == [True]


def test_logical_estimator_reorders_layout_and_rejects_parameter_loss():
    """Layout expansion maps observables while preserving logical parameter identities."""
    from qiskit.transpiler import CouplingMap
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    circuit = _ry()
    transpiler = generate_preset_pass_manager(
        optimization_level=0, coupling_map=CouplingMap.from_line(3), initial_layout=[2]
    )
    wrapped = logical_estimator(
        EstimatorResource(StatevectorEstimator(), transpiler=transpiler), max_publications=20
    )
    values = (
        wrapped.run([(circuit, [SparsePauliOp("X"), SparsePauliOp("Z")], [0.3])])
        .result()[0]
        .data.evs
    )
    assert values == pytest.approx([np.sin(0.3), np.cos(0.3)])
    broken = logical_estimator(
        EstimatorResource(
            StatevectorEstimator(),
            transpiler=SimpleNamespace(run=lambda circuit: QuantumCircuit(1)),
        ),
        max_publications=20,
    )
    with pytest.raises(ConfigError, match="parameter set"):
        broken.run([(circuit, SparsePauliOp("Z"), [0.3])])


@pytest.mark.parametrize("method", ["matrix_product_state", "density_matrix"])
def test_finite_shot_aer_geometry_uses_declared_simulation_method(method):
    """Non-statevector sampling methods run measured geometry through the same adapter."""
    result = variational_dynamics(
        _ry(),
        SparsePauliOp("Z"),
        [1.0],
        estimator=ComponentSelection(
            name="aer_shots",
            options={
                "method": method,
                "default_precision": 0.01,
                "seed_simulator": 19,
            },
        ),
        options=VariationalDynamicsOptions(
            time=0.01, steps=1, integrator="euler", numerical_tolerance=0.05
        ),
    )
    assert result.parameters[-1, 0] == pytest.approx(1 + 0.02 * np.sin(1), abs=0.005)


def test_variational_yaml_uses_qpy_and_manifest_contract(tmp_path):
    """The registered experiment persists numeric arrays through the shared artifact engine."""
    path = tmp_path / "state.qpy"
    with path.open("wb") as stream:
        qpy.dump(_ry(), stream)
    output = tmp_path / "artifact.json"
    run_experiment(
        {
            "experiment": {
                "name": "variational_dynamics",
                "options": {
                    "circuit_path": str(path),
                    "observable": {"Z": 1},
                    "observables": [{"X": 1}],
                    "initial_parameters": [1.0],
                    "estimator": "statevector",
                    "dynamics": {"steps": 1, "time": 0.1, "integrator": "euler"},
                },
            }
        },
        output,
    )
    bundle = read_bundle(output)
    assert bundle.description.kind == "variational_trajectory"
    assert bundle.arrays["times"].tolist() == [0, 0.1]
    assert bundle.arrays["expectations"].shape == (2, 2)
    from chemrefine.engines.qiskit.experiment_dynamics import VariationalExperimentOptions

    with pytest.raises(ValueError, match="Pauli labels"):
        VariationalExperimentOptions(
            circuit_path="unused",
            observable={"Z": 1},
            initial_parameters=(0,),
            observables=({"A": 1},),
        )

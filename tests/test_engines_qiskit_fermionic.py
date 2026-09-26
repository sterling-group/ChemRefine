"""Real-stack checks of integral conventions, occupations, and native UCJ energies."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from chemrefine.engines.qiskit.components.fermionic_algorithms import (
    FfsimVQEOptions,
    build_ffsim_vqe,
    validate_ffsim_options,
)
from chemrefine.engines.qiskit.data import ElectronicStructureData, MolecularMetadata
from chemrefine.engines.qiskit.fermionic import (
    fermionic_integrals,
    to_fermion_operator,
    to_ffsim_hamiltonian,
)
from chemrefine.engines.qiskit.native import NativeSolveRequest
from chemrefine.engines.qiskit.options import ActiveSpaceOptions, QiskitOptions
from chemrefine.engines.qiskit.problem import prepare_problem
from chemrefine.engines.qiskit.registry import OPTIMIZERS
from chemrefine.errors import ConfigError

pytest.importorskip("qiskit_nature")
ffsim = pytest.importorskip("ffsim")


@pytest.fixture
def h2_data():
    """Read a stored molecule without launching an electronic-structure workflow."""
    path = Path(__file__).parent / "data/engines/qiskit/h2_integrals.json"
    return ElectronicStructureData(**json.loads(path.read_text()))


@pytest.mark.parametrize("physicist", [False, True])
@pytest.mark.parametrize("reordered", [False, True])
def test_active_hamiltonian_matches_mapped_operator_and_actual_determinant(
    h2_data, physicist, reordered
):
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    if physicist:
        h2_data = replace(
            h2_data,
            two_body_integrals=h2_data.two_body_integrals.transpose(0, 2, 3, 1),
            two_body_order="physicist",
        )
    prepared = prepare_problem(
        h2_data,
        active_space=ActiveSpaceOptions(active_orbitals=[1, 0]) if reordered else None,
    )
    data = fermionic_integrals(prepared)
    assert data.occupations == (((1,), (1,)) if reordered else ((0,), (0,)))
    hamiltonian = to_ffsim_hamiltonian(data)
    assert hamiltonian.constant == 0.0
    linear_operator = ffsim.linear_operator(hamiltonian, data.norb, data.nelec)
    mapped = JordanWignerMapper().map(prepared.fermionic_hamiltonian).to_matrix()
    rng = np.random.default_rng(18)
    vector = rng.normal(size=4) + 1j * rng.normal(size=4)
    vector /= np.linalg.norm(vector)
    qubit_vector = ffsim.qiskit.ffsim_vec_to_qiskit_vec(vector, data.norb, data.nelec)
    np.testing.assert_allclose(
        ffsim.qiskit.ffsim_vec_to_qiskit_vec(linear_operator @ vector, data.norb, data.nelec),
        mapped @ qubit_vector,
        atol=1e-12,
    )
    reference = ffsim.slater_determinant(data.norb, data.occupations)
    assert np.vdot(reference, linear_operator @ reference).real == pytest.approx(
        -1.836967991202984, abs=1e-12
    )
    assert not data.h1.flags.writeable
    assert not data.h2.flags.writeable


def test_compressed_chemist_integrals_preserve_released_fermion_operator(h2_data):
    pytest.importorskip("qiskit_fermions")
    from qiskit_nature.second_q.operators import ElectronicIntegrals, FermionicOp, PolynomialTensor
    from qiskit_nature.second_q.operators.symmetric_two_body import S1Integrals, fold

    prepared = prepare_problem(h2_data)
    prepared.problem.hamiltonian.electronic_integrals = ElectronicIntegrals(
        alpha=PolynomialTensor(
            {
                "+-": h2_data.one_body_integrals,
                "++--": fold(S1Integrals(h2_data.two_body_integrals)),
            }
        )
    )
    data = fermionic_integrals(prepared)
    np.testing.assert_array_equal(data.h2, h2_data.two_body_integrals)
    fermion_operator = to_fermion_operator(data)
    converted = FermionicOp(
        {
            " ".join(f"{'+' if creation else '-'}_{index}" for creation, index in term): coefficient
            for term, coefficient in fermion_operator.iter_terms()
        },
        num_spin_orbitals=2 * data.norb,
    )
    assert converted.normal_order().equiv(prepared.fermionic_hamiltonian.normal_order())


def test_frozen_inactive_and_nuclear_offsets_remain_outside_native_hamiltonian():
    original = ElectronicStructureData(
        2,
        2,
        4,
        np.diag([-2.0, -1.0, 0.0, 1.0]),
        np.zeros((4,) * 4),
        molecular_metadata=MolecularMetadata(("Li", "H"), ((0, 0, 0), (0, 0, 1.6))),
        nuclear_repulsion_energy=1.0,
    )
    prepared = prepare_problem(
        original,
        freeze_core=True,
        active_space=ActiveSpaceOptions(electrons=2, orbitals=2, active_orbitals=[3, 1]),
    )
    data = fermionic_integrals(prepared)
    assert data.occupations == ((1,), (1,))
    assert data.nelec == (1, 1)
    state = ffsim.slater_determinant(data.norb, data.occupations)
    operator = ffsim.linear_operator(to_ffsim_hamiltonian(data), data.norb, data.nelec)
    active = float(np.vdot(state, operator @ state).real)
    assert active == pytest.approx(-2.0)
    assert active + sum(prepared.energy_offsets.values()) == pytest.approx(-5.0)


def test_unrestricted_integrals_are_rejected_instead_of_coerced(h2_data):
    unrestricted = replace(
        h2_data,
        one_body_integrals_beta=h2_data.one_body_integrals + 0.1 * np.eye(2),
        two_body_integrals_beta_beta=h2_data.two_body_integrals,
        two_body_integrals_beta_alpha=h2_data.two_body_integrals,
        overlap_alpha_beta=np.eye(2),
    )
    with pytest.raises(ConfigError, match="unrestricted alpha/beta"):
        fermionic_integrals(prepare_problem(unrestricted))
    with pytest.raises(ConfigError, match="unrestricted orbital overlap"):
        fermionic_integrals(
            prepare_problem(
                replace(
                    unrestricted,
                    one_body_integrals_beta=h2_data.one_body_integrals,
                    overlap_alpha_beta=np.array([[0.0, 1.0], [1.0, 0.0]]),
                )
            )
        )


def test_lucj_vqe_reaches_h2_exact_energy_with_seeded_numeric_initialization(h2_data):
    records: list[dict[str, Any]] = []
    prepared = prepare_problem(h2_data)
    options = QiskitOptions(
        algorithm="ffsim_vqe",
        optimizer={"name": "slsqp", "options": {"maxiter": 200, "ftol": 1e-10}},
    )
    result = build_ffsim_vqe(
        options=FfsimVQEOptions(),
        request=NativeSolveRequest(prepared=prepared, options=options, callback=records.append),
    )
    assert result.active_energy_hartree + sum(prepared.energy_offsets.values()) == pytest.approx(
        -1.1373060357534, abs=1e-8
    )
    assert result.diagnostics["statevector_dimension"] == 4
    assert result.diagnostics["spin_eigenstate_residual"] < 1e-10
    assert result.diagnostics["reference_occupations"] == ((0,), (0,))
    assert result.converged is None  # Qiskit's optimizer result drops SciPy's success status.
    assert records == result.evaluations
    records[0]["metadata"]["simulation"] = "changed by caller"
    assert result.evaluations[0]["metadata"]["simulation"] == "ffsim"


@pytest.mark.parametrize(
    "limits,message",
    [
        ({"max_statevector_dimension": 3}, "max_statevector_dimension"),
        ({"max_parameters": 1}, "max_parameters"),
        ({"max_evaluations": 1}, "max_evaluations"),
    ],
)
def test_native_optimization_enforces_explicit_budgets(h2_data, limits, message):
    """Oversized state spaces, circuits and searches fail before exceeding limits."""
    with pytest.raises(ConfigError, match=message):
        build_ffsim_vqe(
            options=FfsimVQEOptions(**limits),
            request=NativeSolveRequest(
                prepare_problem(h2_data), QiskitOptions(algorithm="ffsim_vqe")
            ),
        )


def test_native_memory_estimate_checked_before_state_allocation(monkeypatch, h2_data):
    """A combinatorial state size can fit the dimension limit and exceed memory."""
    data = fermionic_integrals(prepare_problem(h2_data))
    monkeypatch.setattr(
        "chemrefine.engines.qiskit.components.fermionic_algorithms.fermionic_integrals",
        lambda _: replace(data, norb=10, nelec=(5, 5)),
    )
    with pytest.raises(ConfigError, match="max_memory_mb"):
        build_ffsim_vqe(
            options=FfsimVQEOptions(max_memory_mb=1),
            request=NativeSolveRequest(
                prepare_problem(h2_data), QiskitOptions(algorithm="ffsim_vqe")
            ),
        )


@pytest.fixture
def unchanged_optimizer(monkeypatch):
    """Return the initial state to isolate reference, initialization, and spin validation."""
    calls = []

    class Optimizer:
        def minimize(self, fun, x0):
            calls.append(x0.copy())
            return SimpleNamespace(x=x0, fun=fun(x0), success=False, message="test budget", nfev=1)

    monkeypatch.setattr(OPTIMIZERS, "build", lambda *_args, **_kwargs: Optimizer())
    return calls


@pytest.mark.parametrize("spin_variant", ["balanced", "unbalanced"])
def test_supplied_parameters_and_reordered_reference_are_used(
    h2_data, unchanged_optimizer, spin_variant
):
    prepared = prepare_problem(h2_data, active_space=ActiveSpaceOptions(active_orbitals=[1, 0]))
    options = FfsimVQEOptions(ansatz="ucj", spin_variant=spin_variant, n_reps=2)
    operator_type = (
        ffsim.UCJOpSpinBalanced if spin_variant == "balanced" else ffsim.UCJOpSpinUnbalanced
    )
    initial = np.zeros(operator_type.n_params(norb=2, n_reps=2))
    outcome = build_ffsim_vqe(
        options=options,
        request=NativeSolveRequest(
            prepared, QiskitOptions(algorithm="ffsim_vqe"), initial_point=initial
        ),
    )
    np.testing.assert_array_equal(unchanged_optimizer[0], initial)
    assert outcome.active_energy_hartree == pytest.approx(-1.836967991202984, abs=1e-12)
    assert outcome.diagnostics["initialization"] == "supplied"
    assert outcome.diagnostics["reference_occupations"] == ((1,), (1,))
    assert outcome.converged is False
    assert outcome.termination_reason == "test budget"


def test_spin_contamination_fails_with_explicit_diagnostic(h2_data, unchanged_optimizer):
    request = NativeSolveRequest(prepare_problem(h2_data), QiskitOptions(algorithm="ffsim_vqe"))
    with pytest.raises(ConfigError, match="does not have the requested spin"):
        build_ffsim_vqe(options=FfsimVQEOptions(spin_variant="unbalanced"), request=request)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_reps": 0},
        {"unknown": True},
        {"initial_scale": float("nan")},
        {"interaction_pairs": [None]},
        {"interaction_pairs": [[(1, 0)], None]},
        {"interaction_pairs": [[(0, 1), (0, 1)], None]},
        {"interaction_pairs": [[(-1, 0)], None]},
    ],
)
def test_strict_ucj_options_reject_invalid_graphs(kwargs):
    with pytest.raises(ValidationError):
        FfsimVQEOptions(**kwargs)


@pytest.mark.parametrize(
    "category,selection",
    [
        ("ansatz", "efficient_su2"),
        ("mapper", "parity"),
        ("initial_state", "zero"),
        ("estimator", "basic_backend"),
        ("sampler", {"name": "statevector", "options": {"seed": 1}}),
        ("initial_point", "random"),
    ],
)
def test_native_preflight_rejects_components_that_would_be_ignored(category, selection):
    with pytest.raises(ConfigError, match=f"does not consume the {category} component"):
        validate_ffsim_options(QiskitOptions(algorithm="ffsim_vqe", **{category: selection}))


def test_invalid_problem_dependent_options_fail_before_optimizer(h2_data, unchanged_optimizer):
    prepared = prepare_problem(h2_data)
    request = NativeSolveRequest(prepared, QiskitOptions(algorithm="ffsim_vqe"))
    with pytest.raises(ConfigError, match="below 2"):
        build_ffsim_vqe(
            options=FfsimVQEOptions(interaction_pairs=(((0, 2),), None)), request=request
        )
    with pytest.raises(ConfigError, match="exactly"):
        build_ffsim_vqe(options=FfsimVQEOptions(initial_parameters=(0.0,)), request=request)
    with pytest.raises(ConfigError, match="not both"):
        build_ffsim_vqe(
            options=FfsimVQEOptions(initial_parameters=(0.0,)),
            request=replace(request, initial_point=[0.0]),
        )
    with pytest.raises(ConfigError, match="actual hartree_fock occupations"):
        build_ffsim_vqe(
            options=FfsimVQEOptions(),
            request=replace(
                request, options=QiskitOptions(algorithm="ffsim_vqe", initial_state="zero")
            ),
        )
    assert unchanged_optimizer == []


@pytest.mark.parametrize("value", [[1j], ["bad"]])
def test_objective_parameter_conversion_rejects_nonreal_values(value):
    """Numeric optimizers may not coerce complex or textual parameters."""
    from chemrefine.engines.qiskit.components.fermionic_algorithms import _parameters

    with pytest.raises(ConfigError, match="finite real"):
        _parameters(value, 1, label="test")


def test_ffsim_rejects_gpu_and_preserves_custom_interaction_pairs():
    """Custom opposite-spin graphs are accepted while local ffsim stays CPU-only."""
    from chemrefine.engines.qiskit.components.fermionic_algorithms import _interaction_pairs

    with pytest.raises(ConfigError, match="device: cpu"):
        validate_ffsim_options(QiskitOptions(algorithm="ffsim_vqe", device="cuda"))
    options = FfsimVQEOptions(spin_variant="unbalanced", interaction_pairs=(None, ((1, 0),), ()))
    assert _interaction_pairs(options, 2) == (None, [(1, 0)], [])


def test_zero_initialization_is_explicit(h2_data, unchanged_optimizer):
    """The optional stationary zero initialization is returned honestly in provenance."""
    result = build_ffsim_vqe(
        options=FfsimVQEOptions(initialization="zeros"),
        request=NativeSolveRequest(prepare_problem(h2_data), QiskitOptions(algorithm="ffsim_vqe")),
    )
    assert result.diagnostics["initialization"] == "zeros"
    assert not np.any(unchanged_optimizer[0])


@pytest.mark.parametrize("stage", ["objective", "final", "spin"])
def test_native_rejects_invalid_provider_numerics(monkeypatch, h2_data, stage):
    """NaN from a provider must not become a successful energy artifact."""

    class Optimizer:
        """Choose whether the provider fails in the objective or final verification."""

        def minimize(self, fun, x0):
            """Supply a valid parameter vector and optionally call the objective."""
            if stage == "objective":
                fun(x0)
            return SimpleNamespace(x=x0)

    monkeypatch.setattr(OPTIMIZERS, "build", lambda _: Optimizer())
    if stage == "spin":
        import pyscf.fci.spin_op

        monkeypatch.setattr(
            pyscf.fci.spin_op, "contract_ss", lambda state, *a: np.full_like(state, np.nan)
        )
    else:
        monkeypatch.setattr(ffsim, "apply_unitary", lambda state, *a: np.full_like(state, np.nan))
    with pytest.raises(ConfigError, match=r"non-finite|invalid final"):
        build_ffsim_vqe(
            options=FfsimVQEOptions(),
            request=NativeSolveRequest(
                prepare_problem(h2_data), QiskitOptions(algorithm="ffsim_vqe")
            ),
        )

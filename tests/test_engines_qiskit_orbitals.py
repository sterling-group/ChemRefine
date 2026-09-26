"""Unitary orbital-frame and alternating sampled-subspace optimization tests."""

from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pytest

from chemrefine.engines.qiskit.determinants import (
    FermionicHamiltonian,
    FermionTerm,
    projected_operator,
)
from chemrefine.engines.qiskit.orbitals import (
    OrbitalOptimizationOptions,
    optimize_orbitals,
    rotate_hamiltonian,
)
from chemrefine.errors import ConfigError


def _one_body(matrix):
    """Create spin-orbital coefficients directly, without chemistry tensor guesses."""
    return FermionicHamiltonian(
        len(matrix),
        tuple(
            FermionTerm((int(p),), (int(q),), matrix[p, q])
            for p, q in zip(*np.nonzero(matrix), strict=True)
        ),
    )


def test_rotated_interacting_operator_matches_exterior_power_unitary():
    terms = (
        FermionTerm((0, 2), (2, 0), 0.5),
        FermionTerm((1, 3), (3, 1), 0.8),
        FermionTerm((), (), 0.12),
    )
    rng = np.random.default_rng(40)
    raw = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    unitary = np.linalg.qr(raw)[0]
    one = _one_body(raw + raw.conj().T)
    model = FermionicHamiltonian(4, (*one.terms, *terms))
    rotated = rotate_hamiltonian(model, unitary)
    occupations = list(combinations(range(4), 2))
    basis = tuple(sum(1 << mode for mode in occupied) for occupied in occupations)
    exterior = np.array(
        [[np.linalg.det(unitary[np.ix_(old, new)]) for new in occupations] for old in occupations]
    )
    before = projected_operator(model, basis) @ np.eye(len(basis))
    after = projected_operator(rotated, basis) @ np.eye(len(basis))
    np.testing.assert_allclose(after, exterior.conj().T @ before @ exterior, atol=1e-13)
    np.testing.assert_allclose(np.linalg.eigvalsh(after), np.linalg.eigvalsh(before), atol=1e-13)


@pytest.mark.parametrize("spin_mode", ["shared", "independent", "spin_orbital"])
def test_sampled_state_orbital_optimization_reaches_exact_one_body_state(
    spin_mode,
):
    one = np.array([[-0.3, 0.4j], [-0.4j, 0.7]])
    full = np.kron(np.eye(2), one)
    model = _one_body(full)
    result = optimize_orbitals(model, (5,), options=OrbitalOptimizationOptions(spin_mode=spin_mode))
    assert result.eigensystem.energies[0] == pytest.approx(2 * np.linalg.eigvalsh(one)[0], abs=1e-9)
    assert np.all(np.diff(result.energy_history) < 0)
    np.testing.assert_allclose(result.rotation.conj().T @ result.rotation, np.eye(4), atol=1e-12)
    assert result.objective_evaluations > 0
    assert not result.rotation.flags.writeable
    if spin_mode == "shared":
        np.testing.assert_allclose(result.rotation[:2, :2], result.rotation[2:, 2:], atol=1e-12)
    assert result.eigensystem.states[0].expectation(model) == pytest.approx(
        result.energy_history[-1]
    )


def test_real_rotations_and_small_budget_keep_monotone_accepted_energy():
    model = _one_body(np.array([[-0.3, 0.4], [0.4, 0.7]]))
    result = optimize_orbitals(
        model,
        (1,),
        options=OrbitalOptimizationOptions(
            spin_mode="spin_orbital", complex_rotations=False, max_iterations=1
        ),
    )
    assert len(result.energy_history) == 2
    assert not result.converged
    assert np.max(np.abs(result.rotation.imag)) == 0
    assert result.eigensystem.energies[0] < -0.4
    diagonal = _one_body(np.diag([-1.0, 1.0]))
    result = optimize_orbitals(
        diagonal, (1,), options=OrbitalOptimizationOptions(spin_mode="spin_orbital")
    )
    assert result.converged
    assert result.energy_history == (-1.0,)


@pytest.mark.parametrize("rotation", [np.ones((2, 2)), np.eye(3), np.full((2, 2), np.nan)])
def test_nonunitary_rotations_are_rejected(rotation):
    with pytest.raises(ConfigError, match="finite unitary"):
        rotate_hamiltonian(_one_body(np.eye(2)), rotation)


def test_rotation_and_optimization_preflight_guards():
    model = _one_body(np.eye(2))
    with pytest.raises(ConfigError, match="storage"):
        rotate_hamiltonian(model, np.eye(2), max_memory_mb=0)
    with pytest.raises(ConfigError, match="target_root"):
        optimize_orbitals(model, (1,), target_root=1)
    with pytest.raises(ConfigError, match="alpha-then-beta"):
        optimize_orbitals(_one_body(np.eye(3)), (1,))
    with pytest.raises(ConfigError, match="two rotatable"):
        optimize_orbitals(model, (1,))


def test_nonfinite_optimizer_result_is_a_configuration_failure(monkeypatch):
    import scipy.optimize

    monkeypatch.setattr(
        scipy.optimize, "minimize", lambda *a, **kw: SimpleNamespace(fun=np.nan, x=np.zeros(2))
    )
    with pytest.raises(ConfigError, match="non-finite"):
        optimize_orbitals(
            _one_body(np.eye(2)), (1,), options=OrbitalOptimizationOptions(spin_mode="spin_orbital")
        )


def test_original_basis_rdms_match_explicit_exterior_power_state():
    from dataclasses import replace

    from chemrefine.engines.qiskit.determinants import DeterminantState

    rng = np.random.default_rng(91)
    rotation = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0]
    occupations = list(combinations(range(4), 2))
    basis = tuple(sum(1 << i for i in occupied) for occupied in occupations)
    exterior = np.array(
        [[np.linalg.det(rotation[np.ix_(old, new)]) for new in occupations] for old in occupations]
    )
    vector = rng.normal(size=6) + 1j * rng.normal(size=6)
    vector /= np.linalg.norm(vector)
    rotated = DeterminantState(4, basis, vector, orbital_rotation=rotation)
    original = DeterminantState(4, basis, exterior @ vector)
    np.testing.assert_allclose(rotated.rdms().one_body, original.rdms().one_body, atol=1e-13)
    np.testing.assert_allclose(rotated.rdms().two_body, original.rdms().two_body, atol=1e-13)
    np.testing.assert_allclose(
        rotated.rdms(max_order=1).one_body, original.rdms(max_order=1).one_body, atol=1e-13
    )
    unrotated = replace(rotated, orbital_rotation=None)
    np.testing.assert_allclose(rotated.rdms(basis="state").one_body, unrotated.rdms().one_body)
    assert rotated.expectation(_one_body(np.eye(4)), basis="state") == pytest.approx(2)
    for ket, bra in ((rotated, original), (original, rotated)):
        with pytest.raises(ConfigError, match="common orbital frame"):
            ket.rdms(bra=bra)
    for keyword in ("expectation", "rdms"):
        with pytest.raises(ConfigError, match="basis"):
            if keyword == "expectation":
                rotated.expectation(_one_body(np.eye(4)), basis="invalid")
            else:
                rotated.rdms(basis="invalid")


@pytest.mark.parametrize("rotation", [np.ones((2, 2)), np.eye(3), np.full((2, 2), np.nan)])
def test_state_rejects_invalid_orbital_frame(rotation):
    from chemrefine.engines.qiskit.determinants import DeterminantState

    with pytest.raises(ConfigError, match="orbital_rotation"):
        DeterminantState(2, (1,), [1], orbital_rotation=rotation)


def test_orbital_optimization_is_available_in_explicit_sqd_and_records_frame():
    pytest.importorskip("qiskit_addon_sqd")
    from chemrefine.engines.qiskit.api import ElectronicStructureData, prepare_problem, run_problem

    one = np.array([[-0.3, 0.4j], [-0.4j, 0.7]])
    prepared = prepare_problem(ElectronicStructureData(1, 1, 2, one, np.zeros((2,) * 4)))
    result = run_problem(
        prepared,
        options={
            "algorithm": {
                "name": "sqd",
                "options": {
                    "projection": "explicit",
                    "counts": {"0101": 10},
                    "num_batches": 1,
                    "configuration_recovery": False,
                    "orbital_optimization": {},
                },
            }
        },
    )
    assert result.energy_hartree == pytest.approx(2 * np.linalg.eigvalsh(one)[0], abs=1e-9)
    assert result.states[0].orbital_rotation is not None
    assert (
        result.metadata["solver"]["orbital_optimization"]["state_basis"]
        == "optimized_active_orbitals"
    )
    assert result.metadata["solver"]["root_spin"][0]["spin_eigenstate_residual"] < 1e-10
    assert np.trace(result.states[0].rdms().one_body) == pytest.approx(2)


@pytest.mark.parametrize(
    "options",
    [
        {"orbital_optimization": {}},
        {"projection": "explicit", "orbital_optimization": {"spin_mode": "spin_orbital"}},
        {"projection": "explicit", "orbital_optimization": {}, "max_total_diagonalizations": 31},
    ],
)
def test_orbital_options_preserve_sectors_and_share_total_solve_budget(options):
    from pydantic import ValidationError

    from chemrefine.engines.qiskit.components.subspace_algorithms import SQDOptions

    with pytest.raises(ValidationError):
        SQDOptions(**options)


def test_orbital_optimization_report_mode_does_not_require_a_spin_observable():
    pytest.importorskip("qiskit_addon_sqd")
    from chemrefine.engines.qiskit.api import ElectronicStructureData, prepare_problem, run_problem

    prepared = prepare_problem(
        ElectronicStructureData(1, 1, 2, np.diag([-1, 1]), np.zeros((2,) * 4))
    )
    prepared.problem.properties.angular_momentum = None
    result = run_problem(
        prepared,
        options={
            "algorithm": {
                "name": "sqd",
                "options": {
                    "projection": "explicit",
                    "counts": {"0101": 1},
                    "num_batches": 1,
                    "configuration_recovery": False,
                    "orbital_optimization": {},
                    "spin_constraint": "report",
                },
            }
        },
    )
    assert result.energy_hartree == pytest.approx(-2)
    assert result.metadata["solver"]["root_spin"] == []


def test_non_numeric_orbital_frame_is_classified():
    from chemrefine.engines.qiskit.determinants import DeterminantState

    with pytest.raises(ConfigError, match="finite complex"):
        DeterminantState(2, (1,), [1], orbital_rotation=[["bad", 0], [0, 1]])

"""Double-factorized circuits retain complex hopping, offsets and true convergence."""

from dataclasses import replace

import numpy as np
import pytest

from chemrefine.engines.qiskit.data import ElectronicStructureData
from chemrefine.engines.qiskit.double_factorized import (
    DoubleFactorizedOptions,
    build_double_factorized_evolution,
    simulate_double_factorized_evolution,
)
from chemrefine.errors import ConfigError


def _data():
    """A complex hopping model with two noncommuting real Coulomb factors."""
    factors = np.array([[[1, 0.2], [0.2, 0.1]], [[0.1, 0.4], [0.4, 0.8]]])
    return ElectronicStructureData(
        1,
        1,
        2,
        np.array([[-1, 0.2 + 0.3j], [0.2 - 0.3j, 0.4]]),
        np.einsum("apq,ars->pqrs", factors, factors),
        nuclear_repulsion_energy=0.7,
    )


@pytest.mark.parametrize("order", [1, 2, 4])
def test_double_factorized_real_circuits_converge_to_original_complex_hamiltonian(order):
    options = DoubleFactorizedOptions(
        time=0.8, order=order, exact_reference=True, optimization_level=0
    )
    rough = simulate_double_factorized_evolution(_data(), options)
    refined = simulate_double_factorized_evolution(_data(), options.model_copy(update={"steps": 4}))
    assert refined.metadata["reference_state_error"] < rough.metadata["reference_state_error"] / (
        2**order
    )
    assert refined.metadata["one_body_max_absolute_error"] < 1e-12
    assert refined.metadata["two_body_max_absolute_error"] < 1e-12
    assert refined.metadata["alpha_particle_number"] == pytest.approx(1)
    assert refined.metadata["beta_particle_number"] == pytest.approx(1)
    assert refined.metadata["physical_product_formula_order"] == order
    assert refined.metadata["compiled_two_qubit_gates"] > 0
    assert not refined.statevector.flags.writeable


@pytest.mark.parametrize("factorization", ["eigh", "cholesky", "compressed"])
@pytest.mark.parametrize("z_representation", [False, True])
def test_factorization_choices_and_z_representation_preserve_the_original_generator(
    factorization, z_representation
):
    options = DoubleFactorizedOptions(
        time=0.15,
        steps=4,
        order=2,
        factorization=factorization,
        z_representation=z_representation,
        exact_reference=True,
        energy_shift_hartree=-0.9,
    )
    result = simulate_double_factorized_evolution(_data(), options, cores=2)
    assert result.metadata["reference_state_error"] < 1e-4
    assert result.metadata["energy_offsets_hartree"] == {
        "nuclear_repulsion": 0.7,
        "supplied_shift": -0.9,
    }
    assert result.metadata["cores"] == 2
    assert all(
        not array.flags.writeable
        for array in (result.one_body, result.diagonal_coulomb, result.orbital_rotations)
    )


def test_offsets_are_applied_exactly_once_in_zero_time_and_nonzero_time_circuits():
    from qiskit.quantum_info import Operator

    plain = replace(_data(), nuclear_repulsion_energy=None)
    options = DoubleFactorizedOptions(time=0.3)
    first = build_double_factorized_evolution(plain, options)
    second = build_double_factorized_evolution(
        _data(), options.model_copy(update={"energy_shift_hartree": 0.2})
    )
    np.testing.assert_allclose(
        Operator(second.circuit).data,
        np.exp(-0.3j * 0.9) * Operator(first.circuit).data,
        atol=1e-12,
    )
    zero = build_double_factorized_evolution(_data(), options.model_copy(update={"time": 0}))
    np.testing.assert_allclose(Operator(zero.circuit).data, np.eye(16), atol=1e-12)


def test_open_shell_reordered_reference_and_physicist_integrals():
    data = _data()
    data = replace(
        data,
        num_beta=0,
        orbital_occupations=[0, 1],
        orbital_occupations_beta=[0, 0],
        multiplicity=2,
        two_body_order="physicist",
        two_body_integrals=np.asarray(data.two_body_integrals).transpose(0, 2, 3, 1),
        one_body_integrals_beta=data.one_body_integrals,
        two_body_integrals_beta_beta=np.asarray(data.two_body_integrals).transpose(0, 2, 3, 1),
        two_body_integrals_beta_alpha=np.asarray(data.two_body_integrals).transpose(0, 2, 3, 1),
        overlap_alpha_beta=np.eye(2),
    )
    result = simulate_double_factorized_evolution(
        data, DoubleFactorizedOptions(time=0, exact_reference=True)
    )
    np.testing.assert_allclose(result.metadata["mode_occupations"], [0, 1, 0, 0], atol=1e-12)
    assert result.metadata["reference_state_error"] < 1e-12


def test_factor_truncation_reports_both_tensor_errors_and_certified_norm_bound():
    options = DoubleFactorizedOptions(time=0.2, max_factors=1, factorization_tolerance=1e-12)
    result = build_double_factorized_evolution(_data(), options)
    assert result.metadata["factor_count"] == 1
    assert result.metadata["two_body_max_absolute_error"] > 0.01
    assert result.metadata["one_body_max_absolute_error"] > 0.01
    assert not result.metadata["requested_tensor_tolerance_met"]
    assert result.metadata["factorization_unitary_error_bound"] > 0
    assert result.metadata["product_formula_error_bound"] is None
    with pytest.raises(ConfigError, match="max_integral_error"):
        build_double_factorized_evolution(
            _data(), options.model_copy(update={"max_integral_error": 0.001})
        )


def test_double_factorization_rejects_unsupported_complex_and_unrestricted_interactions():
    data = _data()
    complex_two = np.asarray(data.two_body_integrals, dtype=complex).copy()
    complex_two[0, 1, 0, 1] += 1j
    complex_two[1, 0, 1, 0] -= 1j
    with pytest.raises(ConfigError, match="real two-body"):
        build_double_factorized_evolution(
            replace(data, two_body_integrals=complex_two), DoubleFactorizedOptions()
        )
    pair_broken = np.zeros((2,) * 4, dtype=complex)
    pair_broken[0, 1, 0, 1] = pair_broken[1, 0, 1, 0] = 1
    with pytest.raises(ConfigError, match="chemist pair symmetry"):
        build_double_factorized_evolution(
            replace(data, two_body_integrals=pair_broken), DoubleFactorizedOptions()
        )
    for name in (
        "one_body_integrals_beta",
        "two_body_integrals_beta_beta",
        "two_body_integrals_beta_alpha",
        "overlap_alpha_beta",
    ):
        beta = {
            "one_body_integrals_beta": np.asarray(data.one_body_integrals),
            "two_body_integrals_beta_beta": np.asarray(data.two_body_integrals),
            "two_body_integrals_beta_alpha": np.asarray(data.two_body_integrals),
            "overlap_alpha_beta": np.eye(2),
        }
        beta[name] = beta[name] * 0.9
        with pytest.raises(ConfigError, match="shared alpha/beta"):
            build_double_factorized_evolution(replace(data, **beta), DoubleFactorizedOptions())


def test_cholesky_rejects_indefinite_integrals_while_eigenfactorization_supports_them():
    data = replace(_data(), two_body_integrals=-np.asarray(_data().two_body_integrals))
    with pytest.raises(ConfigError, match="positive semidefinite"):
        build_double_factorized_evolution(data, DoubleFactorizedOptions(factorization="cholesky"))
    result = simulate_double_factorized_evolution(
        data, DoubleFactorizedOptions(time=0.1, exact_reference=True)
    )
    assert result.metadata["reference_state_error"] < 1e-3


@pytest.mark.parametrize("cores", [True, 0, 1.5])
def test_double_factorization_requires_actual_positive_core_grants(cores):
    with pytest.raises(ConfigError, match="cores"):
        build_double_factorized_evolution(_data(), DoubleFactorizedOptions(), cores=cores)


def test_double_factorized_guards_factorization_synthesis_and_simulation_work(monkeypatch):
    from types import SimpleNamespace

    with pytest.raises(ConfigError, match="max_operations"):
        build_double_factorized_evolution(_data(), DoubleFactorizedOptions(max_operations=1))
    with pytest.raises(ConfigError, match="storage estimate"):
        build_double_factorized_evolution(
            _data(), DoubleFactorizedOptions(steps=4, max_memory_mb=1)
        )
    with pytest.raises(ConfigError, match="max_statevector_qubits"):
        simulate_double_factorized_evolution(
            _data(), DoubleFactorizedOptions(max_statevector_qubits=1)
        )
    large = ElectronicStructureData(1, 0, 4, np.eye(4), np.zeros((4,) * 4))
    with pytest.raises(ConfigError, match="reference simulation"):
        simulate_double_factorized_evolution(
            large, DoubleFactorizedOptions(exact_reference=True, max_memory_mb=1)
        )
    monkeypatch.setattr(
        "qiskit.transpile", lambda *args, **kwargs: SimpleNamespace(size=lambda: 1_000_001)
    )
    with pytest.raises(ConfigError, match=r"compiled.*max_operations"):
        build_double_factorized_evolution(_data(), DoubleFactorizedOptions())


@pytest.mark.parametrize("factorization", ["eigh", "cholesky", "compressed"])
def test_zero_interaction_complex_hopping_evolves_exactly_without_extra_factors(factorization):
    data = replace(_data(), two_body_integrals=np.zeros((2,) * 4))
    result = simulate_double_factorized_evolution(
        data,
        DoubleFactorizedOptions(
            factorization=factorization, exact_reference=True, max_integral_error=1e-10
        ),
    )
    assert result.metadata["reference_state_error"] < 1e-12
    assert result.metadata["factor_count"] == 0
    assert result.metadata["compression_optimizer"] is None
    pure = simulate_double_factorized_evolution(data, DoubleFactorizedOptions(time=-0.2))
    assert "reference_state_error" not in pure.metadata


def test_double_factorized_options_forbid_unknown_fields_and_nonphysical_order():
    from pydantic import ValidationError

    for patch in ({"order": 3}, {"steps": True}, {"unknown": 1}, {"time": np.nan}):
        with pytest.raises(ValidationError):
            DoubleFactorizedOptions.model_validate(patch)
    options = DoubleFactorizedOptions()
    with pytest.raises(ValidationError):
        options.steps = 2


def test_reported_factorization_bound_limits_independent_full_fock_generator_error():
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from scipy.linalg import expm

    from chemrefine.engines.qiskit.problem import prepare_problem

    data = _data()
    controls = DoubleFactorizedOptions(time=0.2, max_factors=1)
    result = build_double_factorized_evolution(data, controls)
    u = result.orbital_rotations
    two = np.einsum("kij,kpi,kqi,krj,ksj->pqrs", result.diagonal_coulomb, u, u.conj(), u, u.conj())
    one = result.one_body + 0.5 * np.einsum("prrq->pq", two)
    approximated = replace(data, one_body_integrals=one, two_body_integrals=two)
    mapper = JordanWignerMapper()
    original_matrix = mapper.map(prepare_problem(data).fermionic_hamiltonian).to_matrix()
    approximate_matrix = mapper.map(prepare_problem(approximated).fermionic_hamiltonian).to_matrix()
    actual_error = np.linalg.norm(original_matrix - approximate_matrix, ord=2)
    assert actual_error <= result.metadata["factorization_hamiltonian_norm_error_bound"]
    unitary_error = np.linalg.norm(
        expm(-0.2j * original_matrix) - expm(-0.2j * approximate_matrix), ord=2
    )
    assert unitary_error <= result.metadata["factorization_unitary_error_bound"]

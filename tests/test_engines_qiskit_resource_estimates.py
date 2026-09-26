"""Resource contracts: conservative query budgets, domains and explicit hardware."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from pydantic import ValidationError

from chemrefine.engines.qiskit.bundles import write_bundle
from chemrefine.engines.qiskit.data import ElectronicStructureData
from chemrefine.engines.qiskit.factorized_resources import (
    FactorizedResourceOptions,
    THCFactors,
    estimate_factorized_resources,
    factorization_metrics,
    load_thc_factors,
    restricted_resource_integrals,
    save_thc_factors,
)
from chemrefine.engines.qiskit.resources import (
    MagicFactory,
    PauliResourceOptions,
    QPEBudget,
    SurfaceCodeOptions,
    WalkOracleCost,
    estimate_pauli_resources,
    estimate_surface_code,
    qpe_queries,
)
from chemrefine.errors import ConfigError


def integral_data():
    """Construct a positive Coulomb tensor with known exact THC factors."""
    factors = THCFactors(np.eye(2), np.array([[0.6, 0.2], [0.2, 0.5]]))
    eri = np.einsum(
        "Pp,Pq,PQ,Qr,Qs->pqrs",
        factors.leaf,
        factors.leaf,
        factors.central,
        factors.leaf,
        factors.leaf,
    )
    data = ElectronicStructureData(1, 1, 2, np.diag([-1.0, -0.5]), eri)
    return data, factors


def hardware(**changes):
    """Supply every physical premise rather than importing a default machine."""
    values = {
        "logical_qubits": 10,
        "logical_cycles": 1000,
        "code_distance": 15,
        "physical_error_probability": 0.001,
        "threshold_probability": 0.01,
        "logical_error_prefactor": 0.1,
        "physical_qubits_per_patch_d2": 2,
        "routing_patch_multiplier": 1.5,
        "cycle_time_seconds": 1e-6,
        "failure_budget": 0.01,
        "hardware_provenance": "test machine model",
    }
    return SurfaceCodeOptions(**(values | changes))


def test_pauli_normalization_offsets_and_explicit_oracle():
    """Identity shifts are free; controlled-walk costs do not include QFT overhead."""
    result = estimate_pauli_resources(
        PauliResourceOptions(
            hamiltonian={"II": 5, "XX": -0.5, "ZI": 1.5, "YY": 0},
            budget=QPEBudget(energy_error_hartree=0.01),
            oracle=WalkOracleCost(logical_qubits=5, t=7, toffoli=2, provenance="supplied circuit"),
        )
    )
    assert result["normalization_hartree"] == 2
    assert result["identity_offset_hartree"] == 5
    assert result["system_qubits"] == 2
    assert result["nonidentity_terms"] == 2
    assert result["lcu_selection_qubits"] == 1
    assert result["controlled_walk_gate_counts"]["t"] == 7 * result["controlled_walk_queries"]
    assert result["logical_qubits_including_phase_register"] == 5 + result["phase_bits"]
    assert result["executable_circuit"] is False


def test_constant_pauli_and_no_oracle():
    """Scalar Hamiltonians require no state preparation or QPE calls."""
    budget = QPEBudget(energy_error_hartree=0.01)
    result = estimate_pauli_resources(PauliResourceOptions(hamiltonian={"I": 2}, budget=budget))
    assert result["controlled_walk_queries"] == 0
    assert result["conditional_failure_bound"] == 0
    assert "controlled_walk_gate_counts" not in result
    result = estimate_pauli_resources(
        PauliResourceOptions(
            hamiltonian={"Z": 0},
            budget=budget,
            oracle=WalkOracleCost(logical_qubits=1, provenance="zero"),
        )
    )
    assert result["logical_qubits_including_phase_register"] == 0


def test_qpe_error_and_success_union_bounds():
    """Precision and repeated target sampling both meet their explicit allocations."""
    budget = QPEBudget(
        energy_error_hartree=0.01,
        representation_error_hartree=0.001,
        synthesis_error_hartree=0.002,
        target_overlap=0.2,
        failure_probability=0.03,
    )
    result = qpe_queries(2.1, budget)
    assert 2 * np.pi * 2.1 * 2.0 ** (-result["accuracy_bits"]) <= 0.007
    assert result["conditional_failure_bound"] <= budget.failure_probability
    assert (1 - 0.2) ** (result["repetitions"] - 1) > budget.failure_probability / 2
    assert result["controlled_walk_queries"] == result["repetitions"] * (
        2 ** result["phase_bits"] - 1
    )
    # Verify the phase-error tail directly for several worst-alignment samples.
    m = 4
    t = m + int(np.ceil(np.log2(2 + 1 / (2 * 0.1))))
    outcomes = np.arange(2**t) / 2**t
    for phase in (0.001, 0.121, 0.499, 0.731, 0.999):
        distance = (phase - outcomes + 0.5) % 1 - 0.5
        probabilities = (np.sinc((2**t) * distance) / np.sinc(distance)) ** 2
        assert probabilities[np.abs(distance) > 2.0 ** (-m)].sum() <= 0.1


@pytest.mark.parametrize("normalization", [-1, float("inf"), float("nan")])
def test_invalid_normalization(normalization):
    """Nonphysical norms fail before integer resource calculations."""
    with pytest.raises(ValueError, match="normalization"):
        qpe_queries(normalization, QPEBudget(energy_error_hartree=0.01))


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"target_overlap": 0.01, "max_repetitions": 1}, "max_repetitions"),
        ({"max_phase_bits": 1}, "max_phase_bits"),
    ],
)
def test_qpe_allocation_limits(changes, match):
    """Tiny overlap or excessive precision cannot create unbounded reports."""
    with pytest.raises(ValueError, match=match):
        qpe_queries(1, QPEBudget(energy_error_hartree=0.01, **changes))


@pytest.mark.parametrize(
    "changes",
    [
        {"hamiltonian": {"": 1}},
        {"hamiltonian": {"Z": 1, "XX": 1}},
        {"hamiltonian": {"A": 1}},
        {"hamiltonian": {"Z": float("nan")}},
        {"hamiltonian": {"X": 1, "Z": 1}, "max_terms": 1},
        {"hamiltonian": {"ZZ": 1}, "oracle": {"logical_qubits": 1, "provenance": "bad"}},
    ],
)
def test_pauli_input_validation(changes):
    """Preflight catches malformed labels, coefficients, budgets and widths."""
    with pytest.raises(ValidationError):
        PauliResourceOptions(budget=QPEBudget(energy_error_hartree=0.01), **changes)


def test_precision_allocation_and_provider_contract():
    """Finite factorization precision cannot silently be assigned zero error."""
    with pytest.raises(ValidationError, match="exhaust"):
        QPEBudget(energy_error_hartree=0.01, synthesis_error_hartree=0.01)
    with pytest.raises(ValidationError, match="synthesis error"):
        FactorizedResourceOptions(budget=QPEBudget(energy_error_hartree=0.01))
    with pytest.raises(ValidationError, match="DF only"):
        FactorizedResourceOptions(
            method="thc",
            qualtran_cost_graph=True,
            budget=QPEBudget(energy_error_hartree=0.01, synthesis_error_hartree=0.001),
        )


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"method": "thc", "factorization_threshold": 0.1}, "only to DF"),
        ({"amplitude_bits_outer": 10}, "qualtran_cost_graph"),
        ({"amplitude_bits_inner": 10}, "qualtran_cost_graph"),
    ],
)
def test_provider_options_reject_ignored_knobs(changes, match):
    """Precision settings cannot be silently ignored by the selected cost model."""
    with pytest.raises(ValidationError, match=match):
        FactorizedResourceOptions(
            budget=QPEBudget(energy_error_hartree=0.01, synthesis_error_hartree=0.001), **changes
        )


def test_factorization_preflight_without_optional_imports():
    """Invalid representation inputs fail before importing a resource provider."""
    data, factors = integral_data()
    budget = QPEBudget(energy_error_hartree=0.01, synthesis_error_hartree=0.001)
    with pytest.raises(ConfigError, match="incompatible"):
        estimate_factorized_resources(
            data, FactorizedResourceOptions(budget=budget), thc_factors=factors
        )
    with pytest.raises(ConfigError, match="positive-semidefinite"):
        estimate_factorized_resources(
            replace(data, two_body_integrals=-data.two_body_integrals),
            FactorizedResourceOptions(budget=budget),
        )
    with pytest.raises(ConfigError, match="requires factors"):
        estimate_factorized_resources(data, FactorizedResourceOptions(method="thc", budget=budget))
    large = THCFactors(np.ones((100, 2)), np.eye(100))
    with pytest.raises(ConfigError, match="THC reconstruction workspace"):
        estimate_factorized_resources(
            data,
            FactorizedResourceOptions(method="thc", budget=budget, max_working_bytes=2000),
            thc_factors=large,
        )


def test_factorization_symmetry_validation_is_stricter_than_general_input():
    """Cost models validate the exact shared-real symmetry they depend on."""
    data, _ = integral_data()
    eri = np.asarray(data.two_body_integrals).copy()
    eri[0, 1, 1, 1] = 1e-11
    with pytest.raises(ConfigError, match="chemist ERI symmetry"):
        restricted_resource_integrals(replace(data, two_body_integrals=eri), max_bytes=10000)


def test_physical_model_without_factories():
    """Patch footprint and cycle error model are independently reproducible."""
    result = estimate_surface_code(hardware())
    assert result["physical_qubits"] == 15 * 2 * 15**2
    assert result["runtime_seconds_lower_bound"] == pytest.approx(0.001)
    assert result["failure_union_bound_at_runtime_lower_bound"] == pytest.approx(1.5e-5)
    assert result["meets_failure_budget_at_runtime_lower_bound"] is True


def test_factories_set_throughput_and_union_bound():
    """A factory bottleneck increases data exposure time as well as footprint."""
    factory = MagicFactory(
        count=2,
        physical_qubits_per_factory=1000,
        cycles_per_state=10,
        output_error_probability=1e-6,
        provenance="declared distillation design",
    )
    result = estimate_surface_code(
        hardware(t_states=10000, t_factory=factory, ccz_states=2, ccz_factory=factory)
    )
    assert result["runtime_cycles_lower_bound"] == 50000
    assert result["physical_qubits"] == 15 * 2 * 15**2 + 4000
    assert result["factory_failure_union_bound"] == pytest.approx(0.010002)
    assert result["meets_failure_budget_at_runtime_lower_bound"] is False
    failing = estimate_surface_code(hardware(logical_error_prefactor=100000000))
    assert failing["failure_union_bound_at_runtime_lower_bound"] == 1


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"code_distance": 4}, "odd"),
        ({"physical_error_probability": 0.01}, "below"),
        ({"t_states": 1}, "t_factory"),
        ({"ccz_states": 1}, "ccz_factory"),
    ],
)
def test_physical_model_requires_complete_assumptions(changes, match):
    """No factory, threshold or code-distance defaults are silently invented."""
    with pytest.raises(ValidationError, match=match):
        hardware(**changes)


def test_restricted_integrals_order_and_equal_materialized_spin_blocks():
    """Physicist ordering and explicit identical beta tensors preserve chemistry."""
    data, _ = integral_data()
    h1, eri = restricted_resource_integrals(data, max_bytes=10000)
    physicist = replace(
        data,
        two_body_integrals=eri.transpose(0, 2, 3, 1),
        two_body_order="physicist",
        one_body_integrals_beta=h1,
        two_body_integrals_beta_beta=eri.transpose(0, 2, 3, 1),
        two_body_integrals_beta_alpha=eri.transpose(0, 2, 3, 1),
        overlap_alpha_beta=np.eye(2),
    )
    actual = restricted_resource_integrals(physicist, max_bytes=10000)
    np.testing.assert_array_equal(actual[0], h1)
    np.testing.assert_array_equal(actual[1], eri)
    assert factorization_metrics(h1, eri, eri)["representation_error_bound_hartree"] == 0
    changed = eri * 0.99
    metrics = factorization_metrics(h1, eri, changed)
    assert metrics["representation_error_bound_hartree"] == pytest.approx(0.03)
    assert metrics["eri_frobenius_residual_hartree"] > 0


def test_resource_integral_rejections():
    """Shared-real providers decline valid larger complex/unrestricted domains."""
    data, _ = integral_data()
    with pytest.raises(ConfigError, match="workspace"):
        restricted_resource_integrals(data, max_bytes=1)
    with pytest.raises(ConfigError, match="two spatial"):
        restricted_resource_integrals(
            ElectronicStructureData(1, 0, 1, [[1]], np.zeros((1, 1, 1, 1))), max_bytes=10000
        )
    complex_h1 = np.asarray(data.one_body_integrals, dtype=complex).copy()
    complex_h1[0, 1], complex_h1[1, 0] = 1j, -1j
    with pytest.raises(ConfigError, match="real"):
        restricted_resource_integrals(replace(data, one_body_integrals=complex_h1), max_bytes=10000)
    unrestricted = replace(
        data,
        one_body_integrals_beta=2 * np.asarray(data.one_body_integrals),
        two_body_integrals_beta_beta=data.two_body_integrals,
        two_body_integrals_beta_alpha=data.two_body_integrals,
        overlap_alpha_beta=np.eye(2),
    )
    with pytest.raises(ConfigError, match="equal alpha"):
        restricted_resource_integrals(unrestricted, max_bytes=10000)
    with pytest.raises(ConfigError, match="shared spatial"):
        restricted_resource_integrals(
            replace(
                unrestricted,
                one_body_integrals_beta=data.one_body_integrals,
                overlap_alpha_beta=np.diag([1.0, -1.0]),
            ),
            max_bytes=10000,
        )


@pytest.mark.parametrize(
    "leaf,central,match",
    [
        (np.ones(2), np.eye(2), "shape"),
        (np.ones((2, 2)), np.ones((1, 1)), "central"),
        (np.ones((2, 2)), [[1, 1], [0, 1]], "symmetric"),
        (np.zeros((2, 2)), np.eye(2), "zero leaf"),
        ([[np.nan]], [[1]], "finite"),
        ([[1j]], [[1]], "real"),
    ],
)
def test_thc_factor_validation(leaf, central, match):
    """Malformed factors fail before reconstruction or provider imports."""
    with pytest.raises(ConfigError, match=match):
        THCFactors(leaf, central)


def test_thc_bundle_round_trip_and_kind(tmp_path):
    """Portable factors are checksummed, typed, copied and immutable."""
    _, factors = integral_data()
    path = save_thc_factors(tmp_path / "factors.json", factors)
    restored = load_thc_factors(path)
    np.testing.assert_array_equal(restored.leaf, factors.leaf)
    np.testing.assert_array_equal(restored.central, factors.central)
    assert not restored.leaf.flags.writeable
    write_bundle(path, kind="other", arrays={}, metadata={})
    with pytest.raises(ConfigError, match="thc_factors"):
        load_thc_factors(path)


@pytest.mark.parametrize(
    "method,norb,parameters,changes,match",
    [
        ("df", 8, {"L": 2, "Lxi": 256}, {}, "outer_coefficients"),
        ("df", 8, {"L": 10, "Lxi": 10000}, {"coefficient_bits": 2}, "outer_offsets"),
        ("df", 8, {"L": 32, "Lxi": 3}, {}, "inner_coefficients"),
        ("df", 8, {"L": 32, "Lxi": 100}, {}, "inner_rotations"),
        ("thc", 2, {"M": 2}, {}, "thc_coefficients"),
    ],
)
def test_each_provider_qrom_domain_guard(method, norb, parameters, changes, match):
    """Released QR assumptions are checked without shrinking supplied precision or fake ranks."""
    from chemrefine.engines.qiskit.factorized_resources import provider_qrom_tables

    controls = FactorizedResourceOptions(
        method=method,
        budget=QPEBudget(energy_error_hartree=0.01, synthesis_error_hartree=0.001),
        **changes,
    )
    with pytest.raises(ConfigError, match=match):
        provider_qrom_tables(norb, parameters, controls)


def test_valid_qrom_sizes_are_reported_without_modifying_inputs():
    """The QROM contract describes table entries and payload bits independently of costs."""
    from chemrefine.engines.qiskit.factorized_resources import provider_qrom_tables

    controls = FactorizedResourceOptions(
        budget=QPEBudget(energy_error_hartree=0.01, synthesis_error_hartree=0.001)
    )
    tables = provider_qrom_tables(8, {"L": 32, "Lxi": 256}, controls)
    assert tables["inner_rotations"] == {"entries": 256, "payload_bits": 240}
    assert tables["outer_coefficients"] == {"entries": 33, "payload_bits": 26}

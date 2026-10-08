"""Real optional provider contracts; execute in the isolated Python 3.12 resource stack."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("openfermion")

from chemrefine.engines.qiskit.data import ElectronicStructureData
from chemrefine.engines.qiskit.factorized_resources import (
    FactorizedResourceOptions,
    THCFactors,
    estimate_factorized_resources,
)
from chemrefine.engines.qiskit.resources import QPEBudget
from chemrefine.errors import ConfigError


def problem():
    """Use exact rank-32 factors satisfying the provider's finite-precision QROM domain."""
    leaf = np.random.default_rng(81).normal(size=(32, 8))
    leaf /= np.linalg.norm(leaf, axis=1)[:, None]
    factors = THCFactors(leaf, np.diag(np.linspace(0.1, 0.6, 32)))
    eri = np.einsum(
        "Pp,Pq,PQ,Qr,Qs->pqrs",
        factors.leaf,
        factors.leaf,
        factors.central,
        factors.leaf,
        factors.leaf,
    )
    return ElectronicStructureData(4, 4, 8, np.diag(np.linspace(-1.0, -0.25, 8)), eri), factors


def options(**changes):
    """Reserve separate approximation and finite-precision error allowances."""
    return FactorizedResourceOptions(
        budget=QPEBudget(
            energy_error_hartree=0.001,
            representation_error_hartree=1e-8,
            synthesis_error_hartree=1e-5,
        ),
        **changes,
    )


@pytest.mark.parametrize("method", ["df", "thc"])
def test_released_openfermion_costs_match_direct_calls(method):
    """Provider totals, rank conventions and conservative QPE are separately auditable."""
    from openfermion.resource_estimates import df, thc

    data, factors = problem()
    opts = options(method=method)
    result = estimate_factorized_resources(
        data, opts, thc_factors=factors if method == "thc" else None
    )
    cost = df.compute_cost if method == "df" else thc.compute_cost
    args = dict(
        n=2 * data.num_spatial_orbitals,
        lam=result["normalization_hartree"],
        dE=opts.budget.qpe_error_hartree,
        chi=opts.coefficient_bits,
        beta=opts.rotation_bits,
        **result["factorization_parameters"],
    )
    first = cost(**args, stps=opts.initial_cost_guess)
    expected = cost(**args, stps=first[0])
    assert result["provider_toffoli_per_step"] == expected[0]
    assert result["provider_toffoli_total_single_run"] == expected[1]
    assert result["provider_logical_qubits_including_system_and_phase"] == expected[2]
    assert result["representation_error_bound_hartree"] <= 1e-8
    assert result["provider_failure_probability"] is None
    assert result["conservative_standard_qpe"]["conditional_failure_bound"] <= 0.01
    assert result["executable_circuit"] is False


def test_thc_normalization_is_invariant_to_leaf_rescaling():
    """Equivalent THC gauges reconstruct the same Hamiltonian and LCU norm."""
    data, factors = problem()
    result = estimate_factorized_resources(data, options(method="thc"), thc_factors=factors)
    scales = np.linspace(1.5, 3.0, factors.leaf.shape[0])
    changed = THCFactors(
        factors.leaf * scales[:, None],
        factors.central / scales[:, None] ** 2 / scales[None, :] ** 2,
    )
    other = estimate_factorized_resources(data, options(method="thc"), thc_factors=changed)
    assert other["normalization_hartree"] == pytest.approx(result["normalization_hartree"])


@pytest.mark.parametrize("method", ["df", "thc"])
def test_imported_scalar_offsets_are_reported_without_changing_resource_queries(method):
    """Known scalar energy translations do not change the nonidentity LCU cost."""
    from dataclasses import replace

    data, factors = problem()
    kwargs = {"thc_factors": factors} if method == "thc" else {}
    first = estimate_factorized_resources(data, options(method=method), **kwargs)
    second = estimate_factorized_resources(
        replace(data, nuclear_repulsion_energy=0.7, energy_offsets={"core": -2.0}),
        options(method=method),
        **kwargs,
    )
    assert second["input_energy_offsets_hartree"] == {"core": -2.0}
    assert second["nuclear_repulsion_energy_hartree"] == 0.7
    assert not second["identity_offsets_in_query_cost"]
    for key in (
        "normalization_hartree",
        "provider_toffoli_per_step",
        "provider_toffoli_total_single_run",
        "conservative_standard_qpe",
    ):
        assert second[key] == first[key]


def test_factorization_cannot_spend_unallocated_error():
    """A low-quality supplied THC approximation fails before publishing cost estimates."""
    data, factors = problem()
    bad = THCFactors(factors.leaf, 0.99 * factors.central)
    with pytest.raises(ConfigError, match="exceeds representation"):
        estimate_factorized_resources(data, options(method="thc"), thc_factors=bad)


def test_released_qualtran_cost_graph_remains_an_analytical_graph():
    """The optional actual graph reports native gate categories without a T-count guess."""
    pytest.importorskip("qualtran")
    data, _ = problem()
    result = estimate_factorized_resources(data, options(qualtran_cost_graph=True))
    graph = result["qualtran_cost_graph"]
    assert graph["provider"] == "qualtran"
    assert graph["signature_qubits"] >= 2 * data.num_spatial_orbitals
    assert sum(graph["gate_counts_per_block_encoding"].values()) > 0
    assert graph["executable_circuit"] is False
    assert "external qubitization reflection" in " ".join(graph["limitations"])


@pytest.mark.parametrize("method", ["df", "thc"])
def test_small_cost_model_domain_is_actionable_instead_of_system_exit(method):
    """Mathematically valid small systems must not terminate the ChemRefine worker."""
    factors = THCFactors(np.eye(2), np.eye(2))
    eri = np.einsum(
        "Pp,Pq,PQ,Qr,Qs->pqrs",
        factors.leaf,
        factors.leaf,
        factors.central,
        factors.leaf,
        factors.leaf,
    )
    data = ElectronicStructureData(1, 1, 2, -np.eye(2), eri)
    with pytest.raises(ConfigError, match=r"QROM domain unsupported.*pauli_resources"):
        estimate_factorized_resources(
            data, options(method=method), thc_factors=factors if method == "thc" else None
        )


def test_df_empty_interactions_and_dropped_provider_factors_are_rejected(monkeypatch):
    """Empty factorization results cannot produce valid chemistry query normalizations."""
    from dataclasses import replace

    from openfermion.resource_estimates import df

    data, _ = problem()
    with pytest.raises(ConfigError, match="DF factorization failed"):
        estimate_factorized_resources(
            replace(data, two_body_integrals=np.zeros((8,) * 4)), options()
        )
    monkeypatch.setattr(
        df, "factorize", lambda *a, **kw: (np.zeros((8,) * 4), np.zeros((8, 8, 0)), 0, 0)
    )
    with pytest.raises(ConfigError, match="retained no factors"):
        estimate_factorized_resources(data, options())


def test_zero_factorized_normalization_is_not_sent_to_provider():
    """A scalar-free zero Hamiltonian uses the general Pauli path, not a singular cost formula."""
    from dataclasses import replace

    data, factors = problem()
    zero = THCFactors(factors.leaf, np.zeros_like(factors.central))
    with pytest.raises(ConfigError, match="positive finite normalization"):
        estimate_factorized_resources(
            replace(
                data, one_body_integrals=np.zeros((8, 8)), two_body_integrals=np.zeros((8,) * 4)
            ),
            options(method="thc"),
            thc_factors=zero,
        )


def test_provider_process_exit_is_wrapped_without_changing_factorization(monkeypatch):
    """Unexpected provider domain exits become ordinary actionable worker failures."""
    from openfermion.resource_estimates import df

    def fail(**kwargs):
        """Model a released provider's process-level error boundary."""
        raise SystemExit("unsupported QROM variant")

    monkeypatch.setattr(df, "compute_cost", fail)
    data, _ = problem()
    with pytest.raises(ConfigError, match=r"cost model rejected.*unsupported QROM variant"):
        estimate_factorized_resources(data, options())


@pytest.mark.parametrize("bad", [0, np.nan])
def test_provider_nonpositive_or_nonfinite_costs_are_rejected(monkeypatch, bad):
    """A successful import is insufficient to certify valid cost-domain output."""
    from openfermion.resource_estimates import df

    calls = iter([(1, 1, 1), (1, bad, 1)])
    monkeypatch.setattr(df, "compute_cost", lambda **kw: next(calls))
    data, _ = problem()
    with pytest.raises(ConfigError, match="nonpositive or nonfinite"):
        estimate_factorized_resources(data, options())


@pytest.mark.parametrize("method", ["df", "thc"])
def test_factorized_artifact_loads_integral_and_factor_bundles(tmp_path, method):
    """Both provider branches run through the real durable artifact adapter."""
    from chemrefine.engines.qiskit.bundles import read_bundle
    from chemrefine.engines.qiskit.experiment import run_experiment
    from chemrefine.engines.qiskit.factorized_resources import save_thc_factors
    from chemrefine.engines.qiskit.integral_io import save_integrals

    data, factors = problem()
    path = tmp_path / "result.json"
    selected = options(method=method).model_dump()
    selected["integral_bundle_path"] = str(save_integrals(tmp_path / "integrals.json", data))
    if method == "thc":
        selected["thc_bundle_path"] = str(save_thc_factors(tmp_path / "factors.json", factors))
    run_experiment(
        {"cores": 1, "experiment": {"name": "factorized_resources", "options": selected}},
        output_path=path,
    )
    result = read_bundle(path)
    assert result.description.kind == "resource_estimate"
    assert result.metadata["provider"] == "openfermion"
    assert result.metadata["method"] == method
    assert not result.arrays

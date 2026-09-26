"""Independent exact measurement designs and real circuits for both shadow channels."""

from itertools import combinations

import numpy as np
import pytest

from chemrefine.engines.qiskit.determinants import (
    DeterminantState,
    FermionicHamiltonian,
    FermionTerm,
)
from chemrefine.engines.qiskit.options import ComponentSelection
from chemrefine.engines.qiskit.shadows import (
    FermionicShadowOptions,
    ShadowSetting,
    collect_fermionic_shadows,
    estimate_fermionic_shadows,
    majorana_shadow_snapshot,
    orbital_shadow_snapshot,
    random_shadow_settings,
    shadow_circuit,
)
from chemrefine.errors import ConfigError

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning:qiskit")


def _majorana(index, modes):
    """Independent Jordan-Wigner matrix for a Majorana generator."""
    from qiskit.quantum_info import Pauli

    labels = ["Z"] * (index // 2) + ["X" if index % 2 == 0 else "Y"]
    labels += ["I"] * (modes - len(labels))
    return Pauli("".join(reversed(labels))).to_matrix()


def _matchings(indices):
    """Enumerate all perfect Majorana matchings without redundant signed outcomes."""
    if not indices:
        yield ()
        return
    first = indices[0]
    for second in indices[1:]:
        for rest in _matchings(tuple(index for index in indices if index not in (first, second))):
            yield ((first, second), *rest)


def test_random_majorana_circuits_realize_the_declared_signed_permutation():
    from qiskit.quantum_info import Operator

    options = FermionicShadowOptions(ensemble="majorana_clifford", num_settings=20)
    first = random_shadow_settings(3, options)
    second = random_shadow_settings(3, options)
    for setting, repeated in zip(first, second, strict=True):
        np.testing.assert_array_equal(setting.matrix, repeated.matrix)
        assert not setting.matrix.flags.writeable
        unitary = Operator(shadow_circuit(setting)).data
        for index in range(6):
            target = sum(setting.matrix[index, j] * _majorana(j, 3) for j in range(6))
            np.testing.assert_allclose(
                unitary.conj().T @ _majorana(index, 3) @ unitary, target, atol=1e-12
            )


def test_orbital_circuit_has_the_declared_complex_single_particle_matrix():
    from qiskit.quantum_info import Operator

    options = FermionicShadowOptions(num_particles=1, num_settings=2)
    for setting in random_shadow_settings(3, options):
        matrix = Operator(shadow_circuit(setting)).data
        np.testing.assert_allclose(matrix[np.ix_([1, 2, 4], [1, 2, 4])], setting.matrix, atol=1e-12)
        assert np.max(abs(setting.matrix.imag)) > 0.1


def test_orbital_channel_inversion_matches_exact_complex_qubit_design():
    state = DeterminantState(2, (1, 2), [1 / np.sqrt(2), 1j / np.sqrt(2)])
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    rotations = [np.eye(2), h, h @ np.diag([1, -1j])]
    settings = tuple(ShadowSetting("orbital_haar", value) for value in rotations)
    counts = []
    for value in rotations:
        probabilities = abs(value @ state.amplitudes) ** 2
        counts.append(
            {
                format(1 << mode, "02b"): round(20 * probability)
                for mode, probability in enumerate(probabilities)
                if probability > 1e-12
            }
        )
    result = estimate_fermionic_shadows(
        settings,
        counts,
        FermionicShadowOptions(num_particles=1, num_settings=3, shots_per_setting=20),
    )
    np.testing.assert_allclose(result.rdms.one_body, state.rdms().one_body, atol=1e-13)
    np.testing.assert_allclose(result.rdms.two_body, 0, atol=1e-13)
    assert result.metadata["uncertainty_cluster"] == "randomized_measurement_setting"
    expected = np.std([rdm.one_body.real for rdm in result.setting_rdms], axis=0, ddof=1) / np.sqrt(
        3
    )
    np.testing.assert_allclose(result.standard_errors_real.one_body, expected)
    assert not result.rdms.one_body.flags.writeable


def test_orbital_two_rdm_inverse_matches_lows_independent_diagonal_pair_formula():
    options = FermionicShadowOptions(num_particles=2, num_settings=3)
    for setting in random_shadow_settings(4, options):
        pairs = list(combinations(range(4), 2))
        exterior = np.array(
            [
                [np.linalg.det(setting.matrix[np.ix_(row, column)]) for column in pairs]
                for row in pairs
            ]
        )
        for bits in (3, 5, 9):
            # Low Eq.(4), for m=4,N=2,k=2: weights at overlap s=0,1,2.
            weights = [([1, -1.5, 6][sum(bool(bits & (1 << i)) for i in pair)]) for pair in pairs]
            expected = (exterior.conj().T @ (np.array(weights)[:, None] * exterior)).T
            actual = orbital_shadow_snapshot(setting, bits, num_particles=2).two_body
            assert actual is not None
            extracted = np.array([[actual[*p, *q] for q in pairs] for p in pairs])
            np.testing.assert_allclose(extracted, expected, atol=1e-12)
    full = orbital_shadow_snapshot(ShadowSetting("orbital_haar", np.eye(3)), 7, num_particles=3)
    np.testing.assert_allclose(full.one_body, np.eye(3))
    assert full.two_body is not None
    assert full.two_body[0, 1, 0, 1] == 1


def test_majorana_inverse_matches_complete_matching_design_for_complex_correlated_state():
    from qiskit.quantum_info import Operator

    state = DeterminantState(4, (5, 10), [1 / np.sqrt(2), 1j / np.sqrt(2)])
    vector = np.zeros(16, dtype=complex)
    vector[[5, 10]] = state.amplitudes
    settings, counts = [], []
    populations: set[int] = set()
    for pairs in _matchings(tuple(range(8))):
        permutation = tuple(mode for pair in pairs for mode in pair)
        parity = (-1) ** sum(
            permutation[i] > permutation[j] for i in range(8) for j in range(i + 1, 8)
        )
        matrix = np.eye(8)[list(permutation)]
        matrix[-1] *= parity
        setting = ShadowSetting("majorana_clifford", matrix)
        probabilities = abs(Operator(shadow_circuit(setting)).data @ vector) ** 2
        batch = {
            format(bits, "04b"): round(64 * probability)
            for bits, probability in enumerate(probabilities)
            if probability > 1e-12
        }
        populations.update(bits.count("1") for bits in batch)
        settings.append(setting)
        counts.append(batch)
    result = estimate_fermionic_shadows(
        settings,
        counts,
        FermionicShadowOptions(
            ensemble="majorana_clifford", num_settings=105, shots_per_setting=64
        ),
    )
    np.testing.assert_allclose(result.rdms.one_body, state.rdms().one_body, atol=1e-13)
    np.testing.assert_allclose(result.rdms.two_body, state.rdms().two_body, atol=1e-13)
    assert populations == {0, 2, 4}
    assert all(record["acceptance_fraction"] == 1 for record in result.metadata["acceptance"])
    observable = FermionicHamiltonian(
        4, (FermionTerm((0, 2), (3, 1), 1j), FermionTerm((1, 3), (2, 0), -1j))
    )
    estimate = result.observable(observable)
    assert estimate["mean"] == pytest.approx(state.expectation(observable))
    assert estimate["standard_error"] is not None
    assert estimate["standard_error"] > 0


def test_particle_postselection_is_explicit_and_majorana_accepts_every_population():
    setting = ShadowSetting("orbital_haar", np.eye(2))
    options = FermionicShadowOptions(
        num_particles=1, num_settings=1, shots_per_setting=4, max_order=1
    )
    with pytest.raises(ConfigError, match="fixed N"):
        estimate_fermionic_shadows([setting], [{"01": 3, "00": 1}], options)
    result = estimate_fermionic_shadows(
        [setting], [{"01": 3, "00": 1}], options.model_copy(update={"postselect_particles": True})
    )
    assert result.metadata["acceptance"][0]["acceptance_fraction"] == 0.75
    assert result.standard_errors_real is result.standard_errors_imag is None
    assert result.rdms.two_body is None
    assert result.observable(FermionicHamiltonian(2, ()))["standard_error"] is None
    with pytest.raises(ConfigError, match="no accepted"):
        estimate_fermionic_shadows(
            [setting], [{"00": 4}], options.model_copy(update={"postselect_particles": True})
        )
    for bits in (0, 1):
        snapshot = majorana_shadow_snapshot(ShadowSetting("majorana_clifford", np.eye(2)), bits)
        assert snapshot.one_body[0, 0] == bits
        assert snapshot.two_body is not None
        assert snapshot.two_body[0, 0, 0, 0] == 0
        assert (
            majorana_shadow_snapshot(
                ShadowSetting("majorana_clifford", np.eye(2)), bits, max_order=1
            ).two_body
            is None
        )


@pytest.mark.parametrize("ensemble", ["orbital_haar", "majorana_clifford"])
def test_real_sampler_collection_preserves_settings_and_sampling_metadata(ensemble):
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(2)
    circuit.x(0)
    options = FermionicShadowOptions(
        ensemble=ensemble,
        num_particles=1 if ensemble == "orbital_haar" else None,
        num_settings=3,
        shots_per_setting=8,
    )
    result = collect_fermionic_shadows(
        circuit, ComponentSelection(name="statevector", options={"seed": 4}), options
    )
    assert len(result.settings) == 3
    assert result.metadata["total_shots"] == 24
    assert len(result.metadata["sampling"]) == 3
    assert sum(sum(batch.values()) for batch in result.counts) == 24


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"num_particles": 1, "num_settings": 5, "shots_per_setting": 3, "max_total_shots": 14},
        {"ensemble": "majorana_clifford", "num_particles": 0},
        {"ensemble": "majorana_clifford", "postselect_particles": True},
        {"num_particles": 1, "unknown": True},
    ],
)
def test_shadow_options_enforce_distinct_domains_and_work_budget(options):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        FermionicShadowOptions.model_validate(options)
    immutable = FermionicShadowOptions(num_particles=1)
    with pytest.raises(ValidationError):
        immutable.__setattr__("num_settings", 1)


@pytest.mark.parametrize(
    ("ensemble", "matrix"),
    [
        ("orbital_haar", [["bad"]]),
        ("unknown", np.eye(2)),
        ("orbital_haar", [1, 0]),
        ("orbital_haar", np.zeros((0, 0))),
        ("orbital_haar", np.ones((1, 2))),
        ("orbital_haar", [[np.nan]]),
        ("orbital_haar", [[0.5]]),
        ("majorana_clifford", np.eye(3)),
        ("majorana_clifford", [[1, 1], [1, -1]] / np.sqrt(2)),
        ("majorana_clifford", np.diag([-1, 1])),
    ],
)
def test_shadow_setting_rejects_invalid_or_unsupported_bases(ensemble, matrix):
    with pytest.raises(ConfigError, match=r"shadow|Majorana"):
        ShadowSetting(ensemble, matrix)


@pytest.mark.parametrize("modes", [True, 1.5, 0])
def test_settings_require_a_positive_integer_mode_count(modes):
    with pytest.raises(ConfigError, match="positive integer"):
        random_shadow_settings(modes, FermionicShadowOptions(ensemble="majorana_clifford"))


def test_settings_guard_particle_domain_and_dense_storage_before_allocation():
    with pytest.raises(ConfigError, match="particle number exceeds"):
        random_shadow_settings(2, FermionicShadowOptions(num_particles=3))
    with pytest.raises(ConfigError, match="storage estimate"):
        random_shadow_settings(1000, FermionicShadowOptions(num_particles=2, max_memory_mb=1))


@pytest.mark.parametrize("bits", [-1, 4, 0])
def test_orbital_snapshots_require_fixed_number_outcomes(bits):
    with pytest.raises(ConfigError, match="matching fixed-N"):
        orbital_shadow_snapshot(ShadowSetting("orbital_haar", np.eye(2)), bits, num_particles=1)


@pytest.mark.parametrize("bits", [-1, 4])
def test_majorana_snapshots_require_in_register_outcomes(bits):
    with pytest.raises(ConfigError, match="matching occupation"):
        majorana_shadow_snapshot(ShadowSetting("majorana_clifford", np.eye(4)), bits)


def test_standalone_snapshots_reject_wrong_channels_and_bound_allocations():
    orbital = ShadowSetting("orbital_haar", np.eye(2))
    majorana = ShadowSetting("majorana_clifford", np.eye(4))
    for setting, order in ((majorana, 2), (orbital, 3)):
        with pytest.raises(ConfigError, match="matching fixed-N"):
            orbital_shadow_snapshot(setting, 1, num_particles=1, max_order=order)
    for setting, order in ((orbital, 2), (majorana, 3)):
        with pytest.raises(ConfigError, match="matching occupation"):
            majorana_shadow_snapshot(setting, 1, max_order=order)
    for budget in (0, True, 1.5):
        with pytest.raises(ConfigError, match="positive integer"):
            orbital_shadow_snapshot(orbital, 1, num_particles=1, max_memory_mb=budget)
    with pytest.raises(ConfigError, match="storage estimate"):
        majorana_shadow_snapshot(ShadowSetting("majorana_clifford", np.eye(20)), 1, max_memory_mb=1)


@pytest.mark.parametrize(
    "counts", [{}, {1: 1}, {"0": 1}, {"02": 1}, {"01": True}, {"01": 0.5}, {"01": 0}]
)
def test_shadow_counts_are_physical_fixed_width_frequencies(counts):
    with pytest.raises(ConfigError, match="binary strings"):
        estimate_fermionic_shadows(
            [ShadowSetting("orbital_haar", np.eye(2))],
            [counts],
            FermionicShadowOptions(num_particles=1, num_settings=1),
        )


def test_shadow_batches_validate_shape_channels_and_shot_limits():
    orbital = ShadowSetting("orbital_haar", np.eye(2))
    options = FermionicShadowOptions(num_particles=1, num_settings=1)
    with pytest.raises(ConfigError, match="match num_settings"):
        estimate_fermionic_shadows([], [], options)
    for setting in (ShadowSetting("majorana_clifford", np.eye(4)),):
        with pytest.raises(ConfigError, match="declared ensemble"):
            estimate_fermionic_shadows([setting], [{"01": 1}], options)
    with pytest.raises(ConfigError, match="mode count"):
        estimate_fermionic_shadows(
            [orbital, ShadowSetting("orbital_haar", np.eye(3))],
            [{"01": 1}, {"001": 1}],
            options.model_copy(update={"num_settings": 2}),
        )
    with pytest.raises(ConfigError, match="shot budgets"):
        estimate_fermionic_shadows([orbital], [{"01": 2}], options)
    with pytest.raises(ConfigError, match="shot budgets"):
        estimate_fermionic_shadows(
            [orbital, orbital],
            [{"01": 1}, {"01": 1}],
            options.model_copy(update={"num_settings": 2, "max_total_shots": 1}),
        )
    first_only = estimate_fermionic_shadows(
        [orbital, orbital],
        [{"01": 1}, {"10": 1}],
        options.model_copy(update={"num_settings": 2, "max_order": 1}),
    )
    assert first_only.standard_errors_real.two_body is None
    assert first_only.standard_errors_imag.two_body is None


def test_observable_contraction_validates_shape_order_and_hermiticity():
    from chemrefine.engines.qiskit.determinants import ReducedDensityMatrices
    from chemrefine.engines.qiskit.shadows import rdm_expectation

    operator = FermionicHamiltonian(2, (FermionTerm((), (), 2), FermionTerm((0,), (0,), -1)))
    raw = ReducedDensityMatrices(np.array([[1, 1j], [-1j, 0]]), None)
    assert rdm_expectation(raw, operator) == 1
    with pytest.raises(ConfigError, match="mode counts"):
        rdm_expectation(raw, FermionicHamiltonian(1, ()))
    with pytest.raises(ConfigError, match="two-body RDM"):
        rdm_expectation(raw, FermionicHamiltonian(2, (FermionTerm((0, 1), (1, 0), 1),)))
    for value in (1j, np.inf):
        with pytest.raises(ConfigError, match="finite and real"):
            rdm_expectation(ReducedDensityMatrices(np.diag([value, 0]), None), operator)


def test_collection_rejects_unbound_preparations_and_oversized_simulators():
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter

    parameterized = QuantumCircuit(2)
    parameterized.rx(Parameter("a"), 0)
    for circuit in (parameterized, QuantumCircuit(2, 1)):
        with pytest.raises(ConfigError, match="no classical bits or unbound"):
            collect_fermionic_shadows(
                circuit,
                ComponentSelection(name="statevector"),
                FermionicShadowOptions(num_particles=1),
            )
    for sampler, qubits in (
        (ComponentSelection(name="statevector"), 16),
        (ComponentSelection(name="aer", options={"method": "density_matrix"}), 9),
    ):
        with pytest.raises(ConfigError, match="sampler storage"):
            collect_fermionic_shadows(
                QuantumCircuit(qubits),
                sampler,
                FermionicShadowOptions(
                    ensemble="majorana_clifford", num_settings=1, max_order=1, max_memory_mb=1
                ),
            )


@pytest.mark.parametrize(
    "sampler_options",
    [
        {"method": "automatic", "noise_model": {}},
        {"method": "automatic"},
        {"method": "statevector"},
        {"method": "matrix_product_state"},
        {"method": "density_matrix"},
    ],
)
def test_real_aer_shadow_sampling_supports_declared_simulation_methods(sampler_options):
    from qiskit import QuantumCircuit

    result = collect_fermionic_shadows(
        QuantumCircuit(2),
        ComponentSelection(name="aer", options={"seed_simulator": 7, **sampler_options}),
        FermionicShadowOptions(ensemble="majorana_clifford", num_settings=1, shots_per_setting=2),
    )
    assert result.metadata["total_shots"] == 2


def test_sampling_passes_scheduler_grants_to_selected_provider(monkeypatch):
    from qiskit import QuantumCircuit
    from qiskit.primitives import StatevectorSampler

    from chemrefine.engines.qiskit.context import SamplerResource
    from chemrefine.engines.qiskit.registry import SAMPLERS, ComponentSpec, NoComponentOptions

    builds, closed = [], []

    def build(*, options, device, cores):
        """Record one resource lifetime and the scheduler grant for all settings."""
        assert device == "cuda" and cores == 7
        builds.append(options)
        return SamplerResource(
            StatevectorSampler(seed=np.random.default_rng(7)), close=lambda: closed.append(True)
        )

    monkeypatch.setitem(
        SAMPLERS._specs, "custom_provider", ComponentSpec(NoComponentOptions, build)
    )
    preparation = QuantumCircuit(2)
    preparation.x(0)
    result = collect_fermionic_shadows(
        preparation,
        ComponentSelection(name="custom_provider"),
        FermionicShadowOptions(num_particles=1, num_settings=3),
        device="cuda",
        cores=7,
    )
    assert len(result.counts) == 3 and all(sum(counts.values()) == 1 for counts in result.counts)
    assert len(builds) == 1 and closed == [True]


def test_seeded_one_shot_orbital_shadows_converge_with_independent_setting_noise():
    """Restarting the same sampler seed per setting gives 1.25 instead of occupation one."""
    from qiskit import QuantumCircuit

    preparation = QuantumCircuit(2)
    preparation.x(0)
    result = collect_fermionic_shadows(
        preparation,
        ComponentSelection(name="statevector", options={"seed": 1}),
        FermionicShadowOptions(
            num_particles=1, num_settings=2000, shots_per_setting=1, max_order=1, seed=7
        ),
    )
    assert result.standard_errors_real is not None
    mean = result.rdms.one_body[0, 0].real
    error = result.standard_errors_real.one_body[0, 0]
    assert abs(mean - 1) < 4 * error
    assert abs(mean - 1) < 0.06
    seeds = [batch["sampling_seed"] for batch in result.metadata["sampling"]]
    assert len(set(seeds)) == 2000


def test_shadow_uncertainty_tracks_variation_across_independent_acquisitions():
    """Setting-level errors reflect both random rotations and independent outcome noise."""
    from qiskit import QuantumCircuit

    preparation = QuantumCircuit(2)
    preparation.x(0)
    means, variances = [], []
    for seed in range(20):
        result = collect_fermionic_shadows(
            preparation,
            ComponentSelection(name="statevector", options={"seed": seed}),
            FermionicShadowOptions(
                num_particles=1, num_settings=100, shots_per_setting=1, max_order=1, seed=100 + seed
            ),
        )
        assert result.standard_errors_real is not None
        means.append(result.rdms.one_body[0, 0].real)
        variances.append(result.standard_errors_real.one_body[0, 0] ** 2)
    ratio = np.var(means, ddof=1) / np.mean(variances)
    assert 0.35 < ratio < 2.5
    assert abs(np.mean(means) - 1) < 0.04

"""Released-SQD energy, particle, spin and resource-boundary regression tests."""

from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from chemrefine.engines.qiskit.components.subspace_algorithms import (
    SQDOptions,
    _check_sampling_memory,
    _checked_counts,
    _spin_diagnostics,
    validate_subspace_options,
)
from chemrefine.engines.qiskit.options import QiskitOptions
from chemrefine.errors import ConfigError

# Nature 0.8 uses Qiskit's deprecated circuit superclasses. Keep this
# compatibility allowance local to tests of the released optional stack.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning:qiskit")


@pytest.mark.parametrize(
    "options",
    [
        {"counts": {}},
        {"counts": {"0 1": 10}},
        {"counts": {"0101": 1.5}},
        {"counts": {"0101": True}},
        {"counts": {"0101": 0}},
        {"counts": {"0101": 1}, "parameter_values": []},
        {"parameter_values": [float("nan")]},
        {"max_iterations": 1001},
        {"unused": True},
    ],
)
def test_sqd_options_fail_closed(options):
    with pytest.raises(ValidationError):
        SQDOptions(**options)


@pytest.mark.parametrize(
    "selection",
    [
        {"mapper": "parity"},
        {"estimator": "basic_backend"},
        {"optimizer": "cobyla"},
        {"ansatz": "efficient_su2"},
        {"initial_state": "zero"},
        {"initial_point": "random"},
        {"sampler": "basic_backend"},
        {"device": "cuda"},
    ],
)
def test_supplied_counts_reject_ignored_graph_settings(selection):
    options = QiskitOptions.from_raw(
        {"algorithm": {"name": "sqd", "options": {"counts": {"0101": 1}}}, **selection}
    )
    with pytest.raises(ConfigError):
        validate_subspace_options(options)


def test_particle_postselection_uses_alpha_on_right_and_records_discarded_shots():
    data = SimpleNamespace(norb=2, nelec=(1, 0))
    counts = {"0001": 3, "0100": 2}
    selected, metadata = _checked_counts(counts, data, SQDOptions(configuration_recovery=False))
    assert selected == {"0001": 3}
    assert metadata["invalid_particle_fraction"] == pytest.approx(0.4)
    assert metadata["valid_shots"] == 3
    recovered, _ = _checked_counts(counts, data, SQDOptions(configuration_recovery=True))
    assert recovered == counts


@pytest.mark.parametrize(
    "counts, options, message",
    [
        ({}, {}, "nonempty"),
        ({"01": 1}, {}, "4-bit"),
        ({"001x": 1}, {}, "4-bit"),
        ({"0001": -1}, {}, "positive"),
        ({"0001": True}, {}, "positive"),
        ({"0001": 1.5}, {}, "positive"),
        ({"0001": 3}, {"max_total_shots": 2}, "max_total_shots"),
        ({"0100": 3}, {}, "no determinants"),
    ],
)
def test_counts_reject_malformed_samples_and_absent_target_sector(counts, options, message):
    with pytest.raises(ConfigError, match=message):
        _checked_counts(counts, SimpleNamespace(norb=2, nelec=(1, 0)), SQDOptions(**options))


def test_dense_sampling_budget_is_checked_before_execution():
    request = SimpleNamespace(options=QiskitOptions.from_raw({"algorithm": "sqd"}))
    data = SimpleNamespace(norb=20, nelec=(1, 1))
    with pytest.raises(ConfigError, match="sampling storage"):
        _check_sampling_memory(request, data, SQDOptions(), total_shots=10)


@pytest.mark.parametrize(
    "sampler",
    [
        {"name": "aer", "options": {"method": "density_matrix"}},
        {"name": "aer", "options": {"noise_model": {}}},
        {"name": "aer", "options": {"method": "statevector"}},
        {"name": "aer", "options": {"method": "matrix_product_state"}},
        {"name": "external_test"},
    ],
)
def test_aer_sampling_memory_respects_simulation_representation(sampler):
    """Density matrices need a different storage estimate from pure statevectors."""
    request = SimpleNamespace(options=QiskitOptions(algorithm="sqd", sampler=sampler))
    _check_sampling_memory(
        request, SimpleNamespace(norb=2, nelec=(1, 1)), SQDOptions(), total_shots=10
    )


@pytest.mark.parametrize(
    "data,options,message",
    [
        (SimpleNamespace(norb=64, nelec=(1, 1)), SQDOptions(), "63"),
        (SimpleNamespace(norb=3, nelec=(2, 1)), SQDOptions(symmetrize_spin=True), "equal alpha"),
    ],
)
def test_sampling_rejects_unsupported_dimensions_and_spin(data, options, message):
    """Sampling does not start when the downstream SCI kernel cannot accept it."""
    with pytest.raises(ConfigError, match=message):
        _check_sampling_memory(
            SimpleNamespace(options=QiskitOptions(algorithm="sqd")), data, options, total_shots=10
        )


def test_spin_validation_rejects_bad_norm_and_working_dictionary_growth():
    """S² validation bounds its full sparse image as well as the original state."""
    state = SimpleNamespace(amplitudes=np.array([[1.0]]), ci_strs_a=[1], ci_strs_b=[2])
    data = SimpleNamespace(norb=2, nelec=(1, 1), target_spin_squared=0)
    with pytest.raises(ConfigError, match="spin-validation storage"):
        _spin_diagnostics(state, data, memory_bytes=384)
    state.amplitudes[:] = 0.5
    with pytest.raises(ConfigError, match="normalization"):
        _spin_diagnostics(state, data, memory_bytes=100000)


def test_spin_residual_includes_determinants_outside_selected_space():
    # |alpha_0 beta_1> has <S²>=1 but also mixes singlet and triplet. Its
    # spin-exchanged determinant lies outside this one-dimensional SCI space.
    state = SimpleNamespace(
        amplitudes=np.array([[1.0]]), ci_strs_a=np.array([1]), ci_strs_b=np.array([2])
    )
    data = SimpleNamespace(norb=2, nelec=(1, 1), target_spin_squared=0)
    mean, residual = _spin_diagnostics(state, data, memory_bytes=1_000_000)
    assert mean == pytest.approx(1)
    assert residual == pytest.approx(np.sqrt(2))
    with pytest.raises(ConfigError, match="spin-validation storage"):
        _spin_diagnostics(state, data, memory_bytes=192)


def test_sparse_spin_operator_matches_full_pyscf_application():
    pytest.importorskip("pyscf")
    from pyscf.fci import cistring
    from pyscf.fci.spin_op import contract_ss

    norb, nelec = 4, (2, 2)
    strings = cistring.make_strings(range(norb), 2)
    amplitudes = np.random.default_rng(10).normal(size=(len(strings), len(strings)))
    amplitudes /= np.linalg.norm(amplitudes)
    state = SimpleNamespace(amplitudes=amplitudes, ci_strs_a=strings, ci_strs_b=strings)
    data = SimpleNamespace(norb=norb, nelec=nelec, target_spin_squared=2)
    applied = contract_ss(amplitudes, norb, nelec)
    mean, residual = _spin_diagnostics(state, data, memory_bytes=1_000_000)
    assert mean == pytest.approx(np.vdot(amplitudes, applied).real)
    assert residual == pytest.approx(np.linalg.norm(applied - 2 * amplitudes))


def test_correct_spin_expectation_does_not_hide_mixed_spin_sectors():
    pytest.importorskip("pyscf")
    from pyscf.fci import cistring
    from pyscf.fci.spin_op import contract_ss

    strings = cistring.make_strings(range(4), 2)
    shape = (len(strings), len(strings))
    basis = np.eye(len(strings) ** 2)
    matrix = np.column_stack([contract_ss(row.reshape(shape), 4, (2, 2)).ravel() for row in basis])
    _, vectors = np.linalg.eigh(matrix)
    mixed = np.sqrt(2 / 3) * vectors[:, 0] + np.sqrt(1 / 3) * vectors[:, -1]
    state = SimpleNamespace(amplitudes=mixed.reshape(shape), ci_strs_a=strings, ci_strs_b=strings)
    data = SimpleNamespace(norb=4, nelec=(2, 2), target_spin_squared=2)
    mean, residual = _spin_diagnostics(state, data, memory_bytes=1_000_000)
    assert mean == pytest.approx(2)
    assert residual == pytest.approx(np.sqrt(8))

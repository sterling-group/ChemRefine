"""Released-SQD energy, particle, spin and resource-boundary regression tests."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from chemrefine.engines.qiskit.components.subspace_algorithms import (
    SQDOptions,
    SqDRIFTOptions,
    _checked_counts,
    _spin_diagnostics,
    check_sampling_memory,
    validate_subspace_options,
)
from chemrefine.engines.qiskit.options import QiskitOptions
from chemrefine.errors import ConfigError

# Nature 0.8 uses Qiskit's deprecated circuit superclasses. Keep this
# compatibility allowance local to tests of the released optional stack.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning:qiskit")


@pytest.fixture
def h2_prepared():
    pytest.importorskip("qiskit_addon_sqd")
    pytest.importorskip("qiskit_nature")
    from chemrefine.engines.qiskit.api import ElectronicStructureData, prepare_problem

    path = Path(__file__).parent / "data/engines/qiskit/h2_integrals.json"
    return prepare_problem(ElectronicStructureData(**json.loads(path.read_text())))


def _run_sqd(prepared, algorithm_options=None, **kwargs):
    from chemrefine.engines.qiskit.api import run_problem

    settings = {
        "counts": {"0101": 10, "0110": 10, "1001": 10, "1010": 10},
        "samples_per_batch": 4,
        "configuration_recovery": False,
        "num_batches": 1,
        **(algorithm_options or {}),
    }
    return run_problem(
        prepared,
        options={"algorithm": {"name": "sqd", "options": settings}},
        **kwargs,
    )


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
    "options",
    [
        {"times": []},
        {"times": [-1]},
        {"times": [float("inf")]},
        {"times": [1], "randomizations": 257},
        {"times": [1], "shots": 1001, "randomizations": 1, "max_total_shots": 1000},
    ],
)
def test_sqdrift_options_bound_circuits_and_shots(options):
    with pytest.raises(ValidationError):
        SqDRIFTOptions(**options)


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


def test_complete_sqd_subspace_matches_h2_and_callbacks_keep_active_energy(h2_prepared):
    seen = []

    def callback(record):
        seen.append(record["objective_value_hartree"])
        record["metadata"]["batch"] = "changed by client"

    result = _run_sqd(h2_prepared, callback=callback)
    assert result.energy_hartree == pytest.approx(-1.1373060357534, abs=1e-10)
    assert result.converged is None
    assert result.termination_reason == "sampled_subspace_completed"
    assert result.metadata["solver"]["subspace_dimension"] == 4
    assert result.metadata["solver"]["spin_eigenstate_residual"] < 1e-10
    assert seen[0] + h2_prepared.energy_offsets["nuclear_repulsion_energy"] == pytest.approx(
        result.energy_hartree
    )
    assert result.metadata["evaluations"][0]["metadata"]["batch"] == 0
    json.dumps(result.as_dict(), allow_nan=False)


def test_sqd_recovery_reports_invalid_population_and_reproduces_seed(h2_prepared):
    settings = {
        "counts": {"0101": 10, "0110": 10, "1001": 10, "1010": 10, "0000": 8, "1111": 2},
        "configuration_recovery": True,
        "max_iterations": 3,
        "seed": 12,
    }
    first = _run_sqd(h2_prepared, settings)
    second = _run_sqd(h2_prepared, settings)
    assert first.energy_hartree == pytest.approx(-1.1373060357534, abs=1e-10)
    assert first.metadata["solver"] == second.metadata["solver"]
    assert first.metadata["solver"]["invalid_particle_fraction"] == pytest.approx(0.2)
    assert len(first.metadata["solver"]["iterations"]) >= 2


def test_dimension_budget_is_cartesian_product_not_single_spin_size(h2_prepared):
    result = _run_sqd(h2_prepared, {"max_subspace_dimension": 1})
    assert result.metadata["solver"]["subspace_dimension"] == 1
    assert result.metadata["solver"]["spin_factor_dimension_limit"] == 1
    assert result.energy_hartree >= -1.1373060357534 - 1e-10


def test_memory_guard_runs_before_bitarray_allocation(h2_prepared, monkeypatch):
    from qiskit.primitives.containers import BitArray

    def forbidden(*args, **kwargs):
        raise AssertionError("allocation must follow the resource guard")

    monkeypatch.setattr(BitArray, "from_counts", forbidden)
    with pytest.raises(ConfigError, match="max_memory_mb"):
        _run_sqd(h2_prepared, {"counts": {"0101": 1_000_000}, "max_memory_mb": 1})


def test_dense_sampling_budget_is_checked_before_execution():
    request = SimpleNamespace(options=QiskitOptions.from_raw({"algorithm": "sqd"}))
    data = SimpleNamespace(norb=20, nelec=(1, 1))
    with pytest.raises(ConfigError, match="sampling storage"):
        check_sampling_memory(request, data, SQDOptions(), total_shots=10)


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
    check_sampling_memory(
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
        check_sampling_memory(
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


@pytest.mark.parametrize(
    "initial,algorithm_options,message",
    [
        ([0, 0, 0], {"counts": {"0101": 10}}, "counts cannot"),
        ([0, 0, 0], {"parameter_values": [0, 0, 0]}, "mutually exclusive"),
        (None, {"shots": 100, "max_total_shots": 10}, "shots exceed"),
    ],
)
def test_sqd_rejects_conflicting_inputs_and_shot_budgets(
    h2_prepared, initial, algorithm_options, message
):
    """Each supplied input has one unambiguous role in sample acquisition."""
    from chemrefine.engines.qiskit.api import run_problem

    with pytest.raises(ConfigError, match=message):
        run_problem(
            h2_prepared,
            options={"algorithm": {"name": "sqd", "options": algorithm_options}},
            initial_point=initial,
        )


@pytest.mark.parametrize("api_parameters", [False, True])
def test_fixed_parameters_use_the_declared_input_channel(h2_prepared, api_parameters):
    """Both supported channels sample the supplied vector without implicit optimization."""
    from chemrefine.engines.qiskit.api import run_problem

    opts: dict[str, Any] = {"shots": 32, "num_batches": 1, "max_iterations": 1}
    if not api_parameters:
        opts["parameter_values"] = [0, 0, 0]
    result = run_problem(
        h2_prepared,
        options={"algorithm": {"name": "sqd", "options": opts}},
        initial_point=[0, 0, 0] if api_parameters else None,
    )
    assert result.metadata["solver"]["sampling"]["parameter_source"] == (
        "initial_point" if api_parameters else "parameter_values"
    )


def test_sqd_validates_registered_circuit_capability_and_artifact(h2_prepared, monkeypatch):
    """A provider declaration and its actual artifact must both supply a circuit."""
    from dataclasses import replace

    from chemrefine.engines.qiskit.api import run_problem
    from chemrefine.engines.qiskit.registry import ANSATZE

    original = ANSATZE.spec("uccsd")
    monkeypatch.setitem(ANSATZE._specs, "uccsd", replace(original, capabilities=frozenset()))
    with pytest.raises(ConfigError, match="supplying a circuit"):
        run_problem(h2_prepared, options={"algorithm": "sqd"})
    monkeypatch.setitem(
        ANSATZE._specs,
        "uccsd",
        replace(original, builder=lambda **kw: SimpleNamespace(circuit=None)),
    )
    with pytest.raises(ConfigError, match="supplying a circuit"):
        run_problem(h2_prepared, options={"algorithm": "sqd"})


@pytest.mark.parametrize("failure", ["dimension", "memory", "energy"])
def test_sqd_provider_failures_do_not_escape_resource_or_result_checks(
    h2_prepared, monkeypatch, failure
):
    """Check real adapter contracts around the external diagonalization routine."""
    import qiskit_addon_sqd.fermion as provider

    from chemrefine.engines.qiskit.components.subspace_algorithms import _solve_samples
    from chemrefine.engines.qiskit.fermionic import fermionic_integrals
    from chemrefine.engines.qiskit.native import NativeSolveRequest

    data = fermionic_integrals(h2_prepared)

    def diagonalize(*args, **kwargs):
        """Return a malformed provider result or request an oversized solver batch."""
        if failure == "energy":
            return SimpleNamespace(
                energy=np.nan,
                sci_state=SimpleNamespace(
                    amplitudes=np.array([[1.0]]), ci_strs_a=[1], ci_strs_b=[1]
                ),
            )
        spaces = [] if failure == "dimension" else [(np.arange(100), np.arange(100))]
        return kwargs["sci_solver"](spaces, data.h1, data.h2, 2, (1, 1))

    monkeypatch.setattr(provider, "diagonalize_fermionic_hamiltonian", diagonalize)
    with pytest.raises(ConfigError, match=r"dimension budget|working storage|non-finite"):
        _solve_samples(
            NativeSolveRequest(h2_prepared, QiskitOptions(algorithm="sqd")),
            data,
            {"0101": 1},
            SQDOptions(max_memory_mb=1),
            {},
        )


def test_sqd_large_orbital_count_and_sqdrift_parameters_rejected(h2_prepared):
    """Downstream limits apply to externally supplied samples and randomized circuits."""
    from chemrefine.engines.qiskit.api import run_problem
    from chemrefine.engines.qiskit.components.subspace_algorithms import _chemistry, _solve_samples
    from chemrefine.engines.qiskit.fermionic import fermionic_integrals
    from chemrefine.engines.qiskit.native import NativeSolveRequest

    request = NativeSolveRequest(h2_prepared, QiskitOptions(algorithm="sqd"))
    data = replace(fermionic_integrals(h2_prepared), norb=64)
    with pytest.raises(ConfigError, match="63"):
        _solve_samples(request, data, {"0" * 63 + "1" + "0" * 63 + "1": 1}, SQDOptions(), {})
    with pytest.raises(ConfigError, match="untapered"):
        _chemistry(replace(request, options=QiskitOptions(mapper="parity")))
    with pytest.raises(ConfigError, match="does not accept"):
        run_problem(h2_prepared, options={"algorithm": "sqdrift"}, initial_point=[0])


def test_sqdrift_rejects_nonfinite_operator_before_sampling(h2_prepared, monkeypatch):
    """Corrupted coefficients must not define a qDRIFT probability distribution."""
    pytest.importorskip("qiskit_fermions")
    import qiskit_fermions.operators.terms.grouping as grouping
    import qiskit_fermions.operators.terms.ordering as ordering

    from chemrefine.engines.qiskit.fermionic import fermionic_integrals
    from chemrefine.engines.qiskit.sqdrift import sqdrift_circuits

    monkeypatch.setattr(
        ordering,
        "canonical_order",
        lambda x: SimpleNamespace(get_coeffs=lambda: np.array([np.nan])),
    )
    monkeypatch.setattr(grouping, "group_terms_by_electronic_structure", lambda *a, **k: None)
    with pytest.raises(ConfigError, match="non-finite"):
        list(
            sqdrift_circuits(
                fermionic_integrals(h2_prepared), times=[1], num_groups=1, randomizations=1, seed=0
            )
        )


def test_sqd_rejects_spin_contaminated_selected_state(h2_prepared):
    with pytest.raises(ConfigError, match="spin residual"):
        _run_sqd(h2_prepared, {"counts": {"0110": 10}})


def test_open_shell_sqd_keeps_alpha_beta_order_and_rejects_spin_symmetrization():
    pytest.importorskip("qiskit_addon_sqd")
    from chemrefine.engines.qiskit.api import ElectronicStructureData, prepare_problem

    prepared = prepare_problem(
        ElectronicStructureData(2, 1, 3, np.diag([-1, 0.5, 1]), np.zeros((3,) * 4))
    )
    result = _run_sqd(prepared, {"counts": {"001011": 10, "001101": 10}})
    assert result.energy_hartree == pytest.approx(-1.5)
    assert result.metadata["solver"]["spin_squared"] == pytest.approx(0.75)
    assert result.metadata["solver"]["spin_eigenstate_residual"] < 1e-12
    with pytest.raises(ConfigError, match="equal alpha and beta"):
        _run_sqd(prepared, {"counts": {"001011": 10}, "symmetrize_spin": True})


def test_empty_spin_sector_reports_released_selected_ci_limitation():
    pytest.importorskip("qiskit_addon_sqd")
    from chemrefine.engines.qiskit.api import ElectronicStructureData, prepare_problem

    prepared = prepare_problem(
        ElectronicStructureData(1, 0, 2, np.diag([-1, 0.5]), np.zeros((2,) * 4))
    )
    with pytest.raises(ConfigError, match="at least one alpha and one beta"):
        _run_sqd(prepared, {"counts": {"0001": 10}})


def test_reordered_active_space_restores_inactive_and_nuclear_constants_once():
    pytest.importorskip("qiskit_addon_sqd")
    from chemrefine.engines.qiskit.api import (
        ActiveSpaceOptions,
        ElectronicStructureData,
        MolecularMetadata,
        prepare_problem,
    )

    data = ElectronicStructureData(
        2,
        2,
        4,
        np.diag([-2.0, -1.0, 0.0, 1.0]),
        np.zeros((4,) * 4),
        molecular_metadata=MolecularMetadata(("Li", "H"), ((0, 0, 0), (0, 0, 1.6))),
        nuclear_repulsion_energy=1.0,
    )
    prepared = prepare_problem(
        data,
        freeze_core=True,
        active_space=ActiveSpaceOptions(electrons=2, orbitals=2, active_orbitals=[3, 1]),
    )
    result = _run_sqd(prepared, {"counts": {"1010": 10}})
    assert result.metadata["active_energy_hartree"] == pytest.approx(-2)
    assert result.electronic_energy_hartree == pytest.approx(-6)
    assert result.energy_hartree == pytest.approx(-5)


def test_fixed_ansatz_sampling_records_unoptimized_parameters(h2_prepared):
    from chemrefine.engines.qiskit.api import run_problem

    result = run_problem(
        h2_prepared,
        options={
            "algorithm": {"name": "sqd", "options": {"shots": 64, "num_batches": 1}},
            "sampler": {"name": "statevector", "options": {"seed": 3}},
        },
    )
    assert result.energy_hartree == pytest.approx(-1.1169989967540044, abs=1e-10)
    assert result.ansatz == "uccsd"
    sampling = result.metadata["solver"]["sampling"]
    assert sampling["parameter_source"] == "configured_initial_point"
    assert sampling["parameter_values"] == [0.0, 0.0, 0.0]
    assert result.metadata["solver"]["input_shots"] == 64


@pytest.mark.parametrize("parameters", [[0], [0, float("nan"), 0], [0, 1j, 0]])
def test_fixed_ansatz_rejects_invalid_explicit_parameters(h2_prepared, parameters):
    from chemrefine.engines.qiskit.api import run_problem

    with pytest.raises(ConfigError, match="parameters"):
        run_problem(h2_prepared, options={"algorithm": "sqd"}, initial_point=parameters)


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


def test_sqdrift_preserves_particle_sectors_reference_order_and_seed(h2_prepared):
    pytest.importorskip("qiskit_fermions")
    from qiskit.quantum_info import Statevector

    from chemrefine.engines.qiskit.fermionic import fermionic_integrals
    from chemrefine.engines.qiskit.sqdrift import sqdrift_circuits

    data = fermionic_integrals(h2_prepared)
    data = replace(data, occupations=((1,), (0,)))
    settings = {"times": [0, 1], "num_groups": 40, "randomizations": 2, "seed": 73}
    first = list(sqdrift_circuits(data, **settings))
    second = list(sqdrift_circuits(data, **settings))
    assert len(first) == 4
    assert len({metadata["seed"] for _, metadata in first}) == 4
    assert Statevector.from_instruction(first[0][0]).probabilities_dict() == {"0110": 1.0}
    for (circuit, metadata), (repeated, repeated_metadata) in zip(first, second, strict=True):
        assert metadata == repeated_metadata
        state = Statevector.from_instruction(circuit)
        np.testing.assert_allclose(
            state.data, Statevector.from_instruction(repeated).data, atol=1e-14
        )
        assert circuit.count_ops() == repeated.count_ops()
        assert np.linalg.norm(state.data) == pytest.approx(1)
        for bits, probability in state.probabilities_dict().items():
            if probability > 1e-12:
                assert bits[:2].count("1") == bits[2:].count("1") == 1


def test_sqdrift_runs_real_qiskit_fermions_sampler_and_sqd(h2_prepared):
    pytest.importorskip("qiskit_fermions")
    from chemrefine.engines.qiskit.api import run_problem

    result = run_problem(
        h2_prepared,
        options={
            "algorithm": {
                "name": "sqdrift",
                "options": {
                    "times": [1, 2],
                    "num_groups": 40,
                    "randomizations": 3,
                    "shots": 256,
                    "num_batches": 1,
                    "max_iterations": 2,
                    "seed": 73,
                },
            },
            "sampler": {"name": "statevector", "options": {"seed": 41}},
        },
    )
    assert result.energy_hartree == pytest.approx(-1.1373060357534, abs=1e-10)
    assert result.converged is None
    solver = result.metadata["solver"]
    assert solver["input_shots"] == 1536
    assert solver["invalid_particle_fraction"] == 0
    assert solver["sampling"]["experimental"] is True
    assert solver["sampling"]["diagonal_terms_retained"] is True
    assert len(solver["sampling"]["circuits"]) == 6

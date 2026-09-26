"""Real-stack general sampled-subspace states, excited roots and complex chemistry."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from chemrefine.engines.qiskit.components.subspace_algorithms import SQDOptions
from chemrefine.engines.qiskit.data import ElectronicStructureData
from chemrefine.engines.qiskit.determinants import FermionicHamiltonian, projected_eigensystem
from chemrefine.errors import ConfigError

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning:qiskit")


def _prepared():
    """Prepare the recorded H2 integrals without a chemistry driver."""
    pytest.importorskip("qiskit_addon_sqd")
    from chemrefine.engines.qiskit.problem import prepare_problem

    return prepare_problem(
        ElectronicStructureData(
            **json.loads(
                (Path(__file__).parent / "data/engines/qiskit/h2_integrals.json").read_text()
            )
        )
    )


def _run(prepared, **options):
    """Exercise the full native solver and canonical molecular reporting boundary."""
    from chemrefine.engines.qiskit.workflow import run_problem

    return run_problem(
        prepared,
        options={
            "algorithm": {
                "name": "sqd",
                "options": {
                    "projection": "explicit",
                    "counts": {"0101": 10, "0110": 10, "1001": 10, "1010": 10},
                    "samples_per_batch": 4,
                    "num_batches": 1,
                    "configuration_recovery": False,
                    **options,
                },
            }
        },
    )


def test_all_roots_retain_states_and_restore_offsets_once():
    prepared = _prepared()
    result = _run(prepared, num_roots=4, target_root=2, spin_constraint="report")
    assert result.energy_hartree == pytest.approx(-0.16275315579588445)
    assert result.target_root == 2
    assert len(result.states) == 4
    np.testing.assert_allclose(
        np.array(result.root_energies_hartree) - result.root_electronic_energies_hartree,
        prepared.energy_offsets["nuclear_repulsion_energy"],
    )
    assert result.root_total_energies_hartree == result.root_energies_hartree
    assert "states" not in result.as_dict()
    assert result.metadata["solver"]["property_source"] == "projected_subspace_state"
    assert result.states[0].expectation(
        FermionicHamiltonian.from_operator(prepared.fermionic_hamiltonian, num_modes=4)
    ) == pytest.approx(result.metadata["active_energies_hartree"][0])


def test_default_cartesian_path_also_retains_its_original_state():
    result = _run(_prepared(), projection="cartesian")
    assert len(result.states) == 1
    assert result.states[0].determinants == (5, 9, 6, 10)
    assert result.energy_hartree == pytest.approx(-1.1373060357534)


def test_explicit_projection_has_no_accidental_cartesian_closure():
    result = _run(_prepared(), counts={"0101": 1, "1010": 1})
    assert result.states[0].determinants == (5, 10)
    assert result.metadata["solver"]["subspace_dimension"] == 2
    assert result.energy_hartree == pytest.approx(-1.1373060357534)


def test_explicit_recovery_reproducible_and_spin_report_distinguishes_contamination():
    options = {
        "counts": {"0101": 10, "1010": 10, "1111": 4, "0000": 5},
        "configuration_recovery": True,
        "max_iterations": 3,
        "num_batches": 2,
        "seed": 4,
    }
    first = _run(_prepared(), **options)
    second = _run(_prepared(), **options)
    assert first.metadata["solver"] == second.metadata["solver"]
    assert len(first.metadata["solver"]["iterations"]) >= 2
    assert first.metadata["solver"]["invalid_particle_fraction"] == pytest.approx(9 / 29)
    contaminated = _run(_prepared(), counts={"0110": 1}, spin_constraint="report")
    assert contaminated.metadata["solver"]["root_spin"][0]["spin_squared"] == pytest.approx(1)
    with pytest.raises(ConfigError, match="spin residual"):
        _run(_prepared(), counts={"0110": 1})


def test_one_spin_empty_and_complex_unrestricted_hopping_are_supported():
    pytest.importorskip("qiskit_addon_sqd")
    from chemrefine.engines.qiskit.problem import prepare_problem

    one = np.array([[-1, 0.3j], [-0.3j, 0.2]])
    data = ElectronicStructureData(
        1,
        0,
        2,
        one,
        np.zeros((2,) * 4),
        one_body_integrals_beta=np.diag([0.4, 0.8]),
        two_body_integrals_beta_beta=np.zeros((2,) * 4),
        two_body_integrals_beta_alpha=np.zeros((2,) * 4),
        overlap_alpha_beta=np.eye(2),
    )
    result = _run(prepare_problem(data), counts={"0001": 5, "0010": 5}, num_roots=2)
    np.testing.assert_allclose(result.root_electronic_energies_hartree, np.linalg.eigvalsh(one))
    assert result.root_total_energies_hartree is None
    assert np.max(np.abs(result.states[0].amplitudes.imag)) > 0.1


def test_rotated_complex_coulomb_integrals_keep_hermitian_energy_and_rdms():
    pytest.importorskip("qiskit_nature")
    from chemrefine.engines.qiskit.problem import prepare_problem

    rng = np.random.default_rng(4)
    unitary = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
    two = np.zeros((2,) * 4)
    two[0, 0, 0, 0], two[1, 1, 1, 1] = 0.7, 0.5
    rotated = np.einsum(
        "ap,bq,cr,ds,abcd->pqrs", unitary.conj(), unitary, unitary.conj(), unitary, two
    )
    one = unitary.conj().T @ np.diag([-1.0, 0.3]) @ unitary
    prepared = prepare_problem(ElectronicStructureData(1, 1, 2, one, rotated))
    model = FermionicHamiltonian.from_operator(prepared.fermionic_hamiltonian, num_modes=4)
    roots = projected_eigensystem(model, [5, 6, 9, 10], num_roots=4)
    np.testing.assert_allclose(roots.energies, [-1.3, -0.7, -0.7, 1.1], atol=1e-13)
    assert np.trace(roots.states[0].rdms().one_body) == pytest.approx(2)


@pytest.mark.parametrize(
    "options",
    [
        {"num_roots": 2},
        {"spin_constraint": "report"},
        {"projection": "explicit", "num_roots": 1, "target_root": 1},
    ],
)
def test_multiroot_options_are_explicit_and_consistent(options):
    with pytest.raises(ValidationError):
        SQDOptions(**options)


def test_generic_sampling_and_spin_symmetrization_work_with_real_circuits():
    from chemrefine.engines.qiskit.workflow import run_problem

    prepared = _prepared()
    result = run_problem(
        prepared,
        options={
            "algorithm": {
                "name": "sqd",
                "options": {
                    "projection": "explicit",
                    "shots": 16,
                    "num_batches": 1,
                    "configuration_recovery": False,
                },
            }
        },
    )
    assert result.metadata["solver"]["sampling"]["source"] == "fixed_ansatz"
    assert result.energy_hartree == pytest.approx(-1.1169989967540044)
    symmetrized = _run(prepared, counts={"0110": 1}, symmetrize_spin=True, spin_constraint="report")
    assert symmetrized.states[0].determinants == (6, 9)


def test_new_native_root_contract_rejects_mismatched_state_and_energy_counts():
    from chemrefine.engines.qiskit.native import NativeOutcome, NativeSolveRequest, summarize_native
    from chemrefine.engines.qiskit.options import QiskitOptions

    prepared = _prepared()
    request = NativeSolveRequest(prepared, QiskitOptions())
    with pytest.raises(ConfigError, match="target_root"):
        summarize_native(
            NativeOutcome(-1, active_energies_hartree=(-2,)), request, runtime_seconds=0
        )
    state = _run(prepared).states[0]
    with pytest.raises(ConfigError, match=r"state.*root count"):
        summarize_native(NativeOutcome(-1, states=(state, state)), request, runtime_seconds=0)
    missing_spin = replace(prepared, problem=prepared.problem)
    missing_spin.problem.properties.angular_momentum = None
    with pytest.raises(ConfigError, match="angular momentum"):
        _run(missing_spin)
    assert _run(missing_spin, spin_constraint="report").metadata["solver"]["root_spin"] == []


def test_explicit_spin_and_storage_preflight_reject_before_solving():
    from itertools import combinations

    from chemrefine.engines.qiskit.problem import prepare_problem

    prepared = prepare_problem(ElectronicStructureData(1, 0, 2, np.eye(2), np.zeros((2,) * 4)))
    with pytest.raises(ConfigError, match="equal alpha"):
        _run(prepared, counts={"0001": 1}, symmetrize_spin=True)
    prepared = prepare_problem(ElectronicStructureData(4, 4, 8, np.eye(8), np.zeros((8,) * 4)))
    halves = [sum(1 << i for i in occupied) for occupied in combinations(range(8), 4)]
    counts = {format(alpha | (beta << 8), "016b"): 1 for alpha in halves for beta in halves}
    with pytest.raises(ConfigError, match="input storage"):
        _run(prepared, counts=counts, max_memory_mb=1)


def test_explicit_callback_is_detached_and_sqdrift_uses_same_projector():
    from chemrefine.engines.qiskit.workflow import run_problem

    seen = []

    def callback(record):
        seen.append(record["objective_value_hartree"])
        record["metadata"]["batch"] = 99

    prepared = _prepared()
    result = run_problem(
        prepared,
        callback=callback,
        options={
            "algorithm": {
                "name": "sqd",
                "options": {
                    "projection": "explicit",
                    "counts": {"0101": 5},
                    "configuration_recovery": False,
                    "num_batches": 1,
                },
            }
        },
    )
    assert seen == [result.metadata["active_energy_hartree"]]
    assert result.metadata["evaluations"][0]["metadata"]["batch"] == 0
    result = run_problem(
        prepared,
        options={
            "algorithm": {
                "name": "sqdrift",
                "options": {
                    "projection": "explicit",
                    "times": [0],
                    "num_groups": 1,
                    "randomizations": 1,
                    "num_batches": 1,
                    "max_iterations": 1,
                    "shots": 4,
                },
            }
        },
    )
    assert result.metadata["solver"]["sampling"]["source"] == "sqdrift"


def test_complex_scalar_metadata_still_requires_a_real_domain():
    from chemrefine.engines.qiskit.data import MolecularMetadata

    with pytest.raises(ConfigError, match="valid numbers"):
        MolecularMetadata(("H",), ((0, 0, 1j),))

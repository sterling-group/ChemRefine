"""Fixed Krylov circuits and excitation-expanded sampled spaces against exact references."""

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from chemrefine.engines.qiskit.components.krylov import (
    ExtendedSQDOptions,
    SKQDOptions,
    _reference_excitations,
    krylov_circuits,
)
from chemrefine.engines.qiskit.native import NativeSolveRequest
from chemrefine.engines.qiskit.options import QiskitOptions
from chemrefine.errors import ConfigError

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning:qiskit")


def _prepared():
    """Create a reproducible active H2 problem without running a chemistry driver."""
    pytest.importorskip("qiskit_addon_sqd")
    from chemrefine.engines.qiskit.api import ElectronicStructureData, prepare_problem

    return prepare_problem(
        ElectronicStructureData(
            **json.loads(
                (Path(__file__).parent / "data/engines/qiskit/h2_integrals.json").read_text()
            )
        )
    )


def test_krylov_circuits_are_powers_of_one_synthesized_operator_including_zero():
    from qiskit.quantum_info import Operator, Statevector

    prepared = _prepared()
    request = NativeSolveRequest(prepared, QiskitOptions(algorithm="skqd"))
    options = SKQDOptions(num_steps=3, time_step=0.7, repetitions=2)
    circuits = krylov_circuits(request, options)
    assert len(circuits) == 4
    reference = Operator(circuits[0]).data
    step = Operator(circuits[1]).data @ reference.conj().T
    for power, circuit in enumerate(circuits):
        np.testing.assert_allclose(
            Operator(circuit).data, np.linalg.matrix_power(step, power) @ reference, atol=1e-12
        )
    assert np.argmax(abs(Statevector(circuits[0]).data)) == 5
    lie = krylov_circuits(request, options.model_copy(update={"product_formula": "lie"}))
    assert len(lie) == 4
    assert not np.allclose(Operator(lie[1]).data, Operator(circuits[1]).data, atol=1e-8)


def test_krylov_sampling_diagonalizes_original_hamiltonian_and_records_all_shots():
    from chemrefine.engines.qiskit.workflow import run_problem

    prepared = _prepared()
    result = run_problem(
        prepared,
        options={
            "algorithm": {
                "name": "skqd",
                "options": {
                    "num_steps": 3,
                    "time_step": 1,
                    "shots": 512,
                    "num_batches": 1,
                    "max_iterations": 1,
                    "samples_per_batch": 4,
                },
            },
            "sampler": {"name": "statevector", "options": {"seed": 4}},
        },
    )
    assert result.energy_hartree == pytest.approx(-1.1373060357534)
    metadata = result.metadata["solver"]
    assert metadata["input_shots"] == 2048
    assert metadata["sampling"]["includes_time_zero"]
    assert metadata["sampling"]["fixed_step_operator"]
    assert [record["power"] for record in metadata["sampling"]["circuits"]] == [0, 1, 2, 3]
    with pytest.raises(ConfigError, match="initial parameters"):
        run_problem(prepared, options={"algorithm": "skqd"}, initial_point=[])
    with pytest.raises(ConfigError, match="does not consume"):
        run_problem(prepared, options={"algorithm": "skqd", "ansatz": "efficient_su2"})


def test_extended_sqd_single_reference_reaches_h2_roots_without_inventing_shots():
    from chemrefine.engines.qiskit.workflow import run_problem

    result = run_problem(
        _prepared(),
        options={
            "algorithm": {
                "name": "extended_sqd",
                "options": {
                    "counts": {"0101": 13},
                    "num_roots": 4,
                    "target_root": 1,
                    "num_batches": 1,
                    "configuration_recovery": False,
                },
            }
        },
    )
    np.testing.assert_allclose(
        result.root_energies_hartree,
        [-1.1373060357534, -0.5246155553643471, -0.16275315579588445, 0.49505774161810767],
    )
    assert result.metadata["solver"]["input_shots"] == 13
    assert result.metadata["solver"]["generated_determinants"] == 3
    assert result.metadata["solver"]["sampling"]["expanded_dimension"] == 4
    assert result.states[0].determinants == (5, 6, 9, 10)
    assert abs(np.trace(result.states[1].rdms(bra=result.states[0]).one_body)) < 1e-12


def test_extended_sqd_reordered_occupations_define_excitation_pool():
    prepared = _prepared()
    prepared.problem.orbital_occupations = [0, 1]
    prepared.problem.orbital_occupations_b = [0, 1]
    request = NativeSolveRequest(prepared, QiskitOptions(algorithm="extended_sqd"))
    pool = _reference_excitations(request, ExtendedSQDOptions())
    assert pool == (((0,), (1,)), ((2,), (3,)), ((0, 2), (3, 1)))
    with pytest.raises(ConfigError, match="max_excitation_operators"):
        _reference_excitations(request, ExtendedSQDOptions(max_excitation_operators=1))
    prepared.problem.orbital_occupations = [0.5, 0.5]
    with pytest.raises(ConfigError, match="binary reference"):
        _reference_excitations(request, ExtendedSQDOptions())


@pytest.mark.parametrize(
    "options,message",
    [
        ({"max_generated_determinants": 1}, "max_generated"),
        ({"max_subspace_dimension": 1}, "subspace or memory"),
    ],
)
def test_extended_subspace_budgets_fail_explicitly(options, message):
    from chemrefine.engines.qiskit.workflow import run_problem

    with pytest.raises(ConfigError, match=message):
        run_problem(
            _prepared(),
            options={
                "algorithm": {
                    "name": "extended_sqd",
                    "options": {
                        "counts": {"0101": 10},
                        "num_batches": 1,
                        "configuration_recovery": False,
                        **options,
                    },
                }
            },
        )


@pytest.mark.parametrize(
    "kind,options",
    [
        (SKQDOptions, {"num_steps": 256}),
        (SKQDOptions, {"shots": 100, "max_total_shots": 10}),
        (SKQDOptions, {"product_formula": "lie", "suzuki_order": 4}),
        (ExtendedSQDOptions, {"excitation_ranks": []}),
        (ExtendedSQDOptions, {"excitation_ranks": [1, 1]}),
        (ExtendedSQDOptions, {"max_total_diagonalizations": 30}),
    ],
)
def test_sampling_and_expansion_options_have_finite_complete_budgets(kind, options):
    with pytest.raises(ValidationError):
        kind(**options)


def test_krylov_circuit_storage_guard_precedes_power_materialization():
    prepared = _prepared()
    request = NativeSolveRequest(prepared, QiskitOptions(algorithm="skqd"))
    with pytest.raises(ConfigError, match="circuit storage"):
        krylov_circuits(request, SKQDOptions(num_steps=100, repetitions=10, max_memory_mb=1))


def test_extended_sqd_can_sample_an_actual_circuit_and_optimize_orbitals():
    from chemrefine.engines.qiskit.workflow import run_problem

    result = run_problem(
        _prepared(),
        options={
            "algorithm": {
                "name": "extended_sqd",
                "options": {
                    "shots": 16,
                    "num_batches": 1,
                    "configuration_recovery": False,
                    "orbital_optimization": {"max_iterations": 1},
                },
            }
        },
    )
    assert result.energy_hartree == pytest.approx(-1.1373060357534)
    assert result.metadata["solver"]["sampling"]["acquisition"]["source"] == "fixed_ansatz"
    assert result.states[0].orbital_rotation is not None
    assert result.ansatz == "uccsd"


def test_expansion_handles_excitations_that_annihilate_correlated_determinants():
    from chemrefine.engines.qiskit.workflow import run_problem

    result = run_problem(
        _prepared(),
        options={
            "algorithm": {
                "name": "extended_sqd",
                "options": {
                    "counts": {"0101": 5, "1010": 5},
                    "num_batches": 1,
                    "configuration_recovery": False,
                    "samples_per_batch": 2,
                    "num_roots": 4,
                },
            }
        },
    )
    assert result.metadata["solver"]["sampling"]["source_dimension"] == 2
    assert result.metadata["solver"]["generated_determinants"] == 2

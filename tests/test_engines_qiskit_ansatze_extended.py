"""Rank-selected and number-conserving ansatze obey their scientific contracts."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from chemrefine.engines.qiskit.components.ansatze import (
    EfficientSU2Options,
    UCCOptions,
    UCCSDOptions,
    build_efficient_su2,
    build_ucc,
    build_uccsd,
)
from chemrefine.engines.qiskit.components.ansatze_extended import (
    ExcitationPreservingOptions,
    RealAmplitudesOptions,
    UCCRanksOptions,
    build_excitation_preserving,
    build_real_amplitudes,
    build_ucc_ranks,
)
from chemrefine.engines.qiskit.components.initial_states_extended import (
    DeterminantOptions,
    build_determinant,
)
from chemrefine.engines.qiskit.context import ElectronicStructureContext
from chemrefine.engines.qiskit.options import QiskitOptions
from chemrefine.engines.qiskit.registry import validate_component_graph
from chemrefine.errors import ConfigError

pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning:qiskit"),
    pytest.mark.filterwarnings("ignore::PendingDeprecationWarning:qiskit"),
]


@pytest.mark.parametrize("ranks", [[], [0], [-1], [1, 1], [True], [1.5]])
def test_ucc_rank_options_reject_invalid_or_repeated_ranks(ranks):
    """Only distinct positive integer ranks describe an unambiguous generator list."""
    with pytest.raises(ValidationError):
        UCCRanksOptions(ranks=ranks)


@pytest.mark.parametrize(
    "model", [UCCRanksOptions, RealAmplitudesOptions, ExcitationPreservingOptions]
)
def test_extended_ansatz_options_forbid_unknown_fields(model):
    """Misspelled scientific controls fail during configuration validation."""
    with pytest.raises(ValidationError):
        model(unknown=True)


def test_efficient_su2_can_start_without_a_reference_circuit():
    """The optional initial-state boundary also supports the all-zero basis state."""
    pytest.importorskip("qiskit")
    context = ElectronicStructureContext(None, None, None, 1, (1, 0), 2, 2)
    result = build_efficient_su2(
        options=EfficientSU2Options(reps=1), context=context, initial_state=None
    )
    assert result.circuit.num_qubits == 2
    assert result.circuit.num_parameters > 0


def _context(orbitals=3):
    """Supply a JW context and non-prefix determinant with one electron of each spin."""
    mappers = pytest.importorskip("qiskit_nature.second_q.mappers")
    pytest.importorskip("qiskit")
    problem = SimpleNamespace(
        orbital_occupations=[1] + [0] * (orbitals - 1),
        orbital_occupations_b=[1] + [0] * (orbitals - 1),
    )
    context = ElectronicStructureContext(
        problem, mappers.JordanWignerMapper(), None, orbitals, (1, 1), 2 * orbitals, 1
    )
    initial = build_determinant(
        options=DeterminantOptions(alpha=[1], beta=[orbitals - 1]), context=context
    )
    return context, initial


@pytest.mark.parametrize("include_imaginary", [False, True])
def test_ucc_ranks_uses_the_explicit_determinant_for_every_rank(include_imaginary):
    """Singles and doubles originate in the actual reference, not the problem's HF prefix."""
    context, initial = _context()
    artifacts = build_ucc_ranks(
        options=UCCRanksOptions(ranks=[2, 1], include_imaginary=include_imaginary),
        context=context,
        initial_state=initial,
    )
    excitations = artifacts.circuit.excitation_list
    assert {len(occupied) for occupied, _ in excitations} == {1, 2}
    assert len(excitations[0][0]) == 2
    for occupied, unoccupied in excitations:
        assert set(occupied) <= {1, 5}
        assert not set(unoccupied) & {1, 5}
    assert artifacts.circuit.num_parameters == len(artifacts.operator_pool)
    assert len(artifacts.pool_metadata) == len(excitations)
    zero = artifacts.circuit.assign_parameters(np.zeros(artifacts.circuit.num_parameters))
    from qiskit.quantum_info import Statevector

    assert Statevector.from_instruction(zero).equiv(Statevector.from_instruction(initial))


def test_generalized_rank_pool_is_independent_of_determinant():
    """Generalized excitations connect orbital pairs independently of their occupations."""
    context, initial = _context()
    alternate = build_determinant(options=DeterminantOptions(alpha=[0], beta=[0]), context=context)
    options = UCCRanksOptions(ranks=[1, 2], generalized=True)
    first = build_ucc_ranks(options=options, context=context, initial_state=initial)
    second = build_ucc_ranks(options=options, context=context, initial_state=alternate)
    assert first.circuit.excitation_list == second.circuit.excitation_list


@pytest.mark.parametrize(
    ("builder", "options"),
    [
        (build_ucc_ranks, UCCRanksOptions()),
        (build_uccsd, UCCSDOptions()),
        (build_ucc, UCCOptions(excitations=[((1,), (0,))])),
    ],
)
def test_reference_aware_ucc_can_cross_parallel_compilation_boundary(
    builder, options, qiskit_spawn_pool
):
    """QNSPSA's batched fidelity transpilation serializes circuits to worker processes."""
    context, initial = _context()
    original = builder(options=options, context=context, initial_state=initial).circuit
    restored = pickle.loads(pickle.dumps(original))
    assert restored.excitation_list == original.excitation_list
    assert tuple(restored.parameters) == tuple(original.parameters)
    from qiskit.quantum_info import Statevector

    values = np.random.default_rng(31).uniform(-0.2, 0.2, original.num_parameters)
    assert Statevector(restored.assign_parameters(values)).equiv(
        Statevector(original.assign_parameters(values))
    )
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    manager = generate_preset_pass_manager(
        optimization_level=1, basis_gates=["rz", "sx", "x", "cx"], seed_transpiler=31
    )
    compiled = manager.run([original, restored], num_processes=2)
    assert len(compiled) == 2
    for circuit in compiled:
        assert Statevector(circuit.assign_parameters(values)).equiv(
            Statevector(original.assign_parameters(values))
        )


@pytest.mark.parametrize("mapping", ["parity", "bravyi_kitaev", "z2_tapered"])
def test_excitation_preserving_mapping_fails_during_schema_preflight(mapping):
    """GUI/config validation must reject a known bad encoding before spawning a worker."""
    with pytest.raises(ConfigError, match="unreduced Jordan-Wigner"):
        validate_component_graph(
            QiskitOptions(algorithm="vqe", ansatz="excitation_preserving", mapper=mapping)
        )


@pytest.mark.parametrize("algorithm", ["exact", "vqe"])
def test_excitation_preserving_preflight_checks_only_consumed_ansatz(algorithm):
    """An unused ansatz must not constrain an exact solver's mapper."""
    validate_component_graph(
        QiskitOptions(
            algorithm=algorithm,
            ansatz="excitation_preserving",
            mapper="bravyi_kitaev" if algorithm == "exact" else "jordan_wigner",
        )
    )


@pytest.mark.parametrize("include_imaginary", [False, True])
def test_existing_uccsd_uses_explicit_determinant_metadata(include_imaginary):
    """Existing UCCSD and ADAPT pools must follow the selected determinant as rank UCC does."""
    context, initial = _context()
    artifacts = build_uccsd(
        options=UCCSDOptions(include_imaginary=include_imaginary),
        context=context,
        initial_state=initial,
    )
    for occupied, unoccupied in artifacts.circuit.excitation_list:
        assert set(occupied) <= {1, 5}
        assert not set(unoccupied) & {1, 5}
    assert artifacts.circuit.num_parameters == len(artifacts.operator_pool)
    assert context.problem.orbital_occupations == [1, 0, 0]
    assert context.problem.orbital_occupations_b == [1, 0, 0]


@pytest.mark.parametrize("flatten", [True, False])
def test_efficient_su2_function_preserves_initial_state_and_flatten_semantics(flatten):
    """Functional EfficientSU2 is equivalent to preparing the determinant before its layers."""
    context, initial = _context()
    options = EfficientSU2Options(reps=1, su2_gates=["rx"], flatten=flatten)
    artifacts = build_efficient_su2(options=options, context=context, initial_state=initial)
    from qiskit.circuit.library import efficient_su2
    from qiskit.quantum_info import Statevector

    layers = efficient_su2(6, reps=1, su2_gates=["rx"], entanglement="reverse_linear")
    parameters = np.random.default_rng(37).uniform(-1, 1, layers.num_parameters)
    expected = Statevector.from_instruction(initial).evolve(layers.assign_parameters(parameters))
    actual = Statevector.from_instruction(artifacts.circuit.assign_parameters(parameters))
    assert actual.equiv(expected)
    assert artifacts.circuit.num_parameters == 12
    assert (artifacts.circuit.size() > 1) is flatten


def test_unavailable_ucc_rank_pool_fails_actionably():
    """An impossible requested rank is distinct from an empty occupied/virtual pool."""
    context, initial = _context()
    with pytest.raises(ConfigError, match="cannot exceed"):
        build_ucc_ranks(options=UCCRanksOptions(ranks=[4]), context=context, initial_state=initial)
    with pytest.raises(ConfigError, match="empty excitation pool"):
        build_ucc_ranks(options=UCCRanksOptions(ranks=[3]), context=context, initial_state=initial)


@pytest.mark.parametrize("mode", ["iswap", "fsim"])
@pytest.mark.parametrize("preserve_spin", [True, False])
def test_excitation_preserving_layers_conserve_the_claimed_populations(mode, preserve_spin):
    """Arbitrary parameters cannot leak out of the declared occupation sector."""
    context, initial = _context()
    artifacts = build_excitation_preserving(
        options=ExcitationPreservingOptions(mode=mode, preserve_spin=preserve_spin),
        context=context,
        initial_state=initial,
    )
    from qiskit.quantum_info import Statevector, partial_trace

    parameters = np.random.default_rng(13).uniform(-2, 2, artifacts.circuit.num_parameters)
    state = Statevector.from_instruction(artifacts.circuit.assign_parameters(parameters))
    for index, probability in enumerate(state.probabilities()):
        if probability < 1e-12:
            continue
        alpha = (index & 0b111).bit_count()
        beta = (index >> 3).bit_count()
        assert alpha + beta == 2
        if preserve_spin:
            assert (alpha, beta) == (1, 1)
    if preserve_spin:
        assert partial_trace(state, [3, 4, 5]).purity().real < 1 - 1e-4


@pytest.mark.parametrize("mapping", ["parity", "bravyi_kitaev"])
def test_excitation_preserving_refuses_nonoccupation_encodings(mapping):
    """Preserving encoded Hamming weight is insufficient in parity or BK representations."""
    context, initial = _context()
    from dataclasses import replace

    from qiskit_nature.second_q.mappers import BravyiKitaevMapper, ParityMapper

    mapper = ParityMapper() if mapping == "parity" else BravyiKitaevMapper()
    with pytest.raises(ConfigError, match="Jordan-Wigner"):
        build_excitation_preserving(
            options=ExcitationPreservingOptions(),
            context=replace(context, mapper=mapper),
            initial_state=initial,
        )


def test_real_amplitudes_is_real_valued():
    """The real-valued family is available without claiming chemical symmetry preservation."""
    context, initial = _context()
    artifacts = build_real_amplitudes(
        options=RealAmplitudesOptions(reps=1), context=context, initial_state=initial
    )
    from qiskit.quantum_info import Statevector

    parameters = np.random.default_rng(19).uniform(-1, 1, artifacts.circuit.num_parameters)
    state = Statevector.from_instruction(artifacts.circuit.assign_parameters(parameters))
    np.testing.assert_allclose(state.data.imag, 0, atol=1e-12)
    assert artifacts.operator_pool is None


def test_rank_selected_vqe_matches_h2_exact_energy():
    """The new family solves the stored molecular Hamiltonian through the public workflow."""
    pytest.importorskip("qiskit_algorithms")
    pytest.importorskip("qiskit_nature")
    from chemrefine.engines.qiskit.api import ElectronicStructureData, prepare_problem, run_vqe

    fixture = Path(__file__).parent / "data/engines/qiskit/h2_integrals.json"
    prepared = prepare_problem(ElectronicStructureData(**json.loads(fixture.read_text())))
    result = run_vqe(
        prepared,
        options={
            "ansatz": {"name": "ucc_ranks", "options": {"ranks": [2]}},
            "optimizer": {"name": "slsqp", "options": {"maxiter": 100, "ftol": 1e-10}},
        },
    )
    assert result.energy_hartree == pytest.approx(-1.1373060357534, abs=1e-8)
    assert result.parameter_count == 1

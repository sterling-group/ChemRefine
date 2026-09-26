"""Explicit determinant validation and mapped occupation consistency."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from chemrefine.engines.qiskit.components.initial_states import reference_occupations
from chemrefine.engines.qiskit.components.initial_states_extended import (
    REFERENCE_OCCUPATIONS_KEY,
    DeterminantOptions,
    build_determinant,
)
from chemrefine.engines.qiskit.context import ElectronicStructureContext
from chemrefine.errors import ConfigError


@pytest.mark.parametrize("indices", [[-1], [0, 0], [True], [0.5], ["0"]])
def test_determinant_indices_are_distinct_nonnegative_integers(indices):
    """Do not coerce ambiguous orbital indices into a different determinant."""
    with pytest.raises(ValidationError):
        DeterminantOptions(alpha=indices, beta=[0])


def test_determinant_requires_both_spin_lists_and_rejects_unknown_options():
    """The two populations are always explicit, including empty spin sectors."""
    with pytest.raises(ValidationError):
        DeterminantOptions(alpha=[0])
    with pytest.raises(ValidationError):
        DeterminantOptions(alpha=[0], beta=[], occupations=[1, 0])


def test_custom_reference_rejects_nonmapping_metadata():
    """Custom circuits cannot supply an ambiguous metadata container."""
    context = ElectronicStructureContext(None, None, None, 2, (1, 1), 4, 1)
    with pytest.raises(ConfigError, match="must be a mapping"):
        reference_occupations(context, SimpleNamespace(metadata=[1]))


@pytest.mark.parametrize(
    ("options", "message"),
    [({"alpha": [0, 1], "beta": []}, "particle counts"), ({"alpha": [2], "beta": [0]}, "below 2")],
)
def test_determinant_checks_prepared_sector_before_mapping(options, message):
    """Count and range errors fail without constructing an optional Qiskit object."""
    context = ElectronicStructureContext(None, None, None, 2, (1, 1), 4, 1)
    with pytest.raises(ConfigError, match=message):
        build_determinant(options=DeterminantOptions(**options), context=context)


@pytest.mark.parametrize(
    ("supplied", "message"),
    [
        (None, "contain alpha and beta"),
        ({"alpha": [0, 1]}, "contain alpha and beta"),
        ({"alpha": None, "beta": None}, "binary alpha and beta"),
        ({"alpha": [1, 1], "beta": [1, 0]}, "particle count"),
        ({"alpha": [0, 0.5], "beta": [1, 0]}, "binary alpha and beta"),
    ],
)
def test_explicit_reference_metadata_is_validated_before_excitation_generation(supplied, message):
    """Bad custom-state metadata cannot silently fall back to problem HF occupations."""
    context = ElectronicStructureContext(None, None, None, 2, (1, 1), 4, 1)
    state = SimpleNamespace(metadata={REFERENCE_OCCUPATIONS_KEY: supplied})
    with pytest.raises(ConfigError, match=message):
        reference_occupations(context, state)


@pytest.mark.parametrize("mapping", ["jordan_wigner", "parity", "reduced_parity", "bravyi_kitaev"])
@pytest.mark.parametrize("particles", [(1, 1), (1, 0), (0, 1)])
def test_determinant_encodes_each_occupied_orbital_under_real_mappers(mapping, particles):
    """An explicit determinant overrides the problem reference while retaining its sector."""
    mappers = pytest.importorskip("qiskit_nature.second_q.mappers")
    operators = pytest.importorskip("qiskit_nature.second_q.operators")
    quantum_info = pytest.importorskip("qiskit.quantum_info")
    mapper = {
        "jordan_wigner": mappers.JordanWignerMapper(),
        "parity": mappers.ParityMapper(),
        "reduced_parity": mappers.ParityMapper(num_particles=particles),
        "bravyi_kitaev": mappers.BravyiKitaevMapper(),
    }[mapping]
    width = 2 if mapping == "reduced_parity" else 4
    problem = SimpleNamespace(orbital_occupations=[1, 0], orbital_occupations_b=[1, 0])
    context = ElectronicStructureContext(problem, mapper, None, 2, particles, width, 1)
    options = DeterminantOptions(
        alpha=[1] if particles[0] else [], beta=[0] if particles[1] else []
    )
    circuit = build_determinant(options=options, context=context)
    state = quantum_info.Statevector.from_instruction(circuit)
    expected = [0, particles[0], particles[1], 0]
    observed = []
    for index in range(4):
        number = operators.FermionicOp({f"+_{index} -_{index}": 1.0}, num_spin_orbitals=4)
        observed.append(state.expectation_value(mapper.map(number)).real)
    np.testing.assert_allclose(observed, expected, atol=1e-12)
    assert circuit.metadata[REFERENCE_OCCUPATIONS_KEY] == {
        "alpha": expected[:2],
        "beta": expected[2:],
    }
    assert problem.orbital_occupations == [1, 0]

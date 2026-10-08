"""Independent fermionic algebra, complex tensor, RDM and subspace regressions."""

from itertools import combinations, product
from types import SimpleNamespace

import numpy as np
import pytest

from chemrefine.engines.qiskit.determinants import (
    DeterminantState,
    FermionicHamiltonian,
    FermionTerm,
    apply_operators,
    projected_eigensystem,
    projected_operator,
)
from chemrefine.errors import ConfigError

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning:qiskit")


def _complex_model():
    """Create a complex hopping model with two density interactions."""
    terms = [FermionTerm((i,), (i,), 0.3 * i) for i in range(4)]
    for i, j in ((0, 1), (1, 2), (2, 3), (3, 0)):
        terms.extend([FermionTerm((i,), (j,), 0.2 + 0.3j), FermionTerm((j,), (i,), 0.2 - 0.3j)])
    terms.extend([FermionTerm((0, 2), (2, 0), 0.7), FermionTerm((1, 3), (3, 1), 0.5)])
    return FermionicHamiltonian(4, tuple(terms))


def _qiskit_matrix(model):
    """Map independently through Nature and Qiskit's Jordan-Wigner implementation."""
    pytest.importorskip("qiskit_nature")
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from qiskit_nature.second_q.operators import FermionicOp

    data = {
        " ".join(
            [*(f"+_{i}" for i in term.creation), *(f"-_{i}" for i in term.annihilation)]
        ): term.coefficient
        for term in model.terms
    }
    return (
        JordanWignerMapper().map(FermionicOp(data, num_spin_orbitals=model.num_modes)).to_matrix()
    )


def test_projected_complex_hamiltonian_matches_independent_jordan_wigner():
    model = _complex_model()
    basis = tuple(sum(1 << i for i in occupied) for occupied in combinations(range(4), 2))
    matrix = _qiskit_matrix(model)[np.ix_(basis, basis)]
    operator = projected_operator(model, basis)
    np.testing.assert_allclose(operator @ np.eye(len(basis)), matrix, atol=1e-14)
    np.testing.assert_allclose(operator.rmatvec(np.ones(len(basis))), matrix @ np.ones(len(basis)))
    spectrum = projected_eigensystem(model, basis, num_roots=len(basis))
    np.testing.assert_allclose(spectrum.energies, np.linalg.eigvalsh(matrix), atol=1e-13)
    vectors = np.array([state.amplitudes for state in spectrum.states]).T
    np.testing.assert_allclose(vectors.conj().T @ vectors, np.eye(len(basis)), atol=1e-13)
    assert not spectrum.energies.flags.writeable
    assert max(spectrum.residuals) < 1e-12
    for energy, state in zip(spectrum.energies, spectrum.states, strict=True):
        assert state.expectation(model) == pytest.approx(energy)


def test_sparse_projection_does_not_add_cartesian_determinants_and_keeps_wide_bits():
    model = FermionicHamiltonian(130, (FermionTerm((0,), (0,), -2), FermionTerm((129,), (129,), 3)))
    result = projected_eigensystem(model, [1, 1 << 129], num_roots=2)
    assert result.states[0].determinants == (1, 1 << 129)
    np.testing.assert_allclose(result.energies, [-2, 3])
    paired = FermionicHamiltonian(4, (FermionTerm((0,), (0,), 1),))
    restricted = projected_eigensystem(paired, [0b0101, 0b1010], num_roots=2)
    assert len(restricted.states[0].determinants) == 2


def test_rdms_and_transition_matrices_match_direct_operator_expectations():
    model = _complex_model()
    basis = tuple(sum(1 << i for i in occupied) for occupied in combinations(range(4), 2))
    states = projected_eigensystem(model, basis, num_roots=2).states
    pytest.importorskip("qiskit_nature")
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from qiskit_nature.second_q.operators import FermionicOp

    ket = np.zeros(16, dtype=complex)
    ket[list(basis)] = states[0].amplitudes
    for bra_state in states:
        bra = np.zeros(16, dtype=complex)
        bra[list(basis)] = bra_state.amplitudes
        rdms = states[0].rdms(bra=bra_state)
        for p, q in product(range(4), repeat=2):
            matrix = (
                JordanWignerMapper()
                .map(FermionicOp({f"+_{p} -_{q}": 1}, num_spin_orbitals=4))
                .to_matrix()
            )
            assert rdms.one_body[p, q] == pytest.approx(np.vdot(bra, matrix @ ket), abs=1e-13)
        assert rdms.two_body is not None
        for p, q, r, s in product(range(4), repeat=4):
            matrix = (
                JordanWignerMapper()
                .map(FermionicOp({f"+_{p} +_{q} -_{s} -_{r}": 1}, num_spin_orbitals=4))
                .to_matrix()
            )
            assert rdms.two_body[p, q, r, s] == pytest.approx(np.vdot(bra, matrix @ ket), abs=1e-13)
    rdms = states[0].rdms()
    assert np.trace(rdms.one_body) == pytest.approx(2)
    np.testing.assert_allclose(np.einsum("pqrq->pr", rdms.two_body), rdms.one_body, atol=1e-13)
    assert np.linalg.eigvalsh(rdms.one_body).min() >= -1e-13
    assert np.linalg.eigvalsh(rdms.one_body).max() <= 1 + 1e-13
    blocks = rdms.spin_blocks()
    summed = rdms.spin_summed()
    np.testing.assert_allclose(summed.one_body, blocks["alpha"] + blocks["beta"])
    assert summed.two_body.shape == (2,) * 4
    assert states[0].rdms(max_order=1).spin_summed().two_body is None
    assert not rdms.one_body.flags.writeable and not rdms.two_body.flags.writeable


def test_transition_rdms_between_distinct_bases_and_particle_sectors():
    ket = DeterminantState(4, (1,), np.array([1]))
    bra = DeterminantState(4, (2,), np.array([1j]))
    values = ket.rdms(bra=bra)
    assert values.one_body[1, 0] == -1j
    assert np.count_nonzero(values.one_body) == 1
    vacuum = DeterminantState(4, (0,), np.array([1]))
    assert not np.any(ket.rdms(bra=vacuum).one_body)
    assert ket.num_particles == 1


def test_normal_order_signs_zero_terms_and_hermitian_adjoint():
    model = FermionicHamiltonian(
        4,
        (
            FermionTerm((1, 0), (1, 0), -2),
            FermionTerm((0, 0), (0, 0), 5),
            FermionTerm((0,), (0,), 1),
            FermionTerm((0,), (0,), -1),
            FermionTerm((), (), 0.1),
        ),
    )
    assert model.terms == (FermionTerm((), (), 0.1), FermionTerm((0, 1), (1, 0), 2))
    assert apply_operators(3, (0, 1), (1, 0)) == (3, 1)
    assert apply_operators(1, (0,), ()) is None
    assert apply_operators(0, (), (0,)) is None
    assert model.apply({3: 1j}) == {3: 2.1j}
    with pytest.raises(ConfigError, match="storage budget"):
        model.apply({3: 1}, max_entries=0)
    assert FermionicHamiltonian(1, ()).apply({0: 1}) == {}


@pytest.mark.parametrize(
    "n,terms,message",
    [
        (0, (), "positive"),
        (True, (), "positive"),
        (2, (FermionTerm((0,), (), 1),), "conserve"),
        (3, (FermionTerm((0, 1, 2), (2, 1, 0), 1),), "two bodies"),
        (2, (FermionTerm((3,), (3,), 1),), "outside"),
        (2, (FermionTerm((True,), (0,), 1),), "outside"),
        (2, (FermionTerm((0,), (0,), "bad"),), "finite"),
        (2, (FermionTerm((0,), (0,), np.nan),), "finite"),
        (2, (FermionTerm((0,), (1,), 1),), "Hermitian"),
    ],
)
def test_hamiltonian_rejects_invalid_science(n, terms, message):
    with pytest.raises(ConfigError, match=message):
        FermionicHamiltonian(n, terms)


@pytest.mark.parametrize(
    "n,basis,amplitudes,message",
    [
        (False, (1,), [1], "positive"),
        (2, (), [], "nonempty"),
        (2, (-1,), [1], "nonnegative"),
        (2, (4,), [1], "num_modes"),
        (2, (True,), [1], "Python integers"),
        (2, (1, 1), [1, 0], "unique"),
        (2, (0, 1), [1, 0], "particle sector"),
        (2, (1,), ["bad"], "finite complex"),
        (2, (1,), [np.nan], "finite and match"),
        (2, (1,), [1, 0], "match"),
        (2, (1,), [2], "normalization"),
    ],
)
def test_states_reject_invalid_basis_and_coefficients(n, basis, amplitudes, message):
    with pytest.raises(ConfigError, match=message):
        DeterminantState(n, basis, amplitudes)


def test_observable_and_rdm_resource_guards():
    state = DeterminantState(3, (1,), [1])
    with pytest.raises(ConfigError, match="mode counts"):
        state.expectation(FermionicHamiltonian(4, ()))
    with pytest.raises(ConfigError, match="matching modes"):
        state.rdms(bra=DeterminantState(4, (1,), [1]))
    with pytest.raises(ConfigError, match="max_order"):
        state.rdms(max_order=3)
    with pytest.raises(ConfigError, match="max_memory_mb"):
        state.rdms(max_memory_mb=0)
    with pytest.raises(ConfigError, match="even"):
        state.rdms().spin_blocks()


def test_adapter_keeps_complex_coefficients_and_rejects_unknown_actions():
    pytest.importorskip("qiskit_nature")
    from qiskit_nature.second_q.operators import FermionicOp

    op = FermionicOp({"+_0 -_1": 1j, "+_1 -_0": -1j}, num_spin_orbitals=2)
    model = FermionicHamiltonian.from_operator(op, num_modes=2)
    assert model.terms[0].coefficient == 1j
    bad = SimpleNamespace(normal_order=lambda: SimpleNamespace(terms=lambda: [([("x", 0)], 1)]))
    with pytest.raises(ConfigError, match="unsupported"):
        FermionicHamiltonian.from_operator(bad, num_modes=2)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"num_roots": 0}, "roots"),
        ({"num_roots": 3}, "roots"),
        ({"max_subspace_dimension": 1}, "dimension"),
        ({"max_memory_mb": 0}, "memory"),
        ({"tolerance": 0}, "positive"),
        ({"tolerance": np.nan}, "positive"),
        ({"max_iterations": 0}, "positive"),
    ],
)
def test_projector_budget_errors(kwargs, message):
    with pytest.raises(ConfigError, match=message):
        projected_eigensystem(FermionicHamiltonian(2, ()), [1, 2], **kwargs)


def test_iterative_projection_matches_sparse_diagonal_and_reports_nonconvergence(monkeypatch):
    import scipy.sparse.linalg
    from scipy.sparse.linalg import ArpackNoConvergence

    model = FermionicHamiltonian(65, tuple(FermionTerm((i,), (i,), i) for i in range(65)))
    basis = tuple(1 << i for i in range(65))
    result = projected_eigensystem(model, basis, num_roots=2)
    np.testing.assert_allclose(result.energies, [0, 1], atol=1e-10)

    def fail(*args, **kwargs):
        raise ArpackNoConvergence("fixture", np.array([]), np.zeros((65, 0)))

    monkeypatch.setattr(scipy.sparse.linalg, "eigsh", fail)
    with pytest.raises(ConfigError, match="did not converge"):
        projected_eigensystem(model, basis)


def test_invalid_solver_residual_is_not_returned_as_success(monkeypatch):
    monkeypatch.setattr(
        np.linalg, "eigh", lambda matrix: (np.array([42.0]), np.ones((1, 1), dtype=complex))
    )
    with pytest.raises(ConfigError, match="invalid energy or residual"):
        projected_eigensystem(FermionicHamiltonian(1, ()), [0])


def test_natural_occupations_and_number_correlations_use_state_rdm_conventions():
    state = DeterminantState(4, (5, 10), np.sqrt([0.8, 0.2]))
    rdms = state.rdms()
    np.testing.assert_allclose(rdms.natural_occupations(), [0.8, 0.8, 0.2, 0.2])
    correlations = rdms.number_correlations()
    assert correlations[0, 2] == pytest.approx(0.8)
    assert correlations[0, 0] == pytest.approx(0.8)
    assert correlations[0, 1] == 0
    np.testing.assert_allclose(np.sum(correlations, axis=1), 2 * np.diag(rdms.one_body))
    with pytest.raises(ConfigError, match="two-body"):
        state.rdms(max_order=1).number_correlations()
    transition = DeterminantState(4, (1,), [1]).rdms(bra=DeterminantState(4, (2,), [1]))
    with pytest.raises(ConfigError, match="Hermitian state"):
        transition.natural_occupations()

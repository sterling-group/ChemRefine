"""Independent complete-unitary checks of coupled-exchange synthesis and pool selection."""

from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm

from chemrefine.engines.qiskit.adaptive import append_block, select_blocks
from chemrefine.engines.qiskit.ceo import (
    exchange_pool,
    mvp_circuit,
    ovp_circuit,
    pauli_support,
    qubit_exchange,
    single_exchange_circuit,
)
from chemrefine.engines.qiskit.components.adaptive import AdaptiveOptions
from chemrefine.errors import ConfigError

pytest.importorskip("qiskit_nature")
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator, SparsePauliOp
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper


def context(n=2, mapper=None):
    """Only the declared mapper and orbital-count contract is needed to generate pools."""
    return SimpleNamespace(
        num_spatial_orbitals=n, num_qubits=2 * n, mapper=mapper or JordanWignerMapper()
    )


def pool(n=2, *, coupled=True, imaginary=False, mapper=None):
    """Build a deliberately bounded pool for algebraic tests."""
    return exchange_pool(
        context(n, mapper), coupled=coupled, include_imaginary=imaginary, max_pool_size=10000
    )


@pytest.mark.parametrize("source,target,width", [(0, 1, 2), (3, 1, 4)])
def test_single_exchange_full_unitary_and_two_cx(source, target, width):
    theta = Parameter("theta")
    circuit = single_exchange_circuit(width, source, target, theta)
    generator = qubit_exchange((source,), (target,), width)
    assert circuit.count_ops()["cx"] == 2
    for value in (0, 0.37, -1.12):
        np.testing.assert_allclose(
            Operator(circuit.assign_parameters({theta: value})).data,
            expm(-1j * value * generator.to_matrix()),
            atol=2e-15,
        )


@pytest.mark.parametrize("support,width", [((0, 1, 2, 3), 4), ((0, 2, 3, 5), 6)])
def test_every_ovp_pair_sign_exact_global_phase_and_nine_cx(support, width):
    exchanges = [
        {
            "source": (support[0], support[index]),
            "target": tuple(q for q in support if q not in (support[0], support[index])),
        }
        for index in (1, 2, 3)
    ]
    theta = Parameter("angle")
    for first, second in combinations(exchanges, 2):
        left = qubit_exchange(first["source"], first["target"], width)
        right = qubit_exchange(second["source"], second["target"], width)
        for sign in (-1, 1):
            circuit = ovp_circuit(width, first, second, sign, theta)
            assert circuit.count_ops()["cx"] == 9
            generator = left + sign * right
            for value in (0, 0.43, -0.71):
                np.testing.assert_allclose(
                    Operator(circuit.assign_parameters({theta: value})).data,
                    expm(-1j * value * generator.to_matrix()),
                    atol=3e-15,
                )


def test_mvp_independent_parameters_matches_joint_exponential_thirteen_cx():
    support, width = (0, 2, 3, 4), 5
    generators = [
        qubit_exchange(
            (support[0], q), tuple(i for i in support if i not in (support[0], q)), width
        )
        for q in support[1:]
    ]
    parameters = [Parameter(f"p{i}") for i in range(3)]
    circuit = mvp_circuit(width, generators, parameters)
    assert circuit.count_ops()["cx"] == 13
    values = [0.37, -0.22, 0.19]
    combined = sum(
        value * generator.to_matrix() for value, generator in zip(values, generators, strict=True)
    )
    np.testing.assert_allclose(
        Operator(circuit.assign_parameters(dict(zip(parameters, values, strict=True)))).data,
        expm(-1j * combined),
        atol=3e-15,
    )


def test_mvp_general_eight_odd_y_coefficients_not_only_exchange_subspace():
    strings = ["YXXX", "XYXX", "XXYX", "XXXY", "YYYX", "YYXY", "YXYY", "XYYY"]
    coefficients = np.random.default_rng(18).normal(size=8)
    generator = SparsePauliOp(strings, coefficients)
    circuit = mvp_circuit(4, [generator], [0.29])
    np.testing.assert_allclose(
        Operator(circuit).data, expm(-0.29j * generator.to_matrix()), atol=3e-15
    )


@pytest.mark.parametrize("n,qe_count,ceo_count", [(2, 4, 6), (4, 90, 174)])
def test_generalized_pool_counts_spin_projection_and_quadrature_groups(n, qe_count, ceo_count):
    qe, ceo = pool(n, coupled=False), pool(n, imaginary=True)
    assert len(qe.operator_pool) == qe_count
    assert len(ceo.operator_pool) == 2 * ceo_count
    number = SparsePauliOp.from_sparse_list(
        [("Z", [q], -0.5) for q in range(2 * n)], num_qubits=2 * n
    )
    ms = SparsePauliOp.from_sparse_list(
        [("Z", [q], -0.25 if q < n else 0.25) for q in range(2 * n)], num_qubits=2 * n
    )
    for operator, item in zip(ceo.operator_pool, ceo.pool_metadata, strict=True):
        assert np.allclose((operator @ number - number @ operator).simplify().coeffs, 0)
        assert np.allclose((operator @ ms - ms @ operator).simplify().coeffs, 0)
        assert pauli_support(operator) == frozenset(item["support"])
        assert all(
            ceo.pool_metadata[i]["quadrature"] == item["quadrature"] for i in item["group_indices"]
        )
    assert pauli_support(SparsePauliOp("IIII", [0])) == frozenset()


def test_pool_budget_and_encoding_rejections_are_explicit():
    with pytest.raises(ConfigError, match="before generation"):
        exchange_pool(context(30), coupled=True, include_imaginary=False, max_pool_size=10)
    with pytest.raises(ConfigError, match="empty"):
        pool(1)
    with pytest.raises(ConfigError, match="Jordan-Wigner"):
        pool(mapper=ParityMapper())
    mismatch = context()
    mismatch.num_qubits = 2
    with pytest.raises(ConfigError, match="unreduced"):
        exchange_pool(mismatch, coupled=False, include_imaginary=False, max_pool_size=20)


def test_tapering_projection_filters_changing_and_zero_generators():
    calls = []

    def project(operator):
        calls.append(operator)
        if len(calls) == 1:
            return None
        if len(calls) == 2:
            return SparsePauliOp("I", [0])
        return SparsePauliOp("Y", [1 if len(calls) == 3 else 0])

    mapper = SimpleNamespace(
        mapper=JordanWignerMapper(), chemrefine_tapering=SimpleNamespace(map_operator=project)
    )
    artifacts = pool(mapper=mapper)
    assert len(artifacts.operator_pool) == 1
    assert artifacts.pool_metadata[0]["candidate"]
    assert artifacts.pool_metadata[0]["group_indices"] == [0]
    mapper.chemrefine_tapering.map_operator = lambda _: SparsePauliOp("Y")
    # Equal parents cancel the difference OVP; retain the sum and parent identities.
    artifacts = pool(mapper=mapper)
    assert len(artifacts.operator_pool) == 5
    assert artifacts.pool_metadata[-1]["role"] == "ovp"


def test_invalid_optimized_synthesis_fails_instead_of_changing_the_generator():
    with pytest.raises(ConfigError, match="canonical"):
        ovp_circuit(
            4, {"source": (0, 1), "target": (2, 3)}, {"source": (0, 1), "target": (2, 3)}, 1, 0.2
        )
    with pytest.raises(ConfigError, match="identical support"):
        mvp_circuit(4, [SparsePauliOp("IXYZ")], [0.2])
    with pytest.raises(ConfigError, match="odd-Y"):
        mvp_circuit(4, [SparsePauliOp("XXXX")], [0.2])


def test_tetris_uses_actual_union_of_pauli_support_and_stable_gradient_order():
    operators = tuple(SparsePauliOp(label) for label in ("IIXX", "XXII", "IXXI", "ZIII"))
    blocks = select_blocks(
        np.array([3, -2, 4, 1]), operators, ({},) * 4, ceo_variant=None, tetris=True, threshold=0
    )
    assert [block["indices"] for block in blocks] == [[2], [3]]
    assert (
        select_blocks(np.zeros(4), operators, ({},) * 4, ceo_variant=None, tetris=True, threshold=0)
        == []
    )


@pytest.mark.parametrize(
    "variant,active,kind,indices",
    [
        ("adaptive", [1, 0], "ovp", [4]),
        ("adaptive", [1, 0.3], "mvp", [2, 3]),
        ("ovp", [1, 0.3], "ovp", [4]),
        ("mvp", [1, 0], "mvp", [2, 3]),
    ],
)
def test_ceo_selects_published_ovp_mvp_cases(variant, active, kind, indices):
    artifacts = pool()
    gradients = np.array([0, 0, *active, sum(active), active[0] - active[1]])
    blocks = select_blocks(
        gradients,
        artifacts.operator_pool,
        artifacts.pool_metadata,
        ceo_variant=variant,
        tetris=False,
        threshold=1e-12,
    )
    assert len(blocks) == 1
    assert blocks[0]["kind"] == kind
    assert blocks[0]["indices"] == indices


def test_explicit_product_formulas_and_exact_commutation_guard():
    generator = SparsePauliOp(["X", "Z"])
    options = AdaptiveOptions()
    block = {"indices": [0]}
    with pytest.raises(ConfigError, match="noncommuting"):
        append_block(
            QuantumCircuit(1),
            block,
            (generator,),
            ({},),
            [0.3],
            optimized_occupation=False,
            options=options,
        )
    errors = []
    for evolution, repetitions in (("lie_trotter", 1), ("suzuki", 8)):
        circuit = QuantumCircuit(1)
        append_block(
            circuit,
            block,
            (generator,),
            ({},),
            [0.3],
            optimized_occupation=False,
            options=options.model_copy(update={"evolution": evolution, "repetitions": repetitions}),
        )
        errors.append(np.linalg.norm(Operator(circuit).data - expm(-0.3j * generator.to_matrix())))
    assert errors[1] < errors[0] / 100

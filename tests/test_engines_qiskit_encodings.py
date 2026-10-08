"""Physical equivalence, code sectors, Clifford preparation and local synthesis limits."""

from __future__ import annotations

import json

import numpy as np
import pytest
from pydantic import ValidationError

from chemrefine.engines.qiskit.encodings import (
    LocalEncodingOptions,
    build_local_encoding,
    validate_encoding,
)
from chemrefine.engines.qiskit.flow import (
    commuting_evolution,
    partition_flow_edges,
    vc_flow_diagonalizer,
)
from chemrefine.engines.qiskit.lattice import (
    LatticeDynamicsOptions,
    build_lattice_dynamics,
    simulate_lattice_dynamics,
    square_lattice,
)
from chemrefine.errors import ConfigError

pytest.importorskip("qiskit_fermions")
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford, Operator, SparsePauliOp, Statevector
from qiskit_fermions.mappers.library import jordan_wigner
from qiskit_fermions.operators import FermionOperator
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

EDGES = [(0, 1), (0, 2), (1, 3), (2, 3)]


def _options(name, **kwargs):
    """Supply rectangular dimensions only to square encodings."""
    return LocalEncodingOptions(
        name=name, **({} if name == "bksf_graph" else {"rows": 2, "columns": 2}), **kwargs
    )


def _jw(operator, modes=4):
    """Map through an independent released Jordan-Wigner implementation."""
    return SparsePauliOp.from_sparse_observable(jordan_wigner(operator, modes)).simplify()


@pytest.mark.parametrize("name", ["bksf_graph", "vc_square", "dk_square"])
@pytest.mark.parametrize("occupied", [(), (0,), (0, 2), (0, 1, 3), (0, 1, 2, 3)])
def test_complex_interacting_dynamics_and_observables_match_jw(name, occupied):
    """Both parities retain complex hopping signs and non-diagonal quartic observables."""
    encoding = build_local_encoding(4, EDGES, options=_options(name), occupied_modes=occupied)
    terms = []
    for index, (first, second) in enumerate(EDGES):
        amplitude = 0.3 + 0.1 * index + 0.2j
        terms.extend(
            [
                ([(True, first), (False, second)], -amplitude),
                ([(True, second), (False, first)], -amplitude.conjugate()),
            ]
        )
    terms.append(([(True, 0), (False, 0), (True, 3), (False, 3)], 0.37))
    fermion = FermionOperator.from_terms(terms)
    initial = Statevector.from_instruction(encoding.prepare_reference(occupied))
    mapped = encoding.map_operator(fermion)
    evolved = Statevector(expm_multiply(-0.43j * mapped.to_matrix(sparse=True), initial.data))
    reference = np.zeros(16, dtype=complex)
    reference[sum(1 << mode for mode in occupied)] = 1
    reference = Statevector(expm_multiply(-0.43j * _jw(fermion).to_matrix(sparse=True), reference))
    observables = [
        FermionOperator.from_terms([([(True, mode), (False, mode)], 1)]) for mode in range(4)
    ]
    observables.extend(
        [
            FermionOperator.from_terms(
                [([(True, 0), (False, 3)], 1j), ([(True, 3), (False, 0)], -1j)]
            ),
            FermionOperator.from_terms(
                [
                    ([(True, 0), (True, 2), (False, 3), (False, 1)], 0.2j),
                    ([(True, 1), (True, 3), (False, 2), (False, 0)], -0.2j),
                ]
            ),
        ]
    )
    for observable in observables:
        assert evolved.expectation_value(encoding.map_operator(observable)) == pytest.approx(
            reference.expectation_value(_jw(observable)), abs=2e-12
        )
    for constraint in (*encoding.stabilizers, *encoding.gauge_fixers):
        assert evolved.expectation_value(constraint) == pytest.approx(1, abs=2e-12)
    for bitstring, probability in initial.probabilities_dict().items():
        if probability > 1e-12:
            assert encoding.decode_occupations(bitstring) == tuple(
                int(mode in occupied) for mode in range(4)
            )


@pytest.mark.parametrize("name", ["bksf_graph", "vc_square", "dk_square"])
def test_two_species_and_density_coupling_preserve_distinct_components(name):
    """Cross-species densities are supported without pretending to allow spin transport."""
    edges = EDGES + [(first + 4, second + 4) for first, second in EDGES]
    encoding = build_local_encoding(8, edges, options=_options(name), occupied_modes=[0, 4])
    density = FermionOperator.from_terms([([(True, 0), (False, 0), (True, 4), (False, 4)], 2.3)])
    state = Statevector.from_instruction(encoding.prepare_reference([0, 4]))
    assert state.expectation_value(encoding.map_operator(density)) == pytest.approx(2.3)
    with pytest.raises(ConfigError, match="component parity"):
        encoding.map_operator(FermionOperator.from_terms([([(True, 0), (False, 4)], 1)]))


def test_custom_encoding_rejects_incorrect_algebra_sectors_and_negative_cycles():
    """Commutation and signed stabilizer validation reject scientifically invalid maps."""
    pauli = SparsePauliOp
    with pytest.raises(ConfigError, match="commutation"):
        validate_encoding([pauli("ZI"), pauli("IZ")], {(0, 1): pauli("XI")})
    with pytest.raises(ConfigError, match="undeclared"):
        validate_encoding([pauli("Z"), pauli("Z")], {(0, 1): pauli("X")})
    with pytest.raises(ConfigError, match="inconsistent"):
        validate_encoding([pauli("Z"), pauli("Z")], {(0, 1): pauli("X")}, component_parities=(1,))
    reference = build_local_encoding(3, [(0, 1), (0, 2), (1, 2)])
    with pytest.raises(ConfigError, match="width"):
        validate_encoding([pauli("Z")], {(0, 1): pauli("II")})
    with pytest.raises(ConfigError, match="unit Pauli"):
        validate_encoding([0.5 * pauli("Z")], {})
    with pytest.raises(ConfigError, match="Hermitian"):
        validate_encoding([1j * pauli("Z")], {})
    with pytest.raises(ConfigError, match="parity"):
        reference.prepare_reference([0])
    # Valid custom graph generators round-trip through the reusable public validator.
    custom = validate_encoding(reference.vertices, reference.edges, component_parities=(0,))
    assert np.linalg.norm(
        Statevector.from_instruction(custom.prepare_reference([0, 2])).data
    ) == pytest.approx(1)


@pytest.mark.parametrize("name", ["bksf_graph", "vc_square", "dk_square"])
def test_mapping_and_reference_failures_are_explicit(name):
    """No implicit pairing truncation, invalid occupations or exponential term expansion."""
    encoding = build_local_encoding(4, EDGES, options=_options(name), occupied_modes=[0])
    for occupied in ([0, 0], [4], [-1], [True]):
        with pytest.raises(ConfigError, match="occupied_modes"):
            encoding.prepare_reference(occupied)
    for term, message in [([(True, 4), (False, 0)], "outside"), ([(True, 0)], "number-conserving")]:
        with pytest.raises(ConfigError, match=message):
            encoding.map_operator(FermionOperator.from_terms([(term, 1)]))
    with pytest.raises(ConfigError, match="finite"):
        encoding.map_operator(FermionOperator.from_terms([([], np.nan)]))
    limited = build_local_encoding(
        4, EDGES, options=_options(name, max_expanded_terms=1), occupied_modes=[0]
    )
    with pytest.raises(ConfigError, match="max_expanded_terms"):
        limited.map_operator(FermionOperator.from_terms([([(True, 0), (False, 0)], 1)]))
    with pytest.raises(ConfigError, match="bitstrings"):
        encoding.decode_occupations("x" * encoding.num_qubits)
    with pytest.raises(ConfigError, match="different modes"):
        encoding.edge_operator(0, 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "vc_square"},
        {"rows": 2},
        {"name": "dk_square", "rows": 2, "columns": 2, "component_parities": (0,)},
        {"rows": True},
    ],
)
def test_local_options_reject_ignored_or_incomplete_dimensions(kwargs):
    """Typed selections cannot silently ignore extra geometry or parity controls."""
    with pytest.raises(ValidationError):
        LocalEncodingOptions(**kwargs)


@pytest.mark.parametrize(
    "args,options,message",
    [
        ((0, []), {}, "positive integer"),
        ((True, []), {}, "positive integer"),
        ((2, [(0, 0)]), {}, "valid pairs"),
        ((2, [(0, 1), (1, 0)]), {}, "valid pairs"),
        ((2, [(0, 1.5)]), {}, "integer modes"),
        ((2, [(0,)]), {}, "integer modes"),
        ((4, EDGES), {"max_qubits": 1}, "max_qubits"),
        ((4, EDGES), {"max_generators": 2}, "max_generators"),
        ((4, EDGES), {"component_parities": (0, 0)}, "one parity"),
        ((5, EDGES), {"name": "vc_square", "rows": 2, "columns": 2}, "tile"),
        ((4, [(0, 3)]), {"name": "dk_square", "rows": 2, "columns": 2}, "nearest-neighbor"),
    ],
)
def test_local_encoding_preflight(args, options, message):
    """Invalid topology and bounded resources fail before Clifford or statevector work."""
    with pytest.raises(ConfigError, match=message):
        build_local_encoding(*args, options=LocalEncodingOptions(**options))


@pytest.mark.parametrize(
    "terms",
    [
        [("XX", 0.5), ("YY", 0.3), ("ZZ", 0.7), ("II", -0.1)],
        [("YX", 0.5), ("XY", -0.2)],
        [("II", 0.3)],
    ],
)
def test_commuting_synthesis_retains_dependent_pauli_signs_and_global_phase(terms):
    """XX*YY=-ZZ must not be treated as three independent stabilizer constraints."""
    operator = SparsePauliOp.from_list(terms)
    circuit = commuting_evolution(operator, -0.37)
    assert Operator(circuit).data == pytest.approx(expm(0.37j * operator.to_matrix()), abs=2e-12)


@pytest.mark.parametrize(
    "direction,pairs",
    [
        ("east", [(0, 1), (2, 3)]),
        ("west", [(1, 0), (3, 2)]),
        ("south", [(0, 2), (1, 3)]),
        ("north", [(2, 0), (3, 1)]),
    ],
)
def test_vc_flow_cliffords_produce_exact_single_qubit_rotations(direction, pairs):
    """All four documented transfer families have two abstract entangling layers."""
    encoding = build_local_encoding(4, EDGES, options=_options("vc_square"))
    operator = sum(
        0.5j * encoding.vertices[first] @ encoding.edge_operator(first, second)
        for first, second in pairs
    ).simplify()
    basis = vc_flow_diagonalizer(2, 2, 1, direction)
    for pauli in operator.paulis:
        image = pauli.evolve(Clifford(basis), frame="s")
        assert np.count_nonzero(image.x | image.z) == 1
    assert basis.depth(lambda instruction: instruction.operation.name == "cx") == 2
    circuit = commuting_evolution(operator, 0.31, diagonalizer=basis)
    assert Operator(circuit).data == pytest.approx(expm(-0.31j * operator.to_matrix()), abs=3e-12)


def test_flow_coloring_and_invalid_synthesis():
    """Generic directed flows are valid matchings; incorrect synthesis inputs fail closed."""
    edges = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]
    groups = partition_flow_edges(edges)
    assert sorted(edge for group in groups for edge in group) == sorted(edges)
    for group in groups:
        assert len({first for first, _ in group}) == len(group)
        assert len({second for _, second in group}) == len(group)
    for bad in [[(0, 0)], [(0, 1), (0, 1)]]:
        with pytest.raises(ConfigError, match="flow edges"):
            partition_flow_edges(bad)
    for operator, time, message in [
        (SparsePauliOp("X", coeffs=[1j]), 1, "Hermitian"),
        (SparsePauliOp("Z"), np.nan, "finite"),
        (SparsePauliOp.from_list([("X", 1), ("Z", 1)]), 1, "commuting"),
    ]:
        with pytest.raises(ConfigError, match=message):
            commuting_evolution(operator, time)
    with pytest.raises(ConfigError, match="width"):
        commuting_evolution(SparsePauliOp("XX"), 1, diagonalizer=QuantumCircuit(1))
    with pytest.raises(ConfigError, match="diagonal images"):
        commuting_evolution(SparsePauliOp("XX"), 1, diagonalizer=QuantumCircuit(2))
    with pytest.raises(ConfigError, match="cardinal"):
        vc_flow_diagonalizer(2, 2, 1, "up")


@pytest.mark.parametrize("name", ["bksf_graph", "vc_square", "dk_square"])
def test_lattice_complex_hopping_matches_jw_with_same_physical_split(name):
    """Production circuits consume actual occupations and each complex hopping component."""
    model = square_lattice(
        2,
        2,
        spinful=False,
        hopping=0.7,
        hopping_imag=0.2,
        density_interaction=0.3,
        site_potentials=[0.1, -0.2, 0.3, -0.1],
    )
    control = LatticeDynamicsOptions(time=0.27, order=2, steps=2, exact_reference=True)
    reference = simulate_lattice_dynamics(model, occupied_modes=[0, 2], options=control)
    result = simulate_lattice_dynamics(
        model,
        occupied_modes=[0, 2],
        options=control.model_copy(update={"mapping": name, "encoding": _options(name)}),
    )
    assert result.mode_occupations == pytest.approx(reference.mode_occupations, abs=2e-11)
    assert result.energy_expectation == pytest.approx(reference.energy_expectation, abs=2e-11)
    assert result.exact_state_fidelity == pytest.approx(reference.exact_state_fidelity, abs=2e-11)
    assert result.metadata["code_constraint_expectations"] == pytest.approx(
        [1] * len(result.metadata["code_constraint_expectations"]), abs=2e-11
    )
    json.dumps(result.as_dict(), allow_nan=False)


@pytest.mark.parametrize("name", ["bksf_graph", "vc_square", "dk_square"])
def test_flow_refinement_converges_with_code_constraints_and_reports_particle_drift(name):
    """The flow split is a real approximation whose accuracy improves with more steps."""
    model = square_lattice(
        2,
        2,
        spinful=False,
        hopping_imag=0.3,
        density_interaction=0.4,
        site_potentials=[0.2, -0.3, 0.1, 0.4],
    )
    results = [
        simulate_lattice_dynamics(
            model,
            occupied_modes=[0],
            options=LatticeDynamicsOptions(
                mapping=name,
                encoding=_options(name),
                synthesis="flow_sets",
                time=0.6,
                order=2,
                steps=steps,
                exact_reference=True,
            ),
        )
        for steps in (1, 4)
    ]
    assert results[0].exact_state_fidelity is not None
    assert results[1].exact_state_fidelity is not None
    assert 1 - results[1].exact_state_fidelity < 1 - results[0].exact_state_fidelity
    for result in results:
        assert result.metadata["code_constraint_expectations"] == pytest.approx(
            [1] * len(result.metadata["code_constraint_expectations"]), abs=2e-11
        )
        assert "particle_number_drift" in result.metadata
        assert result.metadata["flow_particle_number_exact"] is False


def test_lattice_local_preflight_and_actual_encoded_memory_budget():
    """Auxiliary qubits count in every circuit, statevector and exact-reference budget."""
    model = square_lattice(2, 2, spinful=False)
    with pytest.raises(ConfigError, match="max_qubits"):
        build_lattice_dynamics(
            model,
            occupied_modes=[0],
            options=LatticeDynamicsOptions(
                mapping="vc_square", encoding=_options("vc_square"), max_qubits=4
            ),
        )
    for values, message in [
        ({"max_statevector_bytes": 1024}, "max_statevector_bytes"),
        ({"exact_reference": True, "max_exact_qubits": 4}, "max_exact_qubits"),
        ({"max_evolution_blocks": 1}, "max_evolution_blocks"),
    ]:
        with pytest.raises(ConfigError, match=message):
            simulate_lattice_dynamics(
                model,
                occupied_modes=[0],
                options=LatticeDynamicsOptions(
                    mapping="vc_square", encoding=_options("vc_square"), **values
                ),
            )
    for values in [
        {"mapping": "vc_square"},
        {"synthesis": "flow_sets"},
        {"encoding": _options("bksf_graph")},
        {"mapping": "dk_square", "encoding": _options("vc_square")},
    ]:
        with pytest.raises(ValidationError):
            LatticeDynamicsOptions(**values)


@pytest.mark.parametrize("occupied", [[], [0]])
def test_edgeless_bksf_uses_explicit_frozen_scalar_padding(occupied):
    """A zero-dimensional physical sector uses a declared one-qubit gauge register."""
    from chemrefine.engines.qiskit.lattice import FermionicLatticeModel

    model = FermionicLatticeModel(num_sites=1, spinful=False, site_potentials=(0.4,))
    result = simulate_lattice_dynamics(
        model,
        occupied_modes=occupied,
        options=LatticeDynamicsOptions(mapping="bksf_graph", exact_reference=True),
    )
    assert result.particle_number == pytest.approx(len(occupied))
    assert result.energy_expectation == pytest.approx(0.4 * len(occupied))
    assert result.metadata["encoding_details"]["scalar_register_padding"] == 1
    assert result.exact_state_fidelity == pytest.approx(1)


@pytest.mark.parametrize("rows,columns", [(2, 3), (3, 2), (3, 3)])
@pytest.mark.parametrize("name", ["vc_square", "dk_square"])
def test_rectangular_checkerboards_have_complete_clifford_reference_sectors(rows, columns, name):
    """Boundary faces and odd checkerboard imbalance need no exponential gauge projector."""
    model = square_lattice(rows, columns, spinful=False)
    configuration = LocalEncodingOptions(name=name, rows=rows, columns=columns)
    encoding = build_local_encoding(
        model.num_modes,
        [(edge.source, edge.target) for edge in model.edges],
        options=configuration,
        occupied_modes=[0],
    )
    # Clifford stabilizer checks scale polynomially instead of allocating a statevector.
    clifford = Clifford(encoding.prepare_reference([0]))
    for index, vertex in enumerate(encoding.vertices):
        pauli = vertex.paulis[0].evolve(clifford, frame="h")
        assert not np.any(pauli.x)
        assert (-1j) ** int(pauli.phase) * vertex.coeffs[0] == pytest.approx(
            -1 if index == 0 else 1
        )


def test_custom_gauge_and_measurement_basis_validation():
    """Custom encodings expose gauge fixing and refuse silent computational-basis decoding."""
    from dataclasses import replace

    from chemrefine.engines.qiskit.encodings import independent_paulis, pauli_product

    encoding = validate_encoding([SparsePauliOp("IX")], {})
    with pytest.raises(ConfigError, match="basis change"):
        encoding.decode_occupations("00")
    with pytest.raises(ConfigError, match="unexplained reference"):
        replace(encoding, gauge_fixers=()).prepare_reference([])
    with pytest.raises(ConfigError, match="too many"):
        independent_paulis([SparsePauliOp("X"), SparsePauliOp("Z")], 1)
    for vertices, edges, options, message in [
        ([], {}, {}, "no modes"),
        ([pauli_product(0)], {}, {}, "at least one"),
        ([SparsePauliOp("Z")], {(0, 0): SparsePauliOp("X")}, {}, "ordered endpoints"),
        ([SparsePauliOp("Z")], {}, {"max_generators": 0}, "max_generators"),
        ([SparsePauliOp("I")], {}, {"component_parities": (2,)}, "one parity"),
        ([SparsePauliOp("I")], {}, {"component_parities": ()}, "one parity"),
    ]:
        with pytest.raises(ConfigError, match=message):
            validate_encoding(vertices, edges, **options)
    with pytest.raises(ConfigError, match="rows and columns"):
        build_local_encoding(
            4, EDGES, options=LocalEncodingOptions.model_construct(name="vc_square")
        )
    with pytest.raises(ConfigError, match="max_generators"):
        build_local_encoding(4, [], options=_options("dk_square", max_generators=4))
    with pytest.raises(ConfigError, match="max_qubits"):
        build_local_encoding(4, EDGES, options=_options("vc_square", max_qubits=4))
    nonfinite = SparsePauliOp("Z")
    nonfinite.coeffs[0] = np.inf
    with pytest.raises(ConfigError, match="finite"):
        commuting_evolution(nonfinite, 1)


def test_local_empty_flow_time_and_rectangle_guards():
    """Zero-time and omitted-bond circuits remain valid with honest width estimates."""
    from chemrefine.engines.qiskit.lattice import FermionicLatticeModel, lattice_encoding_qubits

    model = FermionicLatticeModel(num_sites=4, spinful=False)
    configuration = LatticeDynamicsOptions(
        mapping="vc_square", encoding=_options("vc_square"), synthesis="flow_sets", time=0
    )
    artifacts = build_lattice_dynamics(model, occupied_modes=[0], options=configuration)
    assert artifacts.metadata["evolution_blocks"] == 0
    assert lattice_encoding_qubits(model, configuration) == 8
    for name in ("vc_square", "dk_square"):
        with pytest.raises(ConfigError, match="match num_sites"):
            lattice_encoding_qubits(
                FermionicLatticeModel(num_sites=6),
                LatticeDynamicsOptions(mapping=name, encoding=_options(name)),
            )
    with pytest.raises(ConfigError, match="dimensions"):
        lattice_encoding_qubits(model, LatticeDynamicsOptions.model_construct(mapping="vc_square"))
    # A valid empty flow at nonzero time also performs only reference preparation.
    artifacts = build_lattice_dynamics(
        model, occupied_modes=[0], options=configuration.model_copy(update={"time": 1})
    )
    assert artifacts.metadata["evolution_blocks"] == 0

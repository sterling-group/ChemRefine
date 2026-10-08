"""Physical lattice dynamics, encoding equivalence and explicit simulation budgets."""

from __future__ import annotations

import json

import numpy as np
import pytest
from pydantic import ValidationError

from chemrefine.engines.qiskit.lattice import (
    FermionicLatticeModel,
    LatticeDynamicsOptions,
    LatticeEdge,
    build_lattice_dynamics,
    chain_lattice,
    simulate_lattice_dynamics,
    square_lattice,
)
from chemrefine.errors import ConfigError


def _require_stack():
    """Only numerical tests require the optional simulation stack."""
    pytest.importorskip("qiskit_fermions")
    pytest.importorskip("qiskit_nature")


@pytest.mark.parametrize("mapping", ["jordan_wigner", "bravyi_kitaev", "parity"])
@pytest.mark.parametrize("order", [1, 2, 4])
def test_single_particle_hopping_matches_analytic_solution(mapping, order):
    """The same physical electron oscillates correctly in each binary encoding."""
    _require_stack()
    result = simulate_lattice_dynamics(
        chain_lattice(2, spinful=False),
        occupied_modes=[0],
        options=LatticeDynamicsOptions(
            time=0.37, mapping=mapping, order=order, exact_reference=True
        ),
    )
    assert result.mode_occupations == pytest.approx(
        (np.cos(0.37) ** 2, np.sin(0.37) ** 2), abs=1e-12
    )
    assert result.particle_number == pytest.approx(1, abs=1e-12)
    assert result.exact_state_fidelity == pytest.approx(1, abs=1e-12)
    assert result.return_probability == pytest.approx(np.cos(0.37) ** 2, abs=1e-12)
    assert result.energy_drift == pytest.approx(0, abs=1e-12)
    assert not result.statevector.flags.writeable


@pytest.mark.parametrize("mapping", ["jordan_wigner", "bravyi_kitaev", "parity"])
def test_spinful_hubbard_maps_have_same_physical_dynamics(mapping):
    """Double occupancy, extended interactions and potentials retain their signs."""
    _require_stack()
    model = chain_lattice(
        2, onsite_interaction=2.1, density_interaction=0.3, site_potentials=[0.2, -0.1]
    )
    options = LatticeDynamicsOptions(time=0.4, steps=4, order=4, exact_reference=True)
    reference = simulate_lattice_dynamics(model, occupied_modes=[0, 2], options=options)
    result = simulate_lattice_dynamics(
        model, occupied_modes=[0, 2], options=options.model_copy(update={"mapping": mapping})
    )
    assert result.mode_occupations == pytest.approx(reference.mode_occupations, abs=1e-11)
    assert sum(result.mode_occupations[:2]) == pytest.approx(1, abs=1e-11)
    assert sum(result.mode_occupations[2:]) == pytest.approx(1, abs=1e-11)
    assert result.metadata["initial_energy"] == pytest.approx(2.5)
    assert result.exact_state_fidelity is not None
    assert result.exact_state_fidelity > 1 - 1e-8
    payload = result.as_dict(include_statevector=True)
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    payload["metadata"]["changed"] = True
    assert "changed" not in result.metadata
    assert "statevector" not in result.as_dict()


def test_suzuki_orders_and_step_refinement_reduce_real_trotter_error():
    """Accuracy improves for noncommuting blocks, rather than an exactly soluble split."""
    _require_stack()
    model = chain_lattice(3, spinful=False, site_potentials=[0.5, -0.3, 0.8])
    results = [
        simulate_lattice_dynamics(
            model,
            occupied_modes=[0],
            options=LatticeDynamicsOptions(
                time=0.7, order=order, steps=steps, exact_reference=True
            ),
        )
        for order, steps in [(1, 1), (2, 1), (4, 1), (2, 8)]
    ]
    errors = []
    for result in results:
        assert result.exact_state_fidelity is not None
        errors.append(1 - result.exact_state_fidelity)
    assert errors[0] > errors[1] > errors[2] > 0
    assert errors[3] < errors[1]


@pytest.mark.parametrize("time", [0.0, -0.4])
def test_empty_graph_vacuum_and_negative_time(time):
    """An empty Hamiltonian preserves any reference, including zero particles."""
    _require_stack()
    result = simulate_lattice_dynamics(
        FermionicLatticeModel(num_sites=1, spinful=False),
        occupied_modes=[],
        options=LatticeDynamicsOptions(time=time, exact_reference=True),
    )
    assert result.particle_number == 0
    assert result.energy_expectation == 0
    assert result.exact_state_fidelity == pytest.approx(1)
    assert result.metadata["evolution_blocks"] == 0


def test_density_interactions_and_spin_potentials_count_each_term_once():
    """A stationary filled determinant has independently known diagonal energy."""
    _require_stack()
    model = chain_lattice(
        2, hopping=0, onsite_interaction=1.2, density_interaction=0.3, site_potentials=[0.2, -0.1]
    )
    result = simulate_lattice_dynamics(model, occupied_modes=[0, 1, 2, 3])
    assert result.energy_expectation == pytest.approx(2 * 1.2 + 4 * 0.3 + 2 * 0.2 - 2 * 0.1)
    assert result.mode_occupations == pytest.approx([1] * 4)
    assert result.exact_state_fidelity is None


def test_graph_helpers_have_no_self_edges_or_duplicate_periodic_bonds():
    """Degenerate periodic dimensions are simple graphs, never doubled hoppings."""
    assert len(chain_lattice(4, periodic=True).edges) == 4
    assert len(chain_lattice(2, periodic=True).edges) == 1
    assert len(square_lattice(2, 2, periodic=True).edges) == 4
    assert len(square_lattice(3, 3, periodic=True).edges) == 18
    assert len(square_lattice(1, 1, periodic=True).edges) == 0
    assert len(square_lattice(2, 3).edges) == 7
    assert square_lattice(1, 2, site_potentials=[0.2, 0.3]).site_potentials == (0.2, 0.3)


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_graph_helpers_reject_invalid_sizes(size):
    """Integer graph dimensions are required before allocating edges."""
    with pytest.raises(ConfigError):
        chain_lattice(size)
    with pytest.raises(ConfigError):
        square_lattice(size, 2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"edges": [{"source": 0, "target": 0}]},
        {"edges": [{"source": 0, "target": 2}]},
        {"edges": [{"source": 0, "target": 1}, {"source": 1, "target": 0}]},
        {"spinful": False, "onsite_interaction": 1},
        {"site_potentials": [0.1]},
        {"onsite_interaction": float("nan")},
    ],
)
def test_lattice_rejects_ambiguous_models(kwargs):
    """Model validation catches malformed graphs and undefined interactions."""
    with pytest.raises(ValidationError):
        FermionicLatticeModel(num_sites=2, **kwargs)


@pytest.mark.parametrize("occupied", [[True], [0.2], [-1], [2], [0, 0]])
def test_lattice_rejects_ambiguous_occupations(occupied):
    """A determinant lists distinct integer modes, including an explicit empty list."""
    with pytest.raises(ConfigError, match="occupied_modes"):
        build_lattice_dynamics(chain_lattice(2, spinful=False), occupied_modes=occupied)


@pytest.mark.parametrize(
    "settings,message",
    [
        ({"max_qubits": 1}, "max_qubits"),
        ({"max_statevector_bytes": 1}, "max_statevector_bytes"),
        ({"exact_reference": True, "max_exact_qubits": 1}, "max_exact_qubits"),
        ({"max_evolution_blocks": 1}, "max_evolution_blocks"),
    ],
)
def test_simulation_limits_are_checked_before_work(settings, message):
    """Each independent resource budget produces an actionable error."""
    _require_stack()
    with pytest.raises(ConfigError, match=message):
        simulate_lattice_dynamics(
            chain_lattice(2), occupied_modes=[0], options=LatticeDynamicsOptions(**settings)
        )


def test_build_qubit_limit_and_unknown_mapping():
    """Circuit construction has a bound independently of statevector execution."""
    with pytest.raises(ConfigError, match="max_qubits"):
        build_lattice_dynamics(
            chain_lattice(3), occupied_modes=[0], options=LatticeDynamicsOptions(max_qubits=2)
        )
    with pytest.raises(ValidationError):
        LatticeDynamicsOptions(mapping="flow_set")
    with pytest.raises(ValidationError):
        LatticeEdge(source=True, target=1)


@pytest.mark.parametrize("failure", ["block", "determinant"])
def test_mapping_contract_rejects_noncommuting_blocks_and_superposed_references(
    monkeypatch, failure
):
    """A changed provider may not silently invalidate exact per-block synthesis."""
    _require_stack()
    from qiskit.quantum_info import SparseObservable, SparsePauliOp

    from chemrefine.engines.qiskit import lattice

    mapper = lattice._mapper("jordan_wigner")

    def corrupted(operator, num_qubits):
        """Inject an unsupported Pauli image at one precisely selected boundary."""
        terms = list(operator.iter_terms())
        if failure == "block" or len(terms[0][0]) == 1:
            return SparseObservable.from_sparse_pauli_op(
                SparsePauliOp.from_list([("XI", 1), ("ZI", 1)])
            )
        return mapper(operator, num_qubits)

    monkeypatch.setattr(lattice, "_mapper", lambda _: corrupted)
    with pytest.raises(ConfigError, match=r"commuting|one bitstring"):
        build_lattice_dynamics(chain_lattice(2, spinful=False), occupied_modes=[0])

"""Physical lattice dynamics, encoding equivalence and explicit simulation budgets."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from chemrefine.engines.qiskit.lattice import (
    FermionicLatticeModel,
    chain_lattice,
    square_lattice,
)
from chemrefine.errors import ConfigError


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

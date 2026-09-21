"""Exact reference calculations must preserve both alpha and beta populations."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from chemrefine.engines.qiskit.components.algorithms import build_exact
from chemrefine.engines.qiskit.context import ElectronicStructureContext, SolverComponents
from chemrefine.engines.qiskit.registry import NoComponentOptions


@pytest.mark.parametrize("particles", [(1, 0), (0, 1), (1, 1)])
def test_exact_criterion_preserves_requested_magnetization(monkeypatch, particles):
    """N and S-squared alone do not distinguish the configured spin populations."""
    fake = ModuleType("qiskit_algorithms")
    fake.NumPyMinimumEigensolver = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "qiskit_algorithms", fake)
    multiplicity = abs(particles[0] - particles[1]) + 1
    context = ElectronicStructureContext(None, None, None, 2, particles, 4, multiplicity)
    solver = build_exact(
        options=NoComponentOptions(), context=context, components=SolverComponents()
    ).solver
    magnetization = (particles[0] - particles[1]) / 2
    spin = (multiplicity - 1) / 2
    observables: dict[str, tuple[float, dict[str, float]]] = {
        "ParticleNumber": (sum(particles), {}),
        "AngularMomentum": (spin * (spin + 1), {}),
        "Magnetization": (magnetization, {}),
    }
    assert solver.filter_criterion(None, 0, observables)
    observables["Magnetization"] = (magnetization + 1, {})
    assert not solver.filter_criterion(None, 0, observables)
    del observables["Magnetization"]
    assert solver.filter_criterion(None, 0, observables)


@pytest.mark.parametrize(("alpha", "beta", "expected_energy"), [(1, 0, 1.0), (0, 1, -2.0)])
def test_exact_rejects_lower_energy_in_the_wrong_spin_population(alpha, beta, expected_energy):
    """A tiny spin-dependent Hamiltonian makes an incorrect population observable in energy."""
    pytest.importorskip("qiskit_nature")
    pytest.importorskip("qiskit_algorithms")
    from chemrefine.engines.qiskit.api import ElectronicStructureData, prepare_problem, solve_exact

    zero = np.zeros((1, 1, 1, 1))
    data = ElectronicStructureData(
        num_alpha=alpha,
        num_beta=beta,
        num_spatial_orbitals=1,
        one_body_integrals=[[1.0]],
        one_body_integrals_beta=[[-2.0]],
        two_body_integrals=zero,
        two_body_integrals_beta_beta=zero,
        two_body_integrals_beta_alpha=zero,
        overlap_alpha_beta=[[1.0]],
        nuclear_repulsion_energy=0.0,
    )
    result = solve_exact(prepare_problem(data), options={"cores": 1})
    assert result.energy_hartree == pytest.approx(expected_energy, abs=1e-12)
    assert result.num_particles == (alpha, beta)

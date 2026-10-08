"""User-directed orbital reduction with explicit original-orbital provenance."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from chemrefine.engines.qiskit.options import ActiveSpaceOptions
from chemrefine.errors import ConfigError

logger = logging.getLogger(__name__)


def _reduce(problem: Any, indices: list[int], particles: tuple[int, int], label: str) -> Any:
    """Transform selected orbitals while retaining every previous energy offset."""
    from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

    if not indices or len(set(indices)) != len(indices):
        raise ConfigError("qiskit active space must contain distinct orbitals and cannot be empty")
    if any(index < 0 or index >= problem.num_spatial_orbitals for index in indices):
        raise ConfigError("qiskit active orbital index exceeds the available orbital count")
    occupations = (problem.orbital_occupations, problem.orbital_occupations_b)
    actual = tuple(round(float(np.asarray(occ)[indices].sum())) for occ in occupations)
    if actual != particles:
        raise ConfigError(
            f"qiskit selected orbital occupations contain {actual} electrons; requested {particles}"
        )
    inactive = [i for i in range(problem.num_spatial_orbitals) if i not in indices]
    alpha, beta = (np.asarray(occ)[inactive] for occ in occupations)
    if not np.array_equal(alpha, beta):
        raise ConfigError("qiskit inactive occupied orbitals must be doubly occupied")
    old_constants = dict(problem.hamiltonian.constants)
    reduced: Any = ActiveSpaceTransformer(
        particles, len(indices), active_orbitals=indices
    ).transform(problem)
    offset = reduced.hamiltonian.constants.pop("ActiveSpaceTransformer")
    reduced.hamiltonian.constants.update(old_constants)
    offset_key = label
    counter = 2
    while offset_key in old_constants:
        offset_key = f"{label}_{counter}"
        counter += 1
    reduced.hamiltonian.constants[offset_key] = offset
    if tuple(reduced.num_particles) != particles:
        raise ConfigError("qiskit active-space transformer returned inconsistent electron counts")
    return reduced


def apply_active_space(
    problem: Any,
    *,
    active_space: ActiveSpaceOptions | None = None,
    freeze_core: bool = False,
) -> tuple[Any, list[int], list[dict[str, Any]]]:
    """Apply freeze-core then supplied selection; explicit indices use the original MO basis.

    The core count follows Qiskit Nature's element-based convention. We apply
    its ActiveSpaceTransformer using the input occupations, avoiding implicit
    reconstruction of open-shell occupations by FreezeCoreTransformer.
    """
    original_indices = list(range(problem.num_spatial_orbitals))
    transformations: list[dict[str, Any]] = []
    if freeze_core:
        from qiskit_nature.second_q.transformers import FreezeCoreTransformer

        molecule = problem.molecule
        if molecule is None:
            raise ConfigError("qiskit freeze_core requires molecular symbols and charge metadata")
        counter = FreezeCoreTransformer()
        expected = sum(counter.Z(symbol) for symbol in molecule.symbols) - molecule.charge
        if expected != sum(problem.num_particles):
            raise ConfigError(
                "qiskit freeze_core requires all-electron molecular data; charge/electrons disagree"
            )
        core = counter.count_core_orbitals(molecule.symbols)
        if core >= problem.num_spatial_orbitals or core > min(problem.num_particles):
            raise ConfigError("qiskit freeze_core would remove unavailable orbitals or electrons")
        indices = original_indices[core:]
        frozen_particles = tuple(count - core for count in problem.num_particles)
        if core:
            problem = _reduce(
                problem,
                indices,
                (frozen_particles[0], frozen_particles[1]),
                "FreezeCoreTransformer",
            )
        original_indices = indices
        transformations.append({"kind": "freeze_core", "frozen_orbitals": list(range(core))})
    if active_space is not None:
        particles = active_space.electrons
        if isinstance(particles, int):
            if particles % 2:
                raise ConfigError("qiskit odd active electron counts require an [alpha, beta] pair")
            requested = (particles // 2, particles // 2)
        else:
            requested = particles
        if active_space.active_orbitals is None:
            inactive_electrons = sum(problem.num_particles) - sum(requested)
            if inactive_electrons < 0 or inactive_electrons % 2:
                raise ConfigError(
                    "qiskit active space must leave a non-negative even inactive count"
                )
            start = inactive_electrons // 2
            indices = list(range(start, start + active_space.orbitals))
        else:
            positions = {original: index for index, original in enumerate(original_indices)}
            if any(index not in positions for index in active_space.active_orbitals):
                raise ConfigError(
                    "qiskit active orbital indices are unavailable or frozen core orbitals"
                )
            indices = [positions[index] for index in active_space.active_orbitals]
        problem = _reduce(problem, indices, requested, "ActiveSpaceTransformer")
        original_indices = [original_indices[index] for index in indices]
        transformations.append(
            {
                "kind": "active_space",
                "active_orbitals": original_indices,
                "num_particles": list(requested),
            }
        )
    if transformations:
        logger.info(
            "Qiskit active space applied: orbitals=%s particles=%s",
            original_indices,
            problem.num_particles,
        )
    return problem, original_indices, transformations

"""Collapse structures that converged onto the same geometry during a step.

The orchestrator calls :func:`apply` once per step, on the engine's raw parsed
structures, **before** :func:`chemrefine.filtering.apply` ranks and slices them by
energy. Two structures that started as distinct conformers (e.g. from GOAT's step-1
search) can relax onto the same minimum in a later optimization step; nothing else in
the pipeline notices, since :mod:`chemrefine.filtering` only ever reasons about energy.

Structures are grouped by exact atomic-number sequence (order matters — this is also
what keeps unrelated systems from ever being compared), then within a group pruned
greedily by Kabsch-aligned RMSD: examined lowest-energy first, a structure is dropped
the moment its aligned RMSD to any structure already kept falls at or below the
configured threshold.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
from numpy.typing import NDArray

from chemrefine.config import DedupeConfig
from chemrefine.quantities import HARTREE_TO_KCALMOL
from chemrefine.state import Structure

logger = logging.getLogger(__name__)

#: Cheap reject before the O(n^2) Kabsch alignment: two structures at the same
#: geometry are the same electronic-structure calculation, so their energies agree to
#: near machine precision. A gap this large (kcal/mol) cannot be a duplicate, so RMSD
#: is skipped — never the reverse (a small gap always still gets the real RMSD check).
_ENERGY_PREFILTER_KCALMOL = 5.0


def apply(
    structures: tuple[Structure, ...], dedupe: DedupeConfig | None
) -> tuple[Structure, ...]:
    """Return ``structures`` with structural duplicates collapsed under ``dedupe``.

    ``dedupe is None`` is the identity — every structure passes through untouched,
    matching :func:`chemrefine.filtering.apply`'s ``sample is None`` convention.
    """
    if dedupe is None or len(structures) <= 1:
        return structures
    if dedupe.by_parent:
        return tuple(_dedupe_by_parent(structures, dedupe))
    return tuple(_dedupe_group(list(structures), dedupe))


def _dedupe_by_parent(
    structures: tuple[Structure, ...], dedupe: DedupeConfig
) -> list[Structure]:
    """Group by parent ID, dedupe each group, return concatenated survivors."""
    groups: dict[str, list[Structure]] = defaultdict(list)
    for struct in structures:
        # Seeds (parent_id=None) form singleton groups by falling back to their own
        # id — the same convention as chemrefine.filtering._filter_by_parent.
        groups[struct.parent_id or struct.id].append(struct)
    survivors: list[Structure] = []
    for parent, group in groups.items():
        kept = _dedupe_group(group, dedupe)
        logger.debug("parent %s: %d structures -> %d after dedupe", parent, len(group), len(kept))
        survivors.extend(kept)
    return survivors


def _dedupe_group(structures: list[Structure], dedupe: DedupeConfig) -> list[Structure]:
    """Bucket by atomic-number sequence, then prune each bucket independently."""
    buckets: dict[tuple[int, ...], list[Structure]] = defaultdict(list)
    for struct in structures:
        buckets[tuple(int(z) for z in struct.atoms.get_atomic_numbers())].append(struct)
    survivors: list[Structure] = []
    for bucket in buckets.values():
        survivors.extend(_prune_bucket(bucket, dedupe))
    return survivors


def _prune_bucket(bucket: list[Structure], dedupe: DedupeConfig) -> list[Structure]:
    """Greedily keep one representative per distinct geometry, lowest-energy first."""
    ranked = sorted(bucket, key=_energy_sort_key)
    kept: list[Structure] = []
    kept_energies: list[float | None] = []
    kept_coords: list[NDArray[np.float64]] = []
    for struct in ranked:
        coords = _coords(struct, dedupe.include_hydrogens)
        energy = struct.energy_hartree
        is_duplicate = False
        for other_energy, other_coords in zip(kept_energies, kept_coords, strict=True):
            if _energy_gap_kcalmol(energy, other_energy) > _ENERGY_PREFILTER_KCALMOL:
                continue
            if _kabsch_rmsd(coords, other_coords) <= dedupe.rmsd_angstrom:
                is_duplicate = True
                break
        if is_duplicate:
            logger.debug(
                "dropping structure %s as a structural duplicate (RMSD <= %.3f A)",
                struct.id,
                dedupe.rmsd_angstrom,
            )
            continue
        kept.append(struct)
        kept_energies.append(energy)
        kept_coords.append(coords)
    return kept


def _energy_sort_key(struct: Structure) -> float:
    """Missing-energy structures (e.g. ``on_failure: best`` backfills) sort last."""
    return struct.energy_hartree if struct.energy_hartree is not None else float("inf")


def _energy_gap_kcalmol(a: float | None, b: float | None) -> float:
    """The energy-prefilter gap; ``0.0`` (never reject) when either energy is missing."""
    if a is None or b is None:
        return 0.0
    return abs(a - b) * HARTREE_TO_KCALMOL


def _coords(struct: Structure, include_hydrogens: bool) -> NDArray[np.float64]:
    """This structure's coordinates, masked to heavy atoms when configured.

    Safe to mask by position alone: structures reaching here share one bucket, hence
    one atomic-number sequence, so index ``i`` names the same atom in every structure
    being compared.
    """
    atoms = struct.atoms
    positions = atoms.get_positions()
    if include_hydrogens:
        return positions
    heavy = atoms.get_atomic_numbers() != 1
    return positions[heavy]


def _kabsch_rmsd(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """RMSD between ``a`` and ``b`` after the optimal (Kabsch) rigid-body alignment."""
    a_centered = a - a.mean(axis=0)
    b_centered = b - b.mean(axis=0)
    covariance = a_centered.T @ b_centered
    u, _, vt = np.linalg.svd(covariance)
    sign = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    correction = np.diag([1.0, 1.0, sign])
    rotation = vt.T @ correction @ u.T
    aligned = a_centered @ rotation.T
    diff = aligned - b_centered
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))

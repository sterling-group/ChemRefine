"""Tests for the structural-duplicate collapse."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from chemrefine.config import DedupeConfig
from chemrefine.dedupe import apply
from chemrefine.state import Structure

_WATER = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.9584, 0.0, 0.0],
        [-0.2396, 0.9273, 0.0],
    ]
)


def _water(id_: str, *, positions=None, energy=-1.0, parent_id=None) -> Structure:
    atoms = Atoms("OHH", positions=_WATER if positions is None else positions)
    return Structure(id=id_, atoms=atoms, parent_id=parent_id, energy_hartree=energy)


def _rotate_translate(positions: np.ndarray) -> np.ndarray:
    """A rigid-body transform that leaves RMSD-after-alignment at (numerically) zero."""
    theta = 0.7
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return positions @ rotation.T + np.array([5.0, -3.0, 2.0])


def _distort(positions: np.ndarray, atom_index: int, delta: np.ndarray) -> np.ndarray:
    """Move one atom relative to the others — a *non*-rigid-body change, unlike
    :func:`_rotate_translate`, so Kabsch alignment cannot hide it."""
    out = positions.copy()
    out[atom_index] += delta
    return out


# ---------------------------------------------------------------------------
# Identity (dedupe=None)
# ---------------------------------------------------------------------------


def test_apply_with_none_dedupe_keeps_everything():
    structures = (_water("0"), _water("1"))
    assert apply(structures, None) == structures


def test_apply_with_single_structure_is_a_noop():
    structures = (_water("0"),)
    assert apply(structures, DedupeConfig()) == structures


# ---------------------------------------------------------------------------
# Core RMSD collapse
# ---------------------------------------------------------------------------


def test_identical_geometry_collapses_to_one():
    # A tiny energy gap, as two independent geometry-converged optimizations of the
    # same minimum would actually show — not the ~600 kcal/mol the prefilter guards.
    structures = (_water("0", energy=-1.0), _water("1", energy=-1.00005))
    survivors = apply(structures, DedupeConfig())
    assert [s.id for s in survivors] == ["1"]  # lower energy kept


def test_rotated_and_translated_duplicate_still_collapses():
    """RMSD is computed after Kabsch alignment, so a rigid-body transform is invisible."""
    structures = (
        _water("0", energy=-1.0),
        _water("1", positions=_rotate_translate(_WATER), energy=-1.00005),
    )
    survivors = apply(structures, DedupeConfig())
    assert [s.id for s in survivors] == ["1"]


def test_structures_beyond_threshold_both_survive():
    # A non-rigid distortion (only the oxygen moves) — a uniform translation would be
    # invisible to Kabsch-aligned RMSD, which is the point of the alignment.
    distorted = _distort(_WATER, 0, np.array([1.0, 0.0, 0.0]))
    structures = (_water("0"), _water("1", positions=distorted))
    survivors = apply(structures, DedupeConfig(rmsd_angstrom=0.125))
    assert {s.id for s in survivors} == {"0", "1"}


def test_threshold_is_configurable():
    # A small non-rigid perturbation: within a loose threshold, outside a tight one.
    perturbed = _distort(_WATER, 0, np.array([0.02, 0.0, 0.0]))
    structures = (_water("0"), _water("1", positions=perturbed))
    assert len(apply(structures, DedupeConfig(rmsd_angstrom=0.5))) == 1
    assert len(apply(structures, DedupeConfig(rmsd_angstrom=0.001))) == 2


# ---------------------------------------------------------------------------
# Bucketing by atomic-number sequence
# ---------------------------------------------------------------------------


def test_different_systems_never_compared():
    """Different atomic-number sequences never collapse, whatever their RMSD would be."""
    water = _water("0")
    other = Structure(id="1", atoms=Atoms("HHO", positions=_WATER), energy_hartree=-1.0)
    survivors = apply((water, other), DedupeConfig(rmsd_angstrom=999.0))
    assert {s.id for s in survivors} == {"0", "1"}


# ---------------------------------------------------------------------------
# include_hydrogens
# ---------------------------------------------------------------------------


def test_include_hydrogens_false_ignores_hydrogen_only_displacement():
    moved_h = _WATER.copy()
    moved_h[1] += np.array([0.3, 0.0, 0.0])  # move only a hydrogen
    structures = (_water("0"), _water("1", positions=moved_h))
    heavy_only = apply(structures, DedupeConfig(rmsd_angstrom=0.125, include_hydrogens=False))
    all_atom = apply(structures, DedupeConfig(rmsd_angstrom=0.125, include_hydrogens=True))
    assert len(heavy_only) == 1
    assert len(all_atom) == 2


# ---------------------------------------------------------------------------
# by_parent
# ---------------------------------------------------------------------------


def test_by_parent_false_collapses_across_parents():
    structures = (
        _water("0", energy=-1.0, parent_id="a"),
        _water("1", energy=-1.00005, parent_id="b"),
    )
    survivors = apply(structures, DedupeConfig(by_parent=False))
    assert [s.id for s in survivors] == ["1"]


def test_by_parent_true_keeps_duplicates_from_different_parents():
    structures = (
        _water("0", energy=-1.0, parent_id="a"),
        _water("1", energy=-1.00005, parent_id="b"),
    )
    survivors = apply(structures, DedupeConfig(by_parent=True))
    assert {s.id for s in survivors} == {"0", "1"}


def test_by_parent_true_still_collapses_within_one_parent():
    structures = (
        _water("0", energy=-1.0, parent_id="a"),
        _water("1", energy=-1.00005, parent_id="a"),
    )
    survivors = apply(structures, DedupeConfig(by_parent=True))
    assert [s.id for s in survivors] == ["1"]


# ---------------------------------------------------------------------------
# Energy pre-filter
# ---------------------------------------------------------------------------


def test_energy_gap_beyond_prefilter_skips_rmsd_even_for_identical_geometry():
    """A large energy gap is a cheap reject: two structures this far apart in energy
    cannot be the same electronic-structure calculation, so RMSD is never computed —
    even if their (unphysically) identical geometry would otherwise collapse."""
    structures = (_water("0", energy=-1.0), _water("1", energy=-1.0 - 1.0))  # ~627 kcal/mol apart
    survivors = apply(structures, DedupeConfig())
    assert {s.id for s in survivors} == {"0", "1"}


def test_missing_energy_never_skips_rmsd():
    """A structure with no energy (e.g. an ``on_failure: best`` backfill) still gets the
    real RMSD check rather than being waved through by the pre-filter."""
    structures = (_water("0", energy=None), _water("1", energy=None))
    survivors = apply(structures, DedupeConfig())
    assert len(survivors) == 1

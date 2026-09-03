"""Tests for the frozen state dataclasses (`Structure`, `PipelineState`, ...)."""

from __future__ import annotations

from ase import Atoms

from chemrefine.state import PipelineState, Structure


def test_pipeline_state_len_matches_structures_length():
    s = PipelineState(
        structures=(
            Structure(id="0", atoms=Atoms("H")),
            Structure(id="1", atoms=Atoms("H")),
        )
    )
    assert len(s) == 2


def test_pipeline_state_empty_len_is_zero():
    assert len(PipelineState()) == 0


def test_pipeline_state_bool_reflects_emptiness():
    assert not PipelineState()
    assert PipelineState(structures=(Structure(id="0", atoms=Atoms("H")),))


def test_pipeline_state_indexes_its_structures_once():
    """``by_id`` is the one index every per-job reader shares, built on first use and kept.

    Rebuilt per reader it made a step's parse quadratic in its parent count; cached on the
    frozen state it cannot go stale, because the structures it indexes never change.
    """
    a = Structure(id="a", atoms=Atoms("H"))
    b = Structure(id="b", atoms=Atoms("H"))
    state = PipelineState(structures=(a, b))

    index = state.by_id

    assert index == {"a": a, "b": b}
    assert state.by_id is index
    assert PipelineState().by_id == {}


def test_structure_parent_id_defaults_to_none():
    """Seed structures don't pass parent_id; the default is None."""
    seed = Structure(id="0", atoms=Atoms("H"))
    assert seed.parent_id is None


def test_structure_parent_id_carries_through_when_set():
    """Derived structures carry their input's lineage explicitly."""
    child = Structure(id="0-1", atoms=Atoms("H"), parent_id="0")
    assert child.parent_id == "0"

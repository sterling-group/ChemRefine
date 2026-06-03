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


def test_structure_parent_id_defaults_to_none():
    """Seed structures don't pass parent_id; the default is None."""
    seed = Structure(id="0", atoms=Atoms("H"))
    assert seed.parent_id is None


def test_structure_parent_id_carries_through_when_set():
    """Derived structures carry their input's lineage explicitly."""
    child = Structure(id="0-1", atoms=Atoms("H"), parent_id="0")
    assert child.parent_id == "0"

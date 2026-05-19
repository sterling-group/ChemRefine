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

"""Tests for the frozen state dataclasses (`Structure`, `PipelineState`, ...)."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms

from chemrefine.state import PipelineState, StepInputs, Structure


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


def test_step_inputs_accessors_return_aligned_tuples():
    """``input_paths`` / ``output_paths`` / ``structure_ids`` split the
    ordered (inp, out, sid) triples into their three component tuples."""
    files = (
        (Path("a.inp"), Path("a.out"), "0"),
        (Path("b.inp"), Path("b.out"), "1"),
    )
    inputs = StepInputs(files=files)
    assert inputs.input_paths == (Path("a.inp"), Path("b.inp"))
    assert inputs.output_paths == (Path("a.out"), Path("b.out"))
    assert inputs.structure_ids == ("0", "1")

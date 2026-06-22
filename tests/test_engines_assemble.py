"""Tests for the engine-independent structure assembler.

:func:`chemrefine.engines._job.build_structures` is the single home for turning each
engine's per-input ``ParsedResult``s into lineage-correct ``Structure``s — minting child
IDs and threading parents through a step's fan-out. The engines (ORCA, script) only produce
``ParsedResult``s; this exercises the shared assembly once.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from chemrefine.engines._job import build_structures
from chemrefine.engines.api import ParsedResult
from chemrefine.state import PipelineState, Structure


def _prev(*structures: Structure) -> PipelineState:
    return PipelineState(structures=structures)


def _result(symbol: str = "H", energy: float = -1.0, **kw) -> ParsedResult:
    return ParsedResult(
        symbols=(symbol,),
        positions=np.zeros((1, 3)),
        energy_hartree=energy,
        forces_ev_per_a=None,
        **kw,
    )


def test_one_to_one_keeps_id_and_inherits_parent():
    """A single parsed result per input keeps the input's id and its parent lineage."""
    prev = _prev(Structure(id="0", atoms=Atoms("H"), parent_id="root"))
    results = build_structures([("0", [_result(energy=-2.0)])], prev)
    assert len(results.structures) == 1
    child = results.structures[0]
    assert child.id == "0"
    assert child.parent_id == "root"  # inherits the input's parent on a 1:1 step
    assert child.energy_hartree == -2.0


def test_fan_out_mints_child_ids_parented_to_the_input():
    """An ensemble fan-out gets ``{input}-{i}`` ids whose parent is the input itself."""
    prev = _prev(Structure(id="0", atoms=Atoms("H"), parent_id="root"))
    results = build_structures([("0", [_result(), _result(), _result()])], prev)
    assert [s.id for s in results.structures] == ["0-0", "0-1", "0-2"]
    assert {s.parent_id for s in results.structures} == {"0"}


def test_multiple_inputs_assemble_independently():
    """Each input's fan-out is numbered under its own id; ids stay unique."""
    prev = _prev(
        Structure(id="0", atoms=Atoms("H")),
        Structure(id="1", atoms=Atoms("H")),
    )
    results = build_structures([("0", [_result(), _result()]), ("1", [_result()])], prev)
    assert [s.id for s in results.structures] == ["0-0", "0-1", "1"]


def test_thermochemistry_and_status_flags_pass_through():
    """Gibbs/enthalpy/ZPE + converged/terminated flags survive assembly verbatim."""
    prev = _prev(Structure(id="0", atoms=Atoms("H")))
    parsed = _result(
        converged=True,
        terminated=True,
        gibbs_hartree=-1.5,
        enthalpy_hartree=-1.4,
        energy_zpe_hartree=-1.3,
    )
    child = build_structures([("0", [parsed])], prev).structures[0]
    assert child.converged is True
    assert child.terminated is True
    assert (child.gibbs_hartree, child.enthalpy_hartree, child.energy_zpe_hartree) == (
        -1.5,
        -1.4,
        -1.3,
    )


def test_input_with_no_parsed_results_contributes_nothing():
    """A parsed-but-empty input (e.g. all frames dropped) yields no structures."""
    prev = _prev(Structure(id="0", atoms=Atoms("H")))
    assert build_structures([("0", [])], prev).structures == ()


def test_unknown_input_id_yields_none_parent():
    """If an input id isn't in the previous state, a 1:1 child gets no parent."""
    prev = _prev(Structure(id="0", atoms=Atoms("H")))
    child = build_structures([("missing", [_result()])], prev).structures[0]
    assert child.id == "missing"
    assert child.parent_id is None

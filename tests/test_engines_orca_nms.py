"""Tests for ``chemrefine.engines.orca.nms`` (pure target-aware displacement)
and the engine's two-round orchestration + step resolution."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms
from synthetic import (
    FREQUENCY_BLOCK,
    NORMAL_MODES_BLOCK_2_ATOMS,
    synthetic_dft_output,
)

from chemrefine.config import StepConfig
from chemrefine.engines.base import get_engine
from chemrefine.engines.orca import nms
from chemrefine.ids import structure_artifact_path
from chemrefine.state import (
    JobBatch,
    PipelineState,
    StepContext,
    StepInputs,
    StepResults,
    Structure,
)


def _struct(positions, sid: str = "0") -> Structure:
    return Structure(id=sid, atoms=Atoms("H2", positions=positions))


def _modes(n_modes: int) -> np.ndarray:
    """(2 atoms, 3, n_modes): mode k moves atom0 +0.1·(k+1) x, atom1 -0.1·(k+1) x."""
    t = np.zeros((2, 3, n_modes))
    for k in range(n_modes):
        t[0, 0, k] = 0.1 * (k + 1)
        t[1, 0, k] = -0.1 * (k + 1)
    return t


# ---------------------------------------------------------------------------
# NmsOptions
# ---------------------------------------------------------------------------


def test_nms_options_defaults():
    o = nms.NmsOptions()
    assert o.target == "minimum"
    assert o.displacement_value == 1.0
    assert o.num_random_displacements == 1
    assert o.ts_mode_index is None
    assert o.seed == 42


def test_nms_options_from_raw_filters_unknown_keys():
    o = nms.NmsOptions.from_raw({"target": "ts", "ts_mode_index": 7, "basis": "ignored"})
    assert o.target == "ts"
    assert o.ts_mode_index == 7


def test_nms_options_rejects_bad_target():
    with pytest.raises(ValueError):
        nms.NmsOptions(target="saddle")  # type: ignore[arg-type]


def test_target_imaginary_count():
    assert nms.target_imaginary_count(nms.NmsOptions(target="minimum")) == 0
    assert nms.target_imaginary_count(nms.NmsOptions(target="ts")) == 1
    assert nms.target_imaginary_count(nms.NmsOptions(target="random")) is None


# ---------------------------------------------------------------------------
# displace_along_mode (pure)
# ---------------------------------------------------------------------------


def test_displace_along_mode_returns_pos_and_neg():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mode = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    pos, neg = nms.displace_along_mode(positions, mode, displacement=2.0)
    np.testing.assert_allclose(pos, [[0.2, 0.0, 0.0], [0.8, 0.0, 0.0]])
    np.testing.assert_allclose(neg, [[-0.2, 0.0, 0.0], [1.2, 0.0, 0.0]])


def test_displace_along_mode_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        nms.displace_along_mode(np.zeros((2, 3)), np.zeros((3, 3)), displacement=1.0)


# ---------------------------------------------------------------------------
# select_displacements — target-aware
# ---------------------------------------------------------------------------


def _rng():
    return np.random.default_rng(0)


def test_select_minimum_displaces_every_imaginary_mode():
    sel = nms.select_displacements(
        _struct([[0, 0, 0], [1, 0, 0]]),
        {5: -42.0, 6: -100.0},
        _modes(8),
        nms.NmsOptions(target="minimum"),
        _rng(),
    )
    assert [label for label, _ in sel] == ["m5_pos", "m5_neg", "m6_pos", "m6_neg"]


def test_select_ts_keeps_largest_imaginary_removes_spurious():
    # mode 6 is most-imaginary → reaction coordinate (kept); mode 5 displaced.
    sel = nms.select_displacements(
        _struct([[0, 0, 0], [1, 0, 0]]),
        {5: -42.0, 6: -200.0},
        _modes(8),
        nms.NmsOptions(target="ts"),
        _rng(),
    )
    assert [label for label, _ in sel] == ["m5_pos", "m5_neg"]


def test_select_ts_honors_explicit_mode_index():
    sel = nms.select_displacements(
        _struct([[0, 0, 0], [1, 0, 0]]),
        {5: -42.0, 6: -200.0},
        _modes(8),
        nms.NmsOptions(target="ts", ts_mode_index=5),
        _rng(),
    )
    # Keep mode 5 (explicit RC) → displace the other imaginary mode 6.
    assert [label for label, _ in sel] == ["m6_pos", "m6_neg"]


def test_select_random_count_and_determinism():
    opts = nms.NmsOptions(target="random", num_random_displacements=2, seed=42)
    a = nms.select_displacements(
        _struct([[0, 0, 0], [1, 0, 0]]), {}, _modes(12), opts, np.random.default_rng(42)
    )
    b = nms.select_displacements(
        _struct([[0, 0, 0], [1, 0, 0]]), {}, _modes(12), opts, np.random.default_rng(42)
    )
    assert len(a) == 4  # 2 modes x (pos, neg)
    assert [label for label, _ in a] == [label for label, _ in b]


def test_select_skips_modes_outside_the_tensor():
    sel = nms.select_displacements(
        _struct([[0, 0, 0], [1, 0, 0]]),
        {37: -100.0},
        _modes(6),
        nms.NmsOptions(target="minimum"),
        _rng(),
    )
    assert sel == []


def test_select_applies_displacement_value():
    sel = nms.select_displacements(
        _struct([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        {5: -42.0},
        _modes(6),
        nms.NmsOptions(displacement_value=2.0),
        _rng(),
    )
    _, pos = sel[0]  # m5_pos; mode 5 moves atom0 by 0.1*6 = 0.6, x2.0 = 1.2
    np.testing.assert_allclose(pos[0], [1.2, 0.0, 0.0])
    np.testing.assert_allclose(pos[1], [1.0 - 1.2, 0.0, 0.0])


def test_select_minimum_with_no_imaginary_returns_empty():
    """No imaginary modes ⇒ nothing to displace (minimum/ts)."""
    sel = nms.select_displacements(
        _struct([[0, 0, 0], [1, 0, 0]]),
        {},
        _modes(8),
        nms.NmsOptions(target="minimum"),
        _rng(),
    )
    assert sel == []


def test_select_random_with_no_candidate_modes_returns_empty():
    """A zero-mode tensor leaves the random sampler no candidates."""
    sel = nms.select_displacements(
        _struct([[0, 0, 0], [1, 0, 0]]),
        {},
        _modes(0),
        nms.NmsOptions(target="random"),
        _rng(),
    )
    assert sel == []


def test_select_skips_mode_with_shape_mismatch():
    """A mode slice whose shape ≠ the geometry is logged and skipped."""
    bad_modes = np.zeros((3, 3, 8))  # 3 "atoms" but the struct has 2
    bad_modes[0, 0, 5] = 0.1
    sel = nms.select_displacements(
        _struct([[0, 0, 0], [1, 0, 0]]),
        {5: -42.0},
        bad_modes,
        nms.NmsOptions(target="minimum"),
        _rng(),
    )
    assert sel == []


# ---------------------------------------------------------------------------
# OrcaEngine two-round orchestration (mocked round-2 submit/parse)
# ---------------------------------------------------------------------------


def _orca_nms_ctx(tmp_path, *, target: str = "minimum") -> StepContext:
    template_dir = tmp_path / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    (template_dir / "step1.inp").write_text("! freq\n%pal\n  nprocs 1\nend\n", encoding="utf-8")
    (template_dir / "cpu.slurm.header").write_text(
        "#!/bin/bash\n#SBATCH --partition=x\n", encoding="utf-8"
    )
    seed = Structure(id="0", atoms=Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]))
    step_cfg = StepConfig(
        step=1, engine="orca", operation="freq", nms=True, options={"target": target}
    )
    return StepContext(
        step_cfg=step_cfg,
        step_dir=tmp_path / "outputs" / "step1",
        template_dir=template_dir,
        scratch_dir=None,
        prev_state=PipelineState(structures=(seed,)),
        charge=0,
        multiplicity=1,
        max_cores=2,
        slurm_template="cpu.slurm.header",
        executables={},
    )


def test_orca_parse_caches_nms_freqs_and_modes(tmp_path):
    """An NMS-step parse caches imaginary freqs + the normal-mode tensor."""
    engine = get_engine("orca")
    ctx = _orca_nms_ctx(tmp_path)
    ctx.step_dir.mkdir(parents=True, exist_ok=True)
    out = ctx.step_dir / "step1_structure_0.out"
    out.write_text(
        synthetic_dft_output([-1.0], [("H", 0, 0, 0), ("H", 0.74, 0, 0)])
        + FREQUENCY_BLOCK
        + NORMAL_MODES_BLOCK_2_ATOMS
        + "\n****ORCA TERMINATED NORMALLY****\n",
        encoding="utf-8",
    )
    inputs = StepInputs(files=((ctx.step_dir / "step1_structure_0.inp", out, "0"),))
    engine.parse(inputs, ctx)
    assert set(engine._imag_freqs["0"]) == {37, 38}
    assert engine._modes["0"] is not None


def test_orca_parse_sets_modes_none_when_block_missing(tmp_path):
    """Freqs present but no NORMAL MODES block ⇒ ``_modes[sid]`` is ``None``."""
    engine = get_engine("orca")
    ctx = _orca_nms_ctx(tmp_path)
    ctx.step_dir.mkdir(parents=True, exist_ok=True)
    out = ctx.step_dir / "step1_structure_0.out"
    out.write_text(
        synthetic_dft_output([-1.0], [("H", 0, 0, 0), ("H", 0.74, 0, 0)])
        + FREQUENCY_BLOCK
        + "\n****ORCA TERMINATED NORMALLY****\n",
        encoding="utf-8",
    )
    inputs = StepInputs(files=((ctx.step_dir / "step1_structure_0.inp", out, "0"),))
    engine.parse(inputs, ctx)
    assert engine._modes["0"] is None


def test_orca_normal_mode_sample_displaces_and_flags(tmp_path):
    """A not-at-target structure is displaced, round 2 runs (mocked), children flagged."""
    engine = get_engine("orca")
    ctx = _orca_nms_ctx(tmp_path)
    # Round-1 parent has one imaginary mode; the round-2 child parsed a freq
    # table with zero imaginary modes (counted-and-zero ⇒ resolved minimum).
    engine._imag_freqs = {"0": {5: -42.0}, "0_m5_pos": {}}
    engine._modes = {"0": _modes(6)}
    round1 = StepResults(structures=(_struct([[0, 0, 0], [0.74, 0, 0]], "0"),))
    child = Structure(
        id="0_m5_pos",
        atoms=Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]),
        parent_id="0",
        terminated=True,
    )
    captured = {}

    def _fake_prepare(c):
        captured["nms_dir"] = c.step_dir
        return StepInputs(files=())

    with (
        patch.object(engine, "prepare", _fake_prepare),
        patch.object(engine, "submit", lambda i, c: JobBatch(jobs={})),
        patch.object(engine, "wait", lambda b: None),
        patch.object(engine, "parse", lambda i, c: StepResults(structures=(child,))),
    ):
        result = engine.normal_mode_sample(round1, ctx)

    assert captured["nms_dir"].name == "nms"
    flagged = {s.id: s.converged for s in result.structures}
    assert flagged == {"0_m5_pos": True}  # freq table parsed, 0 imaginary → resolved


def test_orca_nms_child_without_freq_table_is_unresolved(tmp_path):
    """A round-2 output with no VIBRATIONAL FREQUENCIES table must not resolve.

    ``len({}) == 0`` matching the ``minimum`` target would silently pass a
    structure as a verified minimum with no frequency evidence at all (e.g.
    the template lost its ``Freq`` keyword, or the freq module aborted after
    the optimisation).
    """
    engine = get_engine("orca")
    ctx = _orca_nms_ctx(tmp_path)
    engine._imag_freqs = {"0": {5: -42.0}}  # nothing cached for the child
    engine._modes = {"0": _modes(6)}
    round1 = StepResults(structures=(_struct([[0, 0, 0], [0.74, 0, 0]], "0"),))
    child = Structure(
        id="0_m5_pos",
        atoms=Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]),
        parent_id="0",
        terminated=True,
    )
    with (
        patch.object(engine, "prepare", lambda c: StepInputs(files=())),
        patch.object(engine, "submit", lambda i, c: JobBatch(jobs={})),
        patch.object(engine, "wait", lambda b: None),
        patch.object(engine, "parse", lambda i, c: StepResults(structures=(child,))),
    ):
        result = engine.normal_mode_sample(round1, ctx)
    flagged = {s.id: s.converged for s in result.structures}
    assert flagged == {"0_m5_pos": False}


def test_orca_parse_caches_none_when_freq_table_missing(tmp_path):
    """An NMS-step output without a freq table caches ``None``, not ``{}``."""
    engine = get_engine("orca")
    ctx = _orca_nms_ctx(tmp_path)
    ctx.step_dir.mkdir(parents=True, exist_ok=True)
    out = ctx.step_dir / "step1_structure_0.out"
    out.write_text(
        synthetic_dft_output([-1.0], [("H", 0, 0, 0), ("H", 0.74, 0, 0)])
        + "\n****ORCA TERMINATED NORMALLY****\n",
        encoding="utf-8",
    )
    inputs = StepInputs(files=((ctx.step_dir / "step1_structure_0.inp", out, "0"),))
    engine.parse(inputs, ctx)
    assert engine._imag_freqs["0"] is None


def test_orca_nms_parent_without_freq_table_is_not_already_at_target(tmp_path):
    """Round 1: missing freq data must not count as 'already at the minimum'."""
    engine = get_engine("orca")
    ctx = _orca_nms_ctx(tmp_path)
    engine._imag_freqs = {"0": None}
    engine._modes = {"0": None}
    round1 = StepResults(structures=(_struct([[0, 0, 0], [0.74, 0, 0]], "0"),))
    already, children = engine._nms_displace(round1, ctx)
    assert already == []  # not resolved — and no modes, so not displaced either
    assert children == []


def test_orca_normal_mode_sample_skips_when_no_modes(tmp_path):
    """A structure with no normal-mode tensor can't be displaced (left unresolved)."""
    engine = get_engine("orca")
    ctx = _orca_nms_ctx(tmp_path)
    engine._imag_freqs = {"0": {5: -42.0}}
    engine._modes = {"0": None}
    round1 = StepResults(structures=(_struct([[0, 0, 0], [0.74, 0, 0]], "0"),))
    assert engine.normal_mode_sample(round1, ctx).structures == ()


def test_orca_resolve_nms_from_existing_reads_round2_outputs(tmp_path):
    """rebuild-cache path: re-derive children and parse their existing ``nms/`` outputs."""
    engine = get_engine("orca")
    ctx = _orca_nms_ctx(tmp_path)
    engine._imag_freqs = {"0": {5: -42.0}}
    engine._modes = {"0": _modes(6)}
    round1 = StepResults(structures=(_struct([[0, 0, 0], [0.74, 0, 0]], "0"),))
    nms_dir = ctx.step_dir / "nms"
    nms_dir.mkdir(parents=True, exist_ok=True)
    for cid in ("0_m5_pos", "0_m5_neg"):
        structure_artifact_path(nms_dir, 1, cid, "out").write_text(
            synthetic_dft_output([-1.0], [("H", 0, 0, 0), ("H", 0.74, 0, 0)])
            + "\n****ORCA TERMINATED NORMALLY****\n",
            encoding="utf-8",
        )
    child = Structure(
        id="0_m5_pos",
        atoms=Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]),
        parent_id="0",
        terminated=True,
    )
    with patch.object(engine, "parse", lambda i, c: StepResults(structures=(child,))):
        result = engine.resolve_nms_from_existing(round1, ctx)
    assert any(s.id == "0_m5_pos" for s in result.structures)


def test_orca_resolve_nms_from_existing_with_no_round2_outputs(tmp_path):
    """rebuild-cache path: children with no ``nms/`` outputs on disk are simply
    absent from the result, so their parent stays unresolved downstream."""
    engine = get_engine("orca")
    ctx = _orca_nms_ctx(tmp_path)
    engine._imag_freqs = {"0": {5: -42.0}}
    engine._modes = {"0": _modes(6)}
    round1 = StepResults(structures=(_struct([[0, 0, 0], [0.74, 0, 0]], "0"),))
    result = engine.resolve_nms_from_existing(round1, ctx)
    assert result.structures == ()

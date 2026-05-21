"""Tests for ``chemrefine.engines.orca.nms``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from synthetic import FREQUENCY_BLOCK as _SYNTH_FREQ
from synthetic import NORMAL_MODES_BLOCK_2_ATOMS as _SYNTH_MODES

from chemrefine.config import StepConfig
from chemrefine.engines.orca import nms
from chemrefine.state import PipelineState, StepContext, StepResults, Structure

# ---------------------------------------------------------------------------
# Pure-math helpers
# ---------------------------------------------------------------------------


def test_displace_along_mode_returns_pos_and_neg():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mode = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    pos, neg = nms.displace_along_mode(positions, mode, displacement=2.0)
    np.testing.assert_allclose(pos, [[0.2, 0.0, 0.0], [0.8, 0.0, 0.0]])
    np.testing.assert_allclose(neg, [[-0.2, 0.0, 0.0], [1.2, 0.0, 0.0]])


def test_displace_along_mode_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        nms.displace_along_mode(
            np.zeros((2, 3)), np.zeros((3, 3)), displacement=1.0
        )


def test_least_imaginary_mode_picks_closest_to_zero():
    imag = {37: -118.27, 38: -42.10, 39: -200.0}
    assert nms._least_imaginary_mode(imag) == 38


# ---------------------------------------------------------------------------
# normal_mode_sample — end-to-end with synthetic .out
# ---------------------------------------------------------------------------


def _write_freq_output(step_dir: Path, sid: str, n_atoms: int) -> Path:
    """Write a synthetic frequency .out coherent with the 6-mode tensor.

    The shared :data:`synthetic.FREQUENCY_BLOCK` carries imaginary modes
    at indices 37/38; the shared :data:`synthetic.NORMAL_MODES_BLOCK_2_ATOMS`
    only ships 6 columns, so we override the freq block here to point at
    mode 5 (the one the tensor actually populates).
    """
    step_dir.mkdir(parents=True, exist_ok=True)
    out = step_dir / f"step1_structure_{sid}.out"
    assert n_atoms == 2, "_write_freq_output is wired to the 2-atom synthetic block"
    freq_block = (
        "VIBRATIONAL FREQUENCIES\n"
        "-----------------------\n"
        "\n"
        "Scaling factor for frequencies = 1.0\n"
        "\n"
        "     0:       0.00 cm**-1\n"
        "     5:    -42.10 cm**-1  ***imaginary mode***\n"
        "\n"
    )
    out.write_text(freq_block + _SYNTH_MODES, encoding="utf-8")
    return out


def _ctx_for(tmp_path: Path) -> StepContext:
    step_cfg = StepConfig(
        step=1,
        engine="orca",
        operation="freq",
        nms=True,
    )
    return StepContext(
        step_cfg=step_cfg,
        step_dir=tmp_path / "outputs" / "step1",
        template_dir=tmp_path,
        scratch_dir=None,
        prev_state=PipelineState(),
        charge=0,
        multiplicity=1,
        max_cores=1,
        slurm_template="cpu.slurm.header",
        orca_executable="orca",
    )


def test_normal_mode_sample_expands_each_structure_into_pos_and_neg(tmp_path: Path):
    ctx = _ctx_for(tmp_path)
    seed = Structure(
        id="0",
        atoms=Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]),
    )
    _write_freq_output(ctx.step_dir, "0", n_atoms=2)
    expanded = nms.normal_mode_sample(StepResults(structures=(seed,)), ctx)
    ids = [s.id for s in expanded.structures]
    assert ids == ["0_pos", "0_neg"]


def test_normal_mode_sample_applies_displacement_to_least_imaginary_mode(tmp_path: Path):
    """Mode 38 (-42.10 cm⁻¹) is least-imaginary; mode-5 vector should appear in the pos."""
    ctx = _ctx_for(tmp_path)
    seed = Structure(
        id="0",
        atoms=Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    _write_freq_output(ctx.step_dir, "0", n_atoms=2)
    # Our synthetic block ships 6 modes (0..5); mode 38/39 indices in the
    # frequency block don't exist in this tensor — so the parser should
    # skip cleanly. To exercise the displacement path we need indices
    # that overlap. Replace the synthetic block with a small one whose
    # imaginary mode is mode 5 (which our synthetic tensor populates):
    out = ctx.step_dir / "step1_structure_0.out"
    freq_block = (
        "VIBRATIONAL FREQUENCIES\n"
        "-----------------------\n"
        "\n"
        "Scaling factor for frequencies =  1.000000000\n"
        "\n"
        "     0:       0.00 cm**-1\n"
        "     1:       0.00 cm**-1\n"
        "     2:       0.00 cm**-1\n"
        "     3:       0.00 cm**-1\n"
        "     4:       0.00 cm**-1\n"
        "     5:    -42.10 cm**-1  ***imaginary mode***\n"
        "\n"
    )
    out.write_text(freq_block + _SYNTH_MODES, encoding="utf-8")

    expanded = nms.normal_mode_sample(StepResults(structures=(seed,)), ctx)
    assert len(expanded.structures) == 2
    pos_atoms = expanded.structures[0].atoms.get_positions()
    # mode 5 displaces atom 0 by (0.1, 0.2, 0.3) at displacement_value=1.0
    np.testing.assert_allclose(pos_atoms[0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(pos_atoms[1], [0.9, -0.2, -0.3])


def test_normal_mode_sample_skips_missing_output(tmp_path: Path, caplog):
    """If the freq .out doesn't exist, that structure is skipped (no crash)."""
    import logging

    caplog.set_level(logging.WARNING, logger="chemrefine.engines.orca.nms")
    ctx = _ctx_for(tmp_path)
    seed = Structure(id="ghost", atoms=Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]))
    expanded = nms.normal_mode_sample(StepResults(structures=(seed,)), ctx)
    assert expanded.structures == ()
    assert any("ghost" in r.message for r in caplog.records)


def test_normal_mode_sample_skips_when_no_imaginary_modes(tmp_path: Path):
    ctx = _ctx_for(tmp_path)
    seed = Structure(id="0", atoms=Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]))
    ctx.step_dir.mkdir(parents=True, exist_ok=True)
    # Frequency block with no imaginary modes
    (ctx.step_dir / "step1_structure_0.out").write_text(
        "VIBRATIONAL FREQUENCIES\n-----------------------\n\n"
        "Scaling factor for frequencies = 1.0\n\n"
        "     0:    100.00 cm**-1\n     1:    200.00 cm**-1\n\n",
        encoding="utf-8",
    )
    expanded = nms.normal_mode_sample(StepResults(structures=(seed,)), ctx)
    assert expanded.structures == ()


def test_normal_mode_sample_skips_when_tensor_unparseable(tmp_path: Path):
    """Has imaginary modes but no NORMAL MODES block → skip with a warning."""
    ctx = _ctx_for(tmp_path)
    seed = Structure(id="0", atoms=Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]))
    ctx.step_dir.mkdir(parents=True, exist_ok=True)
    (ctx.step_dir / "step1_structure_0.out").write_text(_SYNTH_FREQ, encoding="utf-8")
    expanded = nms.normal_mode_sample(StepResults(structures=(seed,)), ctx)
    assert expanded.structures == ()


def test_normal_mode_sample_skips_when_imag_index_outside_tensor(tmp_path: Path):
    """Freq says imag mode 37 but tensor only has 6 columns → skip with warning."""
    ctx = _ctx_for(tmp_path)
    seed = Structure(id="0", atoms=Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]))
    ctx.step_dir.mkdir(parents=True, exist_ok=True)
    (ctx.step_dir / "step1_structure_0.out").write_text(
        _SYNTH_FREQ + "\n" + _SYNTH_MODES, encoding="utf-8"
    )
    expanded = nms.normal_mode_sample(StepResults(structures=(seed,)), ctx)
    assert expanded.structures == ()


def test_normal_mode_sample_honors_displacement_value_option(tmp_path: Path):
    """``options.displacement_value`` should scale the ±displacement."""
    step_cfg = StepConfig(
        step=1,
        engine="orca",
        operation="freq",
        nms=True,
        options={"displacement_value": 2.0},
    )
    ctx = StepContext(
        step_cfg=step_cfg,
        step_dir=tmp_path / "outputs" / "step1",
        template_dir=tmp_path,
        scratch_dir=None,
        prev_state=PipelineState(),
        charge=0,
        multiplicity=1,
        max_cores=1,
        slurm_template="cpu.slurm.header",
        orca_executable="orca",
    )
    seed = Structure(id="0", atoms=Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    ctx.step_dir.mkdir(parents=True, exist_ok=True)
    freq_block = (
        "VIBRATIONAL FREQUENCIES\n-----------------------\n\n"
        "Scaling factor for frequencies = 1.0\n\n"
        "     5:    -42.10 cm**-1  ***imaginary mode***\n\n"
    )
    (ctx.step_dir / "step1_structure_0.out").write_text(
        freq_block + _SYNTH_MODES, encoding="utf-8"
    )
    expanded = nms.normal_mode_sample(StepResults(structures=(seed,)), ctx)
    pos = expanded.structures[0].atoms.get_positions()
    # displacement=2.0 → atom 0 moves by 2 * (0.1, 0.2, 0.3) = (0.2, 0.4, 0.6)
    np.testing.assert_allclose(pos[0], [0.2, 0.4, 0.6])

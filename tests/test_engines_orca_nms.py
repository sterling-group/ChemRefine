"""Tests for ORCA's two NMS hooks — ``nms_input_info`` (read the template's
keywords) and ``read_frequencies`` (parse a ``.out``'s imaginary modes + tensor).

The engine-independent two-round algorithm + displacement maths are tested against a
fake engine in ``tests/test_nms.py``; here we only exercise the ORCA-specific half.
"""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from synthetic import FREQUENCY_BLOCK, NORMAL_MODES_BLOCK_2_ATOMS, synthetic_dft_output

from chemrefine.config import StepConfig
from chemrefine.engines.base import get_engine
from chemrefine.ids import structure_artifact_path
from chemrefine.state import PipelineState, StepContext, Structure


def _ctx(tmp_path: Path, template_body: str, *, step: int = 1) -> StepContext:
    template_dir = tmp_path / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    (template_dir / f"step{step}.inp").write_text(template_body, encoding="utf-8")
    (template_dir / "cpu.slurm.header").write_text("#!/bin/bash\n", encoding="utf-8")
    step_cfg = StepConfig(step=step, engine="orca", operation="freq", nms=True)
    seed = Structure(id="0", atoms=Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]))
    return StepContext(
        step_cfg=step_cfg,
        step_dir=tmp_path / "outputs" / f"step{step}",
        template_dir=template_dir,
        scratch_dir=tmp_path / "scratch",
        prev_state=PipelineState(structures=(seed,)),
        charge=0,
        multiplicity=1,
        max_cores=2,
        slurm_template="cpu.slurm.header",
        executables={},
    )


def _freq_output() -> str:
    return (
        synthetic_dft_output([-1.0], [("H", 0, 0, 0), ("H", 0.74, 0, 0)])
        + FREQUENCY_BLOCK
        + NORMAL_MODES_BLOCK_2_ATOMS
        + "\n****ORCA TERMINATED NORMALLY****\n"
    )


# ---------------------------------------------------------------------------
# nms_input_info — read the template keywords
# ---------------------------------------------------------------------------


def test_nms_input_info_plain_opt_freq(tmp_path: Path):
    info = get_engine("orca").nms_input_info(_ctx(tmp_path, "! B3LYP def2-SVP Opt Freq\n"))
    assert info.computes_frequencies is True
    assert info.is_transition_state is False


def test_nms_input_info_optts_is_transition_state(tmp_path: Path):
    info = get_engine("orca").nms_input_info(_ctx(tmp_path, "! B3LYP def2-SVP OptTS Freq\n"))
    assert info.is_transition_state is True
    assert info.computes_frequencies is True


def test_nms_input_info_no_freq_keyword(tmp_path: Path):
    info = get_engine("orca").nms_input_info(_ctx(tmp_path, "! B3LYP def2-SVP Opt\n"))
    assert info.computes_frequencies is False


# ---------------------------------------------------------------------------
# read_frequencies — parse a structure's .out
# ---------------------------------------------------------------------------


def test_read_frequencies_parses_imaginary_and_modes(tmp_path: Path):
    engine = get_engine("orca")
    ctx = _ctx(tmp_path, "! B3LYP def2-SVP Opt Freq\n")
    out = structure_artifact_path(ctx.step_dir, 1, "0", "out")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_freq_output(), encoding="utf-8")
    freq = engine.read_frequencies("0", ctx.step_dir, ctx)
    assert freq.imaginary == {37: -118.27, 38: -42.10}
    assert freq.modes is not None
    assert freq.modes.shape == (2, 3, 6)  # 2 atoms by 3 axes by 6 modes


def test_read_frequencies_none_without_freq_table(tmp_path: Path):
    """An output with no VIBRATIONAL FREQUENCIES table reports None (not {})."""
    engine = get_engine("orca")
    ctx = _ctx(tmp_path, "! B3LYP def2-SVP Opt Freq\n")
    out = structure_artifact_path(ctx.step_dir, 1, "0", "out")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        synthetic_dft_output([-1.0], [("H", 0, 0, 0), ("H", 0.74, 0, 0)]), encoding="utf-8"
    )
    freq = engine.read_frequencies("0", ctx.step_dir, ctx)
    assert freq.imaginary is None
    assert freq.modes is None


def test_read_frequencies_none_when_output_missing(tmp_path: Path):
    engine = get_engine("orca")
    ctx = _ctx(tmp_path, "! B3LYP def2-SVP Opt Freq\n")
    freq = engine.read_frequencies("0", ctx.step_dir, ctx)  # no .out written
    assert freq.imaginary is None
    assert freq.modes is None


def test_read_frequencies_modes_none_on_unparseable_geometry(tmp_path: Path):
    """A freq table with no parseable geometry → imaginary parsed, modes None."""
    engine = get_engine("orca")
    ctx = _ctx(tmp_path, "! B3LYP def2-SVP Opt Freq\n")
    out = structure_artifact_path(ctx.step_dir, 1, "0", "out")
    out.parent.mkdir(parents=True, exist_ok=True)
    # FREQUENCY_BLOCK alone: a VIBRATIONAL FREQUENCIES table but no coordinates.
    out.write_text(FREQUENCY_BLOCK, encoding="utf-8")
    freq = engine.read_frequencies("0", ctx.step_dir, ctx)
    assert freq.imaginary == {37: -118.27, 38: -42.10}
    assert freq.modes is None

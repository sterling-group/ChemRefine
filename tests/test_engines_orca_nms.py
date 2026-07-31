"""Tests for ORCA's NMS surface: ``nms_input_info`` (read the template's keywords) and the
frequency values its parse carries on each structure (``imaginary_freqs`` / ``normal_modes``,
parsed in the same single pass as geometry — there is no separate ``read_frequencies`` hook).

The engine-independent two-round algorithm + displacement maths are tested against a fake
engine in ``tests/test_nms.py``; here we only exercise the ORCA-specific half.
"""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from synthetic import FREQUENCY_BLOCK, NORMAL_MODES_BLOCK_2_ATOMS, synthetic_dft_output

from chemrefine.config import StepConfig
from chemrefine.engines.api import get_engine
from chemrefine.engines.orca import output
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
        template=template_dir / f"step{step}.inp",
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
# frequency values carried on the parsed structure (one pass — no re-parse)
# ---------------------------------------------------------------------------


def test_parse_attaches_imaginary_and_modes():
    """An opt+freq ``.out`` parses geometry/energy AND the frequency block in one pass."""
    parsed = output.parse_dft_from_text(_freq_output())[0]
    assert parsed.imaginary_freqs == {37: -118.27, 38: -42.10}
    assert parsed.normal_modes is not None
    assert parsed.normal_modes.shape == (2, 3, 6)  # 2 atoms by 3 axes by 6 modes


def test_parse_leaves_freqs_none_without_freq_table():
    """A plain opt ``.out`` (no VIBRATIONAL FREQUENCIES) carries None (not {}) for both."""
    text = synthetic_dft_output([-1.0], [("H", 0, 0, 0), ("H", 0.74, 0, 0)])
    parsed = output.parse_dft_from_text(text)[0]
    assert parsed.imaginary_freqs is None
    assert parsed.normal_modes is None


def test_parse_modes_none_when_mode_block_unparseable():
    """A freq table present but no parseable normal-mode block → imaginary set, modes None."""
    text = synthetic_dft_output([-1.0], [("H", 0, 0, 0), ("H", 0.74, 0, 0)]) + FREQUENCY_BLOCK
    parsed = output.parse_dft_from_text(text)[0]
    assert parsed.imaginary_freqs == {37: -118.27, 38: -42.10}
    assert parsed.normal_modes is None

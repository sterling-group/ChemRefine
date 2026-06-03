"""Tests for ``OrcaEngine`` — prepare/parse paths; submit/wait via SLURM mocks."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from ase import Atoms

from chemrefine import slurm
from chemrefine.config import StepConfig
from chemrefine.engines.base import get_engine
from chemrefine.state import JobBatch, PipelineState, StepContext, Structure

FIXTURE = Path(__file__).parent / "data" / "orca.out"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx(
    tmp_path: Path,
    structures: tuple[Structure, ...],
    step_cfg: StepConfig | None = None,
) -> StepContext:
    """Build a StepContext with a usable template + SLURM header on disk."""
    template_dir = tmp_path / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    (template_dir / "step1.inp").write_text(
        "! B3LYP def2-SVP\n%pal\n  nprocs 2\nend\n", encoding="utf-8"
    )
    (template_dir / "cpu.slurm.header").write_text(
        "#!/bin/bash\n"
        "#SBATCH --partition=normal\n"
        "#SBATCH --time=24:00:00\n"
        "module load orca/6.0\n",
        encoding="utf-8",
    )
    step_cfg = step_cfg or StepConfig(step=1, engine="orca", operation="opt_sp")
    step_dir = tmp_path / "outputs" / step_cfg.dir_name()
    return StepContext(
        step_cfg=step_cfg,
        step_dir=step_dir,
        template_dir=template_dir,
        scratch_dir=tmp_path / "scratch",
        prev_state=PipelineState(structures=structures),
        charge=0,
        multiplicity=1,
        max_cores=4,
        slurm_template="cpu.slurm.header",
        executables={},
    )


def _seed_structure(sid: str = "0") -> Structure:
    return Structure(id=sid, atoms=Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]))


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


def test_prepare_writes_inp_and_xyz_per_structure(tmp_path: Path):
    engine = get_engine("orca")
    ctx = _ctx(tmp_path, structures=(_seed_structure("0"), _seed_structure("1")))
    inputs = engine.prepare(ctx)
    assert len(inputs.files) == 2
    for inp_path, _out_path, _sid in inputs.files:
        assert inp_path.exists()
        assert inp_path.suffix == ".inp"
        assert (inp_path.with_suffix(".xyz")).exists()


def test_prepare_input_contains_charge_and_multiplicity(tmp_path: Path):
    engine = get_engine("orca")
    ctx = _ctx(tmp_path, structures=(_seed_structure(),))
    # Override charge / multiplicity via context attributes
    ctx = StepContext(
        step_cfg=ctx.step_cfg,
        step_dir=ctx.step_dir,
        template_dir=ctx.template_dir,
        scratch_dir=ctx.scratch_dir,
        prev_state=ctx.prev_state,
        charge=-2,
        multiplicity=3,
        max_cores=ctx.max_cores,
        slurm_template=ctx.slurm_template,
        executables=ctx.executables,
    )
    inputs = engine.prepare(ctx)
    text = inputs.files[0][0].read_text()
    assert "* xyzfile -2 3" in text


def test_prepare_missing_template_raises(tmp_path: Path):
    engine = get_engine("orca")
    ctx = _ctx(tmp_path, structures=(_seed_structure(),))
    # Delete the template after _ctx wrote it.
    (ctx.template_dir / "step1.inp").unlink()
    with pytest.raises(FileNotFoundError):
        engine.prepare(ctx)


def test_prepare_uses_step_specific_template_when_given(tmp_path: Path):
    engine = get_engine("orca")
    step_cfg = StepConfig(step=2, engine="orca", operation="opt_sp", template="custom.inp")
    ctx = _ctx(tmp_path, structures=(_seed_structure(),), step_cfg=step_cfg)
    (ctx.template_dir / "custom.inp").write_text("! HF\n", encoding="utf-8")
    inputs = engine.prepare(ctx)
    text = inputs.files[0][0].read_text()
    assert "! HF" in text


# ---------------------------------------------------------------------------
# submit / wait
# ---------------------------------------------------------------------------


@patch.object(slurm, "is_finished", return_value=True)
@patch.object(slurm, "submit")
def test_submit_creates_script_per_input_and_returns_batch(
    submit_mock, _is_finished, tmp_path: Path
):
    submit_mock.side_effect = ["1001", "1002"]
    engine = get_engine("orca")
    ctx = _ctx(tmp_path, structures=(_seed_structure("0"), _seed_structure("1")))
    inputs = engine.prepare(ctx)
    batch = engine.submit(inputs, ctx)
    assert isinstance(batch, JobBatch)
    assert set(batch.jobs.values()) == {"1001", "1002"}
    # Each input has an accompanying .slurm script on disk.
    for inp, _out, _sid in inputs.files:
        assert inp.with_suffix(".slurm").exists()


@patch.object(slurm, "is_finished", return_value=True)
@patch.object(slurm, "submit", return_value="9001")
def test_submit_script_contains_orca_executable_invocation(_submit, _is_finished, tmp_path: Path):
    engine = get_engine("orca")
    ctx = _ctx(tmp_path, structures=(_seed_structure(),))
    inputs = engine.prepare(ctx)
    engine.submit(inputs, ctx)
    script_text = inputs.files[0][0].with_suffix(".slurm").read_text()
    assert "orca step1_structure_0.inp" in script_text
    assert "$OUTPUT_DIR/step1_structure_0.out" in script_text
    # ORCA's output_globs ClassVar flows through the shared SlurmBatchEngine.
    assert "*.gbw" in script_text
    assert "*.hess" in script_text


@patch.object(slurm, "is_finished", return_value=True)
@patch.object(slurm, "submit", return_value="9001")
def test_submit_missing_slurm_header_raises(_submit, _is_finished, tmp_path: Path):
    engine = get_engine("orca")
    ctx = _ctx(tmp_path, structures=(_seed_structure(),))
    (ctx.template_dir / "cpu.slurm.header").unlink()
    inputs = engine.prepare(ctx)
    with pytest.raises(FileNotFoundError):
        engine.submit(inputs, ctx)


def test_wait_is_noop(tmp_path: Path):
    """``submit`` already blocks until finished; ``wait`` should not error."""
    engine = get_engine("orca")
    engine.wait(JobBatch(jobs={}))  # no jobs to wait for


# ---------------------------------------------------------------------------
# parse — uses the real fixture for the DFT happy path
# ---------------------------------------------------------------------------


def test_parse_uses_orca_fixture_as_output(tmp_path: Path):
    engine = get_engine("orca")
    ctx = _ctx(tmp_path, structures=(_seed_structure("0"),))
    inputs = engine.prepare(ctx)
    _inp, out, _sid = inputs.files[0]
    # Stage the fixture as the engine's expected output file.
    shutil.copy(FIXTURE, out)
    results = engine.parse(inputs, ctx)
    assert len(results.structures) == 1
    parsed = results.structures[0]
    assert parsed.id == "0"
    assert parsed.energy_hartree is not None
    assert abs(parsed.energy_hartree + 6044.555) < 1e-2
    assert len(parsed.atoms) > 50  # fixture molecule is large


def test_parse_unknown_operation_raises(tmp_path: Path):
    engine = get_engine("orca")
    step_cfg = StepConfig(step=1, engine="orca", operation="weird_op")
    ctx = _ctx(tmp_path, structures=(_seed_structure(),), step_cfg=step_cfg)
    inputs = engine.prepare(ctx)
    shutil.copy(FIXTURE, inputs.files[0][1])
    from chemrefine.errors import OutputParseError

    with pytest.raises(OutputParseError):
        engine.parse(inputs, ctx)


# ---------------------------------------------------------------------------
# Engine registration
# ---------------------------------------------------------------------------


def test_orca_engine_registered():
    from chemrefine.engines.base import ENGINES

    assert "orca" in ENGINES


def test_orca_engine_supports_nms_flag_is_true():
    assert get_engine("orca").supports_nms is True


def test_orca_engine_nms_returns_empty_on_no_freq_output(tmp_path: Path):
    """With no freq output on disk, NMS skips every structure cleanly."""
    engine = get_engine("orca")
    ctx = _ctx(tmp_path, structures=(_seed_structure(),))
    from chemrefine.state import StepResults

    expanded = engine.normal_mode_sample(StepResults(structures=()), ctx)
    assert expanded.structures == ()

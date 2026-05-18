"""Tests for SLURM script generation, sbatch submission, and squeue polling."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chemrefine import slurm
from chemrefine.errors import JobSubmissionError

# ---------------------------------------------------------------------------
# parse_pal
# ---------------------------------------------------------------------------


def test_parse_pal_reads_nprocs(tmp_path: Path):
    inp = tmp_path / "step1.inp"
    inp.write_text("! B3LYP def2-SVP\n%pal\n  nprocs 8\nend\n", encoding="utf-8")
    assert slurm.parse_pal(inp) == 8


def test_parse_pal_reads_inline_directive(tmp_path: Path):
    inp = tmp_path / "step1.inp"
    inp.write_text("! B3LYP PAL4\n", encoding="utf-8")
    assert slurm.parse_pal(inp) == 4


def test_parse_pal_defaults_to_one(tmp_path: Path):
    inp = tmp_path / "step1.inp"
    inp.write_text("! B3LYP def2-SVP\n", encoding="utf-8")
    assert slurm.parse_pal(inp) == 1


# ---------------------------------------------------------------------------
# build_script
# ---------------------------------------------------------------------------


def _write_header(tmp_path: Path) -> Path:
    header = tmp_path / "cpu.slurm.header"
    header.write_text(
        "#!/bin/bash\n"
        "#SBATCH --partition=normal\n"
        "#SBATCH --time=24:00:00\n"
        "#SBATCH --ntasks=1\n"           # gets overridden
        "#SBATCH --cpus-per-task=4\n"    # gets overridden
        "module load orca/6.0\n",
        encoding="utf-8",
    )
    return header


def test_build_script_overrides_ntasks_and_writes_script(tmp_path: Path):
    header = _write_header(tmp_path)
    script = slurm.build_script(
        job_name="step1_structure_0",
        pal=12,
        template_path=header,
        script_path=tmp_path / "out" / "step1_structure_0.slurm",
        input_path=tmp_path / "in" / "step1_structure_0.inp",
        output_dir=tmp_path / "out",
        scratch_dir=tmp_path / "scratch",
        run_block='$ORCA step1_structure_0.inp > $OUTPUT_DIR/step1_structure_0.out',
    )
    assert script.exists()
    text = script.read_text()
    assert "#SBATCH --partition=normal" in text
    assert "#SBATCH --time=24:00:00" in text
    assert "#SBATCH --ntasks=12" in text
    assert "#SBATCH --cpus-per-task=1" in text
    # the user's --ntasks=1 must not survive
    assert "--ntasks=1" not in text or "--ntasks=12" in text
    assert "module load orca/6.0" in text


def test_build_script_includes_run_block(tmp_path: Path):
    header = _write_header(tmp_path)
    script = slurm.build_script(
        job_name="job",
        pal=1,
        template_path=header,
        script_path=tmp_path / "j.slurm",
        input_path=tmp_path / "j.inp",
        output_dir=tmp_path / "out",
        scratch_dir=tmp_path / "scratch",
        run_block="echo CUSTOM_RUN_BLOCK_HERE",
    )
    assert "echo CUSTOM_RUN_BLOCK_HERE" in script.read_text()


def test_build_script_save_scratch_keeps_dir(tmp_path: Path):
    header = _write_header(tmp_path)
    script = slurm.build_script(
        job_name="job",
        pal=1,
        template_path=header,
        script_path=tmp_path / "j.slurm",
        input_path=tmp_path / "j.inp",
        output_dir=tmp_path / "out",
        scratch_dir=tmp_path / "scratch",
        run_block="echo hi",
        save_scratch=True,
    )
    text = script.read_text()
    assert "rm -rf $SCRATCH_DIR" not in text
    assert "scratch dir kept" in text


def test_build_script_missing_template_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        slurm.build_script(
            job_name="job",
            pal=1,
            template_path=tmp_path / "missing.header",
            script_path=tmp_path / "j.slurm",
            input_path=tmp_path / "j.inp",
            output_dir=tmp_path / "out",
            scratch_dir=tmp_path / "scratch",
            run_block="echo hi",
        )


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


def test_submit_parses_job_id_from_sbatch_output():
    fake = MagicMock(returncode=0, stdout="Submitted batch job 12345\n", stderr="")
    with patch.object(subprocess, "run", return_value=fake):
        assert slurm.submit("script.slurm") == "12345"


def test_submit_raises_on_sbatch_failure():
    err = subprocess.CalledProcessError(1, ["sbatch"], stderr="permission denied")
    with patch.object(subprocess, "run", side_effect=err), pytest.raises(JobSubmissionError):
        slurm.submit("script.slurm")


def test_submit_raises_when_output_lacks_job_id():
    fake = MagicMock(returncode=0, stdout="weird output\n", stderr="")
    with patch.object(subprocess, "run", return_value=fake), pytest.raises(JobSubmissionError):
        slurm.submit("script.slurm")


# ---------------------------------------------------------------------------
# is_finished
# ---------------------------------------------------------------------------


def test_is_finished_true_when_job_absent():
    fake = MagicMock(returncode=0, stdout="JOBID\n9999\n", stderr="")
    with patch.object(subprocess, "run", return_value=fake):
        assert slurm.is_finished("12345") is True


def test_is_finished_false_when_job_still_running():
    fake = MagicMock(returncode=0, stdout="JOBID\n12345\n", stderr="")
    with patch.object(subprocess, "run", return_value=fake):
        assert slurm.is_finished("12345") is False


def test_is_finished_treats_squeue_failure_as_not_finished():
    err = subprocess.CalledProcessError(1, ["squeue"])
    with patch.object(subprocess, "run", side_effect=err):
        assert slurm.is_finished("12345") is False

"""Tests for SLURM script generation, sbatch submission, and squeue polling."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chemrefine import slurm
from chemrefine.errors import JobSubmissionError


def _drain_local(job_id: str, *, timeout: float = 5.0) -> None:
    """Block until a background ``local-N`` job is reaped (or the timeout)."""
    deadline = time.monotonic() + timeout
    while not slurm.is_finished(job_id) and time.monotonic() < deadline:
        time.sleep(0.01)

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


def _build_kwargs(tmp_path: Path, **overrides):
    """Default kwargs for ``slurm.build_script`` tests."""
    base = {
        "job_name": "step1_structure_0",
        "pal": 1,
        "template_path": _write_header(tmp_path),
        "script_path": tmp_path / "out" / "step1_structure_0.slurm",
        "input_path": tmp_path / "in" / "step1_structure_0.inp",
        "output_dir": tmp_path / "out",
        "scratch_dir": tmp_path / "scratch",
        "run_block": "echo hi",
        "engine": "orca",
        "operation": "opt_sp",
        "step": 1,
        "structure_id": "0",
        "step_label": "step1_refine",
        "output_globs": ("*.out", "*.xyz", "*.gbw", "*.hess"),
    }
    base.update(overrides)
    return base


def test_build_script_overrides_ntasks_and_writes_script(tmp_path: Path):
    script = slurm.build_script(
        **_build_kwargs(
            tmp_path,
            pal=12,
            run_block='$ORCA step1_structure_0.inp > $OUTPUT_DIR/step1_structure_0.out',
        )
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


def test_build_script_emits_absolute_runlog_output_directives(tmp_path: Path):
    """``#SBATCH --output`` / ``--error`` must point at absolute paths in the step dir."""
    out = (tmp_path / "out").resolve()
    script = slurm.build_script(**_build_kwargs(tmp_path, output_dir=out))
    text = script.read_text()
    assert f'#SBATCH --output="{out}/step1_structure_0.runlog"' in text
    assert f'#SBATCH --error="{out}/step1_structure_0.err"' in text


def test_build_script_includes_runlog_header_and_footer_fields(tmp_path: Path):
    """The generated script must wrap the run_block with job_log header + footer."""
    script = slurm.build_script(
        **_build_kwargs(
            tmp_path,
            engine="mlff",
            operation="opt_sp",
            step=2,
            structure_id="0",
            step_label="step2_refine",
        )
    )
    text = script.read_text()
    assert "engine=mlff" in text
    assert "operation=opt_sp" in text
    assert "step=2" in text
    assert "structure_id=0" in text
    assert "ChemRefine mlff step2_refine starting" in text
    assert "ChemRefine mlff step2_refine finished" in text


def test_build_script_includes_run_block(tmp_path: Path):
    script = slurm.build_script(**_build_kwargs(tmp_path, run_block="echo CUSTOM_RUN_BLOCK_HERE"))
    assert "echo CUSTOM_RUN_BLOCK_HERE" in script.read_text()


def test_build_script_auto_scratch_under_output_dir_when_none(tmp_path: Path):
    """Omitting scratch_dir makes WORK_DIR a sibling of step output."""
    out = (tmp_path / "out").resolve()
    script = slurm.build_script(**_build_kwargs(tmp_path, scratch_dir=None, output_dir=out))
    text = script.read_text()
    assert f'export WORK_DIR="{out}/_work_' in text


def test_build_script_explicit_scratch_uses_chemrefine_subdir(tmp_path: Path):
    scratch = (tmp_path / "scratch").resolve()
    script = slurm.build_script(**_build_kwargs(tmp_path, scratch_dir=scratch))
    assert f'export WORK_DIR="{scratch}/ChemRefine_' in script.read_text()


def test_build_script_save_scratch_keeps_dir(tmp_path: Path):
    script = slurm.build_script(**_build_kwargs(tmp_path, save_scratch=True))
    text = script.read_text()
    assert "rm -rf $WORK_DIR" not in text
    assert "scratch_kept=true" in text


def test_build_script_emits_exit_trap(tmp_path: Path):
    """The footer must be wired through a trap so it fires on failure too."""
    script = slurm.build_script(**_build_kwargs(tmp_path))
    text = script.read_text()
    assert "trap _on_exit EXIT" in text


def test_build_script_missing_template_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        slurm.build_script(**_build_kwargs(tmp_path, template_path=tmp_path / "missing.header"))


def test_build_script_uses_caller_supplied_output_globs(tmp_path: Path):
    """The back-copy line must reflect the engine's declared file extensions."""
    script = slurm.build_script(
        **_build_kwargs(tmp_path, output_globs=("*.json", "*.npz"))
    )
    text = script.read_text()
    assert 'cp *.json *.npz "$OUTPUT_DIR/"' in text
    assert "files_copied=$(ls *.json *.npz 2>/dev/null | wc -l)" in text
    # Engine-specific ORCA globs must not leak in.
    assert "*.gbw" not in text


def test_build_script_appends_extra_header_fields(tmp_path: Path):
    """Engine-specific runlog fields must appear after the fixed skeleton."""
    script = slurm.build_script(
        **_build_kwargs(
            tmp_path,
            extra_header_fields=(("orca_executable", "/opt/orca/orca"),),
        )
    )
    text = script.read_text()
    assert "orca_executable=/opt/orca/orca" in text


def test_build_script_without_extra_header_fields_is_engine_neutral(tmp_path: Path):
    """Default (no extra fields) emits no engine-specific rows."""
    script = slurm.build_script(**_build_kwargs(tmp_path))
    assert "orca_executable=" not in script.read_text()


# ---------------------------------------------------------------------------
# submit (sbatch path)
# ---------------------------------------------------------------------------


def test_submit_parses_job_id_from_sbatch_output():
    fake = MagicMock(returncode=0, stdout="Submitted batch job 12345\n", stderr="")
    with (
        patch("chemrefine.slurm.shutil.which", return_value="/usr/bin/sbatch"),
        patch.object(subprocess, "run", return_value=fake),
    ):
        assert slurm.submit("script.slurm") == "12345"


def test_submit_raises_on_sbatch_failure():
    err = subprocess.CalledProcessError(1, ["sbatch"], stderr="permission denied")
    with (
        patch("chemrefine.slurm.shutil.which", return_value="/usr/bin/sbatch"),
        patch.object(subprocess, "run", side_effect=err),
        pytest.raises(JobSubmissionError),
    ):
        slurm.submit("script.slurm")


def test_submit_raises_when_output_lacks_job_id():
    fake = MagicMock(returncode=0, stdout="weird output\n", stderr="")
    with (
        patch("chemrefine.slurm.shutil.which", return_value="/usr/bin/sbatch"),
        patch.object(subprocess, "run", return_value=fake),
        pytest.raises(JobSubmissionError),
    ):
        slurm.submit("script.slurm")


# ---------------------------------------------------------------------------
# submit (local fallback)
# ---------------------------------------------------------------------------


def test_submit_falls_back_to_local_when_sbatch_missing(tmp_path: Path):
    """No sbatch on PATH → launch via bash in the background, return a local-N job ID."""
    script = tmp_path / "script.slurm"
    script.write_text("#!/bin/bash\ntrue\n", encoding="utf-8")
    with patch("chemrefine.slurm.shutil.which", return_value=None):
        job_id = slurm.submit(script)
    assert job_id.startswith("local-")
    _drain_local(job_id)
    assert slurm.is_finished(job_id) is True


def test_submit_local_writes_runlog_and_err_alongside_script(tmp_path: Path):
    """The background local run redirects stdout/stderr to the SBATCH .runlog / .err paths."""
    script = tmp_path / "step1_structure_0.slurm"
    script.write_text(
        "#!/bin/bash\necho 'hello from the template'\necho 'warning: backend X' >&2\n",
        encoding="utf-8",
    )
    with patch("chemrefine.slurm.shutil.which", return_value=None):
        job_id = slurm.submit(script)
    _drain_local(job_id)
    assert script.with_suffix(".runlog").read_text(encoding="utf-8") == "hello from the template\n"
    assert script.with_suffix(".err").read_text(encoding="utf-8") == "warning: backend X\n"


def test_two_local_jobs_run_in_background_concurrently(tmp_path: Path):
    """Background submission returns immediately, so two local jobs overlap — the old
    synchronous fallback would have serialized them (the first blocking until done)."""
    s1 = tmp_path / "a.slurm"
    s2 = tmp_path / "b.slurm"
    s1.write_text("#!/bin/bash\nsleep 0.5\n", encoding="utf-8")
    s2.write_text("#!/bin/bash\nsleep 0.5\n", encoding="utf-8")
    with patch("chemrefine.slurm.shutil.which", return_value=None):
        j1 = slurm.submit(s1)
        j2 = slurm.submit(s2)
    # Right after submit, both are still running → they run in parallel.
    assert slurm.is_finished(j1) is False
    assert slurm.is_finished(j2) is False
    _drain_local(j1)
    _drain_local(j2)
    assert slurm.is_finished(j1) is True
    assert slurm.is_finished(j2) is True


def test_submit_local_failure_is_not_raised_but_recorded_on_disk(tmp_path: Path):
    """A non-zero local exit no longer raises at submit; the logs still land on disk
    so the failure can surface through the engine's output parsing."""
    script = tmp_path / "script.slurm"
    script.write_text(
        "#!/bin/bash\necho 'partial stdout'\necho 'boom' >&2\nexit 2\n", encoding="utf-8"
    )
    with patch("chemrefine.slurm.shutil.which", return_value=None):
        job_id = slurm.submit(script)  # does not raise
    _drain_local(job_id)
    assert slurm.is_finished(job_id) is True
    assert script.with_suffix(".runlog").read_text(encoding="utf-8") == "partial stdout\n"
    assert script.with_suffix(".err").read_text(encoding="utf-8") == "boom\n"


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


def test_is_finished_true_for_local_job_id():
    """Local fallback IDs are reported finished without touching squeue."""
    with patch.object(subprocess, "run") as run:
        assert slurm.is_finished("local-7") is True
    run.assert_not_called()

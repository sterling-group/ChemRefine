"""Tests for the ``chemrefine.job_log`` taxonomy emitter."""

from __future__ import annotations

from pathlib import Path

from chemrefine import job_log

# ---------------------------------------------------------------------------
# Bash header / footer (string snapshots)
# ---------------------------------------------------------------------------


def test_bash_header_contains_every_required_field(tmp_path: Path):
    snippet = job_log.bash_header(
        engine="mlff",
        operation="opt_sp",
        step=2,
        structure_id="0",
        step_label="step2_refine",
        step_dir=tmp_path / "outputs" / "step2_refine",
        cores=8,
        orca_executable="/opt/orca/orca",
    )
    for key in job_log._HEADER_KEYS:
        assert f"{key}=" in snippet
    assert "ChemRefine mlff step2_refine starting" in snippet
    assert "__cr_mode=slurm" in snippet
    assert "__cr_mode=bash" in snippet
    assert "start_time=$(date +%s)" in snippet


def test_bash_footer_emits_all_footer_keys():
    snippet = job_log.bash_footer(engine="mlff", step_label="step2_refine")
    for key in job_log._FOOTER_KEYS:
        assert f"{key}=" in snippet
    assert "ChemRefine mlff step2_refine finished" in snippet
    assert "elapsed_seconds=$((end_time - start_time))" in snippet


def test_bash_header_inlines_values_not_shell_refs():
    snippet = job_log.bash_header(
        engine="mlff",
        operation="opt_sp",
        step=3,
        structure_id="abc-1",
        step_label="step3_screen",
        step_dir=Path("/abs/outputs/step3_screen"),
        cores=16,
        orca_executable="/usr/bin/orca",
    )
    assert "engine=mlff" in snippet
    assert "operation=opt_sp" in snippet
    assert "step=3" in snippet
    assert "structure_id=abc-1" in snippet
    assert "output=/abs/outputs/step3_screen" in snippet
    assert "cores=16" in snippet
    assert "orca_executable=/usr/bin/orca" in snippet


# ---------------------------------------------------------------------------
# Python header / footer
# ---------------------------------------------------------------------------


def test_python_header_writes_file_with_all_required_fields(tmp_path: Path):
    log = tmp_path / "step1_structure_0.runlog"
    job_log.python_header(
        engine="mlff-direct",
        operation="opt_sp",
        step=1,
        structure_id="0",
        step_label="step1_screen",
        step_dir=tmp_path / "outputs" / "step1_screen",
        log_path=log,
    )
    text = log.read_text(encoding="utf-8")
    for key in job_log._HEADER_KEYS:
        assert f"{key}=" in text
    assert "ChemRefine mlff-direct step1_screen starting" in text
    assert "mode=direct" in text
    assert "cores=1" in text
    assert "orca_executable=—" in text


def test_python_footer_appends_finished_line(tmp_path: Path):
    log = tmp_path / "rl.runlog"
    job_log.python_header(
        engine="mlff-direct",
        operation="opt_sp",
        step=1,
        structure_id="0",
        step_label="step1",
        step_dir=tmp_path,
        log_path=log,
    )
    job_log.python_footer(
        engine="mlff-direct",
        step_label="step1",
        log_path=log,
        exit_code=0,
        elapsed_seconds=42,
        files_copied=3,
        scratch_kept=False,
    )
    text = log.read_text(encoding="utf-8")
    assert "ChemRefine mlff-direct step1 starting" in text
    assert "ChemRefine mlff-direct step1 finished" in text
    assert "exit_code=0" in text
    assert "elapsed_seconds=42" in text
    assert "files_copied=3" in text
    assert "scratch_kept=false" in text


def test_python_footer_renders_kept_scratch_as_true(tmp_path: Path):
    log = tmp_path / "rl.runlog"
    job_log.python_header(
        engine="mlff-direct",
        operation="opt_sp",
        step=1,
        structure_id="0",
        step_label="step1",
        step_dir=tmp_path,
        log_path=log,
    )
    job_log.python_footer(
        engine="mlff-direct",
        step_label="step1",
        log_path=log,
        exit_code=0,
        elapsed_seconds=1,
        scratch_kept=True,
    )
    assert "scratch_kept=true" in log.read_text(encoding="utf-8")


def test_python_header_creates_missing_parent(tmp_path: Path):
    """Should create the runlog's parent directory if it doesn't exist."""
    log = tmp_path / "deep" / "nested" / "x.runlog"
    job_log.python_header(
        engine="pyscf-direct",
        operation="opt_sp",
        step=1,
        structure_id="0",
        step_label="step1",
        step_dir=tmp_path,
        log_path=log,
    )
    assert log.is_file()


def test_format_field_indents_two_spaces():
    assert job_log._format_field("k", "v") == "  k=v"


def test_monotonic_seconds_is_int():
    assert isinstance(job_log.monotonic_seconds(), int)

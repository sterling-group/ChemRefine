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
    )
    for key in job_log._HEADER_KEYS:
        assert f"{key}=" in snippet
    assert "ChemRefine mlff step2_refine starting" in snippet
    assert "__cr_mode=slurm" in snippet
    assert "__cr_mode=bash" in snippet
    assert "start_time=$(date +%s)" in snippet


def test_bash_header_engine_neutral_by_default():
    """Without extra_fields, no engine-specific keys (orca_executable, …) leak."""
    snippet = job_log.bash_header(
        engine="mlff",
        operation="opt_sp",
        step=1,
        structure_id="0",
        step_label="step1_refine",
        step_dir=Path("/abs/outputs/step1_refine"),
        cores=1,
    )
    assert "orca_executable=" not in snippet


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
        extra_fields=(("orca_executable", "/usr/bin/orca"),),
    )
    assert "engine=mlff" in snippet
    assert "operation=opt_sp" in snippet
    assert "step=3" in snippet
    assert "structure_id=abc-1" in snippet
    assert "output=/abs/outputs/step3_screen" in snippet
    assert "cores=16" in snippet
    assert "orca_executable=/usr/bin/orca" in snippet


def test_format_field_indents_two_spaces():
    assert job_log._format_field("k", "v") == "  k=v"

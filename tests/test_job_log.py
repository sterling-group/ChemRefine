"""Tests for the ``chemrefine.job_log`` taxonomy emitter."""

from __future__ import annotations

import re
from pathlib import Path

from chemrefine import ids, job_log, slurm
from chemrefine.state import RunBlock

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


# ---------------------------------------------------------------------------
# The module docstring names a path and a glob — both have to be real
# ---------------------------------------------------------------------------


def test_the_documented_runlog_path_and_glob_are_both_real(tmp_path: Path):
    """The docstring tells a maintainer where runlogs are and how to grep them; check both.

    A drift guard, and it exists because the claim had already drifted: the module
    documented `<step_dir>/step{N}_structure_{ID}.runlog` and a matching glob long after 2.0
    dropped `structure_` from every artifact name, so the published API page (this docstring
    is rendered by mkdocstrings) sent readers to a pattern matching nothing.

    Two assertions, because the claim has two halves and the *location* half is the one that
    drifted. The path is derived the way production derives it —
    `_execution._submit_one` passes `job_name=inp.stem` and `output_dir=out.parent` to
    `build_script`, which writes `#SBATCH --output=` — and the glob is then run as a **real
    glob against a real file**. `fnmatch` is not usable here: its `*` crosses `/`, so a
    pattern one directory level short (`outputs/step*.runlog`) matches a nested path that no
    shell would ever find.
    """
    step_dir = tmp_path / "outputs" / "step1_screen"
    inp = ids.structure_artifact_path(step_dir, 1, "0", "inp")
    out = ids.structure_artifact_path(step_dir, 1, "0", "out")
    inp.parent.mkdir(parents=True)
    inp.write_text("x\n", encoding="utf-8")
    header = tmp_path / "cpu.slurm.header"
    header.write_text("#!/bin/bash\n#SBATCH --time=1\n", encoding="utf-8")

    script = slurm.build_script(
        job_name=inp.stem,
        ntasks=1,
        cpus_per_task=1,
        template_path=header,
        script_path=inp.with_suffix(".slurm"),
        input_path=inp,
        output_dir=out.parent,
        scratch_dir=None,
        run_block=RunBlock(body="true"),
        engine="orca",
        operation="sp",
        step=1,
        structure_id="0",
        step_label="step1_screen",
        output_globs=("*.out",),
    )
    emitted = re.search(r'#SBATCH --output="([^"]+)"', script.read_text())
    assert emitted is not None
    runlog = Path(emitted.group(1))
    runlog.write_text("", encoding="utf-8")  # so the glob below has something to find

    doc = job_log.__doc__ or ""

    # Half one — the *location*: the runlog sits in the structure's own directory, which is
    # exactly what the stale docstring got wrong.
    assert runlog.parent == step_dir / "0"
    assert runlog.name == "step1_0.runlog"
    assert "<step_dir>/<structure_id>/step{N}_{structure_id}.runlog" in doc

    # Half two — the *glob*, run for real. A pattern of the wrong depth finds nothing.
    pattern = re.search(r"``outputs/(\S*\.runlog)``", doc)
    assert pattern is not None, "the docstring no longer names a runlog glob"
    found = set((tmp_path / "outputs").glob(pattern.group(1)))
    assert runlog in found, f"the documented glob outputs/{pattern.group(1)} does not find {runlog}"

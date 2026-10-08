"""Tests for the ``chemrefine.job_log`` taxonomy emitter."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fake_engine import leave_runlog

from chemrefine import ids, job_log, slurm
from chemrefine.errors import ConfigError
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
    # The roster loop above derives its expectation from the same _HEADER_KEYS constant
    # the emitter iterates, so a key deleted from the tuple vanishes from both sides in
    # one edit. The four rows no other test pins are held here as literals — value and
    # all — so the runlog corpus cannot silently lose them. (`mode=` alone would even be
    # satisfied by the `__cr_mode=slurm` probe line with the field row gone.)
    assert "  host=$(hostname)" in snippet
    assert "  job_id=${SLURM_JOB_ID:-$$}" in snippet
    assert "  mode=$__cr_mode" in snippet
    assert "  scratch=$WORK_DIR" in snippet
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


def test_a_field_value_that_would_run_as_bash_is_refused_naming_the_field():
    """The header is an unquoted heredoc: an extra-field value IS interpolated as bash.

    A ``$(...)`` in one executed on the compute node when the job ran — proven with a
    script assembled by the real builder — and the shell-safety table exempted these rows
    under the wrong claim that they were "not interpolated as code". The check lives at
    emission, so a row added by any engine is covered by existing; the refusal is a
    ConfigError because the text originates in the user's YAML.
    """
    with pytest.raises(ConfigError, match="runlog field 'started_from'"):
        job_log.bash_header(
            engine="mlip-train",
            operation="",
            step=1,
            structure_id="train",
            step_label="step1",
            step_dir=Path("/abs/outputs/step1"),
            cores=4,
            extra_fields=(("started_from", "x$(touch pwned)"),),
        )


def test_an_operation_that_would_run_as_bash_is_refused_where_it_is_written():
    """The operation lands in the same heredoc as the rows. A declared one is refused when
    the config loads; an inferred one comes from an engine's inspector, and this is the
    ask it meets."""
    with pytest.raises(ConfigError, match="runlog field 'operation'"):
        job_log.bash_header(
            engine="orca",
            operation="opt_sp$(touch pwned)",
            step=1,
            structure_id="0",
            step_label="step1",
            step_dir=Path("/abs/outputs/step1"),
            cores=4,
        )


def test_field_values_every_engine_actually_emits_still_pass():
    """The emission check must not refuse what the engines legitimately record.

    The rows in circulation: ORCA's validated executable path, Q-Chem's raw recorded
    path, mlip-train's registry key / int / model name — plain text with spaces at most,
    which the rule deliberately allows.
    """
    snippet = job_log.bash_header(
        engine="mlip-train",
        operation="",
        step=1,
        structure_id="train",
        step_label="step1",
        step_dir=Path("/abs/outputs/step1"),
        cores=4,
        extra_fields=(
            ("trainer", "mace_off"),
            ("gpus", 2),
            ("started_from", "/opt/my models/uma-s-1p2"),
        ),
    )
    assert "  gpus=2" in snippet
    assert "  started_from=/opt/my models/uma-s-1p2" in snippet


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
        slurm_header=slurm.read_header(header),
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
    emitted = re.search(r'#SBATCH --output="([^"]+)"', script.read_text(encoding="utf-8"))
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


# ---------------------------------------------------------------------------
# The footer read back: a job's own verdict
# ---------------------------------------------------------------------------


def _job(tmp_path: Path) -> tuple[Path, Path, str]:
    """A prepared job's triple, for the stand-in that writes the runlog its script would."""
    return (tmp_path / "step1_0.inp", tmp_path / "step1_0.out", "0")


def test_a_jobs_runlog_is_named_after_the_job_in_the_directory_it_writes_into(tmp_path: Path):
    assert job_log.runlog_path(tmp_path / "0", "step1_0") == tmp_path / "0" / "step1_0.runlog"


@pytest.mark.parametrize(
    ("exit_code", "scratch_kept"),
    [(0, False), (3, False), (0, True), (143, True)],
    ids=["ran to its end", "failed", "results left in scratch", "stopped, results left"],
)
def test_the_footer_a_script_wrote_is_read_back_as_its_verdict(
    tmp_path: Path, exit_code: int, scratch_kept: bool
):
    """Written by the snippets every generated script embeds, read by the one reader."""
    runlog = leave_runlog(_job(tmp_path), exit_code=exit_code, scratch_kept=scratch_kept)

    assert job_log.read_footer(runlog) == job_log.Footer(exit_code, scratch_kept)


def _whole(tmp_path: Path) -> str:
    """The text of a runlog whose job ran to its end."""
    return leave_runlog(_job(tmp_path)).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "damage",
    [
        lambda text: None,
        lambda text: text[: text.rindex("scratch_kept=")],
        lambda text: text.replace("exit_code=0", "exit_code=zero"),
        lambda text: text.replace("scratch_kept=false", "scratch_kept=maybe"),
        lambda text: text + text[: text.index("finished")].rsplit("[", 1)[0],
        lambda text: text[: text.index("  exit_code=")].rsplit("\n", 2)[0] + "\n",
    ],
    ids=[
        "no runlog",
        "footer cut short",
        "an exit code that is no number",
        "a scratch_kept that is no boolean",
        "a later run that wrote no footer",
        "killed before its footer",
    ],
)
def test_a_runlog_without_a_whole_footer_of_its_last_run_gives_no_verdict(tmp_path: Path, damage):
    """``None`` is no verdict, and never success.

    A scheduler that appends to a job's output file leaves an earlier run's footer above a
    later run's header, and a later run that was killed wrote none of its own: the footer
    above it says nothing about what is on disk.
    """
    text = damage(_whole(tmp_path))
    runlog = tmp_path / "step1_0.runlog"
    if text is None:
        runlog.unlink()
    else:
        runlog.write_text(text, encoding="utf-8")

    assert job_log.read_footer(runlog) is None


def test_a_runlog_that_cannot_be_read_gives_no_verdict(tmp_path: Path):
    """A directory where the runlog belongs is refused by the read, and reads as no verdict."""
    (tmp_path / "step1_0.runlog").mkdir()

    assert job_log.read_footer(tmp_path / "step1_0.runlog") is None


def test_the_footer_is_found_under_a_log_far_longer_than_what_is_read(tmp_path: Path):
    """Only the end of the file is read; a training's log above the footer can be huge."""
    whole = _whole(tmp_path)
    runlog = tmp_path / "step1_0.runlog"
    banner, rest = whole.split("\n", 1)
    runlog.write_text(banner + "\n" + "epoch 1 loss 0.5\n" * 20_000 + rest, encoding="utf-8")
    assert runlog.stat().st_size > 4 * job_log._TAIL_BYTES

    assert job_log.read_footer(runlog) == job_log.Footer(exit_code=0, scratch_kept=False)


def test_a_footer_joined_to_a_programs_unterminated_last_line_is_still_read(tmp_path: Path):
    """A program whose last output carries no newline leaves the footer's line joined to it."""
    whole = _whole(tmp_path)
    head, footer = whole[: whole.rindex("[")], whole[whole.rindex("[") :]
    runlog = tmp_path / "step1_0.runlog"
    runlog.write_text(head + "last words with no newline" + footer, encoding="utf-8")

    assert job_log.read_footer(runlog) == job_log.Footer(exit_code=0, scratch_kept=False)

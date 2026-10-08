"""Tests for SLURM script generation, sbatch submission, and squeue polling."""

from __future__ import annotations

import errno
import logging
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Collection
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chemrefine import job_log, slurm
from chemrefine.errors import ConfigError, JobSubmissionError, OutputTreeError, ThrottleTimeoutError
from chemrefine.slurm import dispatch
from chemrefine.state import RunBlock
from chemrefine.throttle import GpuBudget


def _drain_local(job_id: str, *, timeout: float = 5.0) -> None:
    """Block until a background ``local-N`` job is reaped (or the timeout)."""
    deadline = time.monotonic() + timeout
    while job_id not in slurm.finished_jobs([job_id]) and time.monotonic() < deadline:
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
        "#SBATCH --ntasks=1\n"  # gets overridden
        "#SBATCH --cpus-per-task=4\n"  # gets overridden
        "module load orca/6.0\n",
        encoding="utf-8",
    )
    return header


def _build_kwargs(tmp_path: Path, **overrides):
    """Default kwargs for ``slurm.build_script`` tests."""
    base = {
        "job_name": "step1_structure_0",
        "ntasks": 1,
        "slurm_header": slurm.read_header(_write_header(tmp_path)),
        "script_path": tmp_path / "out" / "step1_structure_0.slurm",
        "input_path": tmp_path / "in" / "step1_structure_0.inp",
        "output_dir": tmp_path / "out",
        "scratch_dir": tmp_path / "scratch",
        "run_block": RunBlock(body="echo hi"),
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
            ntasks=12,
            run_block=RunBlock(
                body="$ORCA step1_structure_0.inp > $OUTPUT_DIR/step1_structure_0.out"
            ),
        )
    )
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "#SBATCH --partition=normal" in text
    assert "#SBATCH --time=24:00:00" in text
    assert "#SBATCH --ntasks=12" in text
    assert "#SBATCH --cpus-per-task=1" in text
    # The header's own --ntasks=1 must not survive. Probed with its trailing newline:
    # "--ntasks=1" bare is a substring of the re-added "--ntasks=12", so the old
    # disjunction here was implied by the assertion above it and could never fail.
    assert "#SBATCH --ntasks=1\n" not in text
    assert "module load orca/6.0" in text


@pytest.mark.parametrize("field", ["output_dir", "scratch_dir"])
def test_a_builder_refuses_a_resolved_path_it_would_interpolate(tmp_path: Path, field: str):
    """The rule is asked on the string bash receives, by both builders.

    Load-time passes see a path as written and as anchored; the builders see the resolved
    one, which is what the script exports. A refused character here is a `ConfigError`
    naming the path — the exit-code contract, not a `ValueError` out of script assembly.
    """
    hostile = tmp_path / "run$1" / field
    with pytest.raises(ConfigError, match=rf"resolved {field}.*\$"):
        slurm.build_script(**_build_kwargs(tmp_path, **{field: hostile}))
    with pytest.raises(ConfigError, match=rf"resolved {field}.*\$"):
        slurm.build_array_script(
            step_label="step1_refine",
            ntasks=1,
            slurm_header=slurm.read_header(_write_header(tmp_path)),
            script_path=tmp_path / "out" / "step1_array.slurm",
            output_dir=tmp_path / "out" if field != "output_dir" else hostile,
            scratch_dir=tmp_path / "scratch" if field != "scratch_dir" else hostile,
            run_block=RunBlock(body="echo hi"),
            engine="orca",
            operation="opt_sp",
            step=1,
            output_globs=("*.out",),
        )


def test_a_symlink_target_passes_both_loaders_and_is_refused_at_render(tmp_path: Path):
    """The route the load-time passes cannot see, closed where the path becomes bash.

    `output_dir: ./outputs` is clean as written and clean once anchored to the config's
    directory — `load_config` validates both readings and refuses nothing — but `outputs` is
    a symlink whose target carries `$`, and the step directory the script exports is the
    resolved one. The builder refuses that string; without it the script would export
    `…/run$1/outputs/step1/…`, and `$1` would expand to nothing on the node.
    """
    import yaml

    from chemrefine.config import load_config
    from chemrefine.step import step_dir_for

    target = tmp_path / "run$1" / "outputs"
    target.mkdir(parents=True)
    project = tmp_path / "proj"
    project.mkdir()
    (project / "outputs").symlink_to(target)
    config_path = project / "input.yaml"
    config_path.write_text(
        yaml.safe_dump({"output_dir": "./outputs", "steps": [{"step": 1, "engine": "orca"}]}),
        encoding="utf-8",
    )

    config = load_config(config_path)  # as written and as anchored: nothing to refuse
    step_dir = step_dir_for(config, config.steps[0])
    assert "$1" in str(step_dir)  # what the script would export

    with pytest.raises(ConfigError, match=r"resolved output_dir.*\$"):
        slurm.build_script(**_build_kwargs(tmp_path, output_dir=step_dir))


def test_a_shared_sbatch_line_keeps_its_unowned_directives(tmp_path: Path):
    """Stripping an owned option must not take its line-mates with it.

    sbatch accepts several options per ``#SBATCH`` line, and dropping the whole line
    whenever one of ours appeared on it silently discarded the co-resident directives —
    a header's ``--time`` vanished because ``--output`` shared its line. The owned
    option goes; whatever else the line declares stays; a line that held only owned
    options disappears entirely, as it always did.
    """
    header = tmp_path / "multi.slurm.header"  # not the name _build_kwargs re-writes
    header.write_text(
        "#!/bin/bash\n"
        "#SBATCH --time=24:00:00 --output=old.log\n"
        "#SBATCH --qos=high --ntasks 4 --partition=normal\n"
        "#SBATCH --ntasks=1 --cpus-per-task=4\n",  # only owned options — the line goes
        encoding="utf-8",
    )
    script = slurm.build_script(
        **_build_kwargs(tmp_path, slurm_header=slurm.read_header(header), ntasks=2)
    )
    text = script.read_text(encoding="utf-8")
    assert "--time=24:00:00" in text
    assert "--output=old.log" not in text
    assert "--qos=high" in text and "--partition=normal" in text
    assert "--ntasks 4" not in text
    # `--ntasks-per-node` and friends must still pass whole — the prefix guard holds
    # in the stripping regex exactly as it did in the old whole-line test below.
    assert "#SBATCH --ntasks=1 --cpus-per-task=4" not in text
    assert "#SBATCH --ntasks=2" in text


def test_a_shared_mem_line_keeps_its_unowned_directives(tmp_path: Path):
    """The memory replacement obeys the same rule as the header strip.

    `_apply_memory` replaces a too-small grant; a `--mem` sharing a line with another
    directive must lose only the memory half.
    """
    header = tmp_path / "shared-mem.slurm.header"  # not the name _build_kwargs re-writes
    header.write_text(
        "#!/bin/bash\n#SBATCH --mem=1 --qos=high\n",
        encoding="utf-8",
    )
    script = slurm.build_script(
        **_build_kwargs(tmp_path, slurm_header=slurm.read_header(header), memory_mb=4000)
    )
    text = script.read_text(encoding="utf-8")
    assert "--qos=high" in text
    assert "--mem=1" not in text
    assert "#SBATCH --mem-per-cpu=4000" in text


def test_build_script_spells_a_threads_layout(tmp_path: Path):
    """``cpus_per_task`` reaches the SBATCH pair, and the runlog reports the product.

    The pair is an engine's ``slurm_layout``: MPI ranks are ``(pal, 1)``, one threaded
    process is ``(1, threads)`` — N single-cpu tasks can be granted across nodes, where a
    threaded program can only use the first node's share.
    """
    script = slurm.build_script(**_build_kwargs(tmp_path, ntasks=1, cpus_per_task=8))
    text = script.read_text(encoding="utf-8")
    assert "#SBATCH --ntasks=1" in text
    assert "#SBATCH --cpus-per-task=8" in text
    assert "cores=8" in text


def _write_header_with(tmp_path: Path, *directives: str) -> Path:
    """A header template carrying extra ``#SBATCH`` lines (for the memory tests)."""
    header = tmp_path / "mem.slurm.header"
    header.write_text(
        "#!/bin/bash\n#SBATCH --partition=normal\n"
        + "".join(f"#SBATCH {d}\n" for d in directives)
        + "module load orca/6.0\n",
        encoding="utf-8",
    )
    return header


def test_no_memory_declaration_leaves_the_header_alone(tmp_path: Path):
    """Engines that declare nothing get exactly the header's memory policy, untouched."""
    header = _write_header_with(tmp_path, "--mem-per-cpu=1000")
    script = slurm.build_script(**_build_kwargs(tmp_path, slurm_header=slurm.read_header(header)))
    text = script.read_text(encoding="utf-8")
    assert "#SBATCH --mem-per-cpu=1000" in text
    assert text.count("--mem-per-cpu") == 1


@pytest.mark.parametrize(
    ("override", "gpus", "expected"),
    [
        ("special.header", 3, "special.header"),  # a per-step header wins over everything
        ("special.header", 0, "special.header"),
        (None, 1, "cuda.slurm.header"),  # GPU demand picks the cuda header
        ("", 1, "cuda.slurm.header"),  # an empty override is no override
        (None, 0, "cpu.slurm.header"),  # else the workflow's own
    ],
)
def test_header_name_for_step_is_the_one_rule(override: str | None, gpus: int, expected: str):
    """Override, then GPU demand, then the default — spelled once, asked by run and validator."""
    assert slurm.header_name_for_step(override, gpus=gpus, default="cpu.slurm.header") == expected


def test_a_single_node_job_pins_the_node_and_a_headers_own_count_yields(tmp_path: Path):
    """``single_node`` writes ``--nodes=1`` and strips the header's own; ``False`` touches nothing.

    The directive is the script's rather than every header's because a header that forgot
    it produced a job wrong only on the partitions that split it. Stripped in place, not by
    line: a ``--nodes=4`` sharing a line with ``--time`` must lose only the node count.
    """
    header = _write_header_with(tmp_path, "--nodes=4 --time=01:00:00", "-N 2")
    pinned = slurm.build_script(
        **_build_kwargs(tmp_path, slurm_header=slurm.read_header(header), single_node=True)
    ).read_text(encoding="utf-8")
    assert "#SBATCH --nodes=1" in pinned
    assert "--nodes=4" not in pinned and "-N 2" not in pinned
    assert "#SBATCH --time=01:00:00" in pinned
    assert pinned.count("--nodes") == 1

    left_alone = slurm.build_script(
        **_build_kwargs(tmp_path, slurm_header=slurm.read_header(header))
    ).read_text(encoding="utf-8")
    assert "#SBATCH --nodes=4 --time=01:00:00" in left_alone
    assert "--nodes=1" not in left_alone


def test_a_sufficient_header_memory_allocation_stands(tmp_path: Path):
    """The cluster's own policy wins whenever it covers the input's requirement."""
    header = _write_header_with(tmp_path, "--mem-per-cpu=4000")
    script = slurm.build_script(
        **_build_kwargs(
            tmp_path,
            slurm_header=slurm.read_header(header),
            ntasks=1,
            cpus_per_task=8,
            memory_mb=16000,
        )
    )
    text = script.read_text(encoding="utf-8")
    assert "#SBATCH --mem-per-cpu=4000" in text  # 8 cpus x 4000 = 32000 >= 16000
    assert text.count("--mem-per-cpu") == 1


def test_a_sufficient_total_memory_header_counts_its_units(tmp_path: Path):
    """A ``--mem=64G`` grant is read as 65536 MB, not compared as the bare number 64."""
    header = _write_header_with(tmp_path, "--mem=64G")
    script = slurm.build_script(
        **_build_kwargs(
            tmp_path,
            slurm_header=slurm.read_header(header),
            ntasks=1,
            cpus_per_task=8,
            memory_mb=60000,
        )
    )
    text = script.read_text(encoding="utf-8")
    assert "#SBATCH --mem=64G" in text
    assert "--mem-per-cpu" not in text


def test_a_short_header_memory_allocation_is_extended(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """A header grant below the input's requirement is replaced, loudly; GPU memory is not.

    ``--mem-per-gpu`` only shares a prefix with the directives this owns — stripping it
    would starve a GPU job of the memory its header deliberately granted.
    """
    header = _write_header_with(tmp_path, "--mem-per-cpu=1000", "--mem-per-gpu=8G")
    with caplog.at_level("INFO"):
        script = slurm.build_script(
            **_build_kwargs(
                tmp_path,
                slurm_header=slurm.read_header(header),
                ntasks=1,
                cpus_per_task=8,
                memory_mb=32000,
            )
        )
    text = script.read_text(encoding="utf-8")
    assert "#SBATCH --mem-per-cpu=4000" in text  # ceil(32000 / 8)
    assert "--mem-per-cpu=1000" not in text
    assert "#SBATCH --mem-per-gpu=8G" in text
    assert "32000" in caplog.text and "8000 MB" in caplog.text


def test_an_absent_header_memory_grant_is_requested(tmp_path: Path):
    """With no header memory at all, the input's requirement becomes the request."""
    script = slurm.build_script(**_build_kwargs(tmp_path, memory_mb=1000))
    assert "#SBATCH --mem-per-cpu=1000" in script.read_text(encoding="utf-8")


def test_a_kilobyte_header_grant_is_floored_to_mb(tmp_path: Path):
    """``2048K`` reads as 2 MB — flooring understates the grant, which only ever extends."""
    header = _write_header_with(tmp_path, "--mem-per-cpu=2048K")
    script = slurm.build_script(
        **_build_kwargs(tmp_path, slurm_header=slurm.read_header(header), memory_mb=2)
    )
    text = script.read_text(encoding="utf-8")
    assert "#SBATCH --mem-per-cpu=2048K" in text  # 2 MB covers the 2 MB requirement
    assert text.count("--mem-per-cpu") == 1


def test_a_kilobyte_header_grant_is_not_read_as_megabytes(tmp_path: Path):
    """The same ``2048K`` grant must be *extended* against a 3 MB requirement.

    The stands-untouched case above pins the unit only from below: a mutant reading
    ``2048K`` as 2048 MB still satisfies ``granted >= 2`` and passes it identically —
    while a real header granting kilobytes would wrongly stand against any requirement
    and the job would OOM. This side can only pass when K genuinely reads as 2 MB.
    """
    header = _write_header_with(tmp_path, "--mem-per-cpu=2048K")
    script = slurm.build_script(
        **_build_kwargs(tmp_path, slurm_header=slurm.read_header(header), memory_mb=3)
    )
    text = script.read_text(encoding="utf-8")
    assert "2048K" not in text, "a 2 MB grant cannot cover 3 MB and must be replaced"
    assert "#SBATCH --mem-per-cpu=3" in text  # ceil(3 / (1 task x 1 cpu))


def test_a_whole_node_grant_satisfies_any_requirement(tmp_path: Path):
    """``--mem=0`` is SLURM's "all the node's memory" — never extended, whatever is asked."""
    header = _write_header_with(tmp_path, "--mem=0")
    script = slurm.build_script(
        **_build_kwargs(tmp_path, slurm_header=slurm.read_header(header), memory_mb=999999)
    )
    text = script.read_text(encoding="utf-8")
    assert "#SBATCH --mem=0" in text
    assert "--mem-per-cpu" not in text


def test_build_script_keeps_ntasks_per_node_directive(tmp_path: Path):
    """``--ntasks-per-*`` directives only share a prefix with ``--ntasks`` — keep them.

    Substring matching on ``--ntasks`` would strip a
    cluster header's ``--ntasks-per-node`` from every generated script.
    """
    header = tmp_path / "per_node.slurm.header"
    header.write_text(
        "#!/bin/bash\n"
        "#SBATCH --ntasks-per-node=16\n"
        "#SBATCH --ntasks-per-core=1\n"
        "#SBATCH --ntasks 1\n"  # space-separated form of an owned flag
        "#SBATCH --output=old.log\n",
        encoding="utf-8",
    )
    script = slurm.build_script(
        **_build_kwargs(tmp_path, slurm_header=slurm.read_header(header), ntasks=8)
    )
    text = script.read_text(encoding="utf-8")
    assert "#SBATCH --ntasks-per-node=16" in text
    assert "#SBATCH --ntasks-per-core=1" in text
    assert "#SBATCH --ntasks 1" not in text
    assert "#SBATCH --ntasks=8" in text
    assert "--output=old.log" not in text


def test_build_script_emits_absolute_runlog_output_directives(tmp_path: Path):
    """``#SBATCH --output`` / ``--error`` must point at absolute paths in the step dir."""
    out = (tmp_path / "out").resolve()
    script = slurm.build_script(**_build_kwargs(tmp_path, output_dir=out))
    text = script.read_text(encoding="utf-8")
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
    text = script.read_text(encoding="utf-8")
    assert "engine=mlff" in text
    assert "operation=opt_sp" in text
    assert "step=2" in text
    assert "structure_id=0" in text
    assert "ChemRefine mlff step2_refine starting" in text
    assert "ChemRefine mlff step2_refine finished" in text


def test_build_script_includes_run_block(tmp_path: Path):
    script = slurm.build_script(
        **_build_kwargs(tmp_path, run_block=RunBlock(body="echo CUSTOM_RUN_BLOCK_HERE"))
    )
    assert "echo CUSTOM_RUN_BLOCK_HERE" in script.read_text(encoding="utf-8")


def test_build_script_auto_scratch_under_output_dir_when_none(tmp_path: Path):
    """Omitting scratch_dir makes WORK_DIR a sibling of step output."""
    out = (tmp_path / "out").resolve()
    script = slurm.build_script(**_build_kwargs(tmp_path, scratch_dir=None, output_dir=out))
    text = script.read_text(encoding="utf-8")
    assert f'export WORK_DIR="{out}/_work_' in text


def test_build_script_explicit_scratch_uses_chemrefine_subdir(tmp_path: Path):
    scratch = (tmp_path / "scratch").resolve()
    script = slurm.build_script(**_build_kwargs(tmp_path, scratch_dir=scratch))
    assert f'export WORK_DIR="{scratch}/ChemRefine_' in script.read_text(encoding="utf-8")


def test_build_script_removes_the_scratch_directory_once_its_results_are_home(tmp_path: Path):
    """No knob keeps scratch: it goes unless a result could not be copied home.

    Keeping an artifact is the engine's to ask for (`RunBlock.cleanup`, `output_dirs`),
    which the array path honours too. The one thing that leaves scratch behind is a copy
    the exit handler could not make, since scratch then holds the only whole copy.
    """
    text = slurm.build_script(**_build_kwargs(tmp_path)).read_text(encoding="utf-8")
    assert '$scratch_kept || { cd "$OUTPUT_DIR" && rm -rf "$WORK_DIR"; }' in text


def test_build_script_emits_exit_trap(tmp_path: Path):
    """The footer must be wired through a trap so it fires on failure too."""
    script = slurm.build_script(**_build_kwargs(tmp_path))
    text = script.read_text(encoding="utf-8")
    assert "trap _on_exit EXIT" in text


def test_build_script_copies_output_dirs_back(tmp_path: Path):
    """``output_dirs`` (e.g. pyscf ``tensors/``) are copied back wholesale on exit."""
    script = slurm.build_script(**_build_kwargs(tmp_path, output_dirs=("tensors",)))
    text = script.read_text(encoding="utf-8")
    assert (
        'if [ -e "tensors" ]; then cp -r -- "tensors" "$OUTPUT_DIR/" || scratch_kept=true; fi'
        in text
    )


def test_a_missing_header_is_refused_where_it_is_read(tmp_path: Path):
    with pytest.raises(ConfigError, match="SLURM header template not found"):
        slurm.read_header(tmp_path / "missing.header")


def test_a_header_is_parsed_into_what_every_script_of_a_batch_shares(tmp_path: Path):
    """The owned options leave; the rest stays, in order, as tuples no script can extend."""
    assert slurm.read_header(_write_header(tmp_path)) == slurm.SlurmHeader(
        sbatch=("#SBATCH --partition=normal", "#SBATCH --time=24:00:00"),
        body=("#!/bin/bash", "module load orca/6.0"),
    )


def test_scripts_built_from_one_parse_carry_only_their_own_directives(tmp_path: Path):
    """One parse serves a whole batch, so no script may carry another's job name or memory."""
    header = slurm.read_header(_write_header(tmp_path))
    texts = [
        slurm.build_script(
            **_build_kwargs(
                tmp_path,
                slurm_header=header,
                memory_mb=4000,
                job_name=f"step1_structure_{i}",
                script_path=tmp_path / "out" / f"step1_structure_{i}.slurm",
            )
        ).read_text(encoding="utf-8")
        for i in (0, 1)
    ]
    for text in texts:
        assert text.count("#SBATCH --job-name=") == 1
        assert text.count("--mem-per-cpu") == 1
    assert "#SBATCH --job-name=step1_structure_1" in texts[1]


def test_build_script_uses_caller_supplied_output_globs(tmp_path: Path):
    """The copy-back loops over the engine's declared file extensions."""
    script = slurm.build_script(**_build_kwargs(tmp_path, output_globs=("*.json", "*.npz")))
    text = script.read_text(encoding="utf-8")
    assert "for item in *.json *.npz; do" in text
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
    text = script.read_text(encoding="utf-8")
    assert "orca_executable=/opt/orca/orca" in text


def test_build_script_without_extra_header_fields_is_engine_neutral(tmp_path: Path):
    """Default (no extra fields) emits no engine-specific rows."""
    script = slurm.build_script(**_build_kwargs(tmp_path))
    assert "orca_executable=" not in script.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# submit (sbatch path)
# ---------------------------------------------------------------------------


def test_submit_parses_job_id_from_sbatch_output():
    """The human `Submitted batch job N` line still parses — the fallback for a site
    wrapper that swallows `--parsable` — and the flag itself is requested."""
    fake = MagicMock(returncode=0, stdout="Submitted batch job 12345\n", stderr="")
    with (
        patch("chemrefine.slurm.dispatch.shutil.which", return_value="/usr/bin/sbatch"),
        patch.object(subprocess, "run", return_value=fake) as run,
    ):
        assert slurm.submit("script.slurm") == "12345"
    assert "--parsable" in run.call_args[0][0]


def test_submit_takes_the_parsable_id_not_the_first_integer():
    """A wrapper banner with a number in it must not become the job id: the old
    unanchored search read this stdout as job 90, so the throttler polled a job that
    didn't exist and the whole batch was ledgered as missing output."""
    fake = MagicMock(returncode=0, stdout="sbatch: 90% of quota used\n12345;cluster\n", stderr="")
    with (
        patch("chemrefine.slurm.dispatch.shutil.which", return_value="/usr/bin/sbatch"),
        patch.object(subprocess, "run", return_value=fake),
    ):
        assert slurm.submit("script.slurm") == "12345"


def test_submit_raises_on_sbatch_failure_with_stderr_and_hint():
    """sbatch's own diagnostic and the `dispatch: local` escape hatch reach the user."""
    err = subprocess.CalledProcessError(1, ["sbatch"], stderr="permission denied")
    with (
        patch("chemrefine.slurm.dispatch.shutil.which", return_value="/usr/bin/sbatch"),
        patch.object(subprocess, "run", side_effect=err),
        pytest.raises(JobSubmissionError, match="permission denied") as excinfo,
    ):
        slurm.submit("script.slurm")
    assert "dispatch: local" in str(excinfo.value)


def test_submit_raises_when_output_lacks_job_id():
    fake = MagicMock(returncode=0, stdout="weird output\n", stderr="")
    with (
        patch("chemrefine.slurm.dispatch.shutil.which", return_value="/usr/bin/sbatch"),
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
    with patch("chemrefine.slurm.dispatch.shutil.which", return_value=None):
        job_id = slurm.submit(script)
    assert job_id.startswith("local-")
    _drain_local(job_id)
    assert job_id in slurm.finished_jobs([job_id])


def test_submit_local_writes_runlog_and_err_alongside_script(tmp_path: Path):
    """The background local run redirects stdout/stderr to the SBATCH .runlog / .err paths."""
    script = tmp_path / "step1_structure_0.slurm"
    script.write_text(
        "#!/bin/bash\necho 'hello from the template'\necho 'warning: backend X' >&2\n",
        encoding="utf-8",
    )
    with patch("chemrefine.slurm.dispatch.shutil.which", return_value=None):
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
    with patch("chemrefine.slurm.dispatch.shutil.which", return_value=None):
        j1 = slurm.submit(s1)
        j2 = slurm.submit(s2)
    # Right after submit, both are still running → they run in parallel.
    assert j1 not in slurm.finished_jobs([j1])
    assert j2 not in slurm.finished_jobs([j2])
    _drain_local(j1)
    _drain_local(j2)
    assert j1 in slurm.finished_jobs([j1])
    assert j2 in slurm.finished_jobs([j2])


def test_submit_local_failure_is_not_raised_but_recorded_on_disk(tmp_path: Path):
    """A non-zero local exit does not raise at submit; the logs land on disk so the
    failure surfaces through the engine's output parsing, like a SLURM one."""
    script = tmp_path / "script.slurm"
    script.write_text(
        "#!/bin/bash\necho 'partial stdout'\necho 'boom' >&2\nexit 2\n", encoding="utf-8"
    )
    with patch("chemrefine.slurm.dispatch.shutil.which", return_value=None):
        job_id = slurm.submit(script)  # does not raise
    _drain_local(job_id)
    assert job_id in slurm.finished_jobs([job_id])
    assert script.with_suffix(".runlog").read_text(encoding="utf-8") == "partial stdout\n"
    assert script.with_suffix(".err").read_text(encoding="utf-8") == "boom\n"


def test_submit_local_closes_handles_and_reraises_when_spawn_fails(tmp_path: Path):
    """If spawning the background job fails, both log handles are closed and the error
    propagates — no leaked file descriptors, no half-registered job."""
    script = tmp_path / "boom.slurm"
    script.write_text("#!/bin/bash\ntrue\n", encoding="utf-8")
    opened: list = []
    real_open = Path.open

    def spy_open(self, *args, **kwargs):
        handle = real_open(self, *args, **kwargs)
        opened.append(handle)
        return handle

    before = set(dispatch._LOCAL_PROCS)
    with (
        patch.object(Path, "open", spy_open),
        patch.object(dispatch.subprocess, "Popen", side_effect=OSError("cannot spawn")),
        pytest.raises(OSError, match="cannot spawn"),
    ):
        dispatch._submit_local(script)
    assert len(opened) == 2 and all(handle.closed for handle in opened)
    assert set(dispatch._LOCAL_PROCS) == before  # no half-registered job


def test_submit_local_closes_the_first_log_when_the_second_cannot_be_opened(tmp_path: Path):
    """The `.err` open failing must not strand the `.runlog` opened one line earlier.

    Both handles were acquired before the guard that closed them, so only the *third*
    failure point — the spawn — was covered. A full disk or a revoked directory takes the
    second `open` instead, and there the first handle had no owner: CPython's refcount
    closed it on the way out, but only after a `ResourceWarning`, which the suite treats as
    an error wherever it is raised.
    """
    script = tmp_path / "half.slurm"
    script.write_text("#!/bin/bash\ntrue\n", encoding="utf-8")
    opened: list = []
    real_open = Path.open

    def open_runlog_only(self, *args, **kwargs):
        if self.suffix == ".err":
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), str(self))
        handle = real_open(self, *args, **kwargs)
        opened.append(handle)
        return handle

    before = set(dispatch._LOCAL_PROCS)
    with (
        patch.object(Path, "open", open_runlog_only),
        pytest.raises(OutputTreeError) as refusal,
    ):
        dispatch._submit_local(script)
    assert str(refusal.value) == f"cannot write {tmp_path / 'half.err'}: No space left on device"
    assert len(opened) == 1 and opened[0].closed, "the .runlog handle outlived its function"
    assert set(dispatch._LOCAL_PROCS) == before


# ---------------------------------------------------------------------------
# finished_jobs: one job's answer
# ---------------------------------------------------------------------------


def test_finished_jobs_passes_noheader_to_squeue():
    """``--noheader`` must be requested, not assumed.

    The output is consumed line-for-line with no header slice, so the flag is
    what keeps ``JOBID`` out of the job-id list. See
    :func:`test_finished_jobs_false_for_the_only_queued_job`.
    """
    fake = MagicMock(returncode=0, stdout="", stderr="")
    with patch.object(subprocess, "run", return_value=fake) as run:
        slurm.finished_jobs(["12345"])
    assert "--noheader" in run.call_args.args[0]


def test_finished_jobs_false_for_the_only_queued_job():
    """The sole queued job must not be mistaken for a header row.

    The previous implementation dropped ``lines[0]`` as a header. On a site whose
    ``squeue`` already suppresses the header, that discarded the first *real*
    job id — reporting a still-running job as finished, which then parsed as a
    missing-output failure.
    """
    fake = MagicMock(returncode=0, stdout="12345\n", stderr="")
    with patch.object(subprocess, "run", return_value=fake):
        assert "12345" not in slurm.finished_jobs(["12345"])


def test_finished_jobs_true_when_queue_is_empty():
    """An empty queue (no header, no rows) means every job is finished."""
    for stdout in ("", "\n"):
        fake = MagicMock(returncode=0, stdout=stdout, stderr="")
        with patch.object(subprocess, "run", return_value=fake):
            assert "12345" in slurm.finished_jobs(["12345"])


def test_finished_jobs_true_when_job_absent():
    fake = MagicMock(returncode=0, stdout="9999\n", stderr="")
    with patch.object(subprocess, "run", return_value=fake):
        assert "12345" in slurm.finished_jobs(["12345"])


def test_finished_jobs_false_when_job_still_running():
    fake = MagicMock(returncode=0, stdout="9999\n12345\n", stderr="")
    with patch.object(subprocess, "run", return_value=fake):
        assert "12345" not in slurm.finished_jobs(["12345"])


def test_finished_jobs_false_while_array_tasks_run():
    """An array's parent id never appears bare in squeue — running tasks print
    as ``12345_0`` and pending ones as ``12345_[5-99%4]``; both must count."""
    fake = MagicMock(returncode=0, stdout="12345_3\n", stderr="")
    with patch.object(subprocess, "run", return_value=fake):
        assert "12345" not in slurm.finished_jobs(["12345"])
    fake = MagicMock(returncode=0, stdout="12345_[4-99%4]\n", stderr="")
    with patch.object(subprocess, "run", return_value=fake):
        assert "12345" not in slurm.finished_jobs(["12345"])


def test_finished_jobs_array_prefix_does_not_match_other_jobs():
    """``123`` must not be held back by an unrelated ``1234`` or ``1234_0``."""
    fake = MagicMock(returncode=0, stdout="1234\n1234_0\n", stderr="")
    with patch.object(subprocess, "run", return_value=fake):
        assert "123" in slurm.finished_jobs(["123"])


def test_finished_jobs_treats_squeue_failure_as_not_finished():
    err = subprocess.CalledProcessError(1, ["squeue"])
    with patch.object(subprocess, "run", side_effect=err):
        assert "12345" not in slurm.finished_jobs(["12345"])


def test_finished_jobs_true_for_local_job_id():
    """Unknown / already-reaped local IDs are reported finished without touching squeue."""
    with patch.object(subprocess, "run") as run:
        assert "local-9999" in slurm.finished_jobs(["local-9999"])
    run.assert_not_called()


# ---------------------------------------------------------------------------
# current_user
# ---------------------------------------------------------------------------


def test_current_user_falls_back_to_uid_when_no_passwd_entry():
    """``getpass.getuser`` raises in passwd-less environments (containers under
    an arbitrary UID); the numeric UID is an equally valid ``squeue -u`` value.
    Resolved lazily so a failing lookup can't break the package import."""
    import getpass
    import os

    slurm.current_user.cache_clear()
    try:
        with patch.object(getpass, "getuser", side_effect=KeyError("getpwuid(): uid not found")):
            assert slurm.current_user() == str(os.getuid())
    finally:
        slurm.current_user.cache_clear()


def test_current_user_returns_login_name():
    import getpass

    slurm.current_user.cache_clear()
    try:
        with patch.object(getpass, "getuser", return_value="alice"):
            assert slurm.current_user() == "alice"
    finally:
        slurm.current_user.cache_clear()


# ---------------------------------------------------------------------------
# GPU budget + device-aware headers
# ---------------------------------------------------------------------------


def test_resolve_gpu_budget_explicit_value_wins():
    """With no inherited allocation the probe is only a guess, so `max_gpus` overrides it."""
    assert slurm.resolve_gpu_budget(3, local=True) == GpuBudget(3, ("0", "1", "2"))


def test_resolve_gpu_budget_unlimited_under_slurm():
    """No device list under SLURM: `--gres` places the GPU, so chemrefine pins nothing."""
    budget = slurm.resolve_gpu_budget(None, local=False)
    assert budget.count >= 1000
    assert budget.devices == ()


def test_resolve_gpu_budget_uses_detected_devices_locally():
    with patch("chemrefine.slurm.dispatch._detected_devices", return_value=("0", "1")):
        assert slurm.resolve_gpu_budget(None, local=True) == GpuBudget(2, ("0", "1"))


def test_resolve_gpu_budget_honours_an_inherited_allocation(monkeypatch):
    """An inherited CUDA_VISIBLE_DEVICES is what this run owns, not a hint about the host.

    `nvidia-smi -L` reports the whole machine, so on a node that granted this run devices 2
    and 3 the probe said 8 and the throttler pinned jobs to indices 0..7 — six GPUs
    belonging to somebody else. `_submit_local` merges its value *over* the inherited one,
    so nothing downstream would have caught it.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    with patch(
        "chemrefine.slurm.dispatch._detected_devices",
        side_effect=AssertionError("probed the host despite an explicit allocation"),
    ):
        assert slurm.resolve_gpu_budget(None, local=True) == GpuBudget(2, ("2", "3"))


def test_resolve_gpu_budget_keeps_uuid_and_mig_tokens_verbatim(monkeypatch):
    """The variable takes UUIDs and MIG handles, and a MIG instance has no other name."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-8a2f, MIG-3b7c ,")
    assert slurm.resolve_gpu_budget(None, local=True) == GpuBudget(2, ("GPU-8a2f", "MIG-3b7c"))


def test_resolve_gpu_budget_reads_an_empty_allocation_as_no_devices(monkeypatch):
    """`CUDA_VISIBLE_DEVICES=""` is a real answer — no GPUs — not an absent variable."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert slurm.resolve_gpu_budget(None, local=True) == GpuBudget(0, ())


def test_max_gpus_narrows_an_allocation_but_cannot_widen_it(monkeypatch, caplog):
    """`max_gpus` is a cap on what was granted, never a claim on more of it.

    Against a *probe* the knob overrides outright — the probe is a guess. Against an
    allocation it can only take less, because the throttler hands out one token per job and
    there is no token for a device this run was not given.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert slurm.resolve_gpu_budget(1, local=True) == GpuBudget(1, ("2",))
    with caplog.at_level(logging.WARNING, logger="chemrefine.slurm.dispatch"):
        assert slurm.resolve_gpu_budget(4, local=True) == GpuBudget(2, ("2", "3"))
    assert "exceeds" in caplog.text


def test_resolve_gpu_budget_probes_nothing_of_its_own(monkeypatch):
    """It takes the caller's resolved `local`, so it runs no PATH probe of its own.

    The caller has already asked `dispatch_locally` — that decision is what routes the whole
    submission — so a probe here would be a second answer to a settled question, free to
    disagree with the path the batch actually took.
    """
    monkeypatch.setattr(
        dispatch.shutil, "which", MagicMock(side_effect=AssertionError("probed PATH"))
    )
    with patch("chemrefine.slurm.dispatch._detected_devices", return_value=("0", "1")):
        assert slurm.resolve_gpu_budget(None, local=True).count == 2
        assert slurm.resolve_gpu_budget(None, local=False).count >= 1000


# ---------------------------------------------------------------------------
# dispatch_locally — the SLURM-vs-local decision
# ---------------------------------------------------------------------------


def test_dispatch_auto_follows_sbatch_availability():
    with patch("chemrefine.slurm.dispatch.shutil.which", return_value="/usr/bin/sbatch"):
        assert slurm.dispatch_locally("auto") is False
    with patch("chemrefine.slurm.dispatch.shutil.which", return_value=None):
        assert slurm.dispatch_locally("auto") is True


def test_dispatch_local_forces_local_runner_despite_sbatch(tmp_path: Path):
    """A stray sbatch on PATH must not hijack a `dispatch: local` run."""
    script = tmp_path / "script.slurm"
    script.write_text("#!/bin/bash\ntrue\n", encoding="utf-8")
    with patch("chemrefine.slurm.dispatch.shutil.which", return_value="/usr/bin/sbatch"):
        job_id = slurm.submit(script, dispatch="local")
    assert job_id.startswith("local-")
    _drain_local(job_id)
    assert job_id in slurm.finished_jobs([job_id])


def test_dispatch_slurm_requires_sbatch():
    """`dispatch: slurm` never silently runs locally — it fails fast instead."""
    with (
        patch("chemrefine.slurm.dispatch.shutil.which", return_value=None),
        pytest.raises(ConfigError, match="sbatch"),
    ):
        slurm.dispatch_locally("slurm")


@pytest.fixture
def uncached_gpu_probe():
    """Clear the device-count cache around a test that varies what ``nvidia-smi`` says.

    The probe is memoized because the device count is a property of the host; a test that
    fakes a *different* host has to say so, on both sides — a stale entry would answer the
    test, and the test's answer would otherwise outlive it.
    """
    dispatch._detected_devices.cache_clear()
    yield
    dispatch._detected_devices.cache_clear()


def test_detected_devices_counts_mig_instances(monkeypatch, uncached_gpu_probe):
    """nvidia-smi -L lists MIG instances when the card is MIG-partitioned → count those."""
    out = (
        "GPU 0: NVIDIA H100 NVL (UUID: GPU-x)\n"
        "  MIG 1g.12gb Device 0: (UUID: MIG-a)\n"
        "  MIG 1g.12gb Device 1: (UUID: MIG-b)\n"
    )
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: MagicMock(returncode=0, stdout=out, stderr="")
    )
    assert dispatch._detected_devices() == ("0", "1")


def test_detected_devices_falls_back_to_one_without_nvidia_smi(monkeypatch, uncached_gpu_probe):
    monkeypatch.setattr(subprocess, "run", MagicMock(side_effect=FileNotFoundError("nvidia-smi")))
    assert dispatch._detected_devices() == ("0",)


def test_detected_devices_forks_nvidia_smi_once_per_process(monkeypatch, uncached_gpu_probe):
    """The device count is a host constant, and it is asked once per batch.

    Uncached, a run forks `nvidia-smi` per step *and* per retry batch to re-learn a number
    that cannot have changed.
    """
    run = MagicMock(return_value=MagicMock(returncode=0, stdout="GPU 0: X (UUID: g)\n", stderr=""))
    monkeypatch.setattr(subprocess, "run", run)
    assert [dispatch._detected_devices() for _ in range(5)] == [("0",)] * 5
    assert run.call_count == 1


def test_submit_local_applies_cuda_visible_devices_env(tmp_path: Path):
    """The local fallback pins CUDA_VISIBLE_DEVICES for the launched process."""
    script = tmp_path / "g.slurm"
    script.write_text('#!/bin/bash\necho "$CUDA_VISIBLE_DEVICES"\n', encoding="utf-8")
    with patch("chemrefine.slurm.dispatch.shutil.which", return_value=None):
        job_id = slurm.submit(script, env={"CUDA_VISIBLE_DEVICES": "1"})
    _drain_local(job_id)
    assert script.with_suffix(".runlog").read_text(encoding="utf-8") == "1\n"


# ---------------------------------------------------------------------------
# Job arrays — build_array_script / write_array_manifests / submit_array
# ---------------------------------------------------------------------------


def test_build_array_script_resolves_task_from_manifest(tmp_path: Path):
    """The array script looks up its structure at runtime: manifest line via
    ``$SLURM_ARRAY_TASK_ID``, per-structure log redirect, run block on the
    resolved basenames — the per-job script's values, computed in bash."""
    script = slurm.build_array_script(
        step_label="step2_refine",
        ntasks=8,
        slurm_header=slurm.read_header(_write_header(tmp_path)),
        script_path=tmp_path / "out" / "step2_refine_array.slurm",
        output_dir=tmp_path / "out",
        scratch_dir=tmp_path / "scratch",
        run_block=RunBlock(body="orca $INP_NAME > $OUTPUT_DIR/$OUT_NAME"),
        engine="orca",
        operation="opt_sp",
        step=2,
        output_globs=("*.out", "*.xyz"),
    )
    text = script.read_text(encoding="utf-8")
    # The manifest is the script's first argument, captured before the template body
    # (`module load …`) can touch the positional parameters, and loud when missing.
    capture = 'CR_MANIFEST="${1:?array manifest path missing}"'
    assert text.index("#SBATCH") < text.index(capture) < text.index("module load orca/6.0")
    assert 'line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CR_MANIFEST")' in text
    assert "IFS=$'\\t' read -r INP OUT SID <<< \"$line\"" in text
    assert 'INP_NAME=$(basename "$INP")' in text
    # Each task resolves its own per-structure dir from its output path.
    assert 'OUT_DIR=$(dirname "$OUT")' in text
    assert 'exec >"$OUT_DIR/${INP_NAME%.*}.runlog"' in text
    assert 'cp "$INP" "$WORK_DIR/"' in text
    assert "orca $INP_NAME > $OUTPUT_DIR/$OUT_NAME" in text
    # Header handling matches the per-job script.
    assert "#SBATCH --partition=normal" in text
    assert "#SBATCH --ntasks=8" in text
    assert "#SBATCH --cpus-per-task=1" in text
    assert "--ntasks=1" not in text  # the header's own directive is overridden
    # Runlog fields resolve at runtime; the fallback log catches early failures.
    assert "structure_id=$SID" in text
    assert "array_%A_%a.log" in text


def test_write_array_manifests_chunks_at_max_array_size(tmp_path: Path):
    files = tuple((tmp_path / f"s{i}.inp", tmp_path / f"s{i}.out", str(i)) for i in range(2500))
    manifests = slurm.write_array_manifests(files, tmp_path, step_label="step1")
    assert [m.name for m, _ in manifests] == [
        "step1_array.manifest.0",
        "step1_array.manifest.1",
        "step1_array.manifest.2",
    ]
    assert [len(chunk) for _, chunk in manifests] == [1000, 1000, 500]
    first_line = manifests[0][0].read_text(encoding="utf-8").splitlines()[0]
    assert first_line == f"{tmp_path / 's0.inp'}\t{tmp_path / 's0.out'}\t0"
    # Indices restart per chunk: line 0 of chunk 1 is task 1000 overall.
    line0_chunk1 = manifests[1][0].read_text(encoding="utf-8").splitlines()[0]
    assert line0_chunk1.endswith("\t1000")


def test_submit_array_passes_the_manifest_as_the_script_argument(tmp_path: Path):
    """The manifest follows the script in argv — sbatch forwards it as ``$1``.

    Not ``--export=ALL,CR_MANIFEST=…``: sbatch splits ``--export`` on commas, so a comma
    in the output path truncated the variable and every task read a manifest that did not
    exist. The argument channel reserves no character, commas in the path included.
    """
    fake = MagicMock(returncode=0, stdout="Submitted batch job 777\n", stderr="")
    manifest = tmp_path / "out,puts" / "m.0"
    with patch.object(subprocess, "run", return_value=fake) as run:
        job_id = slurm.submit_array(
            tmp_path / "a.slurm", n_tasks=10, max_concurrent=4, manifest=manifest
        )
    assert job_id == "777"
    argv = run.call_args[0][0]
    assert "--parsable" in argv
    assert "--export=ALL" in argv
    assert "--array=0-9%4" in argv
    assert argv[-2:] == [str(tmp_path / "a.slurm"), str(manifest)]


def test_submit_array_raises_on_sbatch_failure(tmp_path: Path):
    err = subprocess.CalledProcessError(1, ["sbatch"], stderr="invalid partition specified")
    with (
        patch.object(subprocess, "run", side_effect=err),
        pytest.raises(JobSubmissionError, match="invalid partition specified"),
    ):
        slurm.submit_array(tmp_path / "a.slurm", n_tasks=1, max_concurrent=1, manifest=tmp_path)


def test_submit_array_raises_when_output_lacks_job_id(tmp_path: Path):
    fake = MagicMock(returncode=0, stdout="no id here\n", stderr="")
    with (
        patch.object(subprocess, "run", return_value=fake),
        pytest.raises(JobSubmissionError, match="could not parse job ID"),
    ):
        slurm.submit_array(tmp_path / "a.slurm", n_tasks=1, max_concurrent=1, manifest=tmp_path)


# ---------------------------------------------------------------------------
# finished_jobs — one scheduler query answers the whole batch
# ---------------------------------------------------------------------------


def test_finished_jobs_asks_squeue_once_for_the_whole_batch():
    """The point of the batch form: N jobs cost one squeue, not N.

    Polling per job means a step with N concurrent jobs runs N squeue subprocesses every
    poll interval — at max_cores 512 with pal 1 that is roughly 50 invocations a second
    against the controller, sustained, which sites rate-limit or ban for.
    """
    fake = MagicMock(returncode=0, stdout="1002\n", stderr="")
    ids = [str(1000 + i) for i in range(200)]
    with patch.object(subprocess, "run", return_value=fake) as run:
        done = slurm.finished_jobs(ids)
    assert run.call_count == 1
    assert "1002" not in done  # still queued
    assert len(done) == len(ids) - 1


def test_finished_jobs_matches_array_tasks_by_prefix():
    fake = MagicMock(returncode=0, stdout="12345_3\n777\n", stderr="")
    with patch.object(subprocess, "run", return_value=fake):
        assert slurm.finished_jobs(["12345", "777", "999"]) == {"999"}


def test_poll_jobs_reports_the_array_task_rows_behind_a_parent_id():
    """The rows are what let a caller see an array move while its parent id does not."""
    fake = MagicMock(returncode=0, stdout="12345_3\n12345_[4-999]\n777\n", stderr="")
    with patch.object(subprocess, "run", return_value=fake):
        state = slurm.poll_jobs(["12345", "777", "999"])
    assert state.finished == {"999"}
    assert state.rows == frozenset({"12345_3", "12345_[4-999]", "777"})


def test_poll_jobs_answers_the_whole_batch_in_one_squeue():
    """Both facts come from one query — asking twice would double every loop's poll rate."""
    fake = MagicMock(returncode=0, stdout="1002\n", stderr="")
    with patch.object(subprocess, "run", return_value=fake) as run:
        state = slurm.poll_jobs([str(1000 + i) for i in range(200)])
    assert run.call_count == 1
    assert state.rows == frozenset({"1002"})


def test_poll_jobs_says_it_learned_nothing_when_squeue_fails():
    """A failed poll reports `rows is None` — a third state, not an empty set.

    Empty rows would read as "the queue emptied". Reporting the polled *ids* instead, which
    is what this used to do, is no better in the other direction: `squeue` prints an array's
    tasks as `12345_0`, never the bare parent, so the fabricated set differs from the real
    rows on *every* alternation between a working and a failing poll — and a caller watching
    rows for movement saw it every tick. `None` is the only answer that cannot be mistaken
    for either the queue draining or the queue moving.
    """
    with patch.object(subprocess, "run", side_effect=subprocess.CalledProcessError(1, "squeue")):
        state = slurm.poll_jobs(["1", "2"])
    assert state.finished == frozenset()
    assert state.rows is None


def test_finished_jobs_treats_a_failing_squeue_as_nothing_finished():
    """A transient squeue failure must not be read as "the batch is done" —
    that would parse every still-running job as a missing-output failure."""
    with patch.object(subprocess, "run", side_effect=subprocess.CalledProcessError(1, "squeue")):
        assert slurm.finished_jobs(["1", "2"]) == set()


def test_finished_jobs_skips_squeue_entirely_for_local_only_batches():
    """A laptop run has no scheduler to ask."""
    with patch.object(subprocess, "run") as run:
        assert slurm.finished_jobs(["local-404"]) == {"local-404"}
    run.assert_not_called()


def test_finished_jobs_of_an_empty_batch_asks_nothing():
    with patch.object(subprocess, "run") as run:
        assert slurm.finished_jobs([]) == set()
    run.assert_not_called()


# ---------------------------------------------------------------------------
# terminate_local_jobs — no orphaned compute on an abnormal exit
# ---------------------------------------------------------------------------


def _alive(pid: int) -> bool:
    """Whether ``pid`` still exists as something other than a zombie.

    A reaped-but-not-yet-collected child is present in the pid space and consuming
    nothing; that is stopped. "Still running" means a live state.
    """
    try:
        state = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()[0]
    except (ProcessLookupError, FileNotFoundError, IndexError):
        return False
    return state != "Z"


def _await_stopped(pid: int, *, timeout: float = 5.0) -> None:
    """Assert the calculation stops within ``timeout`` — it must not outlive the job.

    Polled rather than probed once, because the last thing ``terminate_local_jobs`` waits
    on is ``bash``. On the escalation path the group is SIGKILLed and ``bash`` is reaped
    first, so the calculation is torn down a scheduler tick later; a single probe races
    the kernel rather than testing anything. What the contract promises is that it stops,
    not that it has already stopped by the time the call returns.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return
        time.sleep(0.02)
    raise AssertionError(f"calculation {pid} was still running {timeout}s after termination")


def _await_pid(pidfile: Path, *, timeout: float = 10.0) -> int:
    """Block until the calculation has recorded its own pid; return it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        text = pidfile.read_text(encoding="utf-8").strip() if pidfile.is_file() else ""
        if text:
            return int(text)
        time.sleep(0.02)
    raise AssertionError(f"the calculation never wrote {pidfile}")


def _job_with_a_foreground_calculation(
    tmp_path: Path, *, stubborn: bool = False
) -> tuple[Path, Path]:
    """A *generated* script whose run block launches a long foreground child.

    The shape every real engine emits — ``orca <inp> > <out>``, ``python step1.py`` — and
    the reason a bare ``sleep 60`` script cannot test this: that script has no traps and no
    child, so bash dies on SIGTERM whatever is signalled. A generated script installs
    ``trap '_on_exit 143' TERM`` and then *waits* on a foreground child, and bash defers a
    trap until that child returns. Signalling the shell alone therefore does nothing until
    the grace expires.

    ``stubborn`` makes the calculation ignore SIGTERM outright, which is the escalation
    case: quantum-chemistry binaries do install their own handlers.
    """
    pidfile = tmp_path / "calc.pid"
    calc = tmp_path / "calc.py"
    ignore = "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if stubborn else ""
    calc.write_text(
        "import os, signal, sys, time\n"
        f"{ignore}"
        f"open({str(pidfile)!r}, 'w').write(str(os.getpid()))\n"
        "time.sleep(120)\n",
        encoding="utf-8",
    )
    body = f"{sys.executable} {calc} > $OUTPUT_DIR/step1_structure_0.out"
    return _runnable(tmp_path, body), pidfile


def test_terminate_local_jobs_stops_the_calculation_not_just_its_shell(tmp_path: Path):
    """The calculation must die with the job, not outlive it.

    ``Popen`` without ``start_new_session`` leaves the job in the driver's own process
    group, so ``proc.terminate()`` reaches only bash — which is mid-``wait`` on the
    calculation and defers the trap. The grace then expires, bash is SIGKILLed, and the
    calculation survives, reparented to init: cores still burning, and because the EXIT
    trap never runs, no copy-back, no scratch teardown and no runlog footer.
    """
    script, pidfile = _job_with_a_foreground_calculation(tmp_path)
    job_id = dispatch._submit_local(script)
    proc, out_handle, err_handle = dispatch._LOCAL_PROCS[job_id]
    calc_pid = _await_pid(pidfile)
    try:
        slurm.terminate_local_jobs([job_id])

        _await_stopped(calc_pid)  # must not outlive its shell
        assert proc.poll() is not None  # reaped, not left running
        assert out_handle.closed and err_handle.closed
        assert job_id not in dispatch._LOCAL_PROCS
    finally:
        if _alive(calc_pid):  # reached only on a regression, and then it must not linger
            os.kill(calc_pid, signal.SIGKILL)


def test_terminate_local_jobs_runs_the_exit_handler_before_the_job_dies(tmp_path: Path):
    """Stopping the job must still leave the artifacts a cancelled run owes the user.

    The consequence of signalling only the shell is quiet: `.out` is redirected straight to
    `$OUTPUT_DIR`, so parsing still succeeds while the copy-back, the scratch teardown and
    the runlog footer are all skipped. `exit_code=143` is the same invariant
    `test_generated_script_reports_a_cancelled_job_as_failed` asserts for a `scancel`.
    """
    script, pidfile = _job_with_a_foreground_calculation(tmp_path)
    job_id = dispatch._submit_local(script)
    calc_pid = _await_pid(pidfile)
    try:
        slurm.terminate_local_jobs([job_id])
        runlog = script.with_suffix(".runlog").read_text(encoding="utf-8")
        assert "exit_code=143" in runlog, runlog
        assert "files_copied=" in runlog, runlog
    finally:
        if _alive(calc_pid):  # reached only on a regression, and then it must not linger
            os.kill(calc_pid, signal.SIGKILL)


def test_terminate_local_jobs_defaults_to_every_registered_job(tmp_path: Path):
    """The no-argument form is what the interpreter-exit hook uses."""
    script, pidfile = _job_with_a_foreground_calculation(tmp_path)
    ids = [dispatch._submit_local(script)]
    calc_pid = _await_pid(pidfile)
    try:
        slurm.terminate_local_jobs()
        assert all(i not in dispatch._LOCAL_PROCS for i in ids)
        _await_stopped(calc_pid)
    finally:
        if _alive(calc_pid):  # reached only on a regression, and then it must not linger
            os.kill(calc_pid, signal.SIGKILL)


def test_terminate_local_jobs_ignores_unknown_and_finished_ids():
    """Safe to call unconditionally — reaped jobs are simply absent from the registry."""
    slurm.terminate_local_jobs(["local-does-not-exist", "12345"])


def test_terminate_local_jobs_with_an_empty_batch_spares_everything_else(tmp_path: Path):
    """An empty collection means empty — it is not the no-argument sweep.

    The scheduler's unwind passes its live job ids, and an unwind with nothing in flight
    passes an empty collection; reading falsiness as "everything" would make that unwind
    mean "kill every local job this process started".
    """
    script, pidfile = _job_with_a_foreground_calculation(tmp_path)
    job_id = dispatch._submit_local(script)
    calc_pid = _await_pid(pidfile)
    try:
        slurm.terminate_local_jobs(())
        assert job_id in dispatch._LOCAL_PROCS
        assert _alive(calc_pid)
    finally:
        slurm.terminate_local_jobs([job_id])
        if _alive(calc_pid):
            os.kill(calc_pid, signal.SIGKILL)


def test_terminate_local_jobs_closes_handles_of_an_already_exited_job(tmp_path: Path):
    """A job that finished on its own gets no signal — only its handles closed.

    This is the common shape on the unwind path: some of the batch completed
    normally before the exception, and signalling a dead process would be wrong.
    Signalling a *reaped* pid is worse than pointless: the number is free for reuse,
    so the signal could land on someone else's process.
    """
    script = tmp_path / "quick.slurm"
    script.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    job_id = dispatch._submit_local(script)
    proc, out_handle, err_handle = dispatch._LOCAL_PROCS[job_id]
    proc.wait()  # it exits immediately; poll() is now non-None

    with patch.object(dispatch.os, "killpg") as killpg:
        slurm.terminate_local_jobs([job_id])

    killpg.assert_not_called()
    assert out_handle.closed and err_handle.closed
    assert job_id not in dispatch._LOCAL_PROCS


def test_terminate_local_jobs_escalates_to_kill_when_sigterm_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A calculation that ignores SIGTERM still gets reaped — the grace period is bounded.

    Quantum-chemistry binaries do install signal handlers, so a terminate that politely
    waited forever would hang the interpreter on exit. Driven through a real
    SIG_IGN-installing child rather than a stub, because the thing under test is which
    processes the signal reaches, and a stub cannot answer that.
    """
    monkeypatch.setattr(dispatch, "_LOCAL_TERMINATE_GRACE_SECONDS", 0.3)
    script, pidfile = _job_with_a_foreground_calculation(tmp_path, stubborn=True)
    job_id = dispatch._submit_local(script)
    proc, out_handle, err_handle = dispatch._LOCAL_PROCS[job_id]
    calc_pid = _await_pid(pidfile)
    try:
        slurm.terminate_local_jobs([job_id])

        _await_stopped(calc_pid)  # a SIGTERM-ignoring calculation still gets reaped
        assert proc.poll() is not None
        assert out_handle.closed and err_handle.closed
        assert job_id not in dispatch._LOCAL_PROCS
    finally:
        if _alive(calc_pid):  # reached only on a regression, and then it must not linger
            os.kill(calc_pid, signal.SIGKILL)


# ---------------------------------------------------------------------------
# The generated script's exit handler — runs once, records the truth
# ---------------------------------------------------------------------------


def _runnable(tmp_path: Path, body: str, cleanup: str = "", **overrides: object) -> Path:
    """Build a generated script that can actually execute (its input file exists)."""
    kwargs = _build_kwargs(tmp_path, run_block=RunBlock(body=body, cleanup=cleanup), **overrides)
    inp = Path(kwargs["input_path"])
    inp.parent.mkdir(parents=True, exist_ok=True)
    inp.write_text("! SP\n", encoding="utf-8")
    return slurm.build_script(**kwargs)


def _run_generated(
    script: Path, *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Execute a generated script with bash, as the local runner does."""
    return subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, timeout=60, check=False, env=env
    )


def _with_df(tmp_path: Path, answer: str) -> dict[str, str]:
    """An environment whose ``df`` is the bash in ``answer``: what a job is told of its disk.

    No test can fill the filesystem it runs on, so the generated script's question is
    answered by a stand-in placed first on ``PATH``.
    """
    stand_ins = tmp_path / "bin"
    stand_ins.mkdir()
    df = stand_ins / "df"
    df.write_text(f"#!/bin/bash\n{answer}\n", encoding="utf-8")
    df.chmod(0o755)
    return {**os.environ, "PATH": f"{stand_ins}{os.pathsep}{os.environ['PATH']}"}


_DF_HEADER = "echo 'Filesystem 1024-blocks Used Available Capacity Mounted on'"
_DF_FULL = "echo 'tmpfs 1024 1024 0 100% /mnt'"
_DF_ROOM = "echo 'tmpfs 1024 24 1000 3% /mnt'"

_BARE_HEADER = slurm.SlurmHeader(sbatch=(), body=())
"""A header that runs nothing, for a test that reads the whole of a job's stderr."""


def test_generated_script_records_the_real_exit_code(tmp_path: Path):
    """The footer must report the failing command's status, not a swallowed 0."""
    script = _runnable(tmp_path, "exit 7")

    result = _run_generated(script)

    assert result.returncode == 7
    assert "exit_code=7" in result.stdout


def test_generated_script_reports_a_cancelled_job_as_failed(tmp_path: Path):
    """A SIGTERM'd job must not read as a success in its runlog.

    `bash` does fire the EXIT trap on a fatal signal, but with `$?` already reset to 0 — so
    before the explicit TERM trap a `scancel`-ed run recorded `exit_code=0`, i.e. it looked
    like it had finished cleanly. 143 is the conventional 128+SIGTERM.
    """
    script = _runnable(tmp_path, "sleep 30")

    # Signal the whole process group, which is what `scancel` does. Signalling bash alone
    # would leave the foreground `sleep` running -- and bash defers a trap until its
    # foreground child returns, so the handler would not fire for another 30 seconds.
    log = tmp_path / "run.log"
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            ["bash", str(script)], stdout=fh, stderr=subprocess.DEVNULL, start_new_session=True
        )
        time.sleep(0.5)
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    stdout = log.read_text(encoding="utf-8")

    assert "exit_code=143" in stdout, stdout
    # ...and exactly once: TERM fires the handler, then EXIT would fire it a second time
    # without the `_cr_done` latch, running the copy-back and scratch teardown twice.
    assert stdout.count("files_copied=") == 1, stdout


def test_generated_script_runs_engine_cleanup_before_copying_back(tmp_path: Path):
    """`RunBlock.cleanup` is placed inside the script's one EXIT handler.

    It runs *before* the copy-back (so a server releasing files has finished writing), and a
    cleanup that fails must not abort the copy-back that follows it. The engine supplies this
    as data; it has no way to install a trap of its own.
    """
    script = _runnable(tmp_path, "true", cleanup="echo HOOK_RAN\nfalse")

    result = _run_generated(script)

    assert "HOOK_RAN" in result.stdout
    assert result.stdout.index("HOOK_RAN") < result.stdout.index("files_copied=")


def _scratch_dirs(tmp_path: Path) -> list[Path]:
    """The scratch directories a generated script left under the configured scratch root."""
    root = tmp_path / "scratch"
    return sorted(root.iterdir()) if root.is_dir() else []


def test_a_result_that_cannot_come_home_fails_the_job_and_keeps_its_scratch(tmp_path: Path):
    """A copy the output directory refuses is the job's failure, and scratch is not removed.

    A directory already sits where the result belongs, so `cp` refuses it as a full disk
    would. The run block succeeded, so the script exits with the status kept for exactly
    this; the footer says both things, and scratch still holds the result whole.
    """
    script = _runnable(tmp_path, "echo energy > result.out")
    (tmp_path / "out" / "result.out").mkdir(parents=True)

    result = _run_generated(script)

    assert result.returncode == job_log.COPY_BACK_FAILED
    assert "exit_code=0" in result.stdout
    assert "scratch_kept=true" in result.stdout
    assert "result.out" in result.stderr  # cp says which file and why
    (kept,) = _scratch_dirs(tmp_path)
    assert (kept / "result.out").read_text(encoding="utf-8") == "energy\n"


def test_a_declared_directory_that_cannot_come_home_fails_the_job_too(tmp_path: Path):
    """An ``output_dirs`` entry is a result like any file: refused, it keeps scratch."""
    script = _runnable(tmp_path, "mkdir tensors; echo t > tensors/t.npy", output_dirs=("tensors",))
    (tmp_path / "out").mkdir(parents=True, exist_ok=True)
    (tmp_path / "out" / "tensors").write_text("in the way\n", encoding="utf-8")

    result = _run_generated(script)

    assert result.returncode == job_log.COPY_BACK_FAILED
    assert "scratch_kept=true" in result.stdout
    (kept,) = _scratch_dirs(tmp_path)
    assert (kept / "tensors" / "t.npy").read_text(encoding="utf-8") == "t\n"


def test_a_failed_run_block_keeps_its_own_status_when_a_result_stays_behind(tmp_path: Path):
    """The copy-back's status stands in only for a run block that succeeded."""
    script = _runnable(tmp_path, "echo energy > result.out; exit 7")
    (tmp_path / "out" / "result.out").mkdir(parents=True)

    result = _run_generated(script)

    assert result.returncode == 7
    assert "exit_code=7" in result.stdout
    assert "scratch_kept=true" in result.stdout
    assert len(_scratch_dirs(tmp_path)) == 1


def test_a_pattern_or_directory_nothing_wrote_is_no_failure(tmp_path: Path):
    """The patterns are a superset by design, and a declared directory is optional."""
    script = _runnable(tmp_path, "true", output_dirs=("tensors",))

    result = _run_generated(script)

    assert result.returncode == 0
    assert "files_copied=0" in result.stdout
    assert "scratch_kept=false" in result.stdout
    assert "cp:" not in result.stderr
    assert _scratch_dirs(tmp_path) == []


def test_files_copied_counts_the_files_that_came_home(tmp_path: Path):
    """One for each file a pattern brought home; a directory a pattern matches is no file."""
    script = _runnable(tmp_path, "echo a > a.out; echo b > b.xyz; mkdir c.out; echo d > c.out/d")

    result = _run_generated(script)

    assert result.returncode == 0
    assert "files_copied=2" in result.stdout
    assert sorted(p.name for p in (tmp_path / "out").iterdir() if p.suffix != ".slurm") == [
        "a.out",
        "b.xyz",
    ]


@pytest.mark.parametrize(
    ("body", "blocked", "verdict"),
    [
        ("echo e > result.out", False, job_log.Footer(exit_code=0, scratch_kept=False)),
        ("echo e > result.out; exit 7", False, job_log.Footer(exit_code=7, scratch_kept=False)),
        ("echo e > result.out", True, job_log.Footer(exit_code=0, scratch_kept=True)),
    ],
    ids=["ran to its end", "failed", "a result left in scratch"],
)
def test_the_reader_reads_the_verdict_a_real_script_wrote(
    tmp_path: Path, body: str, blocked: bool, verdict: job_log.Footer
):
    """`job_log.read_footer` over what the exit handler itself wrote, not a stand-in's text."""
    script = _runnable(tmp_path, body)
    if blocked:
        (tmp_path / "out" / "result.out").mkdir(parents=True)
    runlog = job_log.runlog_path(tmp_path / "out", "step1_structure_0")

    runlog.write_text(_run_generated(script).stdout, encoding="utf-8")

    assert job_log.read_footer(runlog) == verdict


def test_a_stopped_job_exits_143_whatever_its_program_made_of_the_signal(tmp_path: Path):
    """A stopped job's script exits with the status its footer reports.

    The program here answers TERM by exiting 0, as a trainer that saves its state on a
    signal does. The script was stopped all the same, and both its footer and its exit
    status say so. The signal is sent once the program has said its trap is armed.
    """
    script = _runnable(tmp_path, "bash -c 'trap \"exit 0\" TERM; echo READY; sleep 30 & wait'")

    log = tmp_path / "run.log"
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            ["bash", str(script)], stdout=fh, stderr=subprocess.DEVNULL, start_new_session=True
        )
        deadline = time.monotonic() + 30
        while "READY" not in log.read_text(encoding="utf-8"):
            assert time.monotonic() < deadline, "the program never armed its trap"
            time.sleep(0.05)
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        status = proc.wait(timeout=30)

    assert status == 143
    assert "exit_code=143" in log.read_text(encoding="utf-8")


def test_a_job_that_ends_on_a_full_filesystem_says_so_on_its_stderr(tmp_path: Path):
    """A job whose own scratch filled the disk leaves the reason, though the disk has room again.

    While the filesystem holding a job's directory is full, what the program writes to its
    ``.err`` is refused, and the exit handler then removes the scratch that filled it: an
    empty ``.err`` on a disk with room. The stand-in ``df`` reports no space for as long as
    the job's scratch directory exists, so the line also proves the handler asked before it
    removed anything.
    """
    script = _runnable(tmp_path, "exit 3", slurm_header=_BARE_HEADER)
    scratch = tmp_path / "scratch"
    full_until_the_scratch_goes = (
        f'{_DF_HEADER}\nif ls -d "{scratch}"/ChemRefine_* >/dev/null 2>&1; '
        f"then {_DF_FULL}; else {_DF_ROOM}; fi"
    )

    result = _run_generated(script, env=_with_df(tmp_path, full_until_the_scratch_goes))

    assert result.stderr == (
        f"ChemRefine: the filesystem holding {tmp_path / 'out'} had no space left when this "
        "job ended; what the job wrote there, this file included, may be cut short or missing\n"
    )
    assert result.returncode == 3
    assert "exit_code=3" in result.stdout
    assert not list(scratch.glob("ChemRefine_*"))


@pytest.mark.parametrize(
    "answer",
    [
        f"{_DF_HEADER}\n{_DF_ROOM}",
        f"{_DF_HEADER}\necho 'fuse 0 0 0 - /mnt'",
        "echo 'df: cannot read table of mounted file systems' >&2\nexit 1",
        _DF_HEADER,
    ],
    ids=["room-left", "no-size", "no-answer", "no-filesystem-line"],
)
def test_a_job_says_nothing_of_its_filesystem_unless_df_found_it_full(tmp_path: Path, answer: str):
    """Room left, or a ``df`` that cannot say: no line, and the job ends as it would have.

    The line states what ``df`` measured. A filesystem that reports no size at all has not
    been measured, an empty figure is no measurement, and ``df``'s own complaint is not the
    job's to pass on; none of them costs the job anything — the handler still removes the
    scratch and writes the footer with the real exit code.
    """
    script = _runnable(tmp_path, "exit 3", slurm_header=_BARE_HEADER)

    result = _run_generated(script, env=_with_df(tmp_path, answer))

    assert result.stderr == ""
    assert result.returncode == 3
    assert "exit_code=3" in result.stdout
    assert not list((tmp_path / "scratch").glob("ChemRefine_*"))


def test_the_exit_handler_asks_before_it_frees_anything_and_answers_after_the_scratch_is_gone(
    tmp_path: Path,
):
    """The question precedes the engine's teardown; the line follows the scratch removal.

    Asked later, the scratch that filled the disk is gone and ``df`` finds room. Said
    earlier, the line is one more write to the full filesystem it is about. The handler is
    straight-line bash, so the order of its text is the order it runs in.
    """
    script = _runnable(tmp_path, "true", cleanup="echo TEARDOWN")
    text = script.read_text(encoding="utf-8")
    handler = text[text.index("_on_exit() {") : text.index("trap _on_exit EXIT")]

    asked, teardown, removed, said = (
        handler.index(mark)
        for mark in ('df -Pk "$OUTPUT_DIR"', "echo TEARDOWN", 'rm -rf "$WORK_DIR"', "no space left")
    )
    assert asked < teardown < removed < said


def test_wait_for_jobs_times_out_when_a_deadline_is_set():
    """The array path honours `job_timeout_seconds` too.

    The per-job path waits through the throttler, the array path through this loop. Wiring
    the deadline into only one of them would mean `job_timeout_seconds` silently did nothing
    for a `slurm_array: true` step — the sort of split that makes a knob untrustworthy.
    """
    with pytest.raises(ThrottleTimeoutError):
        slurm.wait_for_jobs(["1"], poll_interval=0.01, poll=_stuck_at("1"), max_wait_seconds=0.05)


def test_wait_for_jobs_returns_when_all_drain():
    """No deadline configured (the default) keeps the old wait-forever behaviour.

    The queue drains on the second poll, so the loop sits through one that finishes
    nothing, with no deadline to trip.
    """
    asked: list[frozenset[str]] = []

    def poll(ids: Collection[str]) -> slurm.QueueState:
        asked.append(frozenset(ids))
        return _all_done(ids) if len(asked) > 1 else _stuck_at(*ids)(ids)

    slurm.wait_for_jobs(["1", "2"], poll_interval=0.01, poll=poll)
    assert asked == [frozenset({"1", "2"})] * 2


def test_wait_for_jobs_deadline_bounds_the_stall_not_the_whole_drain():
    """An array that keeps draining is making progress, however long the drain takes.

    `job_timeout_seconds` means "nothing has finished for this long" on every path — the
    throttler re-anchors on each completion and so does this loop. Bounding the *total*
    instead would make a healthy multi-hour array trip a timeout meant to catch a stuck one,
    and the two paths would disagree about what the same knob means.
    """
    remaining = ["1", "2", "3", "4"]
    # An injected clock, not real sleeps: 0.03 s stretches against a 0.05 s bound left
    # 20 ms for a GC pause or a loaded runner to turn green into ThrottleTimeoutError —
    # the technique test_throttle._clock_that_expires documents, applied here.
    now = [0.0]

    def one_at_a_time(ids: Collection[str]) -> slurm.QueueState:
        now[0] += 0.03  # each stretch is under the bound; four of them exceed it
        done = {remaining.pop()} if remaining else set(ids)
        return slurm.QueueState(frozenset(done), frozenset(set(ids) - done))

    with patch("time.monotonic", side_effect=lambda: now[0]):
        slurm.wait_for_jobs(
            ["1", "2", "3", "4"], poll_interval=0, poll=one_at_a_time, max_wait_seconds=0.05
        )
    # The drain actually drained: an early return that declares victory while jobs
    # still run leaves entries here — a mutant doing exactly that survived the whole
    # suite while this test held no assertion at all.
    assert remaining == []


def test_wait_for_jobs_re_anchors_on_array_tasks_not_on_the_array():
    """A draining array is progress even though its one parent id never leaves `pending`.

    A step of ≤1000 structures is a *single* `sbatch --array`, so the wait holds exactly one
    id and it does not finish until the last task does. Judging progress by that id alone
    made `job_timeout_seconds` a total-runtime bound on this path and a stall bound on every
    other — so a healthy long array failed on a timeout meant to catch a stuck one. The task
    rows underneath the parent are what move.
    """
    tasks = [f"12345_{i}" for i in range(8)]
    # Injected clock for the reason the sibling above gives — the stretches are data here,
    # not wall time to race a runner against.
    now = [0.0]

    def one_task_at_a_time(ids: Collection[str]) -> slurm.QueueState:
        now[0] += 0.03  # each stretch is under the bound; eight of them are far over it
        tasks.pop()
        # The parent leaves the queue only once its last task has exited — the whole reason
        # this path cannot judge progress by ids.
        return slurm.QueueState(frozenset() if tasks else frozenset(ids), frozenset(tasks))

    with patch("time.monotonic", side_effect=lambda: now[0]):
        slurm.wait_for_jobs(
            ["12345"], poll_interval=0, poll=one_task_at_a_time, max_wait_seconds=0.05
        )
    # Every task was seen to exit before the wait returned — the sibling test's
    # full-drain assertion, on the path where only the task rows move.
    assert tasks == []


def test_wait_for_jobs_times_out_on_an_array_whose_tasks_are_all_stuck():
    """The other half of the same rule: unchanged rows are a stall, however many there are.

    Re-anchoring on task movement must not become "an array never times out" — a queue that
    looks identical poll after poll is exactly the stuck batch the knob exists to catch.
    """
    frozen = frozenset({"12345_0", "12345_[1-999]"})
    with pytest.raises(ThrottleTimeoutError, match="12345"):
        slurm.wait_for_jobs(
            ["12345"],
            poll_interval=0.01,
            poll=lambda _ids: slurm.QueueState(frozenset(), frozen),
            max_wait_seconds=0.05,
        )


def test_wait_for_jobs_times_out_when_squeue_only_answers_every_other_poll():
    """An intermittent squeue must not restart the stall clock.

    Driven through the **real** `poll_jobs` against a flapping `subprocess.run`, because
    both halves of this bug have to be held at once: the producer must not invent rows, and
    the consumer must read "learned nothing" as no movement. Feeding `wait_for_jobs` a
    hand-built `QueueState(…, None)` would exercise only the second and pass even with the
    failure branch reverted — which is exactly what a first draft of this test did.

    The old failure branch reported the polled *ids* as rows, so an alternating ok/error
    squeue produced `{"12345_0", "12345_[1-999]"}`, `{"12345"}`, … — a different set every
    tick, which read as an array making progress. `job_timeout_seconds` then never fired on
    the array path, which is the one path where nothing else bounds the wait: the throttler
    reaps as it goes, but a drain has only this deadline.

    The poll count is capped so a regression fails here instead of hanging the suite —
    nothing configures a per-test timeout.
    """
    polls = 0

    def flapping_squeue(*_args, **_kwargs):
        nonlocal polls
        polls += 1
        assert polls < 200, "the stall deadline never fired on an alternating squeue"
        if polls % 2 == 0:
            raise subprocess.CalledProcessError(1, "squeue")
        return MagicMock(returncode=0, stdout="12345_0\n12345_[1-999]\n", stderr="")

    with (
        patch.object(subprocess, "run", side_effect=flapping_squeue),
        pytest.raises(ThrottleTimeoutError, match="12345"),
    ):
        slurm.wait_for_jobs(
            ["12345"], poll_interval=0.001, poll=slurm.poll_jobs, max_wait_seconds=0.05
        )


def test_wait_for_jobs_with_nothing_to_wait_for_returns_immediately():
    """An empty batch must not poll at all — and must not consult the deadline."""
    slurm.wait_for_jobs([], poll_interval=0.01, poll=_never_called, max_wait_seconds=0.0)


def _all_done(ids: Collection[str]) -> slurm.QueueState:
    """Everything the loop asks about has already left the queue."""
    return slurm.QueueState(frozenset(ids), frozenset())


def _stuck_at(*ids: str) -> Callable[[Collection[str]], slurm.QueueState]:
    """Nothing ever finishes and the queue never changes shape."""
    return lambda _ids: slurm.QueueState(frozenset(), frozenset(ids))


def _never_called(_ids: object) -> slurm.QueueState:
    """A `poll` callable that fails the test if the loop polls when it should not."""
    raise AssertionError("wait_for_jobs polled with an empty job set")


# ---------------------------------------------------------------------------
# job leases — the record the resume fence reads
# ---------------------------------------------------------------------------


def test_leases_round_trip_and_an_absent_ledger_reads_empty(tmp_path: Path):
    """Recorded leases come back whole: a SLURM id bare, a local id with its pid.

    The pid is what makes a local lease *probe-able* after the driver that owned the
    process object is gone — it must be captured at recording time, from the live
    registry, because there is nowhere to get it later.
    """
    assert slurm.load_leases(tmp_path) == []

    proc = subprocess.Popen(["sleep", "5"])
    try:
        with patch.dict(dispatch._LOCAL_PROCS, {"local-7": (proc, None, None)}):
            slurm.record_leases(tmp_path, ["12345", "local-7"], accepted="local-7")
    finally:
        proc.kill()
        proc.wait()

    by_id = {lease.id: lease for lease in slurm.load_leases(tmp_path)}
    assert by_id["12345"].pid is None and not by_id["12345"].local
    assert by_id["local-7"].pid == proc.pid and by_id["local-7"].local
    assert all(lease.host for lease in by_id.values())


def test_the_ledger_is_written_whole_as_the_jobs_in_flight(tmp_path: Path):
    """Each write replaces the ledger: a job no longer in flight leaves it."""
    slurm.record_leases(tmp_path, ["local-1", "local-2"], accepted="local-2")
    slurm.record_leases(tmp_path, ["local-2", "local-3"], accepted="local-3")
    assert [lease.id for lease in slurm.load_leases(tmp_path)] == ["local-2", "local-3"]


def test_an_unwinding_release_keeps_the_slurm_leases_only(tmp_path: Path):
    """`keep_slurm` is the unwind's shape: local jobs were just killed, SLURM jobs run on."""
    # No registry entry for local-3, so its pid stays None.
    slurm.record_leases(tmp_path, ["12345", "local-3"], accepted="local-3")

    slurm.release_leases(tmp_path, keep_slurm=True)
    assert [lease.id for lease in slurm.load_leases(tmp_path)] == ["12345"]

    # A second unwind with nothing but local leases must not leave an empty ledger
    # behind — an existing-but-empty file and no file must mean the same thing.
    slurm.release_leases(tmp_path)
    slurm.record_leases(tmp_path, ["local-4"], accepted="local-4")
    slurm.release_leases(tmp_path, keep_slurm=True)
    assert not slurm.lease_path(tmp_path).exists()


def test_a_corrupt_lease_ledger_is_refused_not_read_as_empty(tmp_path: Path):
    """Unreadable leases cannot prove their jobs dead; silence would wave a driver in."""
    from chemrefine.errors import CacheError

    path = slurm.lease_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(CacheError, match="delete it"):
        slurm.load_leases(tmp_path)


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads through any mode")
@pytest.mark.parametrize("locked", ["file", "directory"])
def test_an_unreadable_lease_ledger_is_the_same_worded_refusal(tmp_path: Path, locked: str):
    """Present but unreadable is what the docstring promises to refuse — not a traceback.

    Another account's driver, or a ledger the user is told to inspect from another host:
    the fence cannot prove those jobs dead either, and the message that names the escape
    is the same one a corrupt file gets. So is a ledger under a ``_cache/`` this account
    cannot search, on every Python: asked of ``Path.is_file`` first, that is a traceback on
    3.11 to 3.13 and "no leases" on 3.14, which waves the fence through over a live job.
    """
    from chemrefine.errors import CacheError

    slurm.record_leases(tmp_path, ["local-1"], accepted="local-1")
    path = slurm.lease_path(tmp_path)
    target = path if locked == "file" else path.parent
    target.chmod(0)
    try:
        with pytest.raises(CacheError, match="delete it"):
            slurm.load_leases(tmp_path)
    finally:
        target.chmod(0o755)


@pytest.mark.parametrize(
    "pid",
    ['"abc"', "[1]", "Infinity", "NaN"],
    ids=["a-word", "a-list", "infinity", "not-a-number"],
)
def test_a_lease_pid_that_is_no_number_is_a_corrupt_ledger(tmp_path: Path, pid: str):
    """A pid is read back as the integer it was written as, or the ledger is corrupt.

    Handed on as it came, a word reaches the fence's probe as a ``TypeError`` and the run
    ends in a traceback; an infinity is the one value ``int`` refuses as an overflow.
    """
    from chemrefine.errors import CacheError

    path = slurm.lease_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(f'[{{"id": "local-1", "host": "here", "pid": {pid}}}]', encoding="utf-8")
    with pytest.raises(CacheError, match="delete it"):
        slurm.load_leases(tmp_path)


def test_the_lease_ledger_is_written_readable_for_the_next_driver(tmp_path: Path):
    """The ledger lands with the mode a plain write gives, not the temp file's 0600.

    The resume fence reads it from whichever account next drives the tree, and the
    docs direct a user on another host to inspect it; an owner-only ledger would
    manufacture the very refusal above for everyone but its author.
    """
    import stat

    previous = os.umask(0o027)
    try:
        slurm.record_leases(tmp_path, ["local-1"], accepted="local-1")
        mode = stat.S_IMODE(slurm.lease_path(tmp_path).stat().st_mode)
    finally:
        os.umask(previous)
    assert mode == 0o640


def test_a_failed_lease_write_leaves_no_temp_file_behind(tmp_path: Path):
    """The temp file is the writer's, and it is gone whether or not the rename happened.

    The refusal is the output tree's, naming the ledger; recording a local job, it goes on
    as it came, since the job is the unwind's to stop.
    """
    with (
        patch.object(Path, "replace", side_effect=OSError(errno.ENOSPC, "disk full")),
        pytest.raises(OutputTreeError) as refusal,
    ):
        slurm.record_leases(tmp_path, ["local-1"], accepted="local-1")
    assert str(refusal.value) == f"cannot write {slurm.lease_path(tmp_path)}: disk full"
    assert not list(slurm.lease_path(tmp_path).parent.glob(".tmp_*"))
    assert not slurm.lease_path(tmp_path).exists()


def test_an_unwinding_release_writes_nothing_when_it_drops_nothing(tmp_path: Path):
    """A release that keeps every lease touches no file, on a disk that refuses them all.

    Under SLURM dispatch an unwind keeps every lease, and a rewrite of the same ledger
    could only meet the full or read-only disk the unwind may be stopping for. A step with
    no ledger at all is the same release: nothing to drop, nothing to remove.
    """
    slurm.record_leases(tmp_path, ["12345"], accepted="12345")
    unrecorded = tmp_path / "step2"
    refused = OSError(errno.EROFS, os.strerror(errno.EROFS))
    with (
        patch.object(Path, "replace", side_effect=refused),
        patch.object(Path, "unlink", side_effect=refused),
    ):
        slurm.release_leases(tmp_path, keep_slurm=True)
        slurm.release_leases(unrecorded, keep_slurm=True)
    assert [lease.id for lease in slurm.load_leases(tmp_path)] == ["12345"]
    assert not unrecorded.exists()


def test_a_release_the_tree_refuses_is_named(tmp_path: Path):
    """After a drained batch, a ledger the filesystem will not remove is the batch's refusal."""
    slurm.record_leases(tmp_path, ["local-1"], accepted="local-1")
    ledger = slurm.lease_path(tmp_path)
    with (
        patch.object(
            Path, "unlink", side_effect=OSError(errno.EROFS, os.strerror(errno.EROFS), str(ledger))
        ),
        pytest.raises(OutputTreeError) as refusal,
    ):
        slurm.release_leases(tmp_path)
    assert str(refusal.value) == f"cannot write {ledger}: Read-only file system"
    assert ledger.exists()


def test_scancel_answers_whether_the_cancel_landed():
    """argv with no shell; ``None`` when the scheduler took it, its words when it did not."""
    ok = subprocess.CompletedProcess(["scancel", "4242"], 0, "", "")
    with patch.object(dispatch.subprocess, "run", return_value=ok) as run:
        assert dispatch._scancel("4242") is None
    assert run.call_args.args[0] == ["scancel", "4242"]
    assert "shell" not in run.call_args.kwargs
    refused = subprocess.CalledProcessError(1, "scancel", "", "Invalid job id specified\n")
    with patch.object(dispatch.subprocess, "run", side_effect=refused):
        assert dispatch._scancel("4242") == "Invalid job id specified"
    silent = subprocess.CalledProcessError(1, "scancel", "", "")
    with patch.object(dispatch.subprocess, "run", side_effect=silent):
        assert dispatch._scancel("4242") == "scancel exited 1"
    missing = FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), "scancel")
    with patch.object(dispatch.subprocess, "run", side_effect=missing):
        assert dispatch._scancel("4242") == "[Errno 2] No such file or directory: 'scancel'"


def _array_script(tmp_path: Path) -> Path:
    """``build_array_script`` aimed at ``out/``, the array-path twin of ``_build_kwargs``."""
    return slurm.build_array_script(
        step_label="step1_refine",
        ntasks=1,
        slurm_header=slurm.read_header(_write_header(tmp_path)),
        script_path=tmp_path / "out" / "step1_refine_array.slurm",
        output_dir=tmp_path / "out",
        scratch_dir=tmp_path / "scratch",
        run_block=RunBlock(body="echo hi"),
        engine="orca",
        operation="opt_sp",
        step=1,
        output_globs=("*.out",),
    )


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes through any mode")
@pytest.mark.parametrize(
    ("name", "write"),
    [
        ("step1_structure_0.slurm", lambda tmp_path: slurm.build_script(**_build_kwargs(tmp_path))),
        ("step1_refine_array.slurm", _array_script),
        (
            "step1_array.manifest.0",
            lambda tmp_path: slurm.write_array_manifests(
                ((tmp_path / "0.inp", tmp_path / "0.out", "0"),),
                tmp_path / "out",
                step_label="step1",
            ),
        ),
    ],
    ids=["job-script", "array-script", "array-manifest"],
)
def test_a_script_or_manifest_the_tree_refuses_is_named(
    tmp_path: Path, name: str, write: Callable[[Path], object]
):
    """Each file a submission writes before it submits refuses as the output tree's.

    Refused, nothing is submitted: the script a job would run is never handed over torn.
    """
    out = tmp_path / "out"
    out.mkdir()
    out.chmod(0o555)
    try:
        with pytest.raises(OutputTreeError) as refusal:
            write(tmp_path)
    finally:
        out.chmod(0o755)
    assert str(refusal.value) == f"cannot write {out / name}: Permission denied"


def test_scheduler_reachable_answers_for_squeue_on_path():
    """The fence's verify-vs-warn discriminator is squeue's presence, nothing subtler."""
    with patch.object(dispatch.shutil, "which", return_value="/usr/bin/squeue"):
        assert slurm.scheduler_reachable()
    with patch.object(dispatch.shutil, "which", return_value=None):
        assert not slurm.scheduler_reachable()

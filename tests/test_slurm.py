"""Tests for SLURM script generation, sbatch submission, and squeue polling."""

from __future__ import annotations

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

from chemrefine import slurm
from chemrefine.errors import ConfigError, JobSubmissionError, ThrottleTimeoutError
from chemrefine.slurm import dispatch
from chemrefine.state import RunBlock
from chemrefine.throttle import GpuBudget


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
        "template_path": _write_header(tmp_path),
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
    text = script.read_text()
    assert "#SBATCH --partition=normal" in text
    assert "#SBATCH --time=24:00:00" in text
    assert "#SBATCH --ntasks=12" in text
    assert "#SBATCH --cpus-per-task=1" in text
    # the user's --ntasks=1 must not survive
    assert "--ntasks=1" not in text or "--ntasks=12" in text
    assert "module load orca/6.0" in text


def test_build_script_spells_a_threads_layout(tmp_path: Path):
    """``cpus_per_task`` reaches the SBATCH pair, and the runlog reports the product.

    The pair is an engine's ``slurm_layout``: MPI ranks are ``(pal, 1)``, one threaded
    process is ``(1, threads)`` — N single-cpu tasks can be granted across nodes, where a
    threaded program can only use the first node's share.
    """
    script = slurm.build_script(**_build_kwargs(tmp_path, ntasks=1, cpus_per_task=8))
    text = script.read_text()
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
    script = slurm.build_script(**_build_kwargs(tmp_path, template_path=header))
    text = script.read_text()
    assert "#SBATCH --mem-per-cpu=1000" in text
    assert text.count("--mem-per-cpu") == 1


def test_a_sufficient_header_memory_allocation_stands(tmp_path: Path):
    """The cluster's own policy wins whenever it covers the input's requirement."""
    header = _write_header_with(tmp_path, "--mem-per-cpu=4000")
    script = slurm.build_script(
        **_build_kwargs(tmp_path, template_path=header, ntasks=1, cpus_per_task=8, memory_mb=16000)
    )
    text = script.read_text()
    assert "#SBATCH --mem-per-cpu=4000" in text  # 8 cpus x 4000 = 32000 >= 16000
    assert text.count("--mem-per-cpu") == 1


def test_a_sufficient_total_memory_header_counts_its_units(tmp_path: Path):
    """A ``--mem=64G`` grant is read as 65536 MB, not compared as the bare number 64."""
    header = _write_header_with(tmp_path, "--mem=64G")
    script = slurm.build_script(
        **_build_kwargs(tmp_path, template_path=header, ntasks=1, cpus_per_task=8, memory_mb=60000)
    )
    text = script.read_text()
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
                tmp_path, template_path=header, ntasks=1, cpus_per_task=8, memory_mb=32000
            )
        )
    text = script.read_text()
    assert "#SBATCH --mem-per-cpu=4000" in text  # ceil(32000 / 8)
    assert "--mem-per-cpu=1000" not in text
    assert "#SBATCH --mem-per-gpu=8G" in text
    assert "32000" in caplog.text and "8000 MB" in caplog.text


def test_an_absent_header_memory_grant_is_requested(tmp_path: Path):
    """With no header memory at all, the input's requirement becomes the request."""
    script = slurm.build_script(**_build_kwargs(tmp_path, memory_mb=1000))
    assert "#SBATCH --mem-per-cpu=1000" in script.read_text()


def test_a_kilobyte_header_grant_is_floored_to_mb(tmp_path: Path):
    """``2048K`` reads as 2 MB — flooring understates the grant, which only ever extends."""
    header = _write_header_with(tmp_path, "--mem-per-cpu=2048K")
    script = slurm.build_script(**_build_kwargs(tmp_path, template_path=header, memory_mb=2))
    text = script.read_text()
    assert "#SBATCH --mem-per-cpu=2048K" in text  # 2 MB covers the 2 MB requirement
    assert text.count("--mem-per-cpu") == 1


def test_a_whole_node_grant_satisfies_any_requirement(tmp_path: Path):
    """``--mem=0`` is SLURM's "all the node's memory" — never extended, whatever is asked."""
    header = _write_header_with(tmp_path, "--mem=0")
    script = slurm.build_script(**_build_kwargs(tmp_path, template_path=header, memory_mb=999999))
    text = script.read_text()
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
    script = slurm.build_script(**_build_kwargs(tmp_path, template_path=header, ntasks=8))
    text = script.read_text()
    assert "#SBATCH --ntasks-per-node=16" in text
    assert "#SBATCH --ntasks-per-core=1" in text
    assert "#SBATCH --ntasks 1" not in text
    assert "#SBATCH --ntasks=8" in text
    assert "--output=old.log" not in text


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
    script = slurm.build_script(
        **_build_kwargs(tmp_path, run_block=RunBlock(body="echo CUSTOM_RUN_BLOCK_HERE"))
    )
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


def test_build_script_copies_output_dirs_back(tmp_path: Path):
    """``output_dirs`` (e.g. pyscf ``tensors/``) are copied back wholesale on exit."""
    script = slurm.build_script(**_build_kwargs(tmp_path, output_dirs=("tensors",)))
    text = script.read_text()
    assert 'cp -r "tensors" "$OUTPUT_DIR/" 2>/dev/null || true' in text


def test_build_script_missing_template_raises(tmp_path: Path):
    with pytest.raises(ConfigError):
        slurm.build_script(**_build_kwargs(tmp_path, template_path=tmp_path / "missing.header"))


def test_build_script_uses_caller_supplied_output_globs(tmp_path: Path):
    """The back-copy line must reflect the engine's declared file extensions."""
    script = slurm.build_script(**_build_kwargs(tmp_path, output_globs=("*.json", "*.npz")))
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
    assert slurm.is_finished(job_id) is True


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
    assert slurm.is_finished(j1) is False
    assert slurm.is_finished(j2) is False
    _drain_local(j1)
    _drain_local(j2)
    assert slurm.is_finished(j1) is True
    assert slurm.is_finished(j2) is True


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
    assert slurm.is_finished(job_id) is True
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
            raise OSError("ENOSPC: no space left on device")
        handle = real_open(self, *args, **kwargs)
        opened.append(handle)
        return handle

    before = set(dispatch._LOCAL_PROCS)
    with (
        patch.object(Path, "open", open_runlog_only),
        pytest.raises(OSError, match="ENOSPC"),
    ):
        dispatch._submit_local(script)
    assert len(opened) == 1 and opened[0].closed, "the .runlog handle outlived its function"
    assert set(dispatch._LOCAL_PROCS) == before


# ---------------------------------------------------------------------------
# is_finished
# ---------------------------------------------------------------------------


def test_is_finished_passes_noheader_to_squeue():
    """``--noheader`` must be requested, not assumed.

    The output is consumed line-for-line with no header slice, so the flag is
    what keeps ``JOBID`` out of the job-id list. See
    :func:`test_is_finished_false_for_the_only_queued_job`.
    """
    fake = MagicMock(returncode=0, stdout="", stderr="")
    with patch.object(subprocess, "run", return_value=fake) as run:
        slurm.is_finished("12345")
    assert "--noheader" in run.call_args.args[0]


def test_is_finished_false_for_the_only_queued_job():
    """The sole queued job must not be mistaken for a header row.

    The previous implementation dropped ``lines[0]`` as a header. On a site whose
    ``squeue`` already suppresses the header, that discarded the first *real*
    job id — reporting a still-running job as finished, which then parsed as a
    missing-output failure.
    """
    fake = MagicMock(returncode=0, stdout="12345\n", stderr="")
    with patch.object(subprocess, "run", return_value=fake):
        assert slurm.is_finished("12345") is False


def test_is_finished_true_when_queue_is_empty():
    """An empty queue (no header, no rows) means every job is finished."""
    for stdout in ("", "\n"):
        fake = MagicMock(returncode=0, stdout=stdout, stderr="")
        with patch.object(subprocess, "run", return_value=fake):
            assert slurm.is_finished("12345") is True


def test_is_finished_true_when_job_absent():
    fake = MagicMock(returncode=0, stdout="9999\n", stderr="")
    with patch.object(subprocess, "run", return_value=fake):
        assert slurm.is_finished("12345") is True


def test_is_finished_false_when_job_still_running():
    fake = MagicMock(returncode=0, stdout="9999\n12345\n", stderr="")
    with patch.object(subprocess, "run", return_value=fake):
        assert slurm.is_finished("12345") is False


def test_is_finished_false_while_array_tasks_run():
    """An array's parent id never appears bare in squeue — running tasks print
    as ``12345_0`` and pending ones as ``12345_[5-99%4]``; both must count."""
    fake = MagicMock(returncode=0, stdout="12345_3\n", stderr="")
    with patch.object(subprocess, "run", return_value=fake):
        assert slurm.is_finished("12345") is False
    fake = MagicMock(returncode=0, stdout="12345_[4-99%4]\n", stderr="")
    with patch.object(subprocess, "run", return_value=fake):
        assert slurm.is_finished("12345") is False


def test_is_finished_array_prefix_does_not_match_other_jobs():
    """``123`` must not be held back by an unrelated ``1234`` or ``1234_0``."""
    fake = MagicMock(returncode=0, stdout="1234\n1234_0\n", stderr="")
    with patch.object(subprocess, "run", return_value=fake):
        assert slurm.is_finished("123") is True


def test_is_finished_treats_squeue_failure_as_not_finished():
    err = subprocess.CalledProcessError(1, ["squeue"])
    with patch.object(subprocess, "run", side_effect=err):
        assert slurm.is_finished("12345") is False


def test_is_finished_true_for_local_job_id():
    """Unknown / already-reaped local IDs are reported finished without touching squeue."""
    with patch.object(subprocess, "run") as run:
        assert slurm.is_finished("local-9999") is True
    run.assert_not_called()


# ---------------------------------------------------------------------------
# _current_user
# ---------------------------------------------------------------------------


def test_current_user_falls_back_to_uid_when_no_passwd_entry():
    """``getpass.getuser`` raises in passwd-less environments (containers under
    an arbitrary UID); the numeric UID is an equally valid ``squeue -u`` value.
    Resolved lazily so a failing lookup can't break the package import."""
    import getpass
    import os

    dispatch._current_user.cache_clear()
    try:
        with patch.object(getpass, "getuser", side_effect=KeyError("getpwuid(): uid not found")):
            assert dispatch._current_user() == str(os.getuid())
    finally:
        dispatch._current_user.cache_clear()


def test_current_user_returns_login_name():
    import getpass

    dispatch._current_user.cache_clear()
    try:
        with patch.object(getpass, "getuser", return_value="alice"):
            assert dispatch._current_user() == "alice"
    finally:
        dispatch._current_user.cache_clear()


# ---------------------------------------------------------------------------
# GPU budget + device-aware headers
# ---------------------------------------------------------------------------


def test_header_name_for_device():
    assert slurm.header_name_for_device("cuda") == "cuda.slurm.header"
    assert slurm.header_name_for_device("CUDA") == "cuda.slurm.header"
    assert slurm.header_name_for_device("cpu") == "cpu.slurm.header"


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
    assert slurm.is_finished(job_id) is True


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
        template_path=_write_header(tmp_path),
        script_path=tmp_path / "out" / "step2_refine_array.slurm",
        output_dir=tmp_path / "out",
        scratch_dir=tmp_path / "scratch",
        run_block=RunBlock(body="orca $INP_NAME > $OUTPUT_DIR/$OUT_NAME"),
        engine="orca",
        operation="opt_sp",
        step=2,
        output_globs=("*.out", "*.xyz"),
    )
    text = script.read_text()
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


def test_submit_array_passes_array_and_export_flags(tmp_path: Path):
    fake = MagicMock(returncode=0, stdout="Submitted batch job 777\n", stderr="")
    with patch.object(subprocess, "run", return_value=fake) as run:
        job_id = slurm.submit_array(
            tmp_path / "a.slurm",
            n_tasks=10,
            max_concurrent=4,
            manifest=tmp_path / "m.0",
        )
    assert job_id == "777"
    argv = run.call_args[0][0]
    assert "--parsable" in argv
    assert f"--export=ALL,CR_MANIFEST={tmp_path / 'm.0'}" in argv
    assert "--array=0-9%4" in argv


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

    The scheduler passes its live job ids from a `finally`, and on every clean exit that
    collection *is* empty; reading falsiness as "everything" made the ordinary end of a batch
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


def _runnable(tmp_path: Path, body: str, cleanup: str = "") -> Path:
    """Build a generated script that can actually execute (its input file exists)."""
    kwargs = _build_kwargs(tmp_path, run_block=RunBlock(body=body, cleanup=cleanup))
    inp = Path(kwargs["input_path"])
    inp.parent.mkdir(parents=True, exist_ok=True)
    inp.write_text("! SP\n", encoding="utf-8")
    return slurm.build_script(**kwargs)


def _run_generated(script: Path) -> subprocess.CompletedProcess[str]:
    """Execute a generated script with bash, as the local runner does."""
    return subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, timeout=60, check=False
    )


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


def test_wait_for_jobs_times_out_when_a_deadline_is_set():
    """The array path honours `job_timeout_seconds` too.

    The per-job path waits through the throttler, the array path through this loop. Wiring
    the deadline into only one of them would mean `job_timeout_seconds` silently did nothing
    for a `slurm_array: true` step — the sort of split that makes a knob untrustworthy.
    """
    with pytest.raises(ThrottleTimeoutError):
        slurm.wait_for_jobs(["1"], poll_interval=0.01, poll=_stuck_at("1"), max_wait_seconds=0.05)


def test_wait_for_jobs_returns_when_all_drain():
    """No deadline configured (the default) keeps the old wait-forever behaviour."""
    slurm.wait_for_jobs(["1", "2"], poll_interval=0.01, poll=_all_done)


def test_wait_for_jobs_deadline_bounds_the_stall_not_the_whole_drain():
    """An array that keeps draining is making progress, however long the drain takes.

    `job_timeout_seconds` means "nothing has finished for this long" on every path — the
    throttler re-anchors on each completion and so does this loop. Bounding the *total*
    instead would make a healthy multi-hour array trip a timeout meant to catch a stuck one,
    and the two paths would disagree about what the same knob means.
    """
    remaining = ["1", "2", "3", "4"]

    def one_at_a_time(ids: Collection[str]) -> slurm.QueueState:
        time.sleep(0.03)  # each stretch is under the bound; four of them exceed it
        done = {remaining.pop()} if remaining else set(ids)
        return slurm.QueueState(frozenset(done), frozenset(set(ids) - done))

    slurm.wait_for_jobs(
        ["1", "2", "3", "4"], poll_interval=0, poll=one_at_a_time, max_wait_seconds=0.05
    )


def test_wait_for_jobs_re_anchors_on_array_tasks_not_on_the_array():
    """A draining array is progress even though its one parent id never leaves `pending`.

    A step of ≤1000 structures is a *single* `sbatch --array`, so the wait holds exactly one
    id and it does not finish until the last task does. Judging progress by that id alone
    made `job_timeout_seconds` a total-runtime bound on this path and a stall bound on every
    other — so a healthy long array failed on a timeout meant to catch a stuck one. The task
    rows underneath the parent are what move.
    """
    tasks = [f"12345_{i}" for i in range(8)]

    def one_task_at_a_time(ids: Collection[str]) -> slurm.QueueState:
        time.sleep(0.03)  # each stretch is under the bound; eight of them are far over it
        tasks.pop()
        # The parent leaves the queue only once its last task has exited — the whole reason
        # this path cannot judge progress by ids.
        return slurm.QueueState(frozenset() if tasks else frozenset(ids), frozenset(tasks))

    slurm.wait_for_jobs(["12345"], poll_interval=0, poll=one_task_at_a_time, max_wait_seconds=0.05)


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

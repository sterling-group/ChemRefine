"""Tests for SLURM script generation, sbatch submission, and squeue polling."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from chemrefine import slurm
from chemrefine.errors import ConfigError, JobSubmissionError, ThrottleTimeoutError
from chemrefine.slurm import dispatch
from chemrefine.state import RunBlock


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
        "pal": 1,
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
            pal=12,
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


def test_build_script_keeps_ntasks_per_node_directive(tmp_path: Path):
    """``--ntasks-per-*`` directives only share a prefix with ``--ntasks`` — keep them.

    Regression: substring matching on ``--ntasks`` used to silently strip a
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
    script = slurm.build_script(**_build_kwargs(tmp_path, template_path=header, pal=8))
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
    fake = MagicMock(returncode=0, stdout="Submitted batch job 12345\n", stderr="")
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
    """A non-zero local exit no longer raises at submit; the logs still land on disk
    so the failure can surface through the engine's output parsing."""
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
    """Regression: the sole queued job must not be mistaken for a header row.

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
    assert slurm.resolve_gpu_budget(3) == 3


def test_resolve_gpu_budget_unlimited_under_slurm():
    with patch("chemrefine.slurm.dispatch.shutil.which", return_value="/usr/bin/sbatch"):
        assert slurm.resolve_gpu_budget(None) >= 1000


def test_resolve_gpu_budget_uses_detected_count_locally():
    with (
        patch("chemrefine.slurm.dispatch.shutil.which", return_value=None),
        patch("chemrefine.slurm.dispatch._detect_local_gpus", return_value=2),
    ):
        assert slurm.resolve_gpu_budget(None) == 2


def test_resolve_gpu_budget_respects_forced_local():
    """`dispatch: local` uses the detected device count even with sbatch on PATH."""
    with (
        patch("chemrefine.slurm.dispatch.shutil.which", return_value="/usr/bin/sbatch"),
        patch("chemrefine.slurm.dispatch._detect_local_gpus", return_value=2),
    ):
        assert slurm.resolve_gpu_budget(None, dispatch="local") == 2


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


def test_detect_local_gpus_counts_mig_instances(monkeypatch):
    """nvidia-smi -L lists MIG instances when the card is MIG-partitioned → count those."""
    out = (
        "GPU 0: NVIDIA H100 NVL (UUID: GPU-x)\n"
        "  MIG 1g.12gb Device 0: (UUID: MIG-a)\n"
        "  MIG 1g.12gb Device 1: (UUID: MIG-b)\n"
    )
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: MagicMock(returncode=0, stdout=out, stderr="")
    )
    assert dispatch._detect_local_gpus() == 2


def test_detect_local_gpus_falls_back_to_one_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(subprocess, "run", MagicMock(side_effect=FileNotFoundError("nvidia-smi")))
    assert dispatch._detect_local_gpus() == 1


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
        pal=8,
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

    Polling per job meant a step with N concurrent jobs ran N squeue
    subprocesses every poll interval — at max_cores 512 with pal 1 that is
    roughly 50 invocations a second against the controller, sustained, which
    sites rate-limit or ban for.
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


def test_terminate_local_jobs_kills_and_reaps_a_running_job(tmp_path: Path):
    """An abnormal exit must not leave background children burning cores.

    They are real processes owned by this interpreter; left running they keep
    competing with whatever the user runs next, and their log handles stay open.
    """
    script = tmp_path / "sleeper.slurm"
    script.write_text("#!/bin/bash\nsleep 60\n", encoding="utf-8")
    job_id = dispatch._submit_local(script)
    proc, out_handle, err_handle = dispatch._LOCAL_PROCS[job_id]

    slurm.terminate_local_jobs([job_id])

    assert proc.poll() is not None  # reaped, not left running
    assert out_handle.closed and err_handle.closed
    assert job_id not in dispatch._LOCAL_PROCS


def test_terminate_local_jobs_defaults_to_every_registered_job(tmp_path: Path):
    """The no-argument form is what the interpreter-exit hook uses."""
    script = tmp_path / "sleeper.slurm"
    script.write_text("#!/bin/bash\nsleep 60\n", encoding="utf-8")
    ids = [dispatch._submit_local(script), dispatch._submit_local(script)]
    slurm.terminate_local_jobs()
    assert all(i not in dispatch._LOCAL_PROCS for i in ids)


def test_terminate_local_jobs_ignores_unknown_and_finished_ids():
    """Safe to call unconditionally — reaped jobs are simply absent from the registry."""
    slurm.terminate_local_jobs(["local-does-not-exist", "12345"])


def test_terminate_local_jobs_closes_handles_of_an_already_exited_job(tmp_path: Path):
    """A job that finished on its own gets no signal — only its handles closed.

    This is the common shape on the unwind path: some of the batch completed
    normally before the exception, and signalling a dead process would be wrong.
    """
    script = tmp_path / "quick.slurm"
    script.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    job_id = dispatch._submit_local(script)
    proc, out_handle, err_handle = dispatch._LOCAL_PROCS[job_id]
    proc.wait()  # it exits immediately; poll() is now non-None

    with patch.object(proc, "terminate") as terminate:
        slurm.terminate_local_jobs([job_id])

    terminate.assert_not_called()
    assert out_handle.closed and err_handle.closed
    assert job_id not in dispatch._LOCAL_PROCS


def test_terminate_local_jobs_escalates_to_kill_when_sigterm_is_ignored(tmp_path: Path):
    """A job that ignores SIGTERM still gets reaped — the grace period is bounded.

    Quantum-chemistry binaries do install signal handlers, so a terminate that
    politely waits forever would hang the interpreter on exit. Driven through a
    stub rather than a real signal-trapping shell: whether a given shell forwards
    or ignores SIGTERM is platform behaviour, and this is a test of the escalation
    logic, not of bash.
    """

    class _Stubborn:
        pid = 4242

        def __init__(self) -> None:
            self.terminated_normally = False
            self.killed = False

        def poll(self) -> int | None:
            return None if not self.killed else -9

        def terminate(self) -> None:
            self.terminated_normally = True

        def wait(self, timeout: float | None = None) -> int:
            if timeout is not None and not self.killed:
                raise subprocess.TimeoutExpired("bash", timeout)
            return -9

        def kill(self) -> None:
            self.killed = True

    proc = _Stubborn()
    out_handle = (tmp_path / "j.runlog").open("w", encoding="utf-8")
    err_handle = (tmp_path / "j.err").open("w", encoding="utf-8")
    dispatch._LOCAL_PROCS["local-stubborn"] = (proc, out_handle, err_handle)  # type: ignore[assignment]

    slurm.terminate_local_jobs(["local-stubborn"])

    assert proc.terminated_normally and proc.killed  # asked nicely first, then insisted
    assert out_handle.closed and err_handle.closed
    assert "local-stubborn" not in dispatch._LOCAL_PROCS


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
        slurm.wait_for_jobs(
            ["1"], poll_interval=0.01, finished=lambda _ids: set(), max_wait_seconds=0.05
        )


def test_wait_for_jobs_returns_when_all_drain():
    """No deadline configured (the default) keeps the old wait-forever behaviour."""
    slurm.wait_for_jobs(["1", "2"], poll_interval=0.01, finished=lambda ids: set(ids))


def test_wait_for_jobs_with_nothing_to_wait_for_returns_immediately():
    """An empty batch must not poll at all — and must not consult the deadline."""
    slurm.wait_for_jobs([], poll_interval=0.01, finished=_never_called, max_wait_seconds=0.0)


def _never_called(_ids: object) -> set[str]:
    """A `finished` callable that fails the test if the loop polls when it should not."""
    raise AssertionError("wait_for_jobs polled with an empty job set")

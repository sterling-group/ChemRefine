"""Tests for the engine-independent scheduler (:mod:`chemrefine.engines._execution`).

``run_batch`` is not an engine responsibility — every :class:`~chemrefine.engines._job.JobEngine`
delegates its ``submit`` to it. A minimal fake ``JobEngine`` (the only thing ``run_batch``
consumes) drives it here, proving the orchestration works for *any* job engine, not just ORCA.
The ORCA-specific submit paths (script contents, job arrays) are covered in
``test_engines_orca_engine.py``.
"""

from __future__ import annotations

import errno
import os
import subprocess
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import pytest
from ase import Atoms

from chemrefine import slurm
from chemrefine.config import StepConfig
from chemrefine.engines import _execution
from chemrefine.engines._job import JobEngine
from chemrefine.errors import (
    ChemRefineError,
    ConfigError,
    JobSubmissionError,
    OutputTreeError,
    ThrottleTimeoutError,
)
from chemrefine.slurm import dispatch
from chemrefine.state import JobBatch, PipelineState, RunBlock, StepContext, Structure


class _FakeJobEngine(JobEngine):
    """A non-ORCA job engine that supplies only the primitives ``run_batch`` consumes."""

    name: ClassVar[str] = "fake-job"
    label: ClassVar[str] = "FakeJob"
    template_suffix: ClassVar[str] = "inp"
    output_suffix: ClassVar[str] = "out"
    output_globs: ClassVar[tuple[str, ...]] = ("*.out",)
    # Not a ClassVar: the tests set it per instance to vary GPU demand.
    gpu_count: int = 0

    def build_input(self, *, xyz_path, template, input_path, output_path, ctx) -> None:
        input_path.write_text("fake input\n", encoding="utf-8")

    def run_block(self, ctx, inp_path, out_path) -> RunBlock:
        return RunBlock(body=f"echo run {inp_path.name}")

    def pal(self, ctx) -> int:
        return 1

    def gpus(self, ctx) -> int:
        return self.gpu_count

    def parse_one(self, output_path, structure_id, ctx):
        """These tests drive submission only; nothing here reads an output back."""
        raise NotImplementedError


def _ctx(
    tmp_path: Path, *, ids=("0",), max_gpus=None, slurm_array=False, dispatch="auto"
) -> StepContext:
    template_dir = tmp_path / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    (template_dir / "step1.inp").write_text("template\n", encoding="utf-8")
    (template_dir / "cpu.slurm.header").write_text("#!/bin/bash\n", encoding="utf-8")
    (template_dir / "cuda.slurm.header").write_text(
        "#!/bin/bash\n#SBATCH --gres=gpu:1\n", encoding="utf-8"
    )
    seeds = tuple(Structure(id=i, atoms=Atoms("H", positions=[[0, 0, 0]])) for i in ids)
    return StepContext(
        step_cfg=StepConfig(step=1, engine="fake-job", operation="opt_sp"),
        step_dir=tmp_path / "outputs" / "step1",
        template_dir=template_dir,
        template=template_dir / "step1.inp",
        operation="opt_sp",
        scratch_dir=tmp_path / "scratch",
        prev_state=PipelineState(structures=seeds),
        charge=0,
        multiplicity=1,
        max_cores=4,
        max_gpus=max_gpus,
        slurm_template="cpu.slurm.header",
        executables={},
        slurm_array=slurm_array,
        dispatch=dispatch,
    )


@patch.object(slurm, "finished_jobs", side_effect=lambda ids, **_: set(ids))
@patch.object(slurm, "submit")
def test_run_batch_submits_one_job_per_structure(submit_mock, _finished_jobs, tmp_path: Path):
    """One SLURM script + one submission per input, all mapped in the returned batch."""
    submit_mock.side_effect = ["1001", "1002"]
    engine = _FakeJobEngine()
    ctx = _ctx(tmp_path, ids=("0", "1"))
    inputs = engine.prepare(ctx)
    batch = _execution.run_batch(engine, inputs, ctx)
    assert isinstance(batch, JobBatch)
    assert set(batch.jobs.values()) == {"1001", "1002"}
    for inp, _out, _sid in inputs.files:
        assert (
            inp.with_suffix(".slurm")
            .read_text(encoding="utf-8")
            .splitlines()[-1]
            .startswith("echo run")
        )


class _ThreadedJobEngine(_FakeJobEngine):
    """A job engine that spells its cores as one threaded task, Q-Chem-style."""

    name: ClassVar[str] = "fake-threaded"
    label: ClassVar[str] = "FakeThreaded"

    def pal(self, ctx) -> int:
        return 4

    def slurm_layout(self, ctx) -> tuple[int, int]:
        return (1, min(self.pal(ctx), ctx.max_cores))


class _OneNodeJobEngine(_FakeJobEngine):
    """Ranks that must share a node — the ExtOpt shape."""

    def single_node(self, ctx) -> bool:
        return True


@patch.object(slurm, "finished_jobs", side_effect=lambda ids, **_: set(ids))
@patch.object(slurm, "submit", return_value="1001")
def test_a_single_node_declaration_reaches_the_generated_script(_submit, _finished, tmp_path: Path):
    """The engine's `single_node` decides whether the script pins `--nodes=1`."""
    for engine, pinned in ((_OneNodeJobEngine(), True), (_FakeJobEngine(), False)):
        ctx = _ctx(tmp_path)
        inputs = engine.prepare(ctx)
        _execution.run_batch(engine, inputs, ctx)
        text = inputs.files[0][0].with_suffix(".slurm").read_text(encoding="utf-8")
        assert ("#SBATCH --nodes=1" in text) is pinned


@patch.object(slurm, "finished_jobs", side_effect=lambda ids, **_: set(ids))
@patch.object(slurm, "submit", return_value="1001")
def test_a_threads_layout_reaches_the_generated_script(_submit, _finished, tmp_path: Path):
    """The engine's `slurm_layout` decides the SBATCH pair, not a hardcoded ranks spelling."""
    engine = _ThreadedJobEngine()
    ctx = _ctx(tmp_path)
    inputs = engine.prepare(ctx)
    _execution.run_batch(engine, inputs, ctx)
    text = inputs.files[0][0].with_suffix(".slurm").read_text(encoding="utf-8")
    assert "#SBATCH --ntasks=1" in text
    assert "#SBATCH --cpus-per-task=4" in text
    assert "cores=4" in text  # the runlog reports the product


@patch.object(slurm, "finished_jobs", side_effect=lambda ids, **_: set(ids))
@patch.object(slurm, "submit", return_value="1001")
def test_a_layout_exceeding_the_budget_is_refused(_submit, _finished, tmp_path: Path):
    """A named factorization cannot be silently clamped — over budget is a ConfigError.

    The default layout clamps itself to `max_cores` (the historical behavior); a layout
    that names both factors is the job's shape, and squeezing the product would quietly
    hand an MPI x threads job a different shape than its input was built for.
    """
    engine = _ThreadedJobEngine()
    ctx = _ctx(tmp_path)  # max_cores=4
    inputs = engine.prepare(ctx)
    with (
        patch.object(_ThreadedJobEngine, "slurm_layout", return_value=(2, 4)),
        pytest.raises(ConfigError, match=r"8 cores, more than max_cores=4"),
    ):
        _execution.run_batch(engine, inputs, ctx)


def test_a_gpu_step_with_no_known_budget_is_not_refused_for_it(tmp_path: Path):
    """``None`` is "no bound known" (``validate`` with ``max_gpus`` unset), not zero GPUs."""
    engine = _FakeJobEngine()
    engine.gpu_count = 1
    shape = _execution.check_batch(engine, _ctx(tmp_path), local=True, gpu_budget=None)
    assert shape == _execution.BatchShape(ntasks=1, cpus_per_task=1, gpus=1)
    assert shape.cores == 1


def test_a_degenerate_layout_is_an_engine_bug(tmp_path: Path):
    """A zero in either factor is a broken engine, reported as such — not a throttler crash."""
    engine = _ThreadedJobEngine()
    ctx = _ctx(tmp_path)
    with (
        patch.object(_ThreadedJobEngine, "slurm_layout", return_value=(0, 4)),
        pytest.raises(ChemRefineError, match="both must be at least 1"),
    ):
        _execution.check_batch(engine, ctx, local=True, gpu_budget=None)


@patch.object(slurm, "finished_jobs", side_effect=lambda ids, **_: set(ids))
@patch.object(slurm, "submit", return_value="1001")
def test_declared_memory_reaches_the_generated_script(_submit, _finished, tmp_path: Path):
    """An engine's `memory_mb` becomes the job's `--mem-per-cpu`; the default touches nothing."""
    engine = _FakeJobEngine()
    ctx = _ctx(tmp_path)
    inputs = engine.prepare(ctx)
    with patch.object(_FakeJobEngine, "memory_mb", return_value=2000):
        _execution.run_batch(engine, inputs, ctx)
    text = inputs.files[0][0].with_suffix(".slurm").read_text(encoding="utf-8")
    assert "#SBATCH --mem-per-cpu=2000" in text


def test_header_name_picks_cuda_for_a_gpu_step(tmp_path: Path):
    """A GPU-demanding engine auto-selects the cuda header; a per-step override wins."""
    engine = _FakeJobEngine()
    ctx = _ctx(tmp_path)
    assert _execution._header_name(engine, ctx) == "cpu.slurm.header"
    engine.gpu_count = 1
    assert _execution._header_name(engine, ctx) == "cuda.slurm.header"
    override = ctx.step_cfg.model_copy(update={"slurm_template": "special.header"})
    assert _execution._header_name(engine, replace(ctx, step_cfg=override)) == "special.header"


@patch.object(slurm, "finished_jobs", side_effect=lambda ids, **_: set(ids))
@patch.object(slurm, "submit", return_value="local-1")
def test_run_batch_dispatch_local_skips_array_and_submits_locally(
    submit_mock, _finished_jobs, tmp_path: Path
):
    """`dispatch: local` takes the per-job path (no sbatch --array) even with
    slurm_array set and an sbatch binary on PATH, and threads the mode to submit."""
    engine = _FakeJobEngine()
    ctx = _ctx(tmp_path, slurm_array=True, dispatch="local")
    inputs = engine.prepare(ctx)
    with patch.object(dispatch, "sbatch_available", return_value=True):
        batch = _execution.run_batch(engine, inputs, ctx)
    assert set(batch.jobs.values()) == {"local-1"}
    assert submit_mock.call_args.kwargs["dispatch"] == "local"


@patch.object(slurm, "finished_jobs", side_effect=lambda ids, **_: set(ids))
@patch.object(slurm, "submit", return_value="9001")
def test_run_batch_rejects_a_gpu_step_over_the_budget(_submit, _finished_jobs, tmp_path: Path):
    """Demanding more GPUs than the budget is a ConfigError, not a throttler traceback.

    Uses a single-GPU step against ``max_gpus=0`` so this exercises the *budget*
    check specifically — a >1-GPU step would trip the local-pinning guard first
    (see :func:`test_run_batch_rejects_multi_gpu_step_locally`).
    """
    engine = _FakeJobEngine()
    engine.gpu_count = 1
    ctx = _ctx(tmp_path, max_gpus=0)
    inputs = engine.prepare(ctx)
    with pytest.raises(ConfigError, match="budget"):
        _execution.run_batch(engine, inputs, ctx)


@patch.object(slurm, "finished_jobs", side_effect=lambda ids, **_: set(ids))
@patch.object(slurm, "submit", return_value="9001")
def test_run_batch_rejects_a_gpu_step_over_the_budget_as_an_array_too(
    _submit, _finished_jobs, tmp_path: Path
):
    """The same configuration, judged the same way whichever path submits it.

    The GPU checks used to sit *after* the array/queue fork, so `max_gpus: 0` with a CUDA
    step raised per job and submitted silently under `slurm_array: true` — a step judged by
    the path it happened to take rather than by what it asked for.
    """
    engine = _FakeJobEngine()
    engine.gpu_count = 1
    ctx = _ctx(tmp_path, max_gpus=0, slurm_array=True, dispatch="slurm")
    inputs = engine.prepare(ctx)
    with (
        patch.object(dispatch, "sbatch_available", return_value=True),
        patch.object(slurm, "submit_array") as submit_array,
        pytest.raises(ConfigError, match="budget"),
    ):
        _execution.run_batch(engine, inputs, ctx)
    submit_array.assert_not_called()


@patch.object(slurm, "finished_jobs", side_effect=lambda ids, **_: set(ids))
@patch.object(slurm, "submit", return_value="9001")
def test_run_batch_rejects_multi_gpu_step_locally(_submit, _finished_jobs, tmp_path: Path):
    """Local dispatch pins exactly one device per job, so >1 GPU must fail loudly.

    ``Throttler.assign_device`` hands out a single index and ``run_batch`` exports
    it as one ``CUDA_VISIBLE_DEVICES`` value; without this guard the job would be
    charged for N GPUs but pinned to one.
    """
    engine = _FakeJobEngine()
    engine.gpu_count = 2
    ctx = _ctx(tmp_path, max_gpus=4, dispatch="local")
    inputs = engine.prepare(ctx)
    with pytest.raises(ConfigError, match="pins one device"):
        _execution.run_batch(engine, inputs, ctx)


@patch.object(slurm, "finished_jobs", side_effect=lambda ids, **_: set(ids))
@patch.object(slurm, "submit", return_value="9001")
def test_run_batch_allows_multi_gpu_step_under_slurm(_submit, _finished_jobs, tmp_path: Path):
    """Under SLURM the scheduler places the GPUs, so >1 per job is fine."""
    engine = _FakeJobEngine()
    engine.gpu_count = 2
    ctx = _ctx(tmp_path, max_gpus=4, dispatch="slurm")
    inputs = engine.prepare(ctx)
    with patch.object(dispatch, "sbatch_available", return_value=True):
        batch = _execution.run_batch(engine, inputs, ctx)
    assert batch.jobs


# ---------------------------------------------------------------------------
# An abnormal exit must not orphan local jobs
# ---------------------------------------------------------------------------


def test_run_batch_terminates_local_jobs_when_a_submission_fails(tmp_path: Path):
    """A mid-batch failure unwinds through the cleanup, not past it.

    _LOCAL_PROCS is module-global, so without the unwind any exception between the
    first submit and the drain — a throttle timeout, a submission error, Ctrl-C — leaves real
    background children running. They keep competing for the cores of whatever the user runs
    next, and their log handles stay open.
    """
    engine = _FakeJobEngine()
    ctx = _ctx(tmp_path, ids=("0", "1", "2"))
    inputs = engine.prepare(ctx)
    terminated: list[tuple[str, ...]] = []

    def failing_submit(script_path, *, env=None, dispatch="auto"):
        if len(terminated) or script_path.stem.endswith("_1"):
            raise JobSubmissionError("sbatch refused the job")
        return "local-1"

    with (
        patch.object(slurm, "submit", side_effect=failing_submit),
        patch.object(slurm, "finished_jobs", side_effect=lambda ids, **_: set()),
        patch.object(
            slurm, "terminate_local_jobs", side_effect=lambda ids=(): terminated.append(tuple(ids))
        ),
        pytest.raises(JobSubmissionError),
    ):
        _execution.run_batch(engine, inputs, ctx)

    assert terminated, "the cleanup never ran — local jobs would be orphaned"
    assert "local-1" in terminated[0], "the already-submitted job must be swept up"
    # The killed local job's lease goes with it; nothing else was submitted, so the
    # ledger must be gone rather than left as an empty file.
    assert not slurm.lease_path(ctx.step_dir).exists()


def test_an_unwinding_batch_keeps_its_slurm_leases(tmp_path: Path):
    """The unwind kills local jobs but not SLURM ones — and the ledger says exactly that.

    A driver that dies mid-batch leaves its SLURM jobs running by design; their recorded
    leases are the only evidence the resume fence has, so the exception path must keep
    them while dropping the leases of the local jobs it just terminated.
    """
    engine = _FakeJobEngine()
    ctx = _ctx(tmp_path, ids=("0", "1", "2"))
    inputs = engine.prepare(ctx)

    with (
        patch.object(
            slurm,
            "submit",
            side_effect=["12345", "local-1", JobSubmissionError("sbatch refused the job")],
        ),
        patch.object(slurm, "finished_jobs", side_effect=lambda ids, **_: set()),
        patch.object(slurm, "terminate_local_jobs"),
        pytest.raises(JobSubmissionError),
    ):
        _execution.run_batch(engine, inputs, ctx)

    assert [lease.id for lease in slurm.load_leases(ctx.step_dir)] == ["12345"]


@patch.object(slurm, "finished_jobs", side_effect=lambda ids, **_: set(ids))
@patch.object(slurm, "submit")
def test_the_ledger_holds_the_jobs_in_flight_not_the_whole_batch(submit, _finished, tmp_path: Path):
    """With one core, each job is reaped before the next is submitted.

    So each submission finds the ledger holding the one job before it. A ledger of every
    job the batch ever submitted grows with the batch, and after a hard-killed driver the
    fence probes every pid it ever ran, recycled ones included.
    """
    engine = _FakeJobEngine()
    ctx = replace(_ctx(tmp_path, ids=("0", "1", "2")), max_cores=1)
    seen: list[list[str]] = []

    def _submit(*_args: object, **_kwargs: object) -> str:
        seen.append([lease.id for lease in slurm.load_leases(ctx.step_dir)])
        return f"j{len(seen) - 1}"

    submit.side_effect = _submit
    _execution.run_batch(engine, engine.prepare(ctx), ctx)
    assert seen == [[], ["j0"], ["j1"]]


@patch.object(slurm, "finished_jobs", side_effect=lambda ids, **_: set(ids))
@patch.object(slurm, "submit")
def test_a_header_edited_mid_batch_reaches_the_next_batch_not_the_running_one(
    submit, _finished, tmp_path: Path
):
    """A batch is built from the header it read when it started.

    Every script of a running batch carries the reading its plan was checked with; an
    edit lands with the step's next batch.
    """
    engine = _FakeJobEngine()
    ctx = _ctx(tmp_path, ids=("0", "1"))
    header = ctx.template_dir / "cpu.slurm.header"
    header.write_text("#!/bin/bash\n#SBATCH --partition=first\n", encoding="utf-8")

    def _submit(script_path: Path, **_: object) -> str:
        header.write_text("#!/bin/bash\n#SBATCH --partition=second\n", encoding="utf-8")
        return f"j-{script_path.stem}"

    submit.side_effect = _submit
    _execution.run_batch(engine, engine.prepare(ctx), ctx)
    scripts = sorted(ctx.step_dir.rglob("*.slurm"))
    assert len(scripts) == 2
    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert "#SBATCH --partition=first" in text
        assert "--partition=second" not in text

    _execution.run_batch(engine, engine.prepare(ctx), ctx)
    assert all(
        "#SBATCH --partition=second" in script.read_text(encoding="utf-8")
        for script in sorted(ctx.step_dir.rglob("*.slurm"))
    )


def _ledger_full_from(job_id: str) -> Callable[[Path, str | Path], Path]:
    """``Path.replace`` refusing every ledger write from the first one that records ``job_id``.

    The ledger's rename is where its write lands, and the disk stays full afterwards, as a
    real one does — which is what an unwind's own release meets after the refusal.
    """
    real_replace = Path.replace
    full = False

    def replace(self: Path, target: str | Path) -> Path:
        nonlocal full
        if Path(target).name == slurm.ACTIVE_JOBS_NAME:
            full = full or f'"{job_id}"' in self.read_text(encoding="utf-8")
            if full:
                raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), str(self))
        return real_replace(self, target)

    return replace


def _scancel_answers(answer: object) -> Callable[..., object]:
    """``subprocess.run`` with ``scancel`` answered by ``answer``; anything else runs for real."""
    real_run = subprocess.run

    def run(argv, *args, **kwargs):
        if argv[0] != "scancel":
            return real_run(argv, *args, **kwargs)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    return run


def test_a_failed_ledger_write_leaves_its_local_job_in_the_unwinds_sweep(tmp_path: Path):
    """Registered before it is leased, so the unwind terminates the job it was recording.

    A local job is the unwind's to stop, so the refused write cancels nothing itself: the
    output tree's refusal goes on as it came.
    """
    engine = _FakeJobEngine()
    ctx = _ctx(tmp_path, dispatch="local")
    terminated: list[tuple[str, ...]] = []
    with (
        patch.object(slurm, "submit", return_value="local-1"),
        patch.object(Path, "replace", _ledger_full_from("local-1")),
        patch.object(dispatch.subprocess, "run", side_effect=_scancel_answers(None)) as run,
        patch.object(
            slurm, "terminate_local_jobs", side_effect=lambda ids=(): terminated.append(tuple(ids))
        ),
        pytest.raises(OutputTreeError) as refusal,
    ):
        _execution.run_batch(engine, engine.prepare(ctx), ctx)
    assert str(refusal.value) == (
        f"cannot write {slurm.lease_path(ctx.step_dir)}: No space left on device"
    )
    assert terminated == [("local-1",)]
    assert not [c for c in run.call_args_list if c.args[0][0] == "scancel"]


@pytest.mark.parametrize(
    ("scancel", "said"),
    [
        (
            subprocess.CompletedProcess(["scancel", "4242"], 0, "", ""),
            "so the run cancelled it (scancel 4242).",
        ),
        (
            subprocess.CalledProcessError(1, "scancel", "", "Unable to contact slurm controller\n"),
            "cancelling it failed (Unable to contact slurm controller): `scancel 4242`, or "
            "wait until `squeue -j 4242` no longer lists it, before running the command again.",
        ),
        (
            FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), "scancel"),
            "cancelling it failed ([Errno 2] No such file or directory: 'scancel'): `scancel 4242`",
        ),
    ],
    ids=["cancelled", "controller-unreachable", "no-scancel"],
)
def test_a_slurm_job_whose_lease_cannot_be_written_is_cancelled_or_named(
    tmp_path: Path, scancel: object, said: str
):
    """Accepted but unrecorded, a SLURM job is invisible to the next run's fence.

    The unwind leaves SLURM jobs running because their leases fence the next run; this one
    has none. So the refused write cancels exactly that job, and the refusal names it and
    what became of the cancel. The job recorded before it keeps its lease.
    """
    engine = _FakeJobEngine()
    ctx = replace(_ctx(tmp_path, ids=("0", "1"), dispatch="slurm"), max_cores=2)
    with (
        patch.object(dispatch, "sbatch_available", return_value=True),
        patch.object(slurm, "submit", side_effect=["4241", "4242"]),
        patch.object(slurm, "finished_jobs", side_effect=lambda ids, **_: set(ids)),
        patch.object(Path, "replace", _ledger_full_from("4242")),
        patch.object(dispatch.subprocess, "run", side_effect=_scancel_answers(scancel)) as run,
        pytest.raises(OutputTreeError) as refusal,
    ):
        _execution.run_batch(engine, engine.prepare(ctx), ctx)

    assert [c.args[0] for c in run.call_args_list] == [["scancel", "4242"]]
    assert str(refusal.value).startswith(
        f"cannot write {slurm.lease_path(ctx.step_dir)}: No space left on device. SLURM had "
        "already accepted job 4242, which no ledger records, so "
    )
    assert said in str(refusal.value)
    assert isinstance(refusal.value.__cause__, OSError), "the disk's refusal is the cause"
    assert [lease.id for lease in slurm.load_leases(ctx.step_dir)] == ["4241"]


def test_an_array_chunk_whose_lease_cannot_be_written_is_cancelled_before_any_wait(
    tmp_path: Path,
):
    """The array path has no unwind, so the refused write is the one place to act."""
    engine = _FakeJobEngine()
    ctx = _ctx(tmp_path, ids=("0", "1"), slurm_array=True, dispatch="slurm")
    ok = subprocess.CompletedProcess(["scancel", "4242"], 0, "", "")
    with (
        patch.object(dispatch, "sbatch_available", return_value=True),
        patch("chemrefine.slurm.script._MAX_ARRAY_SIZE", 1),
        patch.object(slurm, "submit_array", side_effect=["4241", "4242"]),
        patch.object(slurm, "wait_for_jobs") as wait,
        patch.object(Path, "replace", _ledger_full_from("4242")),
        patch.object(dispatch.subprocess, "run", side_effect=_scancel_answers(ok)) as run,
        pytest.raises(OutputTreeError, match=r"so the run cancelled it \(scancel 4242\)"),
    ):
        _execution.run_batch(engine, engine.prepare(ctx), ctx)
    assert [c.args[0] for c in run.call_args_list] == [["scancel", "4242"]]
    wait.assert_not_called()
    assert [lease.id for lease in slurm.load_leases(ctx.step_dir)] == ["4241"]


def test_a_release_the_tree_refuses_does_not_replace_the_error_that_unwound_the_batch(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """The batch's own error is its answer; a ledger the disk will not remove is logged.

    The unwind drops the killed local job's lease on the read-only disk it may be stopping
    for, and that refusal must not take the place of the submission failure in flight.
    """
    engine = _FakeJobEngine()
    ctx = _ctx(tmp_path, ids=("0", "1"), dispatch="local")
    ledger = slurm.lease_path(ctx.step_dir)
    real_unlink = Path.unlink

    def read_only(self: Path, missing_ok: bool = False) -> None:
        if self == ledger:
            raise OSError(errno.EROFS, os.strerror(errno.EROFS), str(self))
        real_unlink(self, missing_ok=missing_ok)

    with (
        patch.object(
            slurm, "submit", side_effect=["local-1", JobSubmissionError("sbatch refused the job")]
        ),
        patch.object(slurm, "finished_jobs", side_effect=lambda ids, **_: set()),
        patch.object(slurm, "terminate_local_jobs"),
        patch.object(Path, "unlink", read_only),
        pytest.raises(JobSubmissionError),
    ):
        _execution.run_batch(engine, engine.prepare(ctx), ctx)
    assert (
        f"could not clean up during JobSubmissionError: cannot write {ledger}: "
        "Read-only file system" in caplog.text
    )


def test_run_batch_touches_nothing_on_the_happy_path(tmp_path: Path):
    """A clean drain sweeps no jobs and leaves no leases behind.

    The termination sweep lives on the unwind path only — it exists to reap local jobs
    an exception would orphan, and after an ordinary drain the queue has already reaped
    everything. What the clean path *does* do is release the batch's job leases: with
    every job finished there is nothing left for the resume fence to check.
    """
    engine = _FakeJobEngine()
    ctx = _ctx(tmp_path, ids=("0",))
    inputs = engine.prepare(ctx)
    seen: list[tuple[str, ...]] = []

    with (
        patch.object(slurm, "submit", return_value="local-1"),
        patch.object(slurm, "finished_jobs", side_effect=lambda ids, **_: set(ids)),
        patch.object(
            slurm, "terminate_local_jobs", side_effect=lambda ids=(): seen.append(tuple(ids))
        ),
    ):
        _execution.run_batch(engine, inputs, ctx)

    assert seen == []  # the queue reaped everything; the sweep is for unwinds only
    assert not slurm.lease_path(ctx.step_dir).exists()


# ---------------------------------------------------------------------------
# job_timeout_seconds — the deadline that makes exit code 8 reachable
# ---------------------------------------------------------------------------


@patch.object(slurm, "finished_jobs", side_effect=lambda ids, **_: set())
@patch.object(slurm, "submit", side_effect=lambda *a, **k: "1001")
def test_run_batch_raises_when_jobs_outlive_the_configured_timeout(
    _submit, _finished, tmp_path: Path
):
    """A job that never finishes must eventually fail the run, not block it forever.

    `Throttler` has always accepted `max_wait_seconds` and `ThrottleTimeoutError` has always
    owned exit code 8 — but no caller ever passed a deadline, so the error was unreachable
    and a stuck job blocked the pipeline indefinitely with no diagnostic. The unwind that
    reaps local jobs could not run either, because nothing raised.
    """
    engine = _FakeJobEngine()
    ctx = replace(_ctx(tmp_path), job_timeout_seconds=0.05)
    inputs = engine.prepare(ctx)

    with pytest.raises(ThrottleTimeoutError):
        _execution.run_batch(engine, inputs, ctx)


def test_run_batch_waits_indefinitely_when_no_timeout_is_configured(tmp_path: Path):
    """`None` is the default and waits for as long as the job takes, with no deadline.

    The job finishes on the second poll, so the run has to sit through an empty one: a
    `None` read as a zero-second budget raises there.
    """
    engine = _FakeJobEngine()
    ctx = _ctx(tmp_path)
    assert ctx.job_timeout_seconds is None
    polls: list[set[str]] = []

    def finished(ids, **_):
        polls.append(set(ids))
        return set(ids) if len(polls) > 1 else set()

    with (
        patch.object(slurm, "submit", side_effect=lambda *a, **k: "1001"),
        patch.object(slurm, "finished_jobs", side_effect=finished),
        patch("chemrefine.throttle.time.sleep"),
    ):
        assert _execution.run_batch(engine, engine.prepare(ctx), ctx).jobs
    assert len(polls) >= 2, "it waited through a poll that finished nothing"


# ---------------------------------------------------------------------------
# Streaming: the queue that can grow
# ---------------------------------------------------------------------------


class _Sink:
    """A ``CompletionSink`` that hands back a canned follow-up for named jobs."""

    def __init__(self, follow_ups: dict[str, tuple] | None = None) -> None:
        self.seen: list[str] = []
        self._follow_ups = dict(follow_ups or {})

    def on_complete(self, job):
        self.seen.append(job[2])
        return self._follow_ups.pop(job[2], ())


@patch.object(slurm, "finished_jobs", side_effect=lambda ids, **_: set(ids))
@patch.object(slurm, "submit")
def test_follow_ups_queue_behind_the_work_already_pending(submit, _finished, tmp_path: Path):
    """A re-run waits its turn; it does not jump into the slot its own job just freed.

    Original work is known to be needed and a follow-up is speculative, so admitting the
    retry first would delay the batch for work that may fail again. With one slot, that makes
    the submission order observable: the retry of the *first* structure must come last.
    """
    submit.side_effect = ["j0", "j1", "j2", "j-retry"]
    engine = _FakeJobEngine()
    ctx = replace(_ctx(tmp_path, ids=("0", "1", "2")), max_cores=1)
    inputs = engine.prepare(ctx)
    retry = (tmp_path / "retry.inp", tmp_path / "retry.out", "0-retry")
    sink = _Sink({"0": (retry,)})

    _execution.run_batch(engine, inputs, ctx, sink=sink)

    submitted = [call.args[0].stem for call in submit.call_args_list]
    assert submitted == ["step1_0", "step1_1", "step1_2", "retry"]
    assert sink.seen == ["0", "1", "2", "0-retry"]


@patch.object(slurm, "finished_jobs", side_effect=lambda ids, **_: set(ids))
@patch.object(slurm, "submit")
def test_a_sink_that_asks_for_nothing_matches_the_unstreamed_path(submit, _fin, tmp_path: Path):
    """Streaming must be a scheduling change only — same jobs, same mapping."""
    engine = _FakeJobEngine()
    ctx = _ctx(tmp_path, ids=("0", "1"))

    submit.side_effect = ["a", "b"]
    plain = _execution.run_batch(engine, engine.prepare(ctx), ctx)
    submit.side_effect = ["a", "b"]
    streamed = _execution.run_batch(engine, engine.prepare(ctx), ctx, sink=_Sink())

    assert streamed.jobs == plain.jobs


@patch.object(slurm, "finished_jobs", side_effect=lambda ids, **_: set(ids))
@patch.object(slurm, "submit")
def test_array_mode_still_drives_the_sink(submit, _finished, tmp_path: Path):
    """The array path cannot stream, but it must not swallow the sink either.

    ORCA *is* a streaming engine, so a `slurm_array: true` step still arrives here with a
    sink. Dropping it would leave every structure unparsed — the ledger empty and the step
    failing for a reason nothing points at — so the batch is swept once it has drained and
    the follow-ups go out as another array.
    """
    submit.side_effect = ["arr-0", "arr-1"]
    engine = _FakeJobEngine()
    ctx = _ctx(tmp_path, ids=("0", "1"), slurm_array=True, dispatch="slurm")
    inputs = engine.prepare(ctx)
    retry = (tmp_path / "retry.inp", tmp_path / "retry.out", "0-retry")
    sink = _Sink({"0": (retry,)})

    with (
        patch.object(dispatch, "sbatch_available", return_value=True),
        patch.object(slurm, "submit_array", side_effect=["100", "200"]),
        patch.object(slurm, "wait_for_jobs"),
    ):
        batch = _execution.run_batch(engine, inputs, ctx, sink=sink)

    assert sink.seen == ["0", "1", "0-retry"]
    # The follow-up array's ids are merged in, not discarded with the recursive batch.
    assert set(batch.jobs.values()) == {"100", "200"}


@patch.object(slurm, "finished_jobs", side_effect=lambda ids, **_: set(ids))
@patch.object(slurm, "submit")
def test_multi_chunk_arrays_share_one_core_budget(_submit, _finished, tmp_path: Path):
    """A step past the per-array task cap is several arrays, all queued at once.

    Each carries its own `%limit`, so dividing by `pal` alone hands the *whole* budget to
    every chunk: a 2500-structure step at `pal: 8` against `max_cores: 512` ran 3 x 64
    tasks = 1536 cores, on a step whose config docstring promises the scheduler is
    enforcing exactly that number. The chunk size is patched down rather than building
    2500 structures, because what is under test is the arithmetic, not the chunking.
    """
    engine = _FakeJobEngine()
    ctx = replace(
        _ctx(tmp_path, ids=tuple(str(i) for i in range(6)), slurm_array=True, dispatch="slurm"),
        max_cores=12,
    )
    inputs = engine.prepare(ctx)

    with (
        patch.object(dispatch, "sbatch_available", return_value=True),
        patch("chemrefine.slurm.script._MAX_ARRAY_SIZE", 2),
        patch.object(slurm, "submit_array", side_effect=["1", "2", "3"]) as submit_array,
        patch.object(slurm, "wait_for_jobs"),
    ):
        _execution.run_batch(engine, inputs, ctx)

    limits = [call.kwargs["max_concurrent"] for call in submit_array.call_args_list]
    assert len(limits) == 3, "the chunking this test depends on did not happen"
    # _FakeJobEngine.pal is 1, clamped to max_cores; 12 // (1 * 3) = 4 per chunk. The
    # exact shares, not a `sum(...) <= budget` bound: any under-allocating mutant of the
    # arithmetic — an extra chunk factor, a wrong divisor — still satisfies the
    # inequality while a 2500-structure step silently runs at a fraction of max_cores.
    assert limits == [4, 4, 4]
    assert sum(limits) * engine.pal(ctx) <= ctx.max_cores  # and the shares fit the budget


def test_an_array_batch_is_leased_before_the_wait_and_released_after(tmp_path: Path):
    """The parent id is on record while the wait runs, and gone once it drains.

    Recording after the wait would be recording nothing: the wait is where a driver
    dies, and the lease exists precisely so that death leaves evidence behind.
    """
    engine = _FakeJobEngine()
    ctx = _ctx(tmp_path, ids=("0", "1"), slurm_array=True, dispatch="slurm")
    inputs = engine.prepare(ctx)
    at_wait_time: list[str] = []

    def observing_wait(ids, **_):
        at_wait_time.extend(lease.id for lease in slurm.load_leases(ctx.step_dir))

    with (
        patch.object(dispatch, "sbatch_available", return_value=True),
        patch.object(slurm, "submit_array", return_value="900"),
        patch.object(slurm, "wait_for_jobs", side_effect=observing_wait),
    ):
        _execution.run_batch(engine, inputs, ctx)

    assert at_wait_time == ["900"]
    assert not slurm.lease_path(ctx.step_dir).exists()


def test_an_interrupted_array_wait_keeps_the_lease(tmp_path: Path):
    """A wait that dies leaves the parent id on record — array jobs are all SLURM's."""
    engine = _FakeJobEngine()
    ctx = _ctx(tmp_path, ids=("0",), slurm_array=True, dispatch="slurm")
    inputs = engine.prepare(ctx)

    with (
        patch.object(dispatch, "sbatch_available", return_value=True),
        patch.object(slurm, "submit_array", return_value="901"),
        patch.object(slurm, "wait_for_jobs", side_effect=ThrottleTimeoutError("stalled")),
        pytest.raises(ThrottleTimeoutError),
    ):
        _execution.run_batch(engine, inputs, ctx)

    assert [lease.id for lease in slurm.load_leases(ctx.step_dir)] == ["901"]


@patch.object(slurm, "finished_jobs", side_effect=lambda ids, **_: set(ids))
@patch.object(slurm, "submit")
def test_pending_work_that_can_never_be_submitted_raises(submit, _finished, tmp_path: Path):
    """Silence is the danger here: the jobs would simply never run.

    Unreachable through the real call path (``pal`` is clamped to the budget), so the
    condition is forced. Breaking out of the loop instead would strand the queue and surface
    a layer away as a structure with no result and no explanation.

    Not a ``ConfigError``: the config was already judged above, with its own exit code.
    Reaching here means the scheduler broke its own invariant, which is a different claim.
    """
    submit.side_effect = ["j0"]
    engine = _FakeJobEngine()
    ctx = _ctx(tmp_path, ids=("0", "1"))
    inputs = engine.prepare(ctx)

    with (
        patch.object(_execution.throttle.Throttler, "has_room", return_value=False),
        pytest.raises(ChemRefineError, match="left unsubmitted"),
    ):
        _execution.run_batch(engine, inputs, ctx, sink=_Sink())


@patch.object(slurm, "finished_jobs", side_effect=lambda ids, **_: set(ids))
@patch.object(slurm, "submit")
def test_local_jobs_are_terminated_when_the_sink_raises(submit, _finished, tmp_path: Path):
    """The cleanup guarantee holds on the streaming path too."""
    submit.side_effect = ["local-1", "local-2"]
    engine = _FakeJobEngine()
    ctx = _ctx(tmp_path, ids=("0", "1"), dispatch="local")

    class _Exploding:
        def on_complete(self, job):
            raise RuntimeError("boom")

    with (
        patch.object(slurm, "terminate_local_jobs") as terminate,
        pytest.raises(RuntimeError, match="boom"),
    ):
        _execution.run_batch(engine, engine.prepare(ctx), ctx, sink=_Exploding())
    terminate.assert_called_once()

"""Tests for the engine-independent scheduler (:mod:`chemrefine.engines._execution`).

``run_batch`` is not an engine responsibility — every :class:`~chemrefine.engines._job.JobEngine`
delegates its ``submit`` to it. A minimal fake ``JobEngine`` (the only thing ``run_batch``
consumes) drives it here, proving the orchestration works for *any* job engine, not just ORCA.
The ORCA-specific submit paths (script contents, job arrays) are covered in
``test_engines_orca_engine.py``.
"""

from __future__ import annotations

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
from chemrefine.errors import ConfigError, JobSubmissionError, ThrottleTimeoutError
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

    def build_input(self, *, xyz_path, template_path, input_path, output_path, ctx) -> None:
        input_path.write_text("fake input\n", encoding="utf-8")

    def run_block(self, ctx, inp_path, out_path) -> RunBlock:
        return RunBlock(body=f"echo run {inp_path.name}")

    def pal(self, ctx) -> int:
        return 1

    def gpus(self, ctx) -> int:
        return self.gpu_count


def _ctx(
    tmp_path: Path, *, ids=("0",), max_gpus=None, slurm_array=False, dispatch="auto"
) -> StepContext:
    template_dir = tmp_path / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    (template_dir / "step1.inp").write_text("template\n", encoding="utf-8")
    (template_dir / "cpu.slurm.header").write_text("#!/bin/bash\n", encoding="utf-8")
    (template_dir / "cuda.slurm.header").write_text("#!/bin/bash\n#SBATCH --gres=gpu:1\n")
    seeds = tuple(Structure(id=i, atoms=Atoms("H", positions=[[0, 0, 0]])) for i in ids)
    return StepContext(
        step_cfg=StepConfig(step=1, engine="fake-job", operation="opt_sp"),
        step_dir=tmp_path / "outputs" / "step1",
        template_dir=template_dir,
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
        assert inp.with_suffix(".slurm").read_text().splitlines()[-1].startswith("echo run")


def test_header_name_picks_cuda_for_a_gpu_step(tmp_path: Path):
    """A GPU-demanding engine auto-selects the cuda header; a per-step override wins."""
    engine = _FakeJobEngine()
    ctx = _ctx(tmp_path)
    assert _execution._header_name(engine, ctx) == "cpu.slurm.header"
    engine.gpu_count = 1
    assert _execution._header_name(engine, ctx) == slurm.header_name_for_device("cuda")
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
    with patch.object(slurm, "sbatch_available", return_value=True):
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
    with patch.object(slurm, "sbatch_available", return_value=True):
        batch = _execution.run_batch(engine, inputs, ctx)
    assert batch.jobs


# ---------------------------------------------------------------------------
# An abnormal exit must not orphan local jobs (B7)
# ---------------------------------------------------------------------------


def test_run_batch_terminates_local_jobs_when_a_submission_fails(tmp_path: Path):
    """A mid-batch failure unwinds through the cleanup, not past it.

    _LOCAL_PROCS is module-global and run_batch had no try/finally, so any
    exception between the first submit and wait_all — a throttle timeout, a
    submission error, Ctrl-C — left real background children running. They keep
    competing for the cores of whatever the user runs next, and their log
    handles stay open.
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


def test_run_batch_cleanup_runs_on_the_happy_path_too(tmp_path: Path):
    """The sweep is unconditional; with everything reaped it has nothing to do."""
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

    assert seen == [()]  # wait_all reaped everything; nothing left to terminate


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
    and a stuck job blocked the pipeline indefinitely with no diagnostic. The `finally` that
    reaps local jobs could not fire either, because nothing raised.
    """
    engine = _FakeJobEngine()
    ctx = replace(_ctx(tmp_path), job_timeout_seconds=0.05)
    inputs = engine.prepare(ctx)

    with pytest.raises(ThrottleTimeoutError):
        _execution.run_batch(engine, inputs, ctx)


@patch.object(slurm, "finished_jobs", side_effect=lambda ids, **_: set(ids))
@patch.object(slurm, "submit", side_effect=lambda *a, **k: "1001")
def test_run_batch_waits_indefinitely_when_no_timeout_is_configured(
    _submit, _finished, tmp_path: Path
):
    """`None` is the default and must preserve the old behaviour exactly."""
    engine = _FakeJobEngine()
    ctx = _ctx(tmp_path)
    assert ctx.job_timeout_seconds is None

    assert _execution.run_batch(engine, engine.prepare(ctx), ctx).jobs

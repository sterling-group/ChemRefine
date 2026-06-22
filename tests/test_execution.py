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
from chemrefine.errors import ConfigError
from chemrefine.state import JobBatch, PipelineState, StepContext, Structure


class _FakeJobEngine(JobEngine):
    """A non-ORCA job engine that supplies only the primitives ``run_batch`` consumes."""

    name: ClassVar[str] = "fake-job"
    label: ClassVar[str] = "FakeJob"
    template_suffix: ClassVar[str] = "inp"
    output_suffix: ClassVar[str] = "out"
    output_globs: ClassVar[tuple[str, ...]] = ("*.out",)
    gpu_count: ClassVar[int] = 0

    def build_input(self, *, xyz_path, template_path, input_path, output_path, ctx) -> None:
        input_path.write_text("fake input\n", encoding="utf-8")

    def run_block(self, ctx, inp_path, out_path) -> str:
        return f"echo run {inp_path.name}"

    def pal(self, ctx) -> int:
        return 1

    def gpus(self, ctx) -> int:
        return self.gpu_count


def _ctx(tmp_path: Path, *, ids=("0",), max_gpus=None) -> StepContext:
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
    )


@patch.object(slurm, "is_finished", return_value=True)
@patch.object(slurm, "submit")
def test_run_batch_submits_one_job_per_structure(submit_mock, _is_finished, tmp_path: Path):
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


@patch.object(slurm, "is_finished", return_value=True)
@patch.object(slurm, "submit", return_value="9001")
def test_run_batch_rejects_a_gpu_step_over_the_budget(_submit, _is_finished, tmp_path: Path):
    """Demanding more GPUs than the budget is a ConfigError, not a throttler traceback."""
    engine = _FakeJobEngine()
    engine.gpu_count = 2
    ctx = _ctx(tmp_path, max_gpus=1)
    inputs = engine.prepare(ctx)
    with pytest.raises(ConfigError, match="GPU"):
        _execution.run_batch(engine, inputs, ctx)

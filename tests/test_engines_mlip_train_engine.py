"""Tests for the ``mlip-train`` engine — the step's own wiring.

What the engine owns: validating the step, resolving the trainer, handing the job to the
scheduler, and recording what came out. The library-specific half is
``test_engines_mlip_trainers.py``; the artifact-step lifecycle it plugs into is
``test_step_artifact.py``.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms

from chemrefine.config import StepConfig
from chemrefine.engines.api import (
    ArtifactEngine,
    JobExecutable,
    ProvisionableEngine,
    get_engine,
)
from chemrefine.engines.mlip.train.engine import SIDECAR_NAME, MlipTrainEngine
from chemrefine.errors import ConfigError
from chemrefine.state import JobBatch, PipelineState, StepContext, StepInputs, Structure

_TEMPLATE = """\
name: $RUN_NAME
work_dir: $RUN_DIR
train_file: $TRAIN_SET
valid_file: $VALID_SET
foundation_model: $FOUNDATION_MODEL
device: $DEVICE
max_num_epochs: 2
"""

_FAIRCHEM_TEMPLATE = """\
job:
  run_dir: $RUN_DIR
  timestamp_id: $RUN_NAME
  device_type: $DEVICE
train_file: $TRAIN_SET
valid_file: $VAL_SET
"""
"""FAIRChem's own placeholder names, which are not MACE's.

Its required set is ``TRAIN_SET``/``VAL_SET``/``RUN_DIR``/``RUN_NAME`` — note ``VAL_SET``,
where MACE writes ``VALID_SET``. A shared template would render for one trainer and be
rejected by the other, which is the point: the placeholders belong to the library's config."""


def _ctx(
    tmp_path: Path, *, n: int = 10, template: str = _TEMPLATE, **options: object
) -> StepContext:
    """A training step over ``n`` labelled structures."""
    template_dir = tmp_path / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    path = template_dir / "step1.yaml"
    path.write_text(template, encoding="utf-8")

    opts: dict[str, object] = {"task_name": "mace_off", "device": "cpu", "model_name": "small"}
    opts.update(options)
    structures = tuple(
        Structure(
            id=str(i),
            atoms=Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]),
            energy_hartree=-1.0 - 0.01 * i,
            forces_ev_per_a=np.zeros((2, 3)),
        )
        for i in range(n)
    )
    return StepContext(
        step_cfg=StepConfig(step=1, engine="mlip-train", options=opts),
        step_dir=tmp_path / "outputs" / "step1",
        template_dir=template_dir,
        template=path,
        scratch_dir=None,
        prev_state=PipelineState(structures=structures),
        charge=-1,
        multiplicity=1,
        max_cores=16,
        slurm_template="cpu.slurm.header",
        executables={},
    )


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


def test_the_engine_declares_every_capability_its_wiring_depends_on():
    """Each of the three routes a training step takes is selected by `isinstance`.

    `ArtifactEngine` sends it down `step._run_artifact_step`; `JobExecutable` lets
    `run_batch` schedule it; `ProvisionableEngine` is what makes the preflight check the
    backend and the launch resolve its managed environment. Losing any one of them
    silently changes which code runs rather than failing.
    """
    engine = get_engine("mlip-train")
    assert isinstance(engine, ArtifactEngine)
    assert isinstance(engine, JobExecutable)
    assert isinstance(engine, ProvisionableEngine)


def test_the_backend_requirement_follows_the_task_selection():
    engine = MlipTrainEngine()
    assert engine.backend_requirement({"task_name": "mace_off"}).extra == "mlip-mace"
    assert "mlip-mace" in engine.backend_extras()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_a_step_that_names_no_task_is_refused_with_the_trainable_list(tmp_path: Path):
    """`task_name`'s inference default names a model to *run*, which is a different choice."""
    ctx = _ctx(tmp_path)
    ctx.step_cfg.options.pop("task_name")
    with pytest.raises(ConfigError, match="must name a `task_name`"):
        MlipTrainEngine().prepare(ctx)


def test_a_step_that_names_no_device_is_refused(tmp_path: Path):
    """Neither default is honest: `cpu` grinds for days, `cuda` takes a GPU nobody asked for."""
    ctx = _ctx(tmp_path)
    ctx.step_cfg.options.pop("device")
    with pytest.raises(ConfigError, match="must name a device"):
        MlipTrainEngine().prepare(ctx)


@pytest.mark.parametrize("policy", ["skip", "best"])
def test_an_on_failure_policy_with_nothing_to_act_on_is_refused(policy: str, tmp_path: Path):
    """`skip` and `best` are per-structure policies; a training step has no such failures."""
    ctx = _ctx(tmp_path)
    ctx = replace(ctx, step_cfg=ctx.step_cfg.model_copy(update={"on_failure": policy}))
    with pytest.raises(ConfigError, match="nothing to act on"):
        MlipTrainEngine().prepare(ctx)


def test_a_typoed_knob_fails_the_step_rather_than_being_ignored(tmp_path: Path):
    ctx = _ctx(tmp_path, valid_fractoin=0.2)
    with pytest.raises(ConfigError, match="invalid mliptrain options"):
        MlipTrainEngine().prepare(ctx)


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


def test_prepare_writes_one_job_whose_output_is_the_model(tmp_path: Path):
    """One job over the whole ensemble, not one per structure."""
    ctx = _ctx(tmp_path)
    inputs = MlipTrainEngine().prepare(ctx)

    (config, model, sid) = inputs.files[0]
    assert len(inputs.files) == 1
    assert sid == "train"
    assert config.name == "step1_train.yaml"
    assert model.name == "train.model"
    assert config.parent == model.parent == ctx.step_dir / "train"


@pytest.mark.parametrize(
    ("task", "template"),
    [("mace_off", _TEMPLATE), ("omol", _FAIRCHEM_TEMPLATE)],
)
def test_the_job_writes_into_the_run_directory_whatever_the_trainer_is(
    tmp_path: Path, task: str, template: str
):
    """The job's declared output must sit **in** the run directory, not inside the model's.

    The scheduler reads only this path's parent, and uses it for the runlog, the `.err` and —
    with no `scratch_dir` — `$WORK_DIR` itself. Handing it `artifact()` is right for MACE,
    whose model lands beside its config, and wrong for FAIRChem, whose model is at
    `train/checkpoints/final/inference_ckpt.pt`: the run's own log and scratch were then
    created three directories inside the tree it was about to write its final checkpoint to.

    Parametrised over both trainers because that is the only thing that would have caught it —
    every engine-level test here ran the one backend whose two paths coincide.
    """
    ctx = _ctx(tmp_path, template=template, task_name=task)
    engine = MlipTrainEngine()

    inputs = engine.prepare(ctx)

    (config, job_output, _) = inputs.files[0]
    run_dir = ctx.step_dir / "train"
    assert job_output.parent == run_dir
    assert config.parent == run_dir
    # ...while the *product* may still be nested, and remains the step's success test.
    assert engine.artifact(ctx).is_relative_to(run_dir)
    assert engine.artifact(ctx).name == job_output.name


def test_the_fairchem_dataset_does_not_land_inside_its_own_checkpoint_tree(tmp_path: Path):
    """FAIRChem's `timestamp_id` is `train` and so is the training split — two `train`s.

    Written at the run directory's top the training set would be `train/train/train.db`, which
    is inside the directory FAIRChem creates for its checkpoints. Namespacing the splits under
    `data/` is what keeps the dataset and the model from sharing a tree.
    """
    ctx = _ctx(tmp_path, template=_FAIRCHEM_TEMPLATE, task_name="omol")
    run_dir = ctx.step_dir / "train"

    rendered = MlipTrainEngine().prepare(ctx).files[0][0].read_text()

    assert f"train_file: {run_dir / 'data' / 'train' / 'train.db'}" in rendered
    assert (run_dir / "data" / "train" / "train.db").is_file()
    assert not (run_dir / "train" / "train.db").exists()


def test_prepare_renders_the_template_against_the_dataset_it_wrote(tmp_path: Path):
    """The rendered config must point at *these* files, or the job trains on nothing."""
    ctx = _ctx(tmp_path)
    inputs = MlipTrainEngine().prepare(ctx)

    rendered = inputs.files[0][0].read_text()
    run_dir = ctx.step_dir / "train"
    assert f"train_file: {run_dir / 'train.xyz'}" in rendered
    assert f"valid_file: {run_dir / 'valid.xyz'}" in rendered
    assert f"work_dir: {run_dir}" in rendered
    assert "foundation_model: small" in rendered, "model_name is what a fine-tune starts from"
    assert (run_dir / "train.xyz").is_file()


def test_a_template_that_never_names_the_dataset_fails_before_any_job(tmp_path: Path):
    ctx = _ctx(tmp_path, template="max_num_epochs: 2\n")
    with pytest.raises(ConfigError, match="never references"):
        MlipTrainEngine().prepare(ctx)


def test_the_split_honours_the_steps_own_fractions(tmp_path: Path):
    ctx = _ctx(tmp_path, n=20, valid_fraction=0.25, test_fraction=0.1)
    MlipTrainEngine().prepare(ctx)

    run_dir = ctx.step_dir / "train"
    counts = {p.stem: p.read_text().count("Properties=") for p in run_dir.glob("*.xyz")}
    assert counts == {"train": 13, "valid": 5, "test": 2}


# ---------------------------------------------------------------------------
# submit / parse
# ---------------------------------------------------------------------------


def test_submit_goes_through_the_ordinary_scheduler(tmp_path: Path):
    """Training is throttled, headed and timed out like every other job, by using the same one."""
    ctx = _ctx(tmp_path)
    engine = MlipTrainEngine()
    inputs = engine.prepare(ctx)

    with patch(
        "chemrefine.engines._execution.run_batch", return_value=JobBatch(jobs={})
    ) as run_batch:
        engine.submit(inputs, ctx)

    passed_engine, passed_inputs, passed_ctx = run_batch.call_args.args
    assert passed_engine is engine
    assert passed_inputs is inputs
    assert passed_ctx is ctx


def test_parse_hands_the_ensemble_on_unchanged_and_records_the_model(tmp_path: Path):
    ctx = _ctx(tmp_path)
    engine = MlipTrainEngine()
    engine.prepare(ctx)
    engine.artifact(ctx).write_bytes(b"weights")

    results = engine.parse(StepInputs(files=()), ctx)

    assert results.structures == ctx.prev_state.structures
    sidecar = json.loads((ctx.step_dir / "train" / SIDECAR_NAME).read_text())
    assert sidecar["task_name"] == "mace_off"
    assert sidecar["backend"] == "mlip-mace"
    assert sidecar["started_from"] == "small"
    assert sidecar["charge"] == -1
    assert sidecar["model_path"].endswith("train.model")
    assert len(sidecar["model_digest"]) == 16


def test_the_artifact_is_answerable_without_an_ensemble(tmp_path: Path):
    """`rebuild-cache` asks after submitting nothing, and a failure asks with no split to make."""
    ctx = _ctx(tmp_path, n=0)
    assert MlipTrainEngine().artifact(ctx) == ctx.step_dir / "train" / "train.model"


# ---------------------------------------------------------------------------
# What the scheduler asks of the job
# ---------------------------------------------------------------------------


def test_an_unset_core_count_claims_the_whole_step_budget(tmp_path: Path):
    """The single job *is* the step, so `cores: 1` would leave 15 of 16 idle."""
    assert MlipTrainEngine().pal(_ctx(tmp_path)) == 16


def test_an_explicit_core_count_is_honoured(tmp_path: Path):
    """ "Unset" is read from what the YAML said, not from the inherited default of 1."""
    assert MlipTrainEngine().pal(_ctx(tmp_path, cores=1)) == 1


def test_gpu_demand_follows_the_device_the_step_named(tmp_path: Path):
    engine = MlipTrainEngine()
    assert engine.gpus(_ctx(tmp_path, device="cpu")) == 0
    assert engine.gpus(_ctx(tmp_path, device="cuda")) == 1
    assert engine.gpus(_ctx(tmp_path, device="cuda", gpus=4)) == 4


def test_the_run_block_runs_the_trainers_command_with_thread_limits(tmp_path: Path):
    ctx = _ctx(tmp_path)
    block = MlipTrainEngine().run_block(ctx, Path("step1_train.yaml"), Path("train.model"))

    assert "export OMP_NUM_THREADS=16" in block.body
    assert "-m mace.cli.run_train --config step1_train.yaml" in block.body
    assert block.cleanup == "", "the job script owns the one exit handler"


def test_the_plans_cores_are_the_grant_not_the_ask(tmp_path: Path):
    """A ``cores:`` above ``max_cores`` reaches the plan — and the exports — clamped.

    The plan's ``cores`` feeds the thread exports and the ``$CORES`` placeholder, and
    :meth:`slurm_layout` clamps what the scheduler grants — so the two must say one number.
    The raw ``pal()`` let a training step charged 16 cores export 32 threads.
    """
    ctx = _ctx(tmp_path, cores=32)  # fixture max_cores=16
    block = MlipTrainEngine().run_block(ctx, Path("step1_train.yaml"), Path("train.model"))
    assert "export OMP_NUM_THREADS=16" in block.body
    assert "export MKL_NUM_THREADS=16" in block.body
    assert "=32" not in block.body


def test_the_copy_back_directories_come_from_the_selected_trainer(tmp_path: Path):
    assert MlipTrainEngine().output_dirs(_ctx(tmp_path)) == ("logs", "checkpoints", "results")


def test_the_runlog_records_what_the_job_asked_for(tmp_path: Path):
    """`--gres` comes from the user's own header, so the runlog is where the two can be compared."""
    fields = dict(MlipTrainEngine().extra_header_fields(_ctx(tmp_path, device="cuda", gpus=2)))
    assert fields == {"trainer": "mace_off", "gpus": 2, "started_from": "small"}


def test_training_from_scratch_says_so_in_the_runlog(tmp_path: Path):
    fields = dict(MlipTrainEngine().extra_header_fields(_ctx(tmp_path, model_name="")))
    assert fields["started_from"] == "scratch"


def test_the_training_job_keeps_the_ranks_layout_and_declares_no_memory(tmp_path: Path):
    """The explicit JobExecutable members: the historical SBATCH spelling, no memory ask.

    Declared on the class because it satisfies the protocol directly; flipping training to
    the `(1, cores)` threads spelling is a named follow-up, not an accident of this test.
    """
    engine = MlipTrainEngine()
    ctx = _ctx(tmp_path, cores=4)
    assert engine.slurm_layout(ctx) == (4, 1)
    assert engine.memory_mb(ctx) is None

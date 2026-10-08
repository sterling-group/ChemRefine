"""Tests for the artifact-step lifecycle — one job, one product, structures pass through.

The kind `mlip-train` is: no per-structure inputs, no per-structure outputs, and a file rather
than a set of structures as the thing the step exists to produce. The stub here supplies only
what :class:`~chemrefine.engines.api.ArtifactEngine` asks for, so these tests pin the
*orchestrator's* half of the contract without dragging in a training backend.

It is passed to :func:`~chemrefine.step.run_step` through its ``engine`` parameter rather than
registered, deliberately: registering it would enrol a stub in every registry-parametrized
suite (the contract fixtures, the per-engine invariants) to prove something about `step.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import pytest
from ase import Atoms
from fake_engine import leave_runlog

from chemrefine import cache, ids, job_log
from chemrefine.config import Config, StepConfig
from chemrefine.engines._options import EngineOptions
from chemrefine.engines.api import ArtifactEngine
from chemrefine.errors import CacheError, JobFailureError
from chemrefine.state import (
    JobBatch,
    PipelineState,
    StepContext,
    StepInputs,
    StepResults,
    Structure,
)
from chemrefine.step import (
    StepMode,
    build_context,
    derive_step_key,
    rebuild_cache_step,
    run_step,
)

ARTIFACT_NAME = "model.pt"


class _StubTrainOptions(EngineOptions):
    """The declared knob the staleness tests turn — a trainer's options are declared."""

    epochs: int = 1


class _StubArtifactEngine:
    """The smallest thing that satisfies :class:`ArtifactEngine`.

    ``produces`` is what a real trainer cannot promise: a job can finish having written
    nothing, and telling that apart from success is the whole reason the capability has an
    ``artifact`` hook. ``weights`` distinguishes *which* run wrote the product, which is what
    the staleness test below reads. ``exit_code`` and ``scratch_kept`` are what the job's own
    runlog says of how it ended (``exit_code=None``: it left no runlog at all), the other
    half of the success test.

    Everything is written under one directory of its own beneath the step dir — its config,
    its runlog and its product — which is the shape an artifact step's single job has. A stub
    that scattered those over the step dir instead would sidestep the archiving `step.py` does
    and make these tests agree with code that could not work.

    That directory is deliberately **not** ``ids.TRAINING_ID``. A stub answering the same word
    a constant in `step.py` would name cannot tell "the orchestrator asked the engine" from
    "the orchestrator guessed, and guessed the same" — and an artifact engine that is not a
    trainer would then find its run directory unarchived, with the staleness guard passing
    while protecting nothing. Naming it something else is what makes the tests below able to
    fail.
    """

    RUN_DIR_NAME: ClassVar[str] = "distill"

    name: ClassVar[str] = "fake"
    options_cls: ClassVar[type[EngineOptions]] = _StubTrainOptions

    def __init__(
        self,
        *,
        produces: bool = True,
        weights: str = "weights",
        exit_code: int | None = 0,
        scratch_kept: bool = False,
    ) -> None:
        self.produces = produces
        self.weights = weights
        self.exit_code = exit_code
        self.scratch_kept = scratch_kept
        self.submits = 0

    def run_dir(self, ctx: StepContext) -> Path:
        """The job's own directory under the step — where a real trainer runs."""
        return ctx.step_dir / self.RUN_DIR_NAME

    def artifact(self, ctx: StepContext) -> Path:
        """The product — derived from ``ctx`` alone, so a rebuild can find it too."""
        return self.run_dir(ctx) / ARTIFACT_NAME

    def prepare(self, ctx: StepContext) -> StepInputs:
        """Write the one job's input; the ensemble is not prepared per structure."""
        self.run_dir(ctx).mkdir(parents=True, exist_ok=True)
        config = self.run_dir(ctx) / "train.yaml"
        config.write_text("epochs: 1\n", encoding="utf-8")
        return StepInputs(files=((config, self.artifact(ctx), ids.TRAINING_ID),))

    def submit(self, inputs: StepInputs, ctx: StepContext) -> JobBatch:
        """Stand in for the scheduler: count the run, write the product if it works, and
        leave the runlog the job's script leaves."""
        self.submits += 1
        if self.produces:
            self.artifact(ctx).write_text(f"{self.weights}\n", encoding="utf-8")
        if self.exit_code is not None:
            for job in inputs.files:
                leave_runlog(job, exit_code=self.exit_code, scratch_kept=self.scratch_kept)
        return JobBatch(jobs={})

    def parse(self, inputs: StepInputs, ctx: StepContext) -> StepResults:
        """Hand the prior ensemble on unchanged — the model is the product, not the structures."""
        return StepResults(structures=ctx.prev_state.structures)


def _config(tmp_path: Path, **step_overrides: object) -> Config:
    base: dict[str, object] = {"step": 1, "engine": "fake", "operation": "opt_sp"}
    base.update(step_overrides)
    return Config(
        template_dir=tmp_path / "templates",
        output_dir=tmp_path / "outputs",
        charge=0,
        multiplicity=1,
        max_cores=2,
        steps=[StepConfig(**base)],
    )


def _seeds(n: int) -> PipelineState:
    return PipelineState(
        structures=tuple(
            Structure(id=str(i), atoms=Atoms("H"), energy_hartree=-1.0 - i) for i in range(n)
        )
    )


def _step_dir(config: Config) -> Path:
    return config.step_dir(config.steps[0]).resolve()


# ---------------------------------------------------------------------------
# The capability itself
# ---------------------------------------------------------------------------


def test_the_stub_satisfies_the_capability_it_stands_for():
    """Without this the suite below could pass while `step.py` never took the branch.

    `run_step` routes on `isinstance(engine, ArtifactEngine)`, so a stub that drifted out of
    the Protocol would quietly fall through to the per-structure path and every assertion
    here would be about the wrong code.
    """
    assert isinstance(_StubArtifactEngine(), ArtifactEngine)


# ---------------------------------------------------------------------------
# The pass-through
# ---------------------------------------------------------------------------


def test_an_artifact_step_hands_its_whole_ensemble_to_the_next_step(tmp_path: Path):
    """Every structure in, every structure out, same ids in the same order.

    The bug this pins: the trainer's `parse` was never called, because a step's structures came
    from the per-structure ledger and a training step prepares no per-structure jobs. Ten
    structures went in and zero came out, so the pipeline stopped with "produced no survivors"
    at the step *after* training — which no shipped workflow ever got past.
    """
    config = _config(tmp_path)
    seeds = _seeds(10)
    engine = _StubArtifactEngine()

    outcome = run_step(config, config.steps[0], seeds, engine=engine)

    assert [s.id for s in outcome.state.structures] == [s.id for s in seeds.structures]
    assert engine.submits == 1
    assert not outcome.cache_hit


def test_an_artifact_step_caches_what_it_passed_through(tmp_path: Path):
    """A second run serves the cache instead of retraining."""
    config = _config(tmp_path)
    seeds = _seeds(4)

    first = run_step(config, config.steps[0], seeds, engine=_StubArtifactEngine())
    again = _StubArtifactEngine()
    second = run_step(config, config.steps[0], seeds, engine=again)

    assert second.cache_hit
    assert again.submits == 0
    assert [s.id for s in second.state.structures] == [s.id for s in first.state.structures]


# ---------------------------------------------------------------------------
# A job that produced nothing
# ---------------------------------------------------------------------------


def test_a_missing_product_fails_the_step_and_caches_nothing(tmp_path: Path):
    """The product is the success test, and failing it must leave no cache behind.

    A job that leaves the scheduler's queue having written nothing is indistinguishable from
    one that worked — the old trainer waited on the queue and cached the step either way, so a
    failed training was reported as a success and every later `resume` served it.
    """
    config = _config(tmp_path)

    with pytest.raises(JobFailureError, match=ARTIFACT_NAME):
        run_step(config, config.steps[0], _seeds(3), engine=_StubArtifactEngine(produces=False))

    assert not (_step_dir(config) / "_cache" / "step.json").exists()


# ---------------------------------------------------------------------------
# A product is a finished job's only when the job says so
# ---------------------------------------------------------------------------


def _runlog(config: Config) -> Path:
    """Where the stub's one job logs its run."""
    return job_log.runlog_path(_step_dir(config) / _StubArtifactEngine.RUN_DIR_NAME, "train")


@pytest.mark.parametrize(
    ("ended", "why", "advice"),
    [
        ({"exit_code": 3}, "did not finish", "fix the cause first"),
        ({"scratch_kept": True}, "did not all come home", "rebuild-cache 1"),
        ({"exit_code": None}, "holds no footer", "rebuild-cache 1"),
    ],
    ids=["exited non-zero", "results left in scratch", "no runlog"],
)
def test_a_product_its_job_does_not_vouch_for_fails_the_step_and_caches_nothing(
    tmp_path: Path, ended: dict[str, object], why: str, advice: str
):
    """A file at the product's path is not a trained model unless its job says it finished.

    A job that was stopped, or whose program died, can leave a best-so-far checkpoint there
    all the same, and a copy the output directory cut short is a file too. The job's own
    runlog footer tells them from a finished training; with no such word the step fails and
    caches nothing. The refusal offers ``rebuild-cache`` only for what is unprovable, never
    for a job that said it failed.
    """
    config = _config(tmp_path)
    engine = _StubArtifactEngine(**ended)

    with pytest.raises(JobFailureError, match=why) as refused:
        run_step(config, config.steps[0], _seeds(3), engine=engine)

    assert advice in str(refused.value)
    assert str(_runlog(config)) in str(refused.value)
    assert not (_step_dir(config) / "_cache" / "step.json").exists()


@pytest.mark.parametrize(
    "ended",
    [{"exit_code": None}, {"scratch_kept": True}],
    ids=["no runlog", "results left in scratch"],
)
def test_rebuild_cache_adopts_a_product_nothing_vouches_for_and_says_so(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, ended: dict[str, object]
):
    """The explicit command adopts what is unprovable, on its caller's word, with a warning.

    A tree whose runlog is gone, or whose results were copied home by hand after the job
    could not, is never proven wrong. ``rebuild-cache`` names the step, so it adopts the
    product as it adopts a manifest without provenance, and logs what it could not check.
    """
    config = _config(tmp_path)
    seeds = _seeds(3)
    with pytest.raises(JobFailureError):
        run_step(config, config.steps[0], seeds, engine=_StubArtifactEngine(**ended))

    rebuilt = _StubArtifactEngine()
    with (
        patch("chemrefine.step.get_engine", return_value=rebuilt),
        caplog.at_level("WARNING", logger="chemrefine.step"),
    ):
        outcome = rebuild_cache_step(config, config.steps[0], seeds)

    assert rebuilt.submits == 0
    assert [s.id for s in outcome.state.structures] == [s.id for s in seeds.structures]
    assert (_step_dir(config) / "_cache" / "step.json").exists()
    assert "adopting the product on this command's word" in caplog.text


def test_rebuild_cache_refuses_a_product_whose_job_said_it_failed(tmp_path: Path):
    """A non-zero exit is proof, and no command's word stands against it."""
    config = _config(tmp_path)
    seeds = _seeds(3)
    with pytest.raises(JobFailureError):
        run_step(config, config.steps[0], seeds, engine=_StubArtifactEngine(exit_code=3))

    with (
        patch("chemrefine.step.get_engine", return_value=_StubArtifactEngine()),
        pytest.raises(JobFailureError, match="exit_code=3"),
    ):
        rebuild_cache_step(config, config.steps[0], seeds)

    assert not (_step_dir(config) / "_cache" / "step.json").exists()


def test_rebuild_cache_adopts_a_product_under_a_manifest_that_names_no_job(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """A hand-written manifest may name no job; its product then has no runlog to ask."""
    config = _config(tmp_path)
    seeds = _seeds(2)
    engine = _StubArtifactEngine()
    ctx = build_context(config, config.steps[0], seeds, engine)
    engine.artifact(ctx).parent.mkdir(parents=True)
    engine.artifact(ctx).write_text("weights\n", encoding="utf-8")
    cache.save_manifest(
        StepInputs(files=()),
        ctx.step_dir,
        operation="opt_sp",
        engine="fake",
        stamp=_key_for(config, seeds).manifest_stamp(),
    )

    with (
        patch("chemrefine.step.get_engine", return_value=engine),
        caplog.at_level("WARNING", logger="chemrefine.step"),
    ):
        rebuild_cache_step(config, config.steps[0], seeds)

    assert "its manifest names no job" in caplog.text
    assert (_step_dir(config) / "_cache" / "step.json").exists()


def _key_for(config: Config, seeds) -> cache.StepKey:
    """The key run_step derives for the stub, derived the way run_step derives it."""
    engine = _StubArtifactEngine()
    step_cfg = config.steps[0]
    return derive_step_key(build_context(config, step_cfg, seeds, engine), step_cfg, engine)


def test_a_rerun_that_produces_nothing_does_not_adopt_the_previous_run(tmp_path: Path):
    """A changed configuration whose job dies must fail, not inherit the old product.

    The sequence that used to pass: train once; change the options so the fingerprint misses;
    the new job dies having written nothing; ``artifact.exists()`` finds **run 1's** file and
    calls it success. The step then caches those bytes under the *new* fingerprint, so the run
    is internally consistent and describes a training that never happened — and a consuming
    step that digests the model file cache-hits on it too.

    Nothing here reads the file's contents; the assertion is that the step raises at all. That
    is only possible if the previous run's directory was moved aside before the second one
    prepared, which is what `_run_artifact_step` does with `attempts.archive_previous`.
    """
    config = _config(tmp_path)
    seeds = _seeds(3)
    run_step(config, config.steps[0], seeds, engine=_StubArtifactEngine(weights="run-1"))
    assert (_step_dir(config) / _StubArtifactEngine.RUN_DIR_NAME / ARTIFACT_NAME).exists()

    # A different configuration: the cache misses, so the step really re-runs.
    changed = _config(tmp_path, options={"epochs": 99})
    with pytest.raises(JobFailureError, match=ARTIFACT_NAME):
        run_step(changed, changed.steps[0], seeds, engine=_StubArtifactEngine(produces=False))

    # Nothing vouches for a model here any more: run 1's cache went when its model was
    # moved aside, and run 2 produced none to cache. What must *not* have happened is run
    # 1's bytes being cached under run 2's fingerprint, which adopting the stale model
    # would have done and a `resume` would then serve.
    assert not (_step_dir(changed) / "_cache" / "step.json").exists()

    run_dir = _step_dir(changed) / _StubArtifactEngine.RUN_DIR_NAME
    assert not (run_dir / ARTIFACT_NAME).exists(), "run 1's model is not left at the canonical path"
    assert list(run_dir.glob(f"attempt*/{ARTIFACT_NAME}")), "but it is kept, inside its attempt"


def test_resume_reruns_an_artifact_step_that_produced_nothing(tmp_path: Path):
    """No cache is what makes the recovery work: the next run redoes the step.

    This is the half a ledgered failure would get wrong. Ledgering writes a cache with zero
    structures, which `resume` then serves — the same wrong state by a longer route.
    """
    config = _config(tmp_path)
    seeds = _seeds(3)

    with pytest.raises(JobFailureError):
        run_step(config, config.steps[0], seeds, engine=_StubArtifactEngine(produces=False))

    retry = _StubArtifactEngine()
    outcome = run_step(config, config.steps[0], seeds, engine=retry)

    assert retry.submits == 1, "resume must run the step again, not serve a cache"
    assert len(outcome.state.structures) == 3


# ---------------------------------------------------------------------------
# resume adopts a finished product
# ---------------------------------------------------------------------------


def _run_dir(config: Config) -> Path:
    return _step_dir(config) / _StubArtifactEngine.RUN_DIR_NAME


def _product(config: Config) -> str:
    """Which run's weights stand at the product's path."""
    return (_run_dir(config) / ARTIFACT_NAME).read_text(encoding="utf-8").strip()


def test_resume_adopts_a_finished_product_whose_cache_was_lost(tmp_path: Path):
    """A training that finished is read back by a resume, as a finished output is.

    The cache is gone the way a driver killed before it could write one leaves it, or one
    whose write the output tree refused. The manifest's stamp says the product is this
    configuration's and the job's runlog says the job finished, so nothing is submitted
    and nothing is moved aside.
    """
    config = _config(tmp_path)
    seeds = _seeds(4)
    run_step(config, config.steps[0], seeds, engine=_StubArtifactEngine(weights="first"))
    cache.invalidate(_step_dir(config))

    resumed = _StubArtifactEngine(weights="second")
    outcome = run_step(config, config.steps[0], seeds, engine=resumed)

    assert resumed.submits == 0
    assert _product(config) == "first"
    assert not (_run_dir(config) / "attempt1").exists()
    assert [s.id for s in outcome.state.structures] == [s.id for s in seeds.structures]
    assert (_step_dir(config) / "_cache" / "step.json").exists()


def test_resume_trains_again_over_another_configurations_product(tmp_path: Path):
    """A stamp for another key is another training: it is moved aside and the job runs."""
    config = _config(tmp_path)
    seeds = _seeds(3)
    run_step(config, config.steps[0], seeds, engine=_StubArtifactEngine(weights="first"))
    cache.invalidate(_step_dir(config))

    retuned = _config(tmp_path, options={"epochs": 5})
    engine = _StubArtifactEngine(weights="second")
    run_step(retuned, retuned.steps[0], seeds, engine=engine)

    assert engine.submits == 1
    assert _product(config) == "second"
    assert (_run_dir(config) / "attempt1" / ARTIFACT_NAME).read_text(encoding="utf-8") == "first\n"


@pytest.mark.parametrize(
    "ended",
    [{"exit_code": 3}, {"scratch_kept": True}, {"exit_code": None}],
    ids=["exited non-zero", "results left in scratch", "no runlog"],
)
def test_resume_trains_again_over_a_product_its_job_does_not_vouch_for(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, ended: dict[str, object]
):
    """A resume adopts on proof alone; what is unproven it runs again, and says why."""
    config = _config(tmp_path)
    seeds = _seeds(3)
    with pytest.raises(JobFailureError):
        run_step(
            config, config.steps[0], seeds, engine=_StubArtifactEngine(weights="first", **ended)
        )

    retry = _StubArtifactEngine(weights="second")
    with caplog.at_level("WARNING", logger="chemrefine.step"):
        run_step(config, config.steps[0], seeds, engine=retry)

    assert retry.submits == 1
    assert _product(config) == "second"
    assert "running the job again" in caplog.text


def test_a_training_that_runs_again_drops_the_cache_of_the_model_it_moves_aside(tmp_path: Path):
    """A cache vouches for the model at the product's path, so it goes when that model does.

    The configuration is edited, its training runs and fails, and the edit is taken back.
    The first configuration's cache would still match its key, over a product path that
    holds the second training's file. With the cache gone the first training runs again.
    """
    first = _config(tmp_path)
    second = _config(tmp_path, options={"epochs": 5})
    seeds = _seeds(3)
    run_step(first, first.steps[0], seeds, engine=_StubArtifactEngine(weights="first"))
    with pytest.raises(JobFailureError):
        run_step(
            second,
            second.steps[0],
            seeds,
            engine=_StubArtifactEngine(weights="second, unfinished", exit_code=3),
        )

    again = _StubArtifactEngine(weights="first again")
    outcome = run_step(first, first.steps[0], seeds, engine=again)

    assert again.submits == 1
    assert not outcome.cache_hit
    assert _product(first) == "first again"


def _unstamped_tree(config: Config, seeds: PipelineState) -> _StubArtifactEngine:
    """A finished product under a manifest that carries no stamp, as a hand-written one does."""
    engine = _StubArtifactEngine(weights="adopted by hand")
    ctx = build_context(config, config.steps[0], seeds, engine)
    inputs = engine.prepare(ctx)
    engine.submit(inputs, ctx)
    blank = cache.ManifestStamp(fingerprint="", search_key="", rows={})
    cache.save_manifest(inputs, ctx.step_dir, operation="opt_sp", engine="fake", stamp=blank)
    return engine


def test_resume_refuses_a_product_under_a_manifest_that_proves_nothing(tmp_path: Path):
    """Unprovable is never silently trusted and never silently moved aside.

    The refusal names the command that adopts such a tree, as it does for a manifest
    without row provenance, and the product stays where it is.
    """
    config = _config(tmp_path)
    seeds = _seeds(3)
    _unstamped_tree(config, seeds)

    fresh = _StubArtifactEngine()
    with pytest.raises(CacheError, match="rebuild-cache 1"):
        run_step(config, config.steps[0], seeds, engine=fresh)

    assert fresh.submits == 0
    assert _product(config) == "adopted by hand"
    assert not (_run_dir(config) / "attempt1").exists()


def test_rebuild_cache_stamps_the_product_it_adopts_so_a_resume_can_prove_it(tmp_path: Path):
    """An adoption on the caller's word is recorded: the next lost cache costs a resume."""
    config = _config(tmp_path)
    seeds = _seeds(3)
    engine = _unstamped_tree(config, seeds)

    with patch("chemrefine.step.get_engine", return_value=engine):
        rebuild_cache_step(config, config.steps[0], seeds)

    stamped = cache.load_manifest_provenance(_step_dir(config)).fingerprint
    assert stamped == _key_for(config, seeds).fingerprint
    cache.invalidate(_step_dir(config))
    later = _StubArtifactEngine()
    run_step(config, config.steps[0], seeds, engine=later)
    assert later.submits == 0
    assert _product(config) == "adopted by hand"


@pytest.mark.parametrize("command", ["run", "rerun"])
def test_a_step_asked_to_run_again_runs_its_job_again(tmp_path: Path, command: str):
    """Adoption is a resume's. ``run`` ignores what is on disk and ``rerun`` discards it."""
    config = _config(tmp_path)
    seeds = _seeds(3)
    run_step(config, config.steps[0], seeds, engine=_StubArtifactEngine(weights="first"))

    again = _StubArtifactEngine(weights="second")
    if command == "run":
        run_step(config, config.steps[0], seeds, engine=again, mode=StepMode.EXECUTE)
    else:
        cache.discard_step(_step_dir(config))
        run_step(config, config.steps[0], seeds, engine=again)

    assert again.submits == 1
    assert _product(config) == "second"


# ---------------------------------------------------------------------------
# rebuild-cache adopts a finished product
# ---------------------------------------------------------------------------


def test_rebuild_cache_adopts_a_finished_product_without_rerunning(tmp_path: Path):
    """A run that finished before the driver died is re-cached, not recomputed.

    For a training measured in days this is the difference between a rebuild and a week, and it
    is the reason `artifact` takes only `ctx`: the product has to be findable by a command that
    never submitted anything.
    """
    config = _config(tmp_path)
    seeds = _seeds(5)
    engine = _StubArtifactEngine()

    # Run far enough to leave the manifest + product on disk, then drop the cache as a
    # driver killed after the job but before `finalize` would have.
    run_step(config, config.steps[0], seeds, engine=engine)
    cache.invalidate(_step_dir(config))
    product = _step_dir(config) / _StubArtifactEngine.RUN_DIR_NAME / ARTIFACT_NAME
    assert product.exists(), "the product survives the lost cache"

    rebuilt = _StubArtifactEngine()
    with patch("chemrefine.step.get_engine", return_value=rebuilt):
        outcome = rebuild_cache_step(config, config.steps[0], seeds)

    assert rebuilt.submits == 0, "rebuild-cache must not re-run the job"
    assert [s.id for s in outcome.state.structures] == [s.id for s in seeds.structures]
    assert (_step_dir(config) / "_cache" / "step.json").exists()


def test_rebuild_cache_still_refuses_a_product_from_another_configuration(tmp_path: Path):
    """The fingerprint guard applies to an artifact step like any other.

    Adopting a model trained for different options or a different upstream ensemble would cache
    a run that never happened — the exact thing the guard exists to prevent — so the artifact
    branch sits *after* it, not in front of it.
    """
    config = _config(tmp_path)
    run_step(config, config.steps[0], _seeds(5), engine=_StubArtifactEngine())

    changed = _config(tmp_path, options={"epochs": 99})
    with (
        patch("chemrefine.step.get_engine", return_value=_StubArtifactEngine()),
        pytest.raises(CacheError, match="different configuration"),
    ):
        rebuild_cache_step(changed, changed.steps[0], _seeds(5))


# ---------------------------------------------------------------------------
# The empty-manifest hole — general, not an artifact-step special case
# ---------------------------------------------------------------------------


def test_a_manifest_with_no_jobs_is_not_a_step_to_continue(tmp_path: Path):
    """An interrupted step that prepared no jobs must re-run, not "resume" into nothing.

    `load_manifest` returns `StepInputs(files=())` for a step that wrote `"files": []` — a real
    value, not a missing one. Testing only for `None` sent such a step down the resubmit path,
    where re-parsing zero jobs yielded zero successes and zero failures, so it cached an empty
    result and the work was never done. Asserted with an ordinary engine so the fix stays
    general rather than something the artifact branch happens to sidestep.
    """
    from fake_engine import FakeEngine

    config = _config(tmp_path)
    seeds = _seeds(2)
    ctx_dir = _step_dir(config)
    ctx_dir.mkdir(parents=True, exist_ok=True)
    key = cache.StepKey.of(config.steps[0], seeds.structures, None)
    cache.save_manifest(
        StepInputs(files=()),
        ctx_dir,
        operation="opt_sp",
        engine="fake",
        stamp=key.manifest_stamp(),
    )

    with patch("chemrefine.step._resubmit_failed") as resubmit:
        outcome = run_step(config, config.steps[0], seeds, engine=FakeEngine())

    resubmit.assert_not_called()
    assert len(outcome.state.structures) == 2, "the step must actually run"

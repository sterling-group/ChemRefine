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

from chemrefine import cache, ids
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
from chemrefine.step import rebuild_cache_step, run_step

ARTIFACT_NAME = "model.pt"


class _StubTrainOptions(EngineOptions):
    """The declared knob the staleness tests turn — a trainer's options are declared."""

    epochs: int = 1


class _StubArtifactEngine:
    """The smallest thing that satisfies :class:`ArtifactEngine`.

    ``produces`` is what a real trainer cannot promise: a job can finish having written
    nothing, and telling that apart from success is the whole reason the capability has an
    ``artifact`` hook. ``weights`` distinguishes *which* run wrote the product, which is what
    the staleness test below reads.

    Everything is written under ``step_dir/<TRAINING_ID>/`` because that is the shape
    :data:`~chemrefine.ids.TRAINING_ID` documents for an artifact step's single job — its own
    directory under the step dir, holding its config, its runlog and its product. A stub that
    scattered those over the step dir instead would sidestep the archiving `step.py` does and
    make these tests agree with code that could not work.
    """

    name: ClassVar[str] = "fake"
    options_cls: ClassVar[type[EngineOptions]] = _StubTrainOptions

    def __init__(self, *, produces: bool = True, weights: str = "weights") -> None:
        self.produces = produces
        self.weights = weights
        self.submits = 0

    def run_dir(self, ctx: StepContext) -> Path:
        """The job's own directory under the step — where a real trainer runs."""
        return ctx.step_dir / ids.TRAINING_ID

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
        """Stand in for the scheduler: count the run, and write the product if it works."""
        self.submits += 1
        if self.produces:
            self.artifact(ctx).write_text(f"{self.weights}\n", encoding="utf-8")
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


def _key_for(config: Config, seeds) -> cache.StepKey:
    """The key run_step derives for the stub — its declared options included."""
    step_cfg = config.steps[0]
    return cache.StepKey.of(
        step_cfg,
        seeds.structures,
        None,
        engine_options=_StubTrainOptions.from_raw_lenient(step_cfg.options).model_dump(mode="json"),
    )


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
    assert (_step_dir(config) / ids.TRAINING_ID / ARTIFACT_NAME).exists()

    # A different configuration: the cache misses, so the step really re-runs.
    changed = _config(tmp_path, options={"epochs": 99})
    with pytest.raises(JobFailureError, match=ARTIFACT_NAME):
        run_step(changed, changed.steps[0], seeds, engine=_StubArtifactEngine(produces=False))

    # Run 1's cache is still there — untouched, still keyed to run 1. What must *not* have
    # happened is it being rewritten under run 2's fingerprint, which is what adopting the
    # stale model would have done and what would then make a `resume` serve it.
    cached = cache.load(_step_dir(changed))
    assert cached is not None
    assert cached.fingerprint == _key_for(config, seeds).fingerprint

    run_dir = _step_dir(changed) / ids.TRAINING_ID
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
    product = _step_dir(config) / ids.TRAINING_ID / ARTIFACT_NAME
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
        fingerprint=key.fingerprint,
    )

    with patch("chemrefine.step._resubmit_failed") as resubmit:
        outcome = run_step(config, config.steps[0], seeds, engine=FakeEngine())

    resubmit.assert_not_called()
    assert len(outcome.state.structures) == 2, "the step must actually run"

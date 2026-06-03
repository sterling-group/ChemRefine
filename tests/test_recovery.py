"""Tests for the recovery action dispatcher."""

from __future__ import annotations

from pathlib import Path

import pytest
from ase import Atoms

from chemrefine import io
from chemrefine.config import Config, StepConfig
from chemrefine.errors import ChemRefineError
from chemrefine.recovery import Action, execute, invalidate_step, resolve_target


def _seeded_config(tmp_path: Path, steps: list[StepConfig]) -> Config:
    seed_dir = tmp_path / "seeds"
    io.write_xyz(
        [Atoms("H"), Atoms("H")], ["a", "b"], step_number=0, output_dir=seed_dir
    )
    return Config(
        template_dir=tmp_path / "templates",
        scratch_dir=tmp_path / "scratch",
        output_dir=tmp_path / "outputs",
        input=seed_dir,
        steps=steps,
    )


def _two_step_config(tmp_path: Path) -> Config:
    return _seeded_config(
        tmp_path,
        [
            StepConfig(step=1, name="screen", engine="fake", operation="opt_sp"),
            StepConfig(step=2, name="refine", engine="fake", operation="opt_sp"),
        ],
    )


# ---------------------------------------------------------------------------
# resolve_target
# ---------------------------------------------------------------------------


def test_resolve_target_by_number(tmp_path: Path):
    cfg = _two_step_config(tmp_path)
    step = resolve_target(cfg, 2)
    assert step.step == 2


def test_resolve_target_by_name(tmp_path: Path):
    cfg = _two_step_config(tmp_path)
    step = resolve_target(cfg, "refine")
    assert step.step == 2


def test_resolve_target_by_numeric_string(tmp_path: Path):
    cfg = _two_step_config(tmp_path)
    step = resolve_target(cfg, "1")
    assert step.step == 1


def test_resolve_target_missing_raises(tmp_path: Path):
    cfg = _two_step_config(tmp_path)
    with pytest.raises(ChemRefineError):
        resolve_target(cfg, "missing")


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------


def test_execute_run_invalidates_and_re_executes(tmp_path: Path):
    cfg = _two_step_config(tmp_path)
    # First run populates caches.
    execute(cfg, Action.RESUME)
    step1_dir = (cfg.output_dir / "step1_screen").resolve()
    assert (step1_dir / "_cache" / "step.pkl").is_file()
    # RUN should invalidate and re-execute.
    assert execute(cfg, Action.RUN) == 0
    # Cache should exist again after re-execution.
    assert (step1_dir / "_cache" / "step.pkl").is_file()


def test_execute_resume_keeps_caches(tmp_path: Path):
    cfg = _two_step_config(tmp_path)
    execute(cfg, Action.RESUME)
    # Caches now exist; calling resume again should hit them.
    assert execute(cfg, Action.RESUME) == 0


def test_execute_rebuild_cache_invalidates_only_target(tmp_path: Path):
    cfg = _two_step_config(tmp_path)
    execute(cfg, Action.RESUME)
    step1_pkl = (cfg.output_dir / "step1_screen").resolve() / "_cache" / "step.pkl"
    step2_pkl = (cfg.output_dir / "step2_refine").resolve() / "_cache" / "step.pkl"
    # Both caches present.
    assert step1_pkl.is_file()
    assert step2_pkl.is_file()

    # Invalidate only step 2.
    execute(cfg, Action.REBUILD_CACHE, target=2)
    assert step1_pkl.is_file()
    assert step2_pkl.is_file()  # re-created by the resume run


def test_execute_rerun_target_by_name(tmp_path: Path):
    cfg = _two_step_config(tmp_path)
    execute(cfg, Action.RESUME)
    # rerun by name should also be accepted.
    assert execute(cfg, Action.RERUN, target="screen") == 0


def test_execute_rebuild_cache_no_target_uses_last_step(tmp_path: Path):
    cfg = _two_step_config(tmp_path)
    execute(cfg, Action.RESUME)
    assert execute(cfg, Action.REBUILD_CACHE) == 0


def test_invalidate_step_removes_cache(tmp_path: Path):
    cfg = _two_step_config(tmp_path)
    execute(cfg, Action.RESUME)
    invalidate_step(cfg, cfg.steps[0])
    step1_dir = (cfg.output_dir / "step1_screen").resolve()
    assert not (step1_dir / "_cache" / "step.pkl").exists()


def test_execute_unknown_action_raises(tmp_path: Path):
    cfg = _two_step_config(tmp_path)
    with pytest.raises(ChemRefineError):
        execute(cfg, "not-an-action")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# rerun — resubmit only the failed jobs, then clear the ledger
# ---------------------------------------------------------------------------


def _register_flaky():
    """A fake engine: ``fail_ids`` produce no output; ``submitted`` logs each
    submitted structure id so tests can assert what was (re)run."""
    from typing import ClassVar

    import numpy as np

    from chemrefine.engines.base import register
    from chemrefine.ids import structure_artifact_path
    from chemrefine.state import JobBatch, StepInputs, StepResults, Structure

    @register("flaky")
    class _Flaky:
        name = "flaky"
        supports_nms = False
        fail_ids: ClassVar[set[str]] = set()
        submitted: ClassVar[list[str]] = []

        def prepare(self, ctx):
            ctx.step_dir.mkdir(parents=True, exist_ok=True)
            files = []
            for s in ctx.prev_state.structures:
                step = ctx.step_cfg.step
                inp = structure_artifact_path(ctx.step_dir, step, s.id, "inp")
                out = structure_artifact_path(ctx.step_dir, step, s.id, "out")
                inp.write_text("in\n", encoding="utf-8")
                files.append((inp, out, s.id))
            return StepInputs(files=tuple(files))

        def submit(self, inputs, ctx):
            for _inp, out, sid in inputs.files:
                _Flaky.submitted.append(sid)
                if sid not in _Flaky.fail_ids:  # failed jobs produce no output
                    out.write_text(f"FINAL ENERGY: {-1.0 - int(sid) * 1e-3}\n", encoding="utf-8")
            return JobBatch(jobs={})

        def wait(self, batch):
            return None

        def parse(self, inputs, ctx):
            seeds = {s.id: s for s in ctx.prev_state.structures}
            out = []
            for _inp, o, sid in inputs.files:
                energy = float(o.read_text().split("FINAL ENERGY:")[1])
                seed = seeds[sid]
                out.append(
                    Structure(
                        id=sid, atoms=seed.atoms, parent_id=seed.parent_id,
                        energy_hartree=energy,
                        forces_ev_per_a=np.zeros((len(seed.atoms), 3)),
                    )
                )
            return StepResults(structures=tuple(out))

        def normal_mode_sample(self, results, ctx):
            raise NotImplementedError

    return _Flaky


def test_resume_is_incremental_resubmits_only_failed(tmp_path: Path):
    """A flaky engine fails "1" on the first run (ledgered, "0" cached); after it
    recovers, `resume` resubmits ONLY "1" (incremental) and clears the ledger."""
    from chemrefine import cache
    from chemrefine.engines.base import ENGINES

    eng = _register_flaky()
    try:
        cfg = _seeded_config(
            tmp_path, [StepConfig(step=1, name="s", engine="flaky", operation="opt_sp")]
        )
        step_dir = (cfg.output_dir / "step1_s").resolve()

        eng.fail_ids = {"1"}
        execute(cfg, Action.RESUME)
        assert cache.load_failed_jobs(step_dir) == [
            {"structure_id": "1", "reason": "output missing"}
        ]
        assert {s.id for s in cache.load(step_dir).results.structures} == {"0"}

        eng.fail_ids = set()  # engine recovers
        eng.submitted = []
        assert execute(cfg, Action.RESUME) == 0  # incremental re-attempt
        assert eng.submitted == ["1"]  # only the failed structure resubmitted
        assert cache.load_failed_jobs(step_dir) == []  # ledger cleared
        assert {s.id for s in cache.load(step_dir).results.structures} == {"0", "1"}
    finally:
        eng.fail_ids, eng.submitted = set(), []
        ENGINES.pop("flaky", None)


def test_rerun_redoes_whole_step(tmp_path: Path):
    """`rerun` invalidates the step and re-executes it end-to-end (all structures)."""
    from chemrefine.engines.base import ENGINES

    eng = _register_flaky()
    try:
        cfg = _seeded_config(
            tmp_path, [StepConfig(step=1, name="s", engine="flaky", operation="opt_sp")]
        )
        execute(cfg, Action.RESUME)
        eng.submitted = []
        assert execute(cfg, Action.RERUN, target=1) == 0
        assert sorted(eng.submitted) == ["0", "1"]  # whole step redone
    finally:
        eng.fail_ids, eng.submitted = set(), []
        ENGINES.pop("flaky", None)


def test_rebuild_cache_reparses_without_submitting(tmp_path: Path):
    """`rebuild-cache` rebuilds a step's cache from existing outputs — no submit."""
    from chemrefine import cache
    from chemrefine.engines.base import ENGINES

    eng = _register_flaky()
    try:
        cfg = _seeded_config(
            tmp_path, [StepConfig(step=1, name="s", engine="flaky", operation="opt_sp")]
        )
        step_dir = (cfg.output_dir / "step1_s").resolve()
        execute(cfg, Action.RESUME)  # produces outputs + cache
        cache.invalidate(step_dir)  # drop the cache, keep outputs on disk
        eng.submitted = []
        assert execute(cfg, Action.REBUILD_CACHE, target=1) == 0
        assert eng.submitted == []  # parse-only, nothing resubmitted
        assert {s.id for s in cache.load(step_dir).results.structures} == {"0", "1"}
    finally:
        eng.fail_ids, eng.submitted = set(), []
        ENGINES.pop("flaky", None)

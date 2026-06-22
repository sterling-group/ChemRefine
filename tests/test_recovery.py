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
    io.write_xyz([Atoms("H"), Atoms("H")], ["a", "b"], step_number=0, output_dir=seed_dir)
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
    assert (step1_dir / "_cache" / "step.json").is_file()
    # RUN should invalidate and re-execute.
    assert execute(cfg, Action.RUN) == 0
    # Cache should exist again after re-execution.
    assert (step1_dir / "_cache" / "step.json").is_file()


def test_execute_resume_keeps_caches(tmp_path: Path):
    cfg = _two_step_config(tmp_path)
    execute(cfg, Action.RESUME)
    # Caches now exist; calling resume again should hit them.
    assert execute(cfg, Action.RESUME) == 0


def test_execute_rebuild_cache_invalidates_only_target(tmp_path: Path):
    cfg = _two_step_config(tmp_path)
    execute(cfg, Action.RESUME)
    step1_cache = (cfg.output_dir / "step1_screen").resolve() / "_cache" / "step.json"
    step2_cache = (cfg.output_dir / "step2_refine").resolve() / "_cache" / "step.json"
    # Both caches present.
    assert step1_cache.is_file()
    assert step2_cache.is_file()

    # Invalidate only step 2.
    execute(cfg, Action.REBUILD_CACHE, target=2)
    assert step1_cache.is_file()
    assert step2_cache.is_file()  # re-created by the resume run


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
    assert not (step1_dir / "_cache" / "step.json").exists()


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

    from chemrefine.engines.api import register
    from chemrefine.ids import structure_artifact_path
    from chemrefine.state import JobBatch, StepInputs, StepResults, Structure

    @register("flaky")
    class _Flaky:
        name = "flaky"
        fail_ids: ClassVar[set[str]] = set()
        submitted: ClassVar[list[str]] = []

        def prepare(self, ctx):
            ctx.step_dir.mkdir(parents=True, exist_ok=True)
            files = []
            for s in ctx.prev_state.structures:
                step = ctx.step_cfg.step
                inp = structure_artifact_path(ctx.step_dir, step, s.id, "inp")
                out = structure_artifact_path(ctx.step_dir, step, s.id, "out")
                inp.parent.mkdir(parents=True, exist_ok=True)
                inp.write_text("in\n", encoding="utf-8")
                files.append((inp, out, s.id))
            return StepInputs(files=tuple(files))

        def submit(self, inputs, ctx):
            for _inp, out, sid in inputs.files:
                _Flaky.submitted.append(sid)
                if sid not in _Flaky.fail_ids:  # failed jobs produce no output
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(f"FINAL ENERGY: {-1.0 - int(sid) * 1e-3}\n", encoding="utf-8")
            return JobBatch(jobs={})

        def parse(self, inputs, ctx):
            seeds = {s.id: s for s in ctx.prev_state.structures}
            out = []
            for _inp, o, sid in inputs.files:
                energy = float(o.read_text().split("FINAL ENERGY:")[1])
                seed = seeds[sid]
                out.append(
                    Structure(
                        id=sid,
                        atoms=seed.atoms,
                        parent_id=seed.parent_id,
                        energy_hartree=energy,
                        forces_ev_per_a=np.zeros((len(seed.atoms), 3)),
                    )
                )
            return StepResults(structures=tuple(out))

        def input_digest(self, ctx):
            return ""

    return _Flaky


def test_resume_is_incremental_resubmits_only_failed(tmp_path: Path):
    """An `on_failure: stop` step fails "1" (ledgered, "0" cached) and halts; after
    the engine recovers, `resume` resubmits ONLY "1" (incremental) and clears it."""
    from chemrefine import cache
    from chemrefine.engines.api import ENGINES
    from chemrefine.errors import ChemRefineError

    eng = _register_flaky()
    try:
        cfg = _seeded_config(
            tmp_path,
            [StepConfig(step=1, name="s", engine="flaky", operation="opt_sp", on_failure="stop")],
        )
        step_dir = (cfg.output_dir / "step1_s").resolve()

        eng.fail_ids = {"1"}
        with pytest.raises(ChemRefineError):  # stop halts after caching "0"
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
    from chemrefine.engines.api import ENGINES

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


def test_rerun_errors_reattempts_only_the_target_step_failures(tmp_path: Path):
    """`rerun-errors N` re-attempts only step N's pending (stop) failures."""
    from chemrefine import cache
    from chemrefine.engines.api import ENGINES
    from chemrefine.errors import ChemRefineError

    eng = _register_flaky()
    try:
        cfg = _seeded_config(
            tmp_path,
            [StepConfig(step=1, name="s", engine="flaky", operation="opt_sp", on_failure="stop")],
        )
        step_dir = (cfg.output_dir / "step1_s").resolve()
        eng.fail_ids = {"1"}
        with pytest.raises(ChemRefineError):
            execute(cfg, Action.RESUME)  # stop halts, "1" pending
        eng.fail_ids, eng.submitted = set(), []
        assert execute(cfg, Action.RERUN_ERRORS, target=1) == 0
        assert eng.submitted == ["1"]  # only the failed structure
        assert cache.load_failed_jobs(step_dir) == []
    finally:
        eng.fail_ids, eng.submitted = set(), []
        ENGINES.pop("flaky", None)


def test_rerun_errors_on_skip_step_reports_nothing_pending(tmp_path: Path, caplog):
    """`rerun-errors` on a skip step must not claim it is re-attempting anything.

    The ledger is visibility-only for ``skip``/``best`` — ``run_step`` never
    resubmits those failures, so the log must say nothing is pending instead
    of announcing a re-attempt that silently never happens.
    """
    import logging

    from chemrefine.engines.api import ENGINES

    eng = _register_flaky()
    try:
        cfg = _seeded_config(
            tmp_path,
            [StepConfig(step=1, name="s", engine="flaky", operation="opt_sp", on_failure="skip")],
        )
        eng.fail_ids = {"1"}
        assert execute(cfg, Action.RESUME) == 0
        eng.fail_ids, eng.submitted = set(), []
        with caplog.at_level(logging.INFO, logger="chemrefine.recovery"):
            assert execute(cfg, Action.RERUN_ERRORS, target=1) == 0
        assert eng.submitted == []  # nothing was actually re-attempted
        assert "nothing is pending" in caplog.text
        assert "re-attempting" not in caplog.text
    finally:
        eng.fail_ids, eng.submitted = set(), []
        ENGINES.pop("flaky", None)


def test_resume_does_not_reattempt_skip_step(tmp_path: Path):
    """A `skip` step records its failures (visible) but `resume` cache-hits it —
    skipped failures are intentional, not pending."""
    from chemrefine import cache
    from chemrefine.engines.api import ENGINES

    eng = _register_flaky()
    try:
        cfg = _seeded_config(
            tmp_path,
            [StepConfig(step=1, name="s", engine="flaky", operation="opt_sp", on_failure="skip")],
        )
        step_dir = (cfg.output_dir / "step1_s").resolve()
        eng.fail_ids = {"1"}
        assert execute(cfg, Action.RESUME) == 0  # skip → continues, no halt
        assert cache.load_failed_jobs(step_dir) == [
            {"structure_id": "1", "reason": "output missing"}
        ]  # failure is visible
        eng.fail_ids, eng.submitted = set(), []
        assert execute(cfg, Action.RESUME) == 0
        assert eng.submitted == []  # cache-hit — the skipped failure is NOT re-run
        assert cache.load_failed_jobs(step_dir) != []  # ledger kept for visibility
    finally:
        eng.fail_ids, eng.submitted = set(), []
        ENGINES.pop("flaky", None)


def _register_fake_nms():
    """A two-round NMS fake driven through the engine hook (``nms_input_info``) + the
    frequency values its ``parse`` attaches to each structure; the generic coordinator does
    the displacement. ``resolved`` controls which parents' displaced children resolve;
    ``fail_round1`` produce no round-1 output; ``submitted`` logs round-1 (parent) submissions
    and ``nms_seen`` the parents that reached the NMS stage."""
    from typing import ClassVar

    import numpy as np
    from ase import Atoms

    from chemrefine.engines.api import NmsInputInfo, register
    from chemrefine.ids import structure_artifact_path
    from chemrefine.state import JobBatch, StepInputs, StepResults, Structure

    @register("fake-nms2")
    class _FakeNms2:
        name = "fake-nms2"
        resolved: ClassVar[set[str]] = set()
        fail_round1: ClassVar[set[str]] = set()
        submitted: ClassVar[list[str]] = []
        nms_seen: ClassVar[list[str]] = []

        def prepare(self, ctx):
            ctx.step_dir.mkdir(parents=True, exist_ok=True)
            files = []
            for s in ctx.prev_state.structures:
                step = ctx.step_cfg.step
                inp = structure_artifact_path(ctx.step_dir, step, s.id, "inp")
                out = structure_artifact_path(ctx.step_dir, step, s.id, "out")
                inp.parent.mkdir(parents=True, exist_ok=True)
                inp.write_text("in\n", encoding="utf-8")
                files.append((inp, out, s.id))
            return StepInputs(files=tuple(files))

        def submit(self, inputs, ctx):
            for _inp, out, sid in inputs.files:
                if "_m" not in sid:  # a round-1 (parent) submission, not a displaced child
                    _FakeNms2.submitted.append(sid)
                if sid not in _FakeNms2.fail_round1:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text("E -1.0\n", encoding="utf-8")
            return JobBatch(jobs={})

        def parse(self, inputs, ctx):
            seeds = {s.id: s for s in ctx.prev_state.structures}
            out = []
            for _inp, _o, sid in inputs.files:
                seed = seeds.get(sid)
                # Frequency values ride on the parsed structure (one pass); NMS reads them.
                if "_m" in sid:  # a displaced child: resolved iff its parent is
                    parent = sid.split("_m")[0]
                    imaginary = {} if parent in _FakeNms2.resolved else {3: -9.0}
                    modes = None
                else:  # a round-1 parent reaching NMS
                    _FakeNms2.nms_seen.append(sid)
                    modes = np.zeros((1, 3, 6))
                    modes[0, 0, 5] = 0.1
                    imaginary = {5: -42.0}
                out.append(
                    Structure(
                        id=sid,
                        atoms=seed.atoms if seed else Atoms("H"),
                        energy_hartree=-1.0,
                        terminated=True,
                        converged=True,
                        imaginary_freqs=imaginary,
                        normal_modes=modes,
                    )
                )
            return StepResults(structures=tuple(out))

        def input_digest(self, ctx):
            return ""

        def nms_input_info(self, ctx):
            return NmsInputInfo(is_transition_state=False, computes_frequencies=True)

    return _FakeNms2


def _nms_step(displacement: float, on_failure: str = "skip") -> StepConfig:
    return StepConfig(
        step=1,
        name="s",
        engine="fake-nms2",
        operation="freq",
        nms=True,
        options={"target": "minimum", "displacement_value": displacement},
        on_failure=on_failure,
    )


def test_resume_after_tuning_reattempts_only_unresolved(tmp_path: Path):
    """Tuning displacement_value (reuse fingerprint unchanged) + resume reuses
    round-1 and re-runs NMS for ONLY the previously-unresolved parent."""
    from chemrefine import cache
    from chemrefine.engines.api import ENGINES

    eng = _register_fake_nms()
    try:
        eng.resolved = {"0"}  # "1" stays unresolved on the first run
        cfg1 = _seeded_config(tmp_path, [_nms_step(1.0)])
        step_dir = (cfg1.output_dir / "step1_s").resolve()
        execute(cfg1, Action.RESUME)
        assert cache.load_failed_jobs(step_dir) == [
            {"structure_id": "1", "reason": "NMS: target stationary point not reached"}
        ]
        # Unified model: the survivor keeps the parent's id (resolved geometry), not a child id.
        assert {s.id for s in cache.load(step_dir).results.structures} == {"0"}

        eng.resolved = {"0", "1"}  # new distance resolves "1"
        eng.submitted, eng.nms_seen = [], []
        execute(_seeded_config(tmp_path, [_nms_step(2.0)]), Action.RESUME)
        assert eng.submitted == []  # round-1 freq reused, not resubmitted
        assert eng.nms_seen == ["1"]  # only the unresolved parent re-attempted
        assert cache.load_failed_jobs(step_dir) == []
        assert {s.id for s in cache.load(step_dir).results.structures} == {"0", "1"}
    finally:
        eng.resolved, eng.fail_round1, eng.submitted, eng.nms_seen = set(), set(), [], []
        ENGINES.pop("fake-nms2", None)


def test_reattempt_resubmits_missing_round1(tmp_path: Path):
    """A parent whose round-1 output is missing gets its round-1 resubmitted on
    the next resume (NMS-unresolved parents do not)."""
    from chemrefine import cache
    from chemrefine.engines.api import ENGINES
    from chemrefine.errors import ChemRefineError

    eng = _register_fake_nms()
    try:
        eng.resolved = {"0", "1"}
        eng.fail_round1 = {"1"}  # "1" produces no round-1 output
        cfg = _seeded_config(tmp_path, [_nms_step(1.0, on_failure="stop")])
        step_dir = (cfg.output_dir / "step1_s").resolve()
        with pytest.raises(ChemRefineError):  # stop halts on the missing round-1
            execute(cfg, Action.RESUME)
        assert cache.load_failed_jobs(step_dir) == [
            {"structure_id": "1", "reason": "output missing"}
        ]

        eng.fail_round1 = set()  # round-1 recovers
        eng.submitted = []
        execute(cfg, Action.RESUME)  # same config → full-valid + ledger → reattempt_nms
        assert eng.submitted == ["1"]  # round-1 resubmitted only for the missing one
        assert cache.load_failed_jobs(step_dir) == []
        assert {s.id for s in cache.load(step_dir).results.structures} == {"0", "1"}
    finally:
        eng.resolved, eng.fail_round1, eng.submitted, eng.nms_seen = set(), set(), [], []
        ENGINES.pop("fake-nms2", None)


def test_rebuild_cache_reparses_without_submitting(tmp_path: Path):
    """`rebuild-cache` rebuilds a step's cache from existing outputs — no submit."""
    from chemrefine import cache
    from chemrefine.engines.api import ENGINES

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

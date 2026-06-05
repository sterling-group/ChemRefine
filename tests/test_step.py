"""Tests for the step lifecycle (prepare → submit → wait → parse → filter → cache)."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms

from chemrefine import cache
from chemrefine.config import Config, StepConfig
from chemrefine.state import PipelineState, Structure
from chemrefine.step import StepOutcome, build_context, run_step


def _config(tmp_path: Path, **step_overrides) -> Config:
    """Build a Config rooted at tmp_path with a single fake-engine step."""
    base = {"step": 1, "engine": "fake", "operation": "opt_sp"}
    base.update(step_overrides)
    return Config(
        template_dir=tmp_path / "templates",
        scratch_dir=tmp_path / "scratch",
        output_dir=tmp_path / "outputs",
        charge=0,
        multiplicity=1,
        max_cores=2,
        steps=[StepConfig(**base)],
    )


def _seed_state(ids: list[str]) -> PipelineState:
    return PipelineState(
        structures=tuple(Structure(id=i, atoms=Atoms("H")) for i in ids)
    )


# ---------------------------------------------------------------------------
# build_context
# ---------------------------------------------------------------------------


def test_build_context_threads_global_charge():
    cfg = Config(
        charge=2,
        multiplicity=3,
        steps=[StepConfig(step=1, engine="fake", operation="opt_sp")],
    )
    ctx = build_context(cfg, cfg.steps[0], PipelineState())
    assert ctx.charge == 2
    assert ctx.multiplicity == 3


def test_build_context_step_override_wins():
    cfg = Config(
        charge=0,
        multiplicity=1,
        steps=[
            StepConfig(step=1, engine="fake", operation="opt_sp", charge=-1, multiplicity=2)
        ],
    )
    ctx = build_context(cfg, cfg.steps[0], PipelineState())
    assert ctx.charge == -1
    assert ctx.multiplicity == 2


def test_build_context_directory_paths_are_absolute(tmp_path: Path):
    cfg = _config(tmp_path)
    ctx = build_context(cfg, cfg.steps[0], PipelineState())
    assert ctx.step_dir.is_absolute()
    assert ctx.step_dir.name == "step1"


def test_build_context_named_step_dir_includes_label(tmp_path: Path):
    cfg = _config(tmp_path, name="screen")
    ctx = build_context(cfg, cfg.steps[0], PipelineState())
    assert ctx.step_dir.name == "step1_screen"


# ---------------------------------------------------------------------------
# run_step end-to-end
# ---------------------------------------------------------------------------


def test_run_step_first_run_writes_cache_and_manifest(tmp_path: Path):
    cfg = _config(tmp_path)
    outcome = run_step(cfg, cfg.steps[0], _seed_state(["0", "1"]))
    assert isinstance(outcome, StepOutcome)
    assert outcome.cache_hit is False
    step_dir = cfg.output_dir.resolve() / "step1"
    assert (step_dir / "_cache" / "step.pkl").is_file()
    assert (step_dir / "_cache" / "manifest.json").is_file()


def test_run_step_returns_filtered_state(tmp_path: Path):
    cfg = _config(
        tmp_path,
        sample={"method": "integer", "count": 1},
    )
    outcome = run_step(cfg, cfg.steps[0], _seed_state(["0", "1", "2"]))
    assert len(outcome.state.structures) == 1


def test_run_step_no_sample_keeps_all_results(tmp_path: Path):
    cfg = _config(tmp_path)
    outcome = run_step(cfg, cfg.steps[0], _seed_state(["0", "1", "2"]))
    assert {s.id for s in outcome.state.structures} == {"0", "1", "2"}


def test_run_step_second_call_hits_cache(tmp_path: Path):
    cfg = _config(tmp_path)
    seeds = _seed_state(["0", "1"])
    first = run_step(cfg, cfg.steps[0], seeds)
    second = run_step(cfg, cfg.steps[0], seeds)
    assert first.cache_hit is False
    assert second.cache_hit is True
    # And the surviving state matches across runs.
    assert {s.id for s in first.state.structures} == {s.id for s in second.state.structures}


def test_run_step_invalidates_cache_when_config_changes(tmp_path: Path):
    cfg_a = _config(tmp_path, charge=0)
    cfg_b = _config(tmp_path, charge=-1)
    seeds = _seed_state(["0"])
    run_step(cfg_a, cfg_a.steps[0], seeds)
    # Different charge → fingerprint differs → cache miss
    outcome = run_step(cfg_b, cfg_b.steps[0], seeds)
    assert outcome.cache_hit is False


def test_run_step_disable_cache_re_executes(tmp_path: Path):
    cfg = _config(tmp_path)
    seeds = _seed_state(["0"])
    run_step(cfg, cfg.steps[0], seeds)
    outcome = run_step(cfg, cfg.steps[0], seeds, use_cache=False)
    assert outcome.cache_hit is False


def test_run_step_creates_output_directory(tmp_path: Path):
    cfg = _config(tmp_path, name="named")
    run_step(cfg, cfg.steps[0], _seed_state(["0"]))
    assert (cfg.output_dir.resolve() / "step1_named").is_dir()


def test_run_step_filter_after_cache_hit(tmp_path: Path):
    """Changing the sample method between runs only re-filters, no engine work needed."""
    # First run with no sampling -> cache stores all 3 structures.
    cfg_all = _config(tmp_path)
    seeds = _seed_state(["0", "1", "2"])
    run_step(cfg_all, cfg_all.steps[0], seeds)
    # But changing `sample` changes the fingerprint too, so it's actually a miss.
    cfg_one = _config(tmp_path, sample={"method": "integer", "count": 1})
    outcome = run_step(cfg_one, cfg_one.steps[0], seeds)
    # The fingerprint differs because the sample config is part of it,
    # so the engine re-runs and the filtered state has 1 survivor.
    assert outcome.cache_hit is False
    assert len(outcome.state.structures) == 1


# ---------------------------------------------------------------------------
# Manifest interaction
# ---------------------------------------------------------------------------


def test_manifest_persists_for_rebuild(tmp_path: Path):
    cfg = _config(tmp_path)
    run_step(cfg, cfg.steps[0], _seed_state(["0", "1"]))
    step_dir = cfg.output_dir.resolve() / "step1"
    inputs = cache.load_manifest(step_dir)
    assert inputs is not None
    assert [t[2] for t in inputs.files] == ["0", "1"]


def test_cache_load_after_run_returns_results(tmp_path: Path):
    cfg = _config(tmp_path)
    run_step(cfg, cfg.steps[0], _seed_state(["0"]))
    cached = cache.load(cfg.output_dir.resolve() / "step1")
    assert cached is not None
    assert [s.id for s in cached.results.structures] == ["0"]


# ---------------------------------------------------------------------------
# NMS reuse fingerprint
# ---------------------------------------------------------------------------


def test_nms_reuse_fingerprint_ignores_search_params():
    from chemrefine.step_nms import _nms_reuse_fingerprint

    base = StepConfig(
        step=1, engine="orca", operation="freq", nms=True,
        options={"target": "minimum", "displacement_value": 1.0},
    )
    tuned = base.model_copy(update={"options": {"target": "minimum", "displacement_value": 2.0}})
    assert _nms_reuse_fingerprint(base, ("0",)) == _nms_reuse_fingerprint(tuned, ("0",))


def test_nms_reuse_fingerprint_changes_on_criterion():
    from chemrefine.step_nms import _nms_reuse_fingerprint

    mn = StepConfig(
        step=1, engine="orca", operation="freq", nms=True, options={"target": "minimum"}
    )
    ts = mn.model_copy(update={"options": {"target": "ts"}})
    assert _nms_reuse_fingerprint(mn, ("0",)) != _nms_reuse_fingerprint(ts, ("0",))


def test_nms_reuse_fingerprint_empty_for_non_nms():
    from chemrefine.step_nms import _nms_reuse_fingerprint

    plain = StepConfig(step=1, engine="orca", operation="opt_sp")
    assert _nms_reuse_fingerprint(plain, ("0",)) == ""


# ---------------------------------------------------------------------------
# on_failure policy + per-structure failure capture
# ---------------------------------------------------------------------------


def _register_fail_engine():
    """Register a fake engine whose ``fail`` ClassVar marks per-sid failures.

    ``fail[sid] == "missing"`` produces no output; ``"unconverged"`` produces an
    output that parses but with ``terminated=False``; anything else succeeds.
    """
    from typing import ClassVar

    from chemrefine.engines.base import register
    from chemrefine.ids import structure_artifact_path
    from chemrefine.state import JobBatch, StepInputs, StepResults, Structure

    @register("fake-fail")
    class _FailEngine:
        name = "fake-fail"
        supports_nms = False
        fail: ClassVar[dict[str, str]] = {}

        def prepare(self, ctx):
            ctx.step_dir.mkdir(parents=True, exist_ok=True)
            files = []
            for s in ctx.prev_state.structures:
                step = ctx.step_cfg.step
                inp = structure_artifact_path(ctx.step_dir, step, s.id, "inp")
                out = structure_artifact_path(ctx.step_dir, step, s.id, "out")
                inp.write_text("in\n", encoding="utf-8")
                if self.fail.get(s.id) != "missing":
                    out.write_text("out\n", encoding="utf-8")
                files.append((inp, out, s.id))
            return StepInputs(files=tuple(files))

        def submit(self, inputs, ctx):
            return JobBatch(jobs={})

        def wait(self, batch):
            return None

        def parse(self, inputs, ctx):
            seeds = {s.id: s for s in ctx.prev_state.structures}
            out = []
            for _inp, _o, sid in inputs.files:
                seed = seeds[sid]
                out.append(
                    Structure(
                        id=sid, atoms=seed.atoms, parent_id=seed.parent_id,
                        energy_hartree=-1.0 - int(sid) * 1e-3,
                        terminated=self.fail.get(sid) != "unconverged",
                        converged=True,
                    )
                )
            return StepResults(structures=tuple(out))

        def normal_mode_sample(self, results, ctx):
            raise NotImplementedError

    return _FailEngine


def test_on_failure_skip_drops_failed_keeps_successes(tmp_path: Path):
    from chemrefine import cache
    from chemrefine.engines.base import ENGINES

    eng = _register_fail_engine()
    try:
        eng.fail = {"1": "unconverged"}
        cfg = _config(tmp_path, engine="fake-fail")  # on_failure defaults to skip
        outcome = run_step(cfg, cfg.steps[0], _seed_state(["0", "1", "2"]))
        assert {s.id for s in outcome.state.structures} == {"0", "2"}
        step_dir = cfg.output_dir.resolve() / "step1"
        assert cache.load_failed_jobs(step_dir) == [
            {"structure_id": "1", "reason": "did not terminate normally"}
        ]
    finally:
        eng.fail = {}
        ENGINES.pop("fake-fail", None)


def test_on_failure_stop_caches_successes_then_halts(tmp_path: Path):
    """`stop` runs the whole batch, caches the successes + ledgers the failure,
    then halts — so resume can re-attempt only the failed one."""
    import pytest

    from chemrefine import cache
    from chemrefine.engines.base import ENGINES
    from chemrefine.errors import ChemRefineError

    eng = _register_fail_engine()
    try:
        eng.fail = {"1": "missing"}  # "0" and "2" succeed, "1" fails
        cfg = _config(tmp_path, engine="fake-fail", on_failure="stop")
        step_dir = cfg.output_dir.resolve() / "step1"
        with pytest.raises(ChemRefineError):
            run_step(cfg, cfg.steps[0], _seed_state(["0", "1", "2"]))
        # batch ran to completion: the successes are cached *before* the halt …
        cached = cache.load(step_dir)
        assert cached is not None
        assert {s.id for s in cached.results.structures} == {"0", "2"}
        # … and the failure is recorded (pending for resume).
        assert cache.load_failed_jobs(step_dir) == [
            {"structure_id": "1", "reason": "output missing"}
        ]
    finally:
        eng.fail = {}
        ENGINES.pop("fake-fail", None)


def test_on_failure_best_backfills_all(tmp_path: Path):
    from chemrefine import cache
    from chemrefine.engines.base import ENGINES

    # Seeds carry an energy (as a real prior step would), so a missing-output
    # backfill (the submitted input) still survives energy filtering.
    seeds = PipelineState(
        structures=tuple(
            Structure(id=i, atoms=Atoms("H"), energy_hartree=-1.0) for i in ["0", "1", "2"]
        )
    )
    eng = _register_fail_engine()
    try:
        # "1" unconverged (parsed → best obtained); "2" missing (→ submitted input).
        eng.fail = {"1": "unconverged", "2": "missing"}
        cfg = _config(tmp_path, engine="fake-fail", on_failure="best")
        outcome = run_step(cfg, cfg.steps[0], seeds)
        assert {s.id for s in outcome.state.structures} == {"0", "1", "2"}
        # best keeps going, but the failures are still visible in the ledger.
        step_dir = cfg.output_dir.resolve() / "step1"
        assert {f["structure_id"] for f in cache.load_failed_jobs(step_dir)} == {"1", "2"}
    finally:
        eng.fail = {}
        ENGINES.pop("fake-fail", None)


def test_resolve_nms_keeps_resolved_drops_unresolved(tmp_path: Path):
    """_resolve_nms: already-resolved pass-through + resolved children kept;
    a round-1 parent with no resolved child becomes a (ledgered) failure."""
    from chemrefine import cache
    from chemrefine.state import StepResults
    from chemrefine.step import build_context
    from chemrefine.step_nms import _resolve_nms

    cfg = _config(tmp_path, engine="orca", operation="freq", nms=True)
    ctx = build_context(cfg, cfg.steps[0], _seed_state(["0", "1", "2"]))
    ctx.step_dir.mkdir(parents=True, exist_ok=True)
    round1 = StepResults(
        structures=tuple(
            Structure(id=i, atoms=Atoms("H"), energy_hartree=-1.0) for i in ["0", "1", "2"]
        )
    )
    def _r(sid, parent, e, conv):
        return Structure(
            id=sid, atoms=Atoms("H"), parent_id=parent,
            energy_hartree=e, converged=conv, terminated=True,
        )

    nms_results = StepResults(structures=(
        _r("0", None, -1.0, True),          # already at the target (round-1 id)
        _r("1_m5_pos", "1", -1.1, True),    # "1" → one resolved child
        _r("1_m5_neg", "1", -1.0, False),   #        one not
        _r("2_m5_pos", "2", -0.9, False),   # "2" → both children unresolved
        _r("2_m5_neg", "2", -0.8, False),
    ))
    out = _resolve_nms(nms_results, round1, ctx, cfg.steps[0])
    assert {s.id for s in out.structures} == {"0", "1_m5_pos"}
    assert cache.load_failed_jobs(ctx.step_dir) == [
        {"structure_id": "2", "reason": "NMS: target stationary point not reached"}
    ]


def test_run_step_nms_branch_runs_when_engine_supports_it(tmp_path: Path):
    """The NMS branch in run_step fires when both step_cfg.nms and engine.supports_nms are true."""
    from chemrefine.engines.base import ENGINES, register
    from chemrefine.state import JobBatch, StepInputs, StepResults

    nms_calls: list[int] = []

    @register("fake-nms")
    class _NmsEngine:
        name = "fake-nms"
        supports_nms = True

        def prepare(self, ctx):
            ctx.step_dir.mkdir(parents=True, exist_ok=True)
            return StepInputs(files=())

        def submit(self, inputs, ctx):
            return JobBatch(jobs={})

        def wait(self, batch):
            return None

        def parse(self, inputs, ctx):
            return StepResults(structures=tuple(ctx.prev_state.structures))

        def normal_mode_sample(self, results, ctx):
            nms_calls.append(1)
            return results

    try:
        cfg = _config(tmp_path, engine="fake-nms", nms=True)
        engine = ENGINES["fake-nms"]()
        run_step(cfg, cfg.steps[0], _seed_state(["0"]), engine=engine)
        assert nms_calls == [1]
    finally:
        ENGINES.pop("fake-nms", None)

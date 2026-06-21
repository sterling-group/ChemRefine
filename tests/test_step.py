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
    return PipelineState(structures=tuple(Structure(id=i, atoms=Atoms("H")) for i in ids))


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
        steps=[StepConfig(step=1, engine="fake", operation="opt_sp", charge=-1, multiplicity=2)],
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


def test_build_context_threads_slurm_array():
    cfg = Config(
        slurm_array=True,
        steps=[StepConfig(step=1, engine="fake", operation="opt_sp")],
    )
    ctx = build_context(cfg, cfg.steps[0], PipelineState())
    assert ctx.slurm_array is True


# ---------------------------------------------------------------------------
# run_step end-to-end
# ---------------------------------------------------------------------------


def test_run_step_first_run_writes_cache_and_manifest(tmp_path: Path):
    cfg = _config(tmp_path)
    outcome = run_step(cfg, cfg.steps[0], _seed_state(["0", "1"]))
    assert isinstance(outcome, StepOutcome)
    assert outcome.cache_hit is False
    step_dir = cfg.output_dir.resolve() / "step1"
    assert (step_dir / "_cache" / "step.json").is_file()
    assert (step_dir / "_cache" / "manifest.json").is_file()


def test_run_step_returns_filtered_state(tmp_path: Path):
    cfg = _config(
        tmp_path,
        sample={"method": "min", "count": 1},
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


def test_run_step_filter_only_change_is_cache_hit(tmp_path: Path):
    """Changing only ``sample:`` re-filters the cached results — no engine re-run.

    The cache stores pre-filter results and filtering re-runs on every load,
    so the fingerprint deliberately excludes the sample config: tuning a
    filter must never redo the step's calculations.
    """
    # First run with no sampling -> cache stores all 3 structures.
    cfg_all = _config(tmp_path)
    seeds = _seed_state(["0", "1", "2"])
    run_step(cfg_all, cfg_all.steps[0], seeds)
    cfg_one = _config(tmp_path, sample={"method": "min", "count": 1})
    outcome = run_step(cfg_one, cfg_one.steps[0], seeds)
    assert outcome.cache_hit is True
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
                inp.parent.mkdir(parents=True, exist_ok=True)
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
                        id=sid,
                        atoms=seed.atoms,
                        parent_id=seed.parent_id,
                        energy_hartree=-1.0 - int(sid) * 1e-3,
                        terminated=self.fail.get(sid) != "unconverged",
                        converged=True,
                    )
                )
            return StepResults(structures=tuple(out))

        def input_digest(self, ctx):
            return ""

    return _FailEngine


def test_on_failure_skip_drops_failed_keeps_successes(tmp_path: Path):
    from chemrefine import cache
    from chemrefine.engines.base import ENGINES

    eng = _register_fail_engine()
    try:
        eng.fail = {"1": "unconverged"}
        cfg = _config(tmp_path, engine="fake-fail", on_failure="skip")
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
    """`stop` runs the whole batch and caches the successes + ledgers the failure;
    the single pipeline-level `halt_if_pending` then halts (run_step itself does not)."""
    import pytest

    from chemrefine import cache
    from chemrefine import step as step_mod
    from chemrefine.engines.base import ENGINES
    from chemrefine.errors import ChemRefineError

    eng = _register_fail_engine()
    try:
        eng.fail = {"1": "missing"}  # "0" and "2" succeed, "1" fails
        cfg = _config(tmp_path, engine="fake-fail", on_failure="stop")
        step_dir = cfg.output_dir.resolve() / "step1"
        # run_step no longer raises — it runs the whole batch, caches the
        # successes *before* any halt, and ledgers the failure …
        run_step(cfg, cfg.steps[0], _seed_state(["0", "1", "2"]))
        cached = cache.load(step_dir)
        assert cached is not None
        assert {s.id for s in cached.results.structures} == {"0", "2"}
        assert cache.load_failed_jobs(step_dir) == [
            {"structure_id": "1", "reason": "output missing"}
        ]
        # … and the run is halted by the single pipeline-level check.
        with pytest.raises(ChemRefineError):
            step_mod.halt_if_pending(cfg, cfg.steps[0], None)
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


# ---------------------------------------------------------------------------
# B10 — auto-retry-once on convergence failure
# ---------------------------------------------------------------------------


def _register_conv_engine():
    """A fake engine: structures fail to converge on the first parse and converge
    on the retry (a second parse) — unless their id is in ``never``."""
    from typing import ClassVar

    import numpy as np

    from chemrefine.engines.base import register
    from chemrefine.ids import structure_artifact_path
    from chemrefine.state import JobBatch, StepInputs, StepResults, Structure

    @register("conv-retry")
    class _ConvEngine:
        name = "conv-retry"
        supports_nms = False
        never: ClassVar[set[str]] = set()  # ids that never converge (even on retry)
        parse_count: ClassVar[dict[str, int]] = {}

        def prepare(self, ctx):
            ctx.step_dir.mkdir(parents=True, exist_ok=True)
            files = []
            for s in ctx.prev_state.structures:
                step = ctx.step_cfg.step
                inp = structure_artifact_path(ctx.step_dir, step, s.id, "inp")
                out = structure_artifact_path(ctx.step_dir, step, s.id, "out")
                inp.parent.mkdir(parents=True, exist_ok=True)
                inp.write_text("in\n", encoding="utf-8")
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
                _ConvEngine.parse_count[sid] = _ConvEngine.parse_count.get(sid, 0) + 1
                converged = sid not in _ConvEngine.never and _ConvEngine.parse_count[sid] >= 2
                seed = seeds[sid]
                out.append(
                    Structure(
                        id=sid,
                        atoms=seed.atoms,
                        parent_id=seed.parent_id,
                        energy_hartree=-1.0 - int(sid) * 1e-3,
                        forces_ev_per_a=np.zeros((len(seed.atoms), 3)),
                        terminated=True,
                        converged=converged,
                    )
                )
            return StepResults(structures=tuple(out))

        def input_digest(self, ctx):
            return ""

    return _ConvEngine


def test_run_step_retries_unconverged_from_best_geometry(tmp_path: Path):
    """A "did not converge" structure is retried once from its best geometry; the
    failed attempt is archived under ``attempt1/`` and the retry (which converges)
    survives."""
    from chemrefine.engines.base import ENGINES

    eng = _register_conv_engine()
    try:
        eng.never = set()
        eng.parse_count = {}
        cfg = _config(tmp_path, engine="conv-retry", on_failure="stop")
        outcome = run_step(cfg, cfg.steps[0], _seed_state(["0"]))
        assert {s.id for s in outcome.state.structures} == {"0"}  # retry converged
        step_dir = cfg.output_dir.resolve() / "step1"
        assert (step_dir / "0" / "attempt1").is_dir()  # failed attempt archived
        assert cache.load_failed_jobs(step_dir) == []  # no pending failures
    finally:
        eng.never = set()
        eng.parse_count = {}
        ENGINES.pop("conv-retry", None)


def test_run_step_unconverged_retry_still_fails_is_ledgered(tmp_path: Path):
    """When the retry also fails to converge, the failure is ledgered (after one
    archived attempt) for `resume` to pick up."""
    from chemrefine.engines.base import ENGINES

    eng = _register_conv_engine()
    try:
        eng.never = {"0"}  # never converges, even on retry
        eng.parse_count = {}
        cfg = _config(tmp_path, engine="conv-retry", on_failure="stop")
        outcome = run_step(cfg, cfg.steps[0], _seed_state(["0"]))
        assert outcome.state.structures == ()
        step_dir = cfg.output_dir.resolve() / "step1"
        assert (step_dir / "0" / "attempt1").is_dir()  # one retry attempt, then stop
        assert not (step_dir / "0" / "attempt2").exists()  # only once per run
        assert cache.load_failed_jobs(step_dir) == [
            {"structure_id": "0", "reason": "did not converge"}
        ]
    finally:
        eng.never = set()
        eng.parse_count = {}
        ENGINES.pop("conv-retry", None)


def test_resume_retries_unconverged_again_into_next_attempt(tmp_path: Path):
    """A later run (resume) re-attempts a still-pending convergence failure, archiving
    into the next attempt dir — proving the retry is per-run, not blocked by attempt1."""
    from chemrefine.engines.base import ENGINES

    eng = _register_conv_engine()
    try:
        eng.never = {"0"}  # never converges
        eng.parse_count = {}
        cfg = _config(tmp_path, engine="conv-retry", on_failure="stop")
        state = _seed_state(["0"])
        step_dir = cfg.output_dir.resolve() / "step1"

        run_step(cfg, cfg.steps[0], state)  # first run → attempt1, then ledgered
        assert (step_dir / "0" / "attempt1").is_dir()

        # Resume: the cached step still has the pending convergence failure, so
        # _resubmit_failed retries it again — into attempt2 (never blocked by attempt1).
        run_step(cfg, cfg.steps[0], state)
        assert (step_dir / "0" / "attempt2").is_dir()
        assert cache.load_failed_jobs(step_dir) == [
            {"structure_id": "0", "reason": "did not converge"}
        ]
    finally:
        eng.never = set()
        eng.parse_count = {}
        ENGINES.pop("conv-retry", None)


def test_check_nms_freq_gate_rejects_when_input_computes_no_frequencies(tmp_path: Path):
    """B9 (generic): nms + no explicit operation + an input that computes no frequencies."""
    import pytest

    from chemrefine.engines.base import NmsInputInfo
    from chemrefine.errors import ConfigError
    from chemrefine.step import _check_nms_freq_gate, build_context

    class _NoFreqEngine:
        def nms_input_info(self, ctx):
            return NmsInputInfo(is_transition_state=False, computes_frequencies=False)

    cfg = _config(tmp_path, engine="orca", nms=True, operation=None)  # no explicit operation
    ctx = build_context(cfg, cfg.steps[0], _seed_state([]))
    with pytest.raises(ConfigError, match="frequency"):
        _check_nms_freq_gate(_NoFreqEngine(), ctx, cfg.steps[0])  # type: ignore[arg-type]

    class _FreqEngine:
        def nms_input_info(self, ctx):
            return NmsInputInfo(is_transition_state=False, computes_frequencies=True)

    assert _check_nms_freq_gate(_FreqEngine(), ctx, cfg.steps[0]) is None  # type: ignore[arg-type]


def test_archive_failed_attempt_numbers_sequentially(tmp_path: Path):
    """Numbered attempt dirs: next-free K, never blocked by an existing/odd one."""
    from chemrefine import step_failures

    sid_dir = tmp_path / "0"
    sid_dir.mkdir()
    (sid_dir / "step1_0.out").write_text("fail", encoding="utf-8")
    dest = step_failures.archive_failed_attempt(sid_dir)
    assert dest.name == "attempt1"
    assert (dest / "step1_0.out").is_file()  # the loose file moved in
    assert not (sid_dir / "step1_0.out").exists()

    (sid_dir / "step1_0.out").write_text("fail2", encoding="utf-8")
    assert step_failures.archive_failed_attempt(sid_dir).name == "attempt2"  # next free

    # A manually-added higher attempt + a non-matching 'attempt*' dir: K = max+1,
    # the odd dir is ignored, and existing attempt dirs are left in place.
    (sid_dir / "attempt5").mkdir()
    (sid_dir / "attemptX").mkdir()  # matches the glob but not attempt<digits>
    (sid_dir / "step1_0.out").write_text("fail3", encoding="utf-8")
    assert step_failures.archive_failed_attempt(sid_dir).name == "attempt6"
    assert (sid_dir / "attempt1").is_dir() and (sid_dir / "attempt5").is_dir()


def test_run_step_nms_branch_routes_through_coordinator(tmp_path: Path, monkeypatch):
    """run_step routes an `nms: true` step through the generic coordinator (nms.run_nms),
    then applies the on_failure policy to its survivors/failures."""
    from chemrefine import nms as nms_mod
    from chemrefine.engines.base import ENGINES, FrequencyData, NmsInputInfo, register
    from chemrefine.state import JobBatch, StepInputs, StepResults
    from chemrefine.step_failures import NmsResolution

    calls: list[int] = []

    def _fake_run_nms(engine, round1, failures, ctx, step_cfg):
        calls.append(1)
        return NmsResolution(survivors=round1.structures, failures=tuple(failures))

    monkeypatch.setattr(nms_mod, "run_nms", _fake_run_nms)

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
            return StepResults(structures=())

        def input_digest(self, ctx):
            return ""

        def nms_input_info(self, ctx):
            return NmsInputInfo(is_transition_state=False, computes_frequencies=True)

        def read_frequencies(self, structure_id, step_dir, ctx):
            return FrequencyData(imaginary={}, modes=None)

    try:
        cfg = _config(tmp_path, engine="fake-nms", nms=True)
        run_step(cfg, cfg.steps[0], _seed_state(["0"]), engine=ENGINES["fake-nms"]())
        assert calls == [1]
    finally:
        ENGINES.pop("fake-nms", None)


def test_on_failure_best_drops_failure_with_no_fallback(tmp_path: Path):
    """best: a failure with no best geometry and no prior-state entry has
    nothing to backfill — it is dropped while the others are kept."""
    from chemrefine.step_failures import Failure, apply_failure_policy

    cfg = _config(tmp_path, on_failure="best")
    ctx = build_context(cfg, cfg.steps[0], _seed_state(["0"]))
    ctx.step_dir.mkdir(parents=True, exist_ok=True)
    failures = [Failure(sid="ghost", reason="output missing", best=None)]
    results = apply_failure_policy([], failures, ctx, cfg.steps[0])
    assert results.structures == ()
    assert cache.load_failed_jobs(ctx.step_dir) == [
        {"structure_id": "ghost", "reason": "output missing"}
    ]

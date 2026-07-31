"""Tests for the step lifecycle (prepare → submit → wait → parse → filter → cache)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from ase import Atoms
from fake_engine import FakeEngine

from chemrefine import cache
from chemrefine.config import Config, StepConfig
from chemrefine.errors import CacheError, ChemRefineError
from chemrefine.state import (
    FailureKind,
    FailureRecord,
    PipelineState,
    StepContext,
    StepInputs,
    StepResults,
    Structure,
)
from chemrefine.step import (
    StepMode,
    StepOutcome,
    build_context,
    halt_if_pending,
    run_step,
    step_dir_for,
)


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
    outcome = run_step(cfg, cfg.steps[0], seeds, mode=StepMode.EXECUTE)
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
    output that parses but with ``terminated_normally=False``; anything else succeeds.
    """
    from typing import ClassVar

    from chemrefine.engines.api import register
    from chemrefine.ids import structure_artifact_path
    from chemrefine.state import JobBatch, StepInputs, StepResults, Structure

    @register("fake-fail")
    class _FailEngine:
        name = "fake-fail"
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
                        terminated_normally=self.fail.get(sid) != "unconverged",
                        converged=True,
                    )
                )
            return StepResults(structures=tuple(out))

        def input_digest(self, ctx):
            return ""

    return _FailEngine


def test_on_failure_skip_drops_failed_keeps_successes(tmp_path: Path):
    from chemrefine.engines.api import ENGINES

    eng = _register_fail_engine()
    try:
        eng.fail = {"1": "unconverged"}
        cfg = _config(tmp_path, engine="fake-fail", on_failure="skip")
        outcome = run_step(cfg, cfg.steps[0], _seed_state(["0", "1", "2"]))
        assert {s.id for s in outcome.state.structures} == {"0", "2"}
        step_dir = cfg.output_dir.resolve() / "step1"
        assert [(r.structure_id, r.kind) for r in cache.load_failure_records(step_dir)] == [
            ("1", FailureKind.NOT_TERMINATED)
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
    from chemrefine.engines.api import ENGINES
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
        assert [(r.structure_id, r.kind) for r in cache.load_failure_records(step_dir)] == [
            ("1", FailureKind.MISSING_OUTPUT)
        ]
        # … and the run is halted by the single pipeline-level check.
        with pytest.raises(ChemRefineError):
            step_mod.halt_if_pending(cfg, cfg.steps[0], StepMode.RESUME)
    finally:
        eng.fail = {}
        ENGINES.pop("fake-fail", None)


def test_on_failure_best_backfills_all(tmp_path: Path):
    from chemrefine import cache
    from chemrefine.engines.api import ENGINES

    # Real step-1 seeds: bootstrapped from the input .xyz, so they carry NO energy
    # yet. A missing-output backfill is the submitted input, and it must survive
    # the (identity) filter — this is the case B5 silently turned into `skip`.
    seeds = PipelineState(
        structures=tuple(Structure(id=i, atoms=Atoms("H")) for i in ["0", "1", "2"])
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
        assert {f.structure_id for f in cache.load_failure_records(step_dir)} == {"1", "2"}
    finally:
        eng.fail = {}
        ENGINES.pop("fake-fail", None)


# ---------------------------------------------------------------------------
# Auto-retry-once on convergence failure
# ---------------------------------------------------------------------------


def _register_conv_engine():
    """A fake engine: structures fail to converge on the first parse and converge
    on the retry (a second parse) — unless their id is in ``never``."""
    from typing import ClassVar

    import numpy as np

    from chemrefine.engines.api import register
    from chemrefine.ids import structure_artifact_path
    from chemrefine.state import JobBatch, StepInputs, StepResults, Structure

    @register("conv-retry")
    class _ConvEngine:
        name = "conv-retry"
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
                        terminated_normally=True,
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
    from chemrefine.engines.api import ENGINES

    eng = _register_conv_engine()
    try:
        eng.never = set()
        eng.parse_count = {}
        cfg = _config(tmp_path, engine="conv-retry", on_failure="stop")
        outcome = run_step(cfg, cfg.steps[0], _seed_state(["0"]))
        assert {s.id for s in outcome.state.structures} == {"0"}  # retry converged
        step_dir = cfg.output_dir.resolve() / "step1"
        assert (step_dir / "0" / "attempt1").is_dir()  # failed attempt archived
        assert cache.load_failure_records(step_dir) == []  # no pending failures
    finally:
        eng.never = set()
        eng.parse_count = {}
        ENGINES.pop("conv-retry", None)


def test_run_step_unconverged_retry_still_fails_is_ledgered(tmp_path: Path):
    """When the retry also fails to converge, the failure is ledgered (after one
    archived attempt) for `resume` to pick up."""
    from chemrefine.engines.api import ENGINES

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
        assert [(r.structure_id, r.kind) for r in cache.load_failure_records(step_dir)] == [
            ("0", FailureKind.NOT_CONVERGED)
        ]
    finally:
        eng.never = set()
        eng.parse_count = {}
        ENGINES.pop("conv-retry", None)


def test_resume_retries_unconverged_again_into_next_attempt(tmp_path: Path):
    """A later run (resume) re-attempts a still-pending convergence failure, archiving
    into the next attempt dir — proving the retry is per-run, not blocked by attempt1."""
    from chemrefine.engines.api import ENGINES

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
        assert [(r.structure_id, r.kind) for r in cache.load_failure_records(step_dir)] == [
            ("0", FailureKind.NOT_CONVERGED)
        ]
    finally:
        eng.never = set()
        eng.parse_count = {}
        ENGINES.pop("conv-retry", None)


def test_check_nms_freq_gate_rejects_when_input_computes_no_frequencies(tmp_path: Path):
    """B9 (generic): nms + an input that computes no frequencies is rejected up front."""
    import pytest

    from chemrefine.engines.api import NmsInputInfo
    from chemrefine.errors import ConfigError
    from chemrefine.step import _check_nms_freq_gate, build_context

    class _NoFreqEngine:
        def nms_input_info(self, ctx):
            return NmsInputInfo(is_transition_state=False, computes_frequencies=False)

        def artifact_paths(self, ctx, structure_id):
            step = ctx.step_cfg.step
            return (
                ctx.step_dir / structure_id / f"step{step}_{structure_id}.inp",
                ctx.step_dir / structure_id / f"step{step}_{structure_id}.out",
            )

    cfg = _config(tmp_path, engine="orca", nms=True, operation=None)
    ctx = build_context(cfg, cfg.steps[0], _seed_state([]))
    with pytest.raises(ConfigError, match="frequency"):
        _check_nms_freq_gate(_NoFreqEngine(), ctx, cfg.steps[0])

    class _FreqEngine:
        def nms_input_info(self, ctx):
            return NmsInputInfo(is_transition_state=False, computes_frequencies=True)

        def artifact_paths(self, ctx, structure_id):
            step = ctx.step_cfg.step
            return (
                ctx.step_dir / structure_id / f"step{step}_{structure_id}.inp",
                ctx.step_dir / structure_id / f"step{step}_{structure_id}.out",
            )

    # Returns None; calling it without raising is the assertion.
    _check_nms_freq_gate(_FreqEngine(), ctx, cfg.steps[0])


def test_check_nms_freq_gate_not_bypassed_by_explicit_operation(tmp_path: Path):
    """An explicit `operation` only picks the parser — it never adds frequencies to
    the input, so it must not bypass the gate (a doomed NMS step would otherwise run
    to completion and leave every structure unresolved)."""
    import pytest

    from chemrefine.engines.api import NmsInputInfo
    from chemrefine.errors import ConfigError
    from chemrefine.step import _check_nms_freq_gate, build_context

    class _NoFreqEngine:
        def nms_input_info(self, ctx):
            return NmsInputInfo(is_transition_state=False, computes_frequencies=False)

        def artifact_paths(self, ctx, structure_id):
            step = ctx.step_cfg.step
            return (
                ctx.step_dir / structure_id / f"step{step}_{structure_id}.inp",
                ctx.step_dir / structure_id / f"step{step}_{structure_id}.out",
            )

    cfg = _config(tmp_path, engine="orca", nms=True, operation="opt_sp")
    ctx = build_context(cfg, cfg.steps[0], _seed_state([]))
    with pytest.raises(ConfigError, match="drop `nms: true`"):
        _check_nms_freq_gate(_NoFreqEngine(), ctx, cfg.steps[0])


def test_run_step_nms_branch_routes_through_coordinator(tmp_path: Path, monkeypatch):
    """run_step routes an `nms: true` step through the generic coordinator (nms.run_nms),
    then applies the on_failure policy to its survivors/failures."""
    from chemrefine import nms as nms_mod
    from chemrefine.engines.api import ENGINES, NmsInputInfo, register
    from chemrefine.nms import NmsResolution
    from chemrefine.state import JobBatch, StepInputs, StepResults

    calls: list[int] = []

    def _fake_run_nms(engine, round1, failures, ctx):
        calls.append(1)
        return NmsResolution(survivors=round1.structures, failures=tuple(failures))

    monkeypatch.setattr(nms_mod, "run_nms", _fake_run_nms)

    @register("fake-nms")
    class _NmsEngine:
        name = "fake-nms"

        def prepare(self, ctx):
            ctx.step_dir.mkdir(parents=True, exist_ok=True)
            return StepInputs(files=())

        def submit(self, inputs, ctx):
            return JobBatch(jobs={})

        def parse(self, inputs, ctx):
            return StepResults(structures=())

        def input_digest(self, ctx):
            return ""

        def nms_input_info(self, ctx):
            return NmsInputInfo(is_transition_state=False, computes_frequencies=True)

        def artifact_paths(self, ctx, structure_id):
            step = ctx.step_cfg.step
            return (
                ctx.step_dir / structure_id / f"step{step}_{structure_id}.inp",
                ctx.step_dir / structure_id / f"step{step}_{structure_id}.out",
            )

    try:
        cfg = _config(tmp_path, engine="fake-nms", nms=True)
        run_step(cfg, cfg.steps[0], _seed_state(["0"]), engine=ENGINES["fake-nms"]())
        assert calls == [1]
    finally:
        ENGINES.pop("fake-nms", None)


def test_on_failure_best_drops_failure_with_no_fallback(tmp_path: Path):
    """best: a failure with no best geometry and no prior-state entry has
    nothing to backfill — it is dropped while the others are kept."""
    from chemrefine.lifecycle import apply_failure_policy
    from chemrefine.state import Failure

    cfg = _config(tmp_path, on_failure="best")
    ctx = build_context(cfg, cfg.steps[0], _seed_state(["0"]))
    ctx.step_dir.mkdir(parents=True, exist_ok=True)
    failures = [Failure(sid="ghost", kind=FailureKind.MISSING_OUTPUT, best=None)]
    results = apply_failure_policy([], failures, ctx, cfg.steps[0])
    assert results.structures == ()
    assert [(r.structure_id, r.kind) for r in cache.load_failure_records(ctx.step_dir)] == [
        ("ghost", FailureKind.MISSING_OUTPUT)
    ]


def test_run_step_writes_canonical_result_records(tmp_path: Path):
    """Every parsed job leaves its engine-independent ``*.result.json`` behind."""
    cfg = _config(tmp_path)
    run_step(cfg, cfg.steps[0], _seed_state(["0", "1"]))
    step_dir = cfg.output_dir.resolve() / "step1"
    for sid in ("0", "1"):
        record_path = step_dir / sid / f"step1_{sid}.result.json"
        assert record_path.is_file()
        record = json.loads(record_path.read_text())
        assert record["result_format"] == cache.RESULT_FORMAT_VERSION
        assert record["id"] == sid
        assert record["energy_hartree"] is not None


# ---------------------------------------------------------------------------
# A re-executed step never re-reads the previous run's output
# ---------------------------------------------------------------------------


def test_rerun_does_not_read_a_previous_runs_output(tmp_path: Path):
    """The invariant: a parsed output must come from *this* run's submission.

    ``parse_with_failures`` decides success by ``out.is_file()``, which cannot tell
    this run's output from a leftover. Without archiving, a re-executed step whose
    job dies before writing anything silently re-reads the previous run's result and
    reports it as current — the worst failure mode for a tool whose output feeds a
    publication.
    """
    from chemrefine.engines.api import ENGINES

    eng = _register_fail_engine()
    try:
        eng.fail = {}
        cfg = _config(tmp_path, engine="fake-fail", on_failure="skip")
        state = _seed_state(["0"])
        step_dir = cfg.output_dir.resolve() / "step1"

        first = run_step(cfg, cfg.steps[0], state, mode=StepMode.EXECUTE)
        assert {s.id for s in first.state.structures} == {"0"}
        assert (step_dir / "0" / "step1_0.out").is_file()

        # Re-execute; this time the job dies without producing an output.
        eng.fail = {"0": "missing"}
        second = run_step(cfg, cfg.steps[0], state, mode=StepMode.EXECUTE)

        assert second.state.structures == ()  # not the stale success
        assert [(r.structure_id, r.kind) for r in cache.load_failure_records(step_dir)] == [
            ("0", FailureKind.MISSING_OUTPUT)
        ]
        # The previous run's work is archived, not destroyed — full provenance.
        assert (step_dir / "0" / "attempt1" / "step1_0.out").is_file()
    finally:
        eng.fail = {}
        ENGINES.pop("fake-fail", None)


def test_rerun_archives_each_run_into_its_own_attempt_dir(tmp_path: Path):
    """Repeated re-execution keeps every attempt, numbered — never clobbers."""
    from chemrefine.engines.api import ENGINES

    eng = _register_fail_engine()
    try:
        eng.fail = {}
        cfg = _config(tmp_path, engine="fake-fail")
        state = _seed_state(["0"])
        step_dir = cfg.output_dir.resolve() / "step1"

        for _ in range(3):
            run_step(cfg, cfg.steps[0], state, mode=StepMode.EXECUTE)

        # Run 1 left the dir bare; runs 2 and 3 each archived the run before them.
        assert (step_dir / "0" / "attempt1").is_dir()
        assert (step_dir / "0" / "attempt2").is_dir()
        assert not (step_dir / "0" / "attempt3").exists()
        assert (step_dir / "0" / "step1_0.out").is_file()  # newest run stays put
    finally:
        eng.fail = {}
        ENGINES.pop("fake-fail", None)


def test_first_run_over_a_clean_tree_archives_nothing(tmp_path: Path):
    """Archiving is conditional on prior artifacts — a fresh run makes no attempt dir."""
    cfg = _config(tmp_path)
    run_step(cfg, cfg.steps[0], _seed_state(["0"]), mode=StepMode.EXECUTE)
    assert not list((cfg.output_dir.resolve() / "step1" / "0").glob("attempt*"))


# ---------------------------------------------------------------------------
# halt_if_pending — the single halt point, reached by every mode
# ---------------------------------------------------------------------------


def test_halt_if_pending_skips_a_cache_only_step(tmp_path: Path):
    cfg = _config(tmp_path, on_failure="stop")
    # CACHE_ONLY is the mode every step a scoped action isn't targeting runs in;
    # halting there would stop `rerun-errors N` before it ever reached step N.
    halt_if_pending(cfg, cfg.steps[0], StepMode.CACHE_ONLY)


def test_halt_if_pending_raises_when_stop_step_has_pending(tmp_path: Path):
    cfg = _config(tmp_path, on_failure="stop")
    step_dir = step_dir_for(cfg, cfg.steps[0])
    cache.save_failure_records(
        step_dir, [FailureRecord(structure_id="1", kind=FailureKind.FAILED, reason="x")]
    )
    with pytest.raises(ChemRefineError, match="halted"):
        halt_if_pending(cfg, cfg.steps[0], StepMode.RESUME)


def test_halt_if_pending_no_pending_returns(tmp_path: Path):
    cfg = _config(tmp_path, on_failure="stop")
    step_dir_for(cfg, cfg.steps[0]).mkdir(parents=True, exist_ok=True)
    halt_if_pending(cfg, cfg.steps[0], StepMode.RESUME)  # no ledger → no raise


# ---------------------------------------------------------------------------
# CACHE_ONLY submits nothing, whatever it finds on disk
# ---------------------------------------------------------------------------
#
# It is the mode every step runs in that a scoped action is *not* targeting, so both
# `rebuild-cache` (documented "no submission") and `rerun-errors N` ("prior steps
# cache-hit") depend on it never reaching the engine. There is more than one way out of
# `run_step` that submits, so these assert the property rather than any one branch.


class _SubmitSpy(FakeEngine):
    """A fake engine that records every submission instead of trusting a log line."""

    def __init__(self) -> None:
        super().__init__()
        self.submissions = 0

    def submit(self, inputs, ctx):
        self.submissions += 1
        return super().submit(inputs, ctx)


def test_cache_only_refuses_a_step_with_no_valid_cache(tmp_path: Path):
    """No cache to honour and no permission to make one — say so, do not run the step."""
    cfg = _config(tmp_path)
    engine = _SubmitSpy()

    with pytest.raises(ChemRefineError, match="does not submit"):
        run_step(cfg, cfg.steps[0], _seed_state(["0"]), engine=engine, mode=StepMode.CACHE_ONLY)

    assert engine.submissions == 0


def test_cache_only_serves_a_valid_cache(tmp_path: Path):
    """The other half: a step it *can* honour still costs nothing."""
    cfg = _config(tmp_path)
    seeds = _seed_state(["0"])
    run_step(cfg, cfg.steps[0], seeds, engine=FakeEngine(), mode=StepMode.EXECUTE)

    engine = _SubmitSpy()
    outcome = run_step(cfg, cfg.steps[0], seeds, engine=engine, mode=StepMode.CACHE_ONLY)

    assert outcome.cache_hit is True
    assert engine.submissions == 0


def test_cache_only_leaves_the_outputs_it_was_not_asked_to_touch(tmp_path: Path):
    """Refusing must also not archive: the outputs are what the caller came to read."""
    cfg = _config(tmp_path)
    seeds = _seed_state(["0"])
    run_step(cfg, cfg.steps[0], seeds, engine=FakeEngine(), mode=StepMode.EXECUTE)
    step_dir = step_dir_for(cfg, cfg.steps[0])
    before = sorted(p.name for p in (step_dir / "0").iterdir())

    cache.invalidate(step_dir)  # the cache is gone; the outputs are not
    with pytest.raises(ChemRefineError):
        run_step(cfg, cfg.steps[0], seeds, engine=_SubmitSpy(), mode=StepMode.CACHE_ONLY)

    assert sorted(p.name for p in (step_dir / "0").iterdir()) == before
    assert not list(step_dir.glob("*/attempt*"))


def _branch_ctx(
    tmp_path: Path, *, options=None, nms: bool = False, engine: str = "fake"
) -> StepContext:
    """A minimal StepContext for unit-level branch tests (no templates needed)."""
    step_cfg = StepConfig(step=1, engine=engine, operation="opt_sp", options=options or {}, nms=nms)
    return StepContext(
        step_cfg=step_cfg,
        step_dir=tmp_path / "outputs" / "step1",
        template_dir=tmp_path / "templates",
        scratch_dir=None,
        prev_state=PipelineState(structures=()),
        charge=0,
        multiplicity=1,
        max_cores=2,
        slurm_template="cpu.slurm.header",
        executables={},
    )


def _branch_cfg(tmp_path: Path, **step_over) -> Config:
    return Config(
        output_dir=tmp_path / "outputs",
        steps=[StepConfig(step=1, engine="fake", operation="opt_sp", **step_over)],
    )


# --- no-manifest guards (step / nms) ----------------------------------------


def test_rebuild_cache_step_raises_without_manifest(tmp_path: Path):
    from chemrefine import step

    cfg = _branch_cfg(tmp_path)
    with pytest.raises(CacheError, match="cannot rebuild-cache"):
        step.rebuild_cache_step(cfg, cfg.steps[0], PipelineState(structures=()))


def test_resubmit_failed_raises_without_manifest(tmp_path: Path):
    from chemrefine import step
    from chemrefine.engines.api import get_engine

    ctx = _branch_ctx(tmp_path)
    ctx.step_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(CacheError, match="no manifest to rehydrate"):
        step._resubmit_failed(
            get_engine("fake"),
            ctx,
            ctx.step_cfg,
            [FailureRecord("0", FailureKind.MISSING_OUTPUT, "output missing")],
            (),
        )


def test_reattempt_nms_raises_without_manifest(tmp_path: Path):
    from chemrefine import nms
    from chemrefine.engines.api import get_engine

    ctx = _branch_ctx(tmp_path, nms=True, engine="orca")
    ctx.step_dir.mkdir(parents=True, exist_ok=True)  # no manifest written
    with pytest.raises(CacheError, match="no manifest"):
        nms.reattempt_nms(get_engine("orca"), ctx, ctx.step_cfg, None, ())


def test_rebuild_cache_step_nms_branch(tmp_path: Path):
    """rebuild-cache routes an NMS step through nms.rebuild_nms (here: an already-resolved
    round-1, so the survivor passes through at its canonical id)."""
    from synthetic import synthetic_dft_output

    from chemrefine import cache, step
    from chemrefine.ids import structure_artifact_path

    template_dir = tmp_path / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config(
        output_dir=tmp_path / "outputs",
        template_dir=template_dir,
        steps=[
            StepConfig(
                step=1, engine="orca", operation="freq", nms=True, options={"target": "minimum"}
            ),
        ],
    )
    step_cfg = cfg.steps[0]
    step_dir = (cfg.output_dir / step_cfg.dir_name()).resolve()
    out = structure_artifact_path(step_dir, 1, "0", "out")
    inp = structure_artifact_path(step_dir, 1, "0", "inp")
    out.parent.mkdir(parents=True, exist_ok=True)
    # A frequency table with NO imaginary modes ⇒ already at the minimum ⇒ resolved.
    out.write_text(
        synthetic_dft_output([-1.0], [("H", 0, 0, 0), ("H", 0.74, 0, 0)])
        + "VIBRATIONAL FREQUENCIES\n-----------------------\n     6:    100.00 cm**-1\n"
        + "\n****ORCA TERMINATED NORMALLY****\n",
        encoding="utf-8",
    )
    inp.write_text("! Opt Freq\n", encoding="utf-8")
    cache.save_manifest(
        StepInputs(files=((inp, out, "0"),)), step_dir, operation="freq", engine="orca"
    )
    seed = Structure(id="0", atoms=Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]))
    outcome = step.rebuild_cache_step(cfg, step_cfg, PipelineState(structures=(seed,)))
    assert outcome.cache_hit is False
    assert {s.id for s in outcome.state.structures} == {"0"}  # resolved, id kept


# --- _nms_reuse_outcome (NMS reuse-fingerprint path) ------------------------


def _pin_nms(monkeypatch, fp: str = "FP") -> None:
    """Pin the NMS reuse fingerprint + re-attempt result for these tests.

    ``step.py`` calls through the :mod:`chemrefine.cache` and :mod:`chemrefine.nms`
    module objects, so patching the module attributes redirects the orchestrator
    without touching its code.
    """
    from chemrefine import cache, nms

    monkeypatch.setattr(
        cache,
        "reuse_fingerprint",
        lambda step_cfg, parent_ids, *, parents_digest="", template_digest="": fp,
    )
    monkeypatch.setattr(
        nms,
        "reattempt_nms",
        lambda engine, ctx, step_cfg, cached, parent_ids: StepResults(
            structures=(
                Structure(id="re", atoms=Atoms("H", positions=[[0, 0, 0]]), energy_hartree=-1.0),
            )
        ),
    )


def _save_reuse_cache(ctx, reuse_fp: str):
    from chemrefine import cache

    ctx.step_dir.mkdir(parents=True, exist_ok=True)
    cache.save(
        step_cfg=ctx.step_cfg,
        parent_ids=(),
        results=StepResults(
            structures=(Structure(id="c", atoms=Atoms("H", positions=[[0, 0, 0]])),)
        ),
        step_dir=ctx.step_dir,
        chemrefine_version="v",
        reuse_fingerprint=reuse_fp,
    )


def test_nms_reuse_outcome_none_without_cache(tmp_path: Path, monkeypatch):
    from chemrefine import step
    from chemrefine.engines.api import get_engine

    _pin_nms(monkeypatch)
    ctx = _branch_ctx(tmp_path, nms=True, engine="orca")
    ctx.step_dir.mkdir(parents=True, exist_ok=True)
    assert (
        step._nms_reuse_outcome(ctx, ctx.step_cfg, (), get_engine("orca"), may_submit=True) is None
    )


def test_nms_reuse_outcome_none_on_corrupt_cache(tmp_path: Path, monkeypatch):
    from chemrefine import cache, step
    from chemrefine.engines.api import get_engine

    _pin_nms(monkeypatch)
    ctx = _branch_ctx(tmp_path, nms=True, engine="orca")
    cache_path = cache._cache_path(ctx.step_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"not json")
    assert (
        step._nms_reuse_outcome(ctx, ctx.step_cfg, (), get_engine("orca"), may_submit=True) is None
    )


def test_nms_reuse_outcome_restamps_when_all_resolved(tmp_path: Path, monkeypatch):
    from chemrefine import step
    from chemrefine.engines.api import get_engine

    _pin_nms(monkeypatch, "FP")
    ctx = _branch_ctx(tmp_path, nms=True, engine="orca")
    _save_reuse_cache(ctx, "FP")  # matching reuse fingerprint, no failed ledger
    out = step._nms_reuse_outcome(ctx, ctx.step_cfg, (), get_engine("orca"), may_submit=True)
    assert out is not None and out.cache_hit is False


def test_nms_reuse_outcome_reattempts_when_ledger_present(tmp_path: Path, monkeypatch):
    from chemrefine import cache, step
    from chemrefine.engines.api import get_engine

    _pin_nms(monkeypatch, "FP")
    ctx = _branch_ctx(tmp_path, nms=True, engine="orca")
    _save_reuse_cache(ctx, "FP")
    cache.save_failure_records(
        ctx.step_dir, [FailureRecord(structure_id="0", kind=FailureKind.FAILED, reason="x")]
    )
    out = step._nms_reuse_outcome(ctx, ctx.step_cfg, (), get_engine("orca"), may_submit=True)
    assert out is not None and any(s.id == "re" for s in out.state.structures)


def test_nms_reuse_outcome_declines_to_reattempt_when_the_step_may_not_submit(
    tmp_path: Path, monkeypatch
):
    """Re-attempting the unresolved parents runs round-2 jobs, so it needs permission.

    This branch fires whenever the *reuse* fingerprint matches — the ordinary state after
    tuning a search parameter — so it is the likeliest way for a step nobody targeted to
    start computing under `rebuild-cache` or `rerun-errors`.
    """
    from chemrefine import cache, step
    from chemrefine.engines.api import get_engine

    _pin_nms(monkeypatch, "FP")
    ctx = _branch_ctx(tmp_path, nms=True, engine="orca")
    _save_reuse_cache(ctx, "FP")
    cache.save_failure_records(
        ctx.step_dir, [FailureRecord(structure_id="0", kind=FailureKind.FAILED, reason="x")]
    )
    out = step._nms_reuse_outcome(ctx, ctx.step_cfg, (), get_engine("orca"), may_submit=False)
    assert out is None


# ---------------------------------------------------------------------------
# Resuming a step the driver died in the middle of
# ---------------------------------------------------------------------------


def _interrupt_after(cfg: Config, seeds: PipelineState, keep: set[str]) -> Path:
    """Simulate a driver killed mid-step: run it, then keep only ``keep``'s outputs.

    What survives is exactly what survives a real interruption — the manifest (written
    before submission) and the outputs of the jobs that had finished — but no `step.json`,
    because that is written once, at the very end.
    """
    run_step(cfg, cfg.steps[0], seeds)
    step_dir = step_dir_for(cfg, cfg.steps[0])
    # Only `step.json` goes: it is written last, so an interruption is exactly "no results
    # document, but the manifest and whatever outputs had finished are still there".
    cache.invalidate(step_dir)
    for struct in seeds.structures:
        if struct.id not in keep:
            for stale in (step_dir / struct.id).glob("*.out"):
                stale.unlink()
    return step_dir


def test_resume_reparses_a_finished_structure_instead_of_resubmitting_it(tmp_path: Path):
    """An interrupted step must continue, not start over.

    `step.json` is written once, at the end of a step. If the driver dies before that —
    walltime on the batch job that runs ChemRefine itself, a node failure, Ctrl-C — then
    `resume` missed the cache, entered the full-run path, and `attempts.archive_previous`
    moved every finished `.out` into `attemptK/` before resubmitting *everything*. The
    completed compute was still on disk and was never read: `parse_with_failures` decides
    success by `out.is_file()` at the canonical path, which had just been emptied.

    On HPC that is cluster-days of work discarded silently.
    """
    cfg = _config(tmp_path)
    seeds = _seed_state(["0", "1", "2"])
    step_dir = _interrupt_after(cfg, seeds, keep={"0", "1"})

    submitted: list[list[str]] = []
    original_submit = FakeEngine.submit

    def _recording_submit(self, inputs, ctx):
        submitted.append([sid for _i, _o, sid in inputs.files])
        return original_submit(self, inputs, ctx)

    with patch.object(FakeEngine, "submit", _recording_submit):
        outcome = run_step(cfg, cfg.steps[0], seeds, mode=StepMode.RESUME)

    assert [s.id for s in outcome.state.structures] == ["0", "1", "2"]
    # Only the unfinished structure goes back to the scheduler.
    assert submitted == [["2"]], submitted
    # ...and the finished ones keep their original attempt: nothing was archived.
    assert not list((step_dir / "0").glob("attempt*"))


def test_run_still_redoes_everything_even_when_outputs_are_present(tmp_path: Path):
    """`chemrefine run` means "start over", and must keep meaning that.

    The resume path leans on the manifest fingerprint to prove the on-disk outputs belong
    to this configuration. EXECUTE deliberately ignores all of it — otherwise the B1
    invariant would be weakened: a re-executed job that dies before writing anything would
    re-read the previous run's result and report it as current.
    """
    cfg = _config(tmp_path)
    seeds = _seed_state(["0", "1"])
    step_dir = _interrupt_after(cfg, seeds, keep={"0", "1"})

    run_step(cfg, cfg.steps[0], seeds, mode=StepMode.EXECUTE)

    assert list((step_dir / "0").glob("attempt*")), "EXECUTE must archive and re-run"


def test_resume_refuses_a_manifest_from_a_different_config(tmp_path: Path):
    """A manifest whose fingerprint does not match must fall back to the full re-run.

    This is what keeps the optimisation honest: outputs on disk are only reusable if they
    were produced for *this* step config and *these* parents.
    """
    cfg = _config(tmp_path)
    seeds = _seed_state(["0", "1"])
    step_dir = _interrupt_after(cfg, seeds, keep={"0", "1"})

    changed = _config(tmp_path, options={"basis": "other"})
    run_step(changed, changed.steps[0], seeds, mode=StepMode.RESUME)

    assert list((step_dir / "0").glob("attempt*")), "a stale manifest must not be trusted"

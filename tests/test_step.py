"""Tests for the step lifecycle (prepare → submit → wait → parse → filter → cache)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
from ase import Atoms
from fake_engine import FakeEngine

from chemrefine import cache, io
from chemrefine.config import Config, StepConfig
from chemrefine.engines.api import get_engine
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
    ctx = build_context(cfg, cfg.steps[0], PipelineState(), get_engine(cfg.steps[0].engine))
    assert ctx.charge == 2
    assert ctx.multiplicity == 3


def test_build_context_step_override_wins():
    cfg = Config(
        charge=0,
        multiplicity=1,
        steps=[StepConfig(step=1, engine="fake", operation="opt_sp", charge=-1, multiplicity=2)],
    )
    ctx = build_context(cfg, cfg.steps[0], PipelineState(), get_engine(cfg.steps[0].engine))
    assert ctx.charge == -1
    assert ctx.multiplicity == 2


def test_build_context_directory_paths_are_absolute(tmp_path: Path):
    cfg = _config(tmp_path)
    ctx = build_context(cfg, cfg.steps[0], PipelineState(), get_engine(cfg.steps[0].engine))
    assert ctx.step_dir.is_absolute()
    assert ctx.step_dir.name == "step1"


def test_build_context_named_step_dir_includes_label(tmp_path: Path):
    cfg = _config(tmp_path, name="screen")
    ctx = build_context(cfg, cfg.steps[0], PipelineState(), get_engine(cfg.steps[0].engine))
    assert ctx.step_dir.name == "step1_screen"


def test_build_context_threads_slurm_array():
    cfg = Config(
        slurm_array=True,
        steps=[StepConfig(step=1, engine="fake", operation="opt_sp")],
    )
    ctx = build_context(cfg, cfg.steps[0], PipelineState(), get_engine(cfg.steps[0].engine))
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


FAIL_ENGINE_DRIFT = 0.25
"""How far the fail engine's ``"unconverged"`` parse moves each coordinate off the seed.

The geometry a failed run *reached* must be distinguishable from the geometry it was
*given*, or a test of ``on_failure: best`` cannot tell "backfilled the best obtained"
from "backfilled the submitted input" — the preference `apply_failure_policy` exists
to make."""


def _register_fail_engine():
    """Register a fake engine whose ``fail`` ClassVar marks per-sid failures.

    ``fail[sid] == "missing"`` produces no output; ``"unconverged"`` produces an
    output that parses but with ``terminated_normally=False`` and a geometry moved
    :data:`FAIL_ENGINE_DRIFT` off the seed (the point the failed run reached); anything
    else succeeds.
    """
    from typing import ClassVar

    from chemrefine.engines.api import register
    from chemrefine.ids import structure_artifact_path
    from chemrefine.state import JobBatch, StepInputs, Structure

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
                atoms = seed.atoms
                if self.fail.get(sid) == "unconverged":
                    # The geometry the failed run reached — off the seed by a fixed
                    # drift, so a test can tell the best-obtained backfill from the
                    # submitted input.
                    atoms = seed.atoms.copy()
                    atoms.set_positions(atoms.get_positions() + FAIL_ENGINE_DRIFT)
                out.append(
                    Structure(
                        id=sid,
                        atoms=atoms,
                        parent_id=seed.parent_id,
                        energy_hartree=-1.0 - int(sid) * 1e-3,
                        terminated_normally=self.fail.get(sid) != "unconverged",
                        converged=True,
                    )
                )
            return StepResults(structures=tuple(out))

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
            ("1", FailureKind.NOT_TERMINATED_NORMALLY)
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


def test_on_failure_best_backfills_the_best_geometry_not_the_seed(tmp_path: Path):
    """``best`` carries the geometry the failed run *reached*, not the input it was given.

    ``apply_failure_policy``'s preference — ``f.best if f.best is not None else`` the
    submitted seed — is the whole difference between ``best`` and a re-labelled ``skip``
    for a structure that produced anything at all. Both arms live on one line, which
    branch coverage cannot see, and every other exerciser asserts ids or counts — so
    dropping the preference (always backfilling the seed) survived the suite. The
    positions are the only witness, and the fail engine's ``"unconverged"`` parse moves
    them :data:`FAIL_ENGINE_DRIFT` off the seed precisely so this can fail.
    """
    from chemrefine.engines.api import ENGINES

    seeds = PipelineState(
        structures=(Structure(id="1", atoms=Atoms("H", positions=[[0.0, 0.0, 0.0]])),)
    )
    eng = _register_fail_engine()
    try:
        eng.fail = {"1": "unconverged"}
        cfg = _config(tmp_path, engine="fake-fail", on_failure="best")
        outcome = run_step(cfg, cfg.steps[0], seeds)
        [survivor] = outcome.state.structures
        assert survivor.id == "1"
        assert survivor.atoms.get_positions()[0] == pytest.approx(
            [FAIL_ENGINE_DRIFT, FAIL_ENGINE_DRIFT, FAIL_ENGINE_DRIFT]
        ), "the backfill must be the best geometry obtained, not the submitted seed"
        # Still ledgered — best keeps going without hiding the failure.
        step_dir = cfg.output_dir.resolve() / "step1"
        assert [f.structure_id for f in cache.load_failure_records(step_dir)] == ["1"]
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
# Changing on_failure over a cached step
# ---------------------------------------------------------------------------
#
# The cache stores post-policy results while the fingerprint deliberately excludes
# `on_failure`, so a policy edit alone never misses. `stop` and `skip` both store the
# successes alone (storage-equivalent → a free hit); `best` stores backfills too, so a
# change across that line re-attempts the ledgered failures and re-finalizes under the
# current policy — never recomputing a success. Without the `_policy_conflict` check,
# stop→best silently served skip semantics.


def _run_policy_change(tmp_path: Path, first: str, then: str, *, mode=StepMode.RESUME):
    """Run a fail-one step under ``first``, then run it again under ``then``.

    Structure ``1`` leaves no output on both runs, so the second run's outcome is purely
    the policy machinery's answer. Returns ``(second_outcome, step_dir)``.
    """
    from chemrefine.engines.api import ENGINES

    seeds = _seed_state(["0", "1", "2"])
    eng = _register_fail_engine()
    try:
        eng.fail = {"1": "missing"}
        first_cfg = _config(tmp_path, engine="fake-fail", on_failure=first)
        run_step(first_cfg, first_cfg.steps[0], seeds)
        then_cfg = _config(tmp_path, engine="fake-fail", on_failure=then)
        outcome = run_step(then_cfg, then_cfg.steps[0], seeds, mode=mode)
        return outcome, then_cfg.output_dir.resolve() / "step1"
    finally:
        eng.fail = {}
        ENGINES.pop("fake-fail", None)


def test_policy_change_stop_to_best_backfills_instead_of_serving_skip(tmp_path: Path):
    """The audit's F1 reproduction: stop→best must yield the backfilled set, not {0, 2}."""
    outcome, step_dir = _run_policy_change(tmp_path, "stop", "best")
    assert outcome.cache_hit is False
    assert {s.id for s in outcome.state.structures} == {"0", "1", "2"}
    cached = cache.load(step_dir)
    assert cached is not None
    assert cached.on_failure == "best"
    assert {s.id for s in cached.results.structures} == {"0", "1", "2"}
    # The re-attempted failure stays visible in the ledger, as under a fresh `best` run.
    assert {f.structure_id for f in cache.load_failure_records(step_dir)} == {"1"}


def test_policy_change_best_to_skip_drops_the_backfill(tmp_path: Path):
    outcome, step_dir = _run_policy_change(tmp_path, "best", "skip")
    assert outcome.cache_hit is False
    assert {s.id for s in outcome.state.structures} == {"0", "2"}
    cached = cache.load(step_dir)
    assert cached is not None
    assert {s.id for s in cached.results.structures} == {"0", "2"}


def test_policy_change_best_to_stop_leaves_the_failure_pending(tmp_path: Path):
    outcome, _step_dir = _run_policy_change(tmp_path, "best", "stop")
    assert {s.id for s in outcome.state.structures} == {"0", "2"}
    cfg = _config(tmp_path, engine="fake-fail", on_failure="stop")
    with pytest.raises(ChemRefineError):
        halt_if_pending(cfg, cfg.steps[0], StepMode.RESUME)


def test_policy_change_stop_to_skip_is_a_free_hit(tmp_path: Path):
    """Both policies store the successes alone, so nothing needs re-attempting."""
    outcome, _step_dir = _run_policy_change(tmp_path, "stop", "skip")
    assert outcome.cache_hit is True
    assert {s.id for s in outcome.state.structures} == {"0", "2"}


def test_policy_change_skip_to_stop_reattempts_the_pending_failures(tmp_path: Path):
    """Same storage class, but `stop` makes the ledger pending — the existing branch."""
    outcome, step_dir = _run_policy_change(tmp_path, "skip", "stop")
    assert outcome.cache_hit is False
    assert {s.id for s in outcome.state.structures} == {"0", "2"}
    assert {f.structure_id for f in cache.load_failure_records(step_dir)} == {"1"}


def test_policy_change_with_a_clean_ledger_is_a_free_hit(tmp_path: Path):
    """With no failures every policy produces identical results, so any edit hits."""
    from chemrefine.engines.api import ENGINES

    seeds = _seed_state(["0", "1"])
    _register_fail_engine()
    try:
        cfg = _config(tmp_path, engine="fake-fail", on_failure="stop")
        run_step(cfg, cfg.steps[0], seeds)
        cfg2 = _config(tmp_path, engine="fake-fail", on_failure="best")
        outcome = run_step(cfg2, cfg2.steps[0], seeds, mode=StepMode.RESUME)
        assert outcome.cache_hit is True
        assert {s.id for s in outcome.state.structures} == {"0", "1"}
    finally:
        ENGINES.pop("fake-fail", None)


def test_policy_change_under_cache_only_raises_instead_of_serving_it(tmp_path: Path):
    """A mode that may not submit cannot repair the shape mismatch — it must say so."""
    with pytest.raises(ChemRefineError, match="no cache this configuration can use"):
        _run_policy_change(tmp_path, "stop", "best", mode=StepMode.CACHE_ONLY)


def test_policy_change_over_a_legacy_cache_stays_a_hit(tmp_path: Path):
    """A document written before `on_failure` was recorded serves any policy."""
    outcome, step_dir = _run_policy_change(tmp_path, "stop", "stop")
    document = json.loads((step_dir / "_cache" / "step.json").read_text(encoding="utf-8"))
    document.pop("on_failure")
    (step_dir / "_cache" / "step.json").write_text(json.dumps(document), encoding="utf-8")
    from chemrefine.engines.api import ENGINES

    eng = _register_fail_engine()
    try:
        eng.fail = {"1": "missing"}
        cfg = _config(tmp_path, engine="fake-fail", on_failure="best")
        outcome = run_step(cfg, cfg.steps[0], _seed_state(["0", "1", "2"]), mode=StepMode.RESUME)
        assert outcome.cache_hit is True
        assert {s.id for s in outcome.state.structures} == {"0", "2"}
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
    from chemrefine.state import JobBatch, StepInputs, Structure

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
    ctx = build_context(cfg, cfg.steps[0], _seed_state([]), get_engine(cfg.steps[0].engine))
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
    ctx = build_context(cfg, cfg.steps[0], _seed_state([]), get_engine(cfg.steps[0].engine))
    with pytest.raises(ConfigError, match="drop `nms: true`"):
        _check_nms_freq_gate(_NoFreqEngine(), ctx, cfg.steps[0])


def test_invalid_nms_options_are_refused_when_the_key_is_derived(tmp_path: Path):
    """A bad NMS knob fails as a ConfigError naming the step, not a bare ValidationError.

    ``derive_step_key`` is the first reader of the validated NMS options on every route —
    before any cache is consulted and before anything submits — so this is where a
    ``displacement_value`` that is not a number must become the documented exit code
    rather than a pydantic traceback three layers from the YAML that caused it.
    """
    from chemrefine.errors import ConfigError
    from chemrefine.step import derive_step_key

    cfg = _config(tmp_path, engine="orca", nms=True, options={"displacement_value": "banana"})
    ctx = build_context(cfg, cfg.steps[0], _seed_state([]), get_engine(cfg.steps[0].engine))
    with pytest.raises(ConfigError, match="invalid NMS options"):
        derive_step_key(ctx, cfg.steps[0], get_engine(cfg.steps[0].engine))


def test_run_step_nms_branch_routes_through_coordinator(tmp_path: Path, monkeypatch):
    """run_step routes an `nms: true` step through the generic coordinator (nms.run_nms),
    then applies the on_failure policy to its survivors/failures."""
    from chemrefine import nms as nms_mod
    from chemrefine.engines.api import ENGINES, NmsInputInfo, register
    from chemrefine.nms import NmsResolution
    from chemrefine.state import JobBatch, StepInputs

    calls: list[int] = []

    def _fake_run_nms(engine, round1, failures, ctx, *, round2=None):
        calls.append(1)
        # The step must hand over the round it already ran, not leave the coordinator to
        # fan out a second time — that is what keeps round 2 in round 1's queue.
        assert round2 is not None, "the step ran the fan-out; it must pass it on"
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
    ctx = build_context(cfg, cfg.steps[0], _seed_state(["0"]), get_engine(cfg.steps[0].engine))
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


@pytest.mark.parametrize(
    "predicate",
    [StepMode.may_submit, StepMode.runs_through_run_step, StepMode.can_halt],
    ids=["may_submit", "runs_through_run_step", "can_halt"],
)
def test_a_value_outside_the_mode_enum_fails_loud_in_every_predicate(predicate):
    """The predicates' wildcard arm holds only `assert_never` — the guard that keeps the
    match exhaustive for mypy, so a fifth mode fails type-checking instead of inheriting
    the permissive answer. Anything that reaches the arm at runtime fails loud, like
    `Throttler.assign_device`'s unreachable guard."""
    with pytest.raises(AssertionError):
        predicate(cast(StepMode, "bogus"))


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
        template=None,
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
    with pytest.raises(CacheError, match="no manifest on disk"):
        step.rebuild_cache_step(cfg, cfg.steps[0], PipelineState(structures=()))


def test_rebuild_cache_step_refuses_outputs_from_another_configuration(tmp_path: Path):
    """Re-parsing must not attribute results to a configuration that never produced them.

    The cache it would write is internally valid, so the next `resume` serves it instead
    of computing what was asked for — the edited step silently never runs.
    """
    from chemrefine import step

    cfg_a = _config(tmp_path, charge=0)
    seeds = _seed_state(["0"])
    run_step(cfg_a, cfg_a.steps[0], seeds, mode=StepMode.EXECUTE)

    cfg_b = _config(tmp_path, charge=-1)  # same outputs on disk, different configuration
    with pytest.raises(CacheError, match="produced for a different configuration"):
        step.rebuild_cache_step(cfg_b, cfg_b.steps[0], seeds)


def test_rebuild_adopts_an_unprovenanced_tree(tmp_path: Path):
    """A manifest without row provenance is unprovable, not wrong — the explicit command adopts.

    That is every pre-provenance tree (and the documented v1 adoption's hand-written
    manifests): its bare stamp was written under rules that no longer exist, so nothing
    can prove it right — and nothing proves it wrong. `rebuild-cache` re-parses under the
    current rules and writes the manifest back *with* provenance, so the adoption is
    recorded and the next question is answered per row.
    """
    import json

    from chemrefine import step

    cfg = _config(tmp_path)
    seeds = _seed_state(["0", "1"])
    run_step(cfg, cfg.steps[0], seeds, mode=StepMode.EXECUTE)
    step_dir = step_dir_for(cfg, cfg.steps[0])
    cache.invalidate(step_dir)

    # Strip the manifest to the pre-provenance shape: a bare, alien stamp and no rows.
    manifest_file = cache.manifest_path(step_dir)
    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    data["fingerprint"] = "feedfacefeedface"
    for row in data["files"]:
        row.pop("row_key", None)
        row.pop("parent_digest", None)
    manifest_file.write_text(json.dumps(data), encoding="utf-8")

    outcome = step.rebuild_cache_step(cfg, cfg.steps[0], seeds)
    assert {s.id for s in outcome.state.structures} == {"0", "1"}
    provenance = cache.load_manifest_provenance(step_dir)
    assert set(provenance.rows) == {"0", "1"}, "the adoption must be recorded per row"


def test_rebuild_cache_step_accepts_outputs_from_this_configuration(tmp_path: Path):
    """The ordinary case — re-parsing after a parser change — still works."""
    from chemrefine import step

    cfg = _config(tmp_path)
    seeds = _seed_state(["0", "1"])
    run_step(cfg, cfg.steps[0], seeds, mode=StepMode.EXECUTE)
    cache.invalidate(step_dir_for(cfg, cfg.steps[0]))

    outcome = step.rebuild_cache_step(cfg, cfg.steps[0], seeds)

    assert [s.id for s in outcome.state.structures] == ["0", "1"]


def test_resubmit_failed_raises_without_manifest(tmp_path: Path):
    from chemrefine import step
    from chemrefine.engines.api import get_engine

    ctx = _branch_ctx(tmp_path)
    ctx.step_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(CacheError, match="no manifest to rehydrate"):
        step._resubmit_failed(get_engine("fake"), ctx, ctx.step_cfg, ())


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


def _nms_rebuild_tree(tmp_path: Path, **options) -> tuple[Config, Path, cache.StepKey]:
    """The `test_rebuild_cache_step_nms_branch` tree, plus the step's derived key."""
    from synthetic import synthetic_dft_output

    from chemrefine import cache, step
    from chemrefine.engines.api import get_engine
    from chemrefine.ids import structure_artifact_path

    template_dir = tmp_path / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config(
        output_dir=tmp_path / "outputs",
        template_dir=template_dir,
        steps=[
            StepConfig(
                step=1,
                engine="orca",
                operation="freq",
                nms=True,
                options={"target": "minimum", **options},
            ),
        ],
    )
    step_cfg = cfg.steps[0]
    step_dir = (cfg.output_dir / step_cfg.dir_name()).resolve()
    out = structure_artifact_path(step_dir, 1, "0", "out")
    inp = structure_artifact_path(step_dir, 1, "0", "inp")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        synthetic_dft_output([-1.0], [("H", 0, 0, 0), ("H", 0.74, 0, 0)])
        + "VIBRATIONAL FREQUENCIES\n-----------------------\n     6:    100.00 cm**-1\n"
        + "\n****ORCA TERMINATED NORMALLY****\n",
        encoding="utf-8",
    )
    inp.write_text("! Opt Freq\n", encoding="utf-8")
    (template_dir / "step1.inp").write_text("! Opt Freq\n", encoding="utf-8")
    engine = get_engine("orca")
    seed = Structure(id="0", atoms=Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]))
    ctx = step.build_context(cfg, step_cfg, PipelineState(structures=(seed,)), engine)
    key = step.derive_step_key(ctx, step_cfg, engine)
    cache.save_manifest(
        StepInputs(files=((inp, out, "0"),)),
        step_dir,
        operation="freq",
        engine="orca",
        **key.manifest_stamp(),
    )
    return cfg, step_dir, key


def test_rebuild_refuses_an_attempt_from_another_search_specification(tmp_path: Path):
    """A search retune must not adopt: same child ids, different displaced geometries.

    Row keys exclude the NMS resolution by design, so the row check cannot see a retune —
    and the re-derived children's ids coincide with the old attempt's, so a rebuild would
    parse outputs answering displacements this configuration never asked for and cache
    them under the new resolution's fingerprint. Resume already refuses to reuse such an
    attempt; the explicit command must not adopt what resume refuses.
    """
    from chemrefine import step

    cfg, _step_dir, _key = _nms_rebuild_tree(tmp_path, displacement_value=1.0)
    retuned = Config(
        output_dir=cfg.output_dir,
        template_dir=cfg.template_dir,
        steps=[
            StepConfig(
                step=1,
                engine="orca",
                operation="freq",
                nms=True,
                options={"target": "minimum", "displacement_value": 0.25},
            ),
        ],
    )
    seed = Structure(id="0", atoms=Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]))
    with pytest.raises(CacheError, match="different search settings"):
        step.rebuild_cache_step(retuned, retuned.steps[0], PipelineState(structures=(seed,)))


def test_rebuild_still_adopts_across_a_criterion_retune(tmp_path: Path):
    """Re-reading an attempt under a new *target* is the very thing rebuild-nms offers.

    A criterion retune never moves a child's geometry — the displacement depends on the
    search knobs and the mode, not on what counts as resolved — so the attempt on disk
    stays a sound answer and the adoption must keep working. Only the search half refuses.
    """
    from chemrefine import step

    cfg, _step_dir, _key = _nms_rebuild_tree(tmp_path)
    retuned = Config(
        output_dir=cfg.output_dir,
        template_dir=cfg.template_dir,
        steps=[
            StepConfig(
                step=1,
                engine="orca",
                operation="freq",
                nms=True,
                options={"target": "ts", "ts_mode_index": 6},
            ),
        ],
    )
    seed = Structure(id="0", atoms=Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]))
    outcome = step.rebuild_cache_step(retuned, retuned.steps[0], PipelineState(structures=(seed,)))
    # Under `ts` the structure (zero imaginary modes) is no longer at its target and its
    # only children would need submitting — rebuild submits nothing, so it lands in the
    # ledger as unresolved rather than being refused outright: the adoption path ran.
    assert outcome.cache_hit is False


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
    walltime on the batch job that runs ChemRefine itself, a node failure, Ctrl-C — there is
    no cache to hit. Without this path `resume` enters the full-run path, and
    `attempts.archive_previous` moves every finished `.out` into `attemptK/` before
    resubmitting *everything*: the completed compute is still on disk and never read, because
    `parse_with_failures` decides success by `out.is_file()` at the canonical path, which
    archiving has just emptied.

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


def test_resume_recomputes_rows_whose_key_the_config_change_moved(tmp_path: Path):
    """A config edit that reaches jobs re-runs their rows — stale outputs archived, never read.

    This is what keeps the optimisation honest: a row is adopted only under proof its
    output is the one this configuration would compute, and a changed effective charge
    breaks that proof for every row.
    """
    cfg = _config(tmp_path)
    seeds = _seed_state(["0", "1"])
    step_dir = _interrupt_after(cfg, seeds, keep={"0", "1"})

    # The change must be one that provably reaches jobs — the effective charge. (An
    # *undeclared* option like a stray `basis:` now moves no key by design: nothing
    # reads it, so nothing it could invalidate exists.)
    changed = _config(tmp_path, charge=-1)
    run_step(changed, changed.steps[0], seeds, mode=StepMode.RESUME)

    assert list((step_dir / "0").glob("attempt*")), "a stale manifest must not be trusted"


class _CountingEngine(FakeEngine):
    """The fake engine, with a record of which structures each submit carried."""

    def __init__(self) -> None:
        self.submitted: list[list[str]] = []

    def submit(self, inputs, ctx):
        """Record the batch's structure ids, then run the fake calculation."""
        self.submitted.append([sid for _inp, _out, sid in inputs.files])
        return super().submit(inputs, ctx)


def test_a_partially_changed_parent_set_computes_exactly_the_changed_rows(tmp_path: Path):
    """{A, B, C} → {A, B, C'} runs C' alone; A and B are adopted from disk.

    The row keys are the proof: A's and B's stored keys match the ones this
    configuration derives, so their outputs are re-parsed; C's parent content changed,
    so its row is condemned sight-unseen and recomputed — and nothing else is.
    """
    cfg = _config(tmp_path)
    seeds = _seed_state(["0", "1", "2"])
    run_step(cfg, cfg.steps[0], seeds, mode=StepMode.EXECUTE)
    cache.invalidate(step_dir_for(cfg, cfg.steps[0]))

    moved = Atoms("H")
    moved.positions[0] = (0.0, 0.0, 0.5)
    changed = PipelineState(
        structures=(
            seeds.structures[0],
            seeds.structures[1],
            Structure(id="2", atoms=moved),
        )
    )
    engine = _CountingEngine()
    outcome = run_step(cfg, cfg.steps[0], changed, mode=StepMode.RESUME, engine=engine)

    assert engine.submitted == [["2"]], "exactly the changed row computes"
    assert [s.id for s in outcome.state.structures] == ["0", "1", "2"]


def _partially_changed(tmp_path: Path) -> tuple[Config, PipelineState, Path]:
    """The {A, B, C'} shape: a finished step, its cache gone, parent 2's geometry moved."""
    cfg = _config(tmp_path)
    seeds = _seed_state(["0", "1", "2"])
    run_step(cfg, cfg.steps[0], seeds, mode=StepMode.EXECUTE)
    step_dir = step_dir_for(cfg, cfg.steps[0])
    cache.invalidate(step_dir)
    moved = Atoms("H")
    moved.positions[0] = (0.0, 0.0, 0.5)
    changed = PipelineState(
        structures=(
            seeds.structures[0],
            seeds.structures[1],
            Structure(id="2", atoms=moved),
        )
    )
    return cfg, changed, step_dir


def test_the_manifest_is_stamped_only_after_condemned_outputs_are_archived(tmp_path: Path):
    """The ordering that closes the stamp-then-crash window, pinned at the stamp itself.

    Stamped first, a driver killed during the resubmission pass left the new row
    provenance vouching for the condemned row's old output still at canonical — and the
    next resume computed `changed = {}` and adopted it: a parse-usable answer to the
    *previous* parent's geometry, served as the new parent's result. So at the moment
    `save_manifest` runs on the incremental path, the condemned output must already be
    in an attempt directory, where a crash re-reads as MISSING_OUTPUT instead.
    """
    cfg, changed, step_dir = _partially_changed(tmp_path)
    stale_out = step_dir / "2" / "step1_2.out"
    assert stale_out.is_file(), "precondition: the old output sits at canonical"

    at_stamp: dict[str, bool] = {}
    real = cache.save_manifest

    def _spying(inputs, sd, **kwargs):
        if kwargs.get("rows"):  # the incremental stamp — finalize()'s cache write has none
            at_stamp["canonical"] = stale_out.is_file()
            at_stamp["archived"] = any((step_dir / "2").glob("attempt*/step1_2.out"))
        return real(inputs, sd, **kwargs)

    with patch.object(cache, "save_manifest", _spying):
        run_step(cfg, cfg.steps[0], changed, mode=StepMode.RESUME)

    assert at_stamp == {"canonical": False, "archived": True}


def test_a_crash_after_the_stamp_still_recomputes_the_condemned_row(tmp_path: Path):
    """The window itself, end to end: stamp → die → resume must resubmit, never adopt.

    The first resume is killed right after the manifest carries the new row keys (the
    resubmission pass never runs — a walltime kill lands exactly there on a large step,
    since the pass opens with a parse of every output). The second resume then faces a
    manifest whose rows all match its own; the proof that the condemned row was archived
    rather than left for adoption is that it goes back to the scheduler.
    """
    cfg, changed, _step_dir = _partially_changed(tmp_path)

    with (
        patch("chemrefine.lifecycle.resubmit_unusable", side_effect=SystemExit(143)),
        pytest.raises(SystemExit),
    ):
        run_step(cfg, cfg.steps[0], changed, mode=StepMode.RESUME)

    engine = _CountingEngine()
    outcome = run_step(cfg, cfg.steps[0], changed, mode=StepMode.RESUME, engine=engine)

    assert engine.submitted == [["2"]], "the stale output must not be adopted"
    assert [s.id for s in outcome.state.structures] == ["0", "1", "2"]


def test_a_key_nothing_reads_resumes_without_computing_anything(tmp_path: Path):
    """An undeclared option cannot reach a job, so every row is adopted — zero submissions.

    The row key holds options only as the engine's declared model reads them; a stray
    YAML key reaches no engine and no template, and re-running 1000 finished jobs over
    it would be pure waste. validate already warns about the key itself.
    """
    cfg = _config(tmp_path)
    seeds = _seed_state(["0", "1"])
    run_step(cfg, cfg.steps[0], seeds, mode=StepMode.EXECUTE)
    cache.invalidate(step_dir_for(cfg, cfg.steps[0]))

    edited = _config(tmp_path, options={"stray_knob": 7})
    engine = _CountingEngine()
    outcome = run_step(edited, edited.steps[0], seeds, mode=StepMode.RESUME, engine=engine)

    assert engine.submitted == [], "nothing may run — no job could read the edit"
    assert [s.id for s in outcome.state.structures] == ["0", "1"]


def test_resume_refuses_an_unprovenanced_manifest_by_name(tmp_path: Path):
    """A pre-provenance tree gets the fix spelled out, not a silent archive-and-recompute.

    The bare stamp was written under rules that no longer exist: resume can neither
    prove the outputs right (adopt) nor wrong (recompute honestly says so) — and the
    one command that may adopt the unprovable, because the user explicitly asks, is
    `rebuild-cache`. Refusing loudly is what turns the old footgun — days of finished
    compute silently swept into attemptK/ — into a one-command recovery.
    """
    import json

    cfg = _config(tmp_path)
    seeds = _seed_state(["0", "1"])
    run_step(cfg, cfg.steps[0], seeds, mode=StepMode.EXECUTE)
    step_dir = step_dir_for(cfg, cfg.steps[0])
    cache.invalidate(step_dir)

    manifest_file = cache.manifest_path(step_dir)
    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    data["fingerprint"] = "feedfacefeedface"
    for row in data["files"]:
        row.pop("row_key", None)
        row.pop("parent_digest", None)
    manifest_file.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(CacheError, match="rebuild-cache"):
        run_step(cfg, cfg.steps[0], seeds, mode=StepMode.RESUME)


def test_a_step_derives_its_cache_key_exactly_once(tmp_path: Path, monkeypatch):
    """The key is a value computed once, not a recipe each route re-follows.

    Derived per route instead, a cold step hashes its parents and its template once for
    each of `_cached_outcome`, the incremental route, the manifest stamp and the write.
    That is wasted work on a 10^4-structure step, but the reason it matters is drift: a
    route deriving its own key can disagree with the one the cache was written under, and
    one that reads `ctx.prev_state` gets a *subset* of the parents on the retry paths.
    """
    from chemrefine import cache

    calls: dict[str, int] = {"structure_digest": 0, "template_digest": 0, "row_key": 0}

    def counting(name):
        original = getattr(cache, name)

        def wrapped(*args, **kwargs):
            calls[name] += 1
            return original(*args, **kwargs)

        return wrapped

    for name in calls:
        monkeypatch.setattr(cache, name, counting(name))

    cfg = _config(tmp_path)
    state = _seed_state(["0", "1"])
    run_step(cfg, cfg.steps[0], state, mode=StepMode.RESUME)  # cold: nothing on disk
    # One derivation: each parent digested once, the template once, one row key per parent.
    assert calls == {"structure_digest": 2, "template_digest": 1, "row_key": 2}


# ---------------------------------------------------------------------------
# A streaming engine takes the streaming scheduler through the whole step
# ---------------------------------------------------------------------------


def _register_streaming_conv_engine():
    """A `conv-retry`-shaped engine that also satisfies `StreamingSubmit`.

    Every other fake in this file is a bare class, so they all take the batched route — good
    for regression safety, and exactly why the streaming route needs a fake of its own or it
    gets no step-level coverage at all. It reports completions in reverse to make the
    scheduler's ordering visible in the results if it ever leaked through.
    """
    from typing import ClassVar

    import numpy as np

    from chemrefine.engines.api import register
    from chemrefine.ids import structure_artifact_path
    from chemrefine.state import JobBatch, StepInputs, Structure

    @register("conv-stream")
    class _StreamEngine:
        name = "conv-stream"
        never: ClassVar[set[str]] = set()
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

        def submit_streaming(self, inputs, ctx, sink):
            pending = list(inputs.files)
            while pending:
                nxt = []
                for job in reversed(pending):
                    nxt.extend(sink.on_complete(job))
                pending = nxt
            return JobBatch(jobs={})

        def parse(self, inputs, ctx):
            seeds = {s.id: s for s in ctx.prev_state.structures}
            out = []
            for _inp, _o, sid in inputs.files:
                _StreamEngine.parse_count[sid] = _StreamEngine.parse_count.get(sid, 0) + 1
                converged = sid not in _StreamEngine.never and _StreamEngine.parse_count[sid] >= 2
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

    return _StreamEngine


def test_run_step_streams_retries_and_matches_the_batched_outcome(tmp_path: Path):
    """The whole step, driven by the streaming scheduler: same cache, same ledger.

    Two structures fail to converge and both are retried from their best geometry. What this
    adds over the lifecycle tests is everything around them — the manifest, the survivors that
    reach the cache, and the ledger being cleared once nothing is pending.
    """
    from chemrefine.engines.api import ENGINES

    eng = _register_streaming_conv_engine()
    try:
        eng.never = set()
        eng.parse_count = {}
        cfg = _config(tmp_path, engine="conv-stream", on_failure="stop")
        outcome = run_step(cfg, cfg.steps[0], _seed_state(["0", "1", "2"]))

        step_dir = cfg.output_dir.resolve() / "step1"
        assert [s.id for s in outcome.state.structures] == ["0", "1", "2"], "manifest order"
        assert all((step_dir / sid / "attempt1").is_dir() for sid in ("0", "1", "2"))
        assert cache.load_failure_records(step_dir) == []
        assert cache.load(step_dir) is not None
    finally:
        eng.never = set()
        eng.parse_count = {}
        ENGINES.pop("conv-stream", None)


def test_run_step_streaming_ledgers_a_structure_that_never_converges(tmp_path: Path):
    """One structure fails twice, the others succeed — and the step reports exactly that."""
    from chemrefine.engines.api import ENGINES

    eng = _register_streaming_conv_engine()
    try:
        eng.never = {"1"}
        eng.parse_count = {}
        cfg = _config(tmp_path, engine="conv-stream", on_failure="skip")
        outcome = run_step(cfg, cfg.steps[0], _seed_state(["0", "1", "2"]))

        step_dir = cfg.output_dir.resolve() / "step1"
        assert [s.id for s in outcome.state.structures] == ["0", "2"]
        assert [r.structure_id for r in cache.load_failure_records(step_dir)] == ["1"]
        # Retried once, not repeatedly, even though the queue could feed itself.
        assert not (step_dir / "1" / "attempt2").exists()
    finally:
        eng.never = set()
        eng.parse_count = {}
        ENGINES.pop("conv-stream", None)


def _register_interruptible_engine():
    """A streaming engine whose run can be cut short after a retry has been prepared.

    Unlike `_register_streaming_conv_engine`, `prepare` writes only the *input*: the output
    appears when a job is run. That is what makes the interrupted state real — a retry that
    was prepared but never ran leaves no output at the canonical path, because round 1's was
    archived into `attempt1/` when the retry was prepared.
    """
    from collections import deque
    from typing import ClassVar

    import numpy as np

    from chemrefine.engines.api import register
    from chemrefine.ids import structure_artifact_path
    from chemrefine.state import JobBatch, StepInputs, Structure

    @register("interrupt-stream")
    class _InterruptEngine:
        name = "interrupt-stream"
        runs: ClassVar[dict[str, int]] = {}
        interrupt_before_running: ClassVar[set[str]] = set()

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

        def _run(self, job):
            _inp, out, sid = job
            _InterruptEngine.runs[sid] = _InterruptEngine.runs.get(sid, 0) + 1
            out.write_text("out\n", encoding="utf-8")

        def submit(self, inputs, ctx):
            for job in inputs.files:
                self._run(job)
            return JobBatch(jobs={})

        def submit_streaming(self, inputs, ctx, sink):
            pending = deque(inputs.files)
            while pending:
                job = pending.popleft()
                self._run(job)
                for follow in sink.on_complete(job):
                    if follow[2] in _InterruptEngine.interrupt_before_running:
                        raise KeyboardInterrupt("walltime kill")
                    pending.append(follow)
            return JobBatch(jobs={})

        def parse(self, inputs, ctx):
            seeds = {s.id: s for s in ctx.prev_state.structures}
            out = []
            for _inp, out_path, sid in inputs.files:
                seed = seeds[sid]
                text = out_path.read_text(encoding="utf-8")
                out.append(
                    Structure(
                        id=sid,
                        atoms=seed.atoms,
                        parent_id=seed.parent_id,
                        energy_hartree=-1.0 - int(sid) * 1e-3,
                        forces_ev_per_a=np.zeros((len(seed.atoms), 3)),
                        terminated_normally=text != "partial\n",
                        converged=_InterruptEngine.runs.get(sid, 0) >= 2,
                    )
                )
            return StepResults(structures=tuple(out))

    return _InterruptEngine


def test_a_run_killed_between_a_retrys_archive_and_its_output_recovers_on_resume(tmp_path: Path):
    """Streaming widens an existing window; this pins that resume still closes it.

    Preparing a retry archives round 1's output into ``attempt1/`` and writes a fresh input at
    the canonical path, so between those two moments the structure has *no* output where the
    manifest says one should be. Streaming does not create that window — ``rerun_from_best``
    has always had it — but it moves it from the tail of the step to almost all of it, because
    retries now start as soon as their own job frees a slot.

    Resume must therefore treat "manifest entry with no output" as work to redo, not as a
    verdict. It does: ``_partial_step_outcome`` files it ``MISSING_OUTPUT`` and
    ``_resubmit_failed`` re-runs it, because only ``NOT_CONVERGED`` is held back for the
    retry pass.
    """
    from chemrefine.engines.api import ENGINES

    eng = _register_interruptible_engine()
    try:
        eng.runs = {}
        eng.interrupt_before_running = {"1"}
        cfg = _config(tmp_path, engine="interrupt-stream", on_failure="stop")
        seeds = _seed_state(["0", "1", "2"])

        with pytest.raises(KeyboardInterrupt):
            run_step(cfg, cfg.steps[0], seeds)

        step_dir = cfg.output_dir.resolve() / "step1"
        assert (step_dir / "1" / "attempt1").is_dir(), "round 1 was archived"
        assert not (step_dir / "1" / "step1_1.out").exists(), "the retry never ran"
        assert cache.load(step_dir) is None, "the step did not finish"

        eng.interrupt_before_running = set()
        outcome = run_step(cfg, cfg.steps[0], seeds)

        assert [s.id for s in outcome.state.structures] == ["0", "1", "2"]
        assert cache.load_failure_records(step_dir) == []
    finally:
        eng.runs = {}
        eng.interrupt_before_running = set()
        ENGINES.pop("interrupt-stream", None)


def test_a_retry_whose_output_was_truncated_is_rerun_rather_than_ledgered(tmp_path: Path):
    """The other half of the same window: the job was killed *after* it wrote something.

    ``_on_exit`` copies back on SIGTERM/INT, so a scancelled retry leaves a partial output at
    the canonical path while its good round-1 geometry sits in ``attempt1/``. The file exists,
    so a resume that asked ``out.is_file()`` would call the structure finished, re-parse it as
    ``NOT_TERMINATED_NORMALLY`` — which ``retryable_best`` refuses, since a crashed job would
    crash again — and hand it to the failure policy with an attempt it never got to use.

    ``resubmit_unusable`` judges by what *parsed* instead, so the structure is re-run.
    """
    from chemrefine.engines.api import ENGINES

    eng = _register_interruptible_engine()
    try:
        eng.runs = {}
        eng.interrupt_before_running = {"1"}
        cfg = _config(tmp_path, engine="interrupt-stream", on_failure="skip")
        seeds = _seed_state(["0", "1", "2"])

        with pytest.raises(KeyboardInterrupt):
            run_step(cfg, cfg.steps[0], seeds)

        step_dir = cfg.output_dir.resolve() / "step1"
        # What a scancel leaves behind: the trap copied a half-written output back.
        (step_dir / "1" / "step1_1.out").write_text("partial\n", encoding="utf-8")
        assert (step_dir / "1" / "attempt1" / "step1_1.out").is_file(), "the good one is right here"

        eng.interrupt_before_running = set()
        outcome = run_step(cfg, cfg.steps[0], seeds)

        assert [s.id for s in outcome.state.structures] == ["0", "1", "2"]
        assert cache.load_failure_records(step_dir) == []
    finally:
        eng.runs = {}
        eng.interrupt_before_running = set()
        ENGINES.pop("interrupt-stream", None)


# ---------------------------------------------------------------------------
# The step's ensemble XYZ files — every route leaves both behind
# ---------------------------------------------------------------------------


def test_run_step_writes_results_and_survivors_ensembles(tmp_path: Path):
    """``stepN_ensemble.xyz`` holds every parsed result; ``stepN_survivors.xyz`` the kept set.

    The pair is the point: the first answers "what did this step produce", the second
    "what does the next step start from", and only together do they make a filter's
    effect visible as geometry rather than as a row count.
    """
    cfg = _config(tmp_path, sample={"method": "min", "count": 1})
    run_step(cfg, cfg.steps[0], _seed_state(["0", "1", "2"]))

    step_dir = cfg.output_dir.resolve() / "step1"
    assert len(io.read_xyz_frames(step_dir / "step1_ensemble.xyz")) == 3
    survivors = (step_dir / "step1_survivors.xyz").read_text(encoding="utf-8")
    # The fake engine's energy is ordered within single-digit ids (its documented safe
    # range), so "2" is the one min keeps — and the caption ties the frame back to its
    # structure directory.
    assert survivors.splitlines()[1].startswith("step1 id=2 E=-1.000")
    assert len(io.read_xyz_frames(step_dir / "step1_survivors.xyz")) == 1


def test_ensemble_files_are_byte_identical_across_a_cache_hit(tmp_path: Path):
    """A cache hit rewrites the files with exactly the bytes the computing run left.

    Deterministic content is what makes writing on every route safe: the sort is stable
    over the cache's manifest order, so a resume or a relocated tree regenerates the
    file rather than perturbing it.
    """
    cfg = _config(tmp_path, sample={"method": "min", "count": 2})
    seeds = _seed_state(["0", "1", "2"])
    run_step(cfg, cfg.steps[0], seeds)
    step_dir = cfg.output_dir.resolve() / "step1"
    names = ("step1_ensemble.xyz", "step1_survivors.xyz")
    before = {name: (step_dir / name).read_bytes() for name in names}

    outcome = run_step(cfg, cfg.steps[0], seeds)

    assert outcome.cache_hit is True
    assert {name: (step_dir / name).read_bytes() for name in names} == before


def test_a_halted_stop_step_still_writes_its_successes_ensemble(tmp_path: Path):
    """The ensemble is written before the pipeline halts, like the cache and steps.csv.

    A run stopped by ``on_failure: stop`` should hand the user the geometries it *did*
    finish — the failed structure is in the ledger, not silently missing from a file
    that never got written.
    """
    from chemrefine.engines.api import ENGINES

    eng = _register_fail_engine()
    try:
        eng.fail = {"1": "missing"}
        cfg = _config(tmp_path, engine="fake-fail", on_failure="stop")
        run_step(cfg, cfg.steps[0], _seed_state(["0", "1", "2"]))

        step_dir = cfg.output_dir.resolve() / "step1"
        text = (step_dir / "step1_ensemble.xyz").read_text(encoding="utf-8")
        assert "id=0 " in text and "id=2 " in text
        assert "id=1 " not in text, "the failed structure has no final geometry to show"
    finally:
        eng.fail = {}
        ENGINES.pop("fake-fail", None)


def test_rebuild_cache_step_regenerates_the_ensemble_files(tmp_path: Path):
    """``rebuild-cache`` leaves the same files a run would — deleted ones come back."""
    from chemrefine import step

    cfg = _config(tmp_path)
    seeds = _seed_state(["0", "1"])
    run_step(cfg, cfg.steps[0], seeds, mode=StepMode.EXECUTE)
    step_dir = step_dir_for(cfg, cfg.steps[0])
    for name in ("step1_ensemble.xyz", "step1_survivors.xyz"):
        (step_dir / name).unlink()
    cache.invalidate(step_dir)

    step.rebuild_cache_step(cfg, cfg.steps[0], seeds)

    assert len(io.read_xyz_frames(step_dir / "step1_ensemble.xyz")) == 2
    assert len(io.read_xyz_frames(step_dir / "step1_survivors.xyz")) == 2


# ---------------------------------------------------------------------------
# derive_step_key — template-referenced aux files reach the identity
# ---------------------------------------------------------------------------


def test_the_orca_engines_template_aux_files_reach_the_step_key(tmp_path: Path):
    """Editing a file the template references moves the key — that is the resume re-run.

    The whole chain with the real engine: ``OrcaEngine`` enumerates the quoted
    reference (``AuxFileConsuming``), ``derive_step_key`` hands the files to
    ``StepKey.of``, and an in-place edit to the guest moves fingerprint and row keys —
    which is exactly what makes ``resume`` re-run the docking step instead of serving
    results computed from the old geometry against a fingerprint that stood still.
    """
    from chemrefine import step

    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    guest = template_dir / "cl.xyz"
    guest.write_text("1\nchloride\nCl 0.0 0.0 0.0\n", encoding="utf-8")
    (template_dir / "step1.inp").write_text(
        '! XTB\n%DOCKER\n\tGUEST "cl.xyz"\nEND\n', encoding="utf-8"
    )
    cfg = Config(
        output_dir=tmp_path / "outputs",
        template_dir=template_dir,
        steps=[StepConfig(step=1, engine="orca", operation="opt_sp")],
    )
    engine = get_engine("orca")
    seed = Structure(id="0", atoms=Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]))
    ctx = build_context(cfg, cfg.steps[0], PipelineState(structures=(seed,)), engine)

    before = step.derive_step_key(ctx, cfg.steps[0], engine)
    assert before == step.derive_step_key(ctx, cfg.steps[0], engine)

    guest.write_text("1\nchloride moved\nCl 0.5 0.0 0.0\n", encoding="utf-8")
    after = step.derive_step_key(ctx, cfg.steps[0], engine)
    assert before.fingerprint != after.fingerprint
    assert before.row_keys != after.row_keys

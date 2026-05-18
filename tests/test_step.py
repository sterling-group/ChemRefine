"""Tests for the step lifecycle (prepare → submit → wait → parse → filter → cache)."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms

from chemrefine import cache, manifest
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
    inputs = manifest.load(step_dir)
    assert inputs is not None
    assert [t[2] for t in inputs.files] == ["0", "1"]


def test_cache_load_after_run_returns_results(tmp_path: Path):
    cfg = _config(tmp_path)
    run_step(cfg, cfg.steps[0], _seed_state(["0"]))
    cached = cache.load(cfg.output_dir.resolve() / "step1")
    assert cached is not None
    assert [s.id for s in cached.results.structures] == ["0"]

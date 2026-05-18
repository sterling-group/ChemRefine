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

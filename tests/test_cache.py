"""Tests for the per-step pickle cache, fingerprint, and JSON manifest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from chemrefine.cache import (
    CACHE_FORMAT_VERSION,
    StepCache,
    fingerprint,
    invalidate,
    is_valid,
    load,
    load_manifest,
    manifest_path,
    save,
    save_manifest,
)
from chemrefine.config import StepConfig
from chemrefine.errors import CacheError
from chemrefine.state import StepInputs, StepResults, Structure


def _results() -> StepResults:
    zero_force = np.zeros((1, 3))
    return StepResults(
        structures=(
            Structure(id="0", atoms=Atoms("H"), energy_hartree=-1.0, forces_ev_per_a=zero_force),
            Structure(id="1", atoms=Atoms("H"), energy_hartree=-1.5, forces_ev_per_a=zero_force),
        )
    )


def _cfg(**overrides) -> StepConfig:
    base = {"step": 1, "engine": "fake", "operation": "opt_sp"}
    base.update(overrides)
    return StepConfig(**base)


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_stable_for_same_inputs():
    cfg = _cfg()
    assert fingerprint(cfg, ("0", "1")) == fingerprint(cfg, ("0", "1"))


def test_fingerprint_changes_when_config_changes():
    a = fingerprint(_cfg(charge=0), ("0",))
    b = fingerprint(_cfg(charge=-1), ("0",))
    assert a != b


def test_fingerprint_changes_when_parents_change():
    cfg = _cfg()
    assert fingerprint(cfg, ("0",)) != fingerprint(cfg, ("0", "1"))


def test_fingerprint_is_sixteen_chars():
    assert len(fingerprint(_cfg(), ("0",))) == 16


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------


def test_save_creates_pickle_and_json(tmp_path: Path):
    step_dir = tmp_path / "step1"
    save(
        step_cfg=_cfg(),
        parent_ids=("0", "1"),
        results=_results(),
        step_dir=step_dir,
        chemrefine_version="2.0.0",
    )
    assert (step_dir / "_cache" / "step.pkl").is_file()
    assert (step_dir / "_cache" / "step.json").is_file()


def test_save_and_load_round_trip(tmp_path: Path):
    step_dir = tmp_path / "step1"
    save(
        step_cfg=_cfg(),
        parent_ids=("0", "1"),
        results=_results(),
        step_dir=step_dir,
        chemrefine_version="2.0.0",
    )
    loaded = load(step_dir)
    assert loaded is not None
    assert loaded.step == 1
    assert loaded.engine == "fake"
    assert [s.id for s in loaded.results.structures] == ["0", "1"]
    assert loaded.results.structures[1].energy_hartree == -1.5


def test_reuse_fingerprint_round_trips(tmp_path: Path):
    step_dir = tmp_path / "step1"
    save(
        step_cfg=_cfg(),
        parent_ids=("0",),
        results=_results(),
        step_dir=step_dir,
        chemrefine_version="2.0.0",
        reuse_fingerprint="abc123",
    )
    assert load(step_dir).reuse_fingerprint == "abc123"


def test_reuse_fingerprint_defaults_empty(tmp_path: Path):
    step_dir = tmp_path / "step1"
    save(
        step_cfg=_cfg(),
        parent_ids=("0",),
        results=_results(),
        step_dir=step_dir,
        chemrefine_version="2.0.0",
    )
    assert load(step_dir).reuse_fingerprint == ""


def test_load_missing_returns_none(tmp_path: Path):
    assert load(tmp_path / "step1") is None


def test_load_corrupt_pickle_raises(tmp_path: Path):
    cache_file = tmp_path / "step1" / "_cache" / "step.pkl"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_bytes(b"not a pickle")
    with pytest.raises(CacheError):
        load(tmp_path / "step1")


def test_load_rejects_old_cache_format(tmp_path: Path):
    import pickle

    step_dir = tmp_path / "step1"
    cache_file = step_dir / "_cache" / "step.pkl"
    cache_file.parent.mkdir(parents=True)
    obj = StepCache(
        cache_format="v3.0",
        chemrefine_version="3.0.0",
        fingerprint="abc",
        step=1,
        name=None,
        engine="fake",
        operation="opt_sp",
        parent_ids=(),
        results=_results(),
    )
    cache_file.write_bytes(pickle.dumps(obj))
    with pytest.raises(CacheError):
        load(step_dir)


def test_load_rejects_pickle_that_is_not_step_cache(tmp_path: Path):
    """A pickle whose payload isn't a StepCache must surface as CacheError."""
    import pickle

    step_dir = tmp_path / "step1"
    cache_file = step_dir / "_cache" / "step.pkl"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_bytes(pickle.dumps({"i": "am not a StepCache"}))
    with pytest.raises(CacheError):
        load(step_dir)


# ---------------------------------------------------------------------------
# is_valid
# ---------------------------------------------------------------------------


def test_is_valid_true_when_cache_matches(tmp_path: Path):
    step_dir = tmp_path / "step1"
    cfg = _cfg()
    save(
        step_cfg=cfg,
        parent_ids=("0",),
        results=_results(),
        step_dir=step_dir,
        chemrefine_version="2.0.0",
    )
    assert is_valid(step_cfg=cfg, parent_ids=("0",), step_dir=step_dir)


def test_is_valid_false_when_config_changes(tmp_path: Path):
    step_dir = tmp_path / "step1"
    save(
        step_cfg=_cfg(charge=0),
        parent_ids=("0",),
        results=_results(),
        step_dir=step_dir,
        chemrefine_version="2.0.0",
    )
    assert not is_valid(step_cfg=_cfg(charge=-1), parent_ids=("0",), step_dir=step_dir)


def test_is_valid_false_when_cache_is_corrupt(tmp_path: Path):
    """A corrupt pickle should make ``is_valid`` return False, not raise."""
    step_dir = tmp_path / "step1"
    cache_dir = step_dir / "_cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "step.pkl").write_bytes(b"not a real pickle")
    assert not is_valid(step_cfg=_cfg(), parent_ids=("0",), step_dir=step_dir)


def test_is_valid_false_when_parents_change(tmp_path: Path):
    step_dir = tmp_path / "step1"
    cfg = _cfg()
    save(
        step_cfg=cfg,
        parent_ids=("0",),
        results=_results(),
        step_dir=step_dir,
        chemrefine_version="2.0.0",
    )
    assert not is_valid(step_cfg=cfg, parent_ids=("0", "1"), step_dir=step_dir)


def test_is_valid_false_when_no_cache(tmp_path: Path):
    assert not is_valid(step_cfg=_cfg(), parent_ids=("0",), step_dir=tmp_path / "step1")


# ---------------------------------------------------------------------------
# invalidate
# ---------------------------------------------------------------------------


def test_invalidate_removes_cache(tmp_path: Path):
    step_dir = tmp_path / "step1"
    save(
        step_cfg=_cfg(),
        parent_ids=("0",),
        results=_results(),
        step_dir=step_dir,
        chemrefine_version="2.0.0",
    )
    invalidate(step_dir)
    assert not (step_dir / "_cache" / "step.pkl").exists()
    assert not (step_dir / "_cache" / "step.json").exists()


def test_invalidate_missing_is_noop(tmp_path: Path):
    invalidate(tmp_path / "missing")  # must not raise


def test_cache_format_version_constant():
    """Bumping CACHE_FORMAT_VERSION is a public ABI break we want to notice."""
    assert CACHE_FORMAT_VERSION == "v2.0"


# ---------------------------------------------------------------------------
# Per-step JSON manifest (save_manifest / load_manifest / manifest_path)
# ---------------------------------------------------------------------------


def _manifest_inputs(tmp_path: Path) -> StepInputs:
    return StepInputs(
        files=(
            (tmp_path / "in/step1_structure_0.inp", tmp_path / "out/step1_structure_0.out", "0"),
            (tmp_path / "in/step1_structure_1.inp", tmp_path / "out/step1_structure_1.out", "1"),
        )
    )


def test_manifest_save_writes_under_cache(tmp_path: Path):
    inputs = _manifest_inputs(tmp_path)
    path = save_manifest(inputs, tmp_path / "step1", operation="opt_sp", engine="fake")
    assert path == manifest_path(tmp_path / "step1")
    assert path.is_file()
    assert path.parent.name == "_cache"


def test_manifest_save_and_load_round_trip(tmp_path: Path):
    inputs = _manifest_inputs(tmp_path)
    save_manifest(inputs, tmp_path / "step1", operation="opt_sp", engine="fake")
    loaded = load_manifest(tmp_path / "step1")
    assert loaded is not None
    assert loaded.files == inputs.files


def test_manifest_load_missing_returns_none(tmp_path: Path):
    assert load_manifest(tmp_path / "step1") is None


def test_manifest_save_records_operation_and_engine(tmp_path: Path):
    import json

    inputs = _manifest_inputs(tmp_path)
    path = save_manifest(inputs, tmp_path / "step1", operation="goat", engine="orca")
    data = json.loads(path.read_text())
    assert data["operation"] == "goat"
    assert data["engine"] == "orca"


def test_manifest_load_corrupt_json_raises_cache_error(tmp_path: Path):
    path = manifest_path(tmp_path / "step1")
    path.parent.mkdir(parents=True)
    path.write_text("not valid json {", encoding="utf-8")
    with pytest.raises(CacheError):
        load_manifest(tmp_path / "step1")


def test_manifest_load_missing_files_key_raises_cache_error(tmp_path: Path):
    """A manifest with no ``files`` key must surface as a ``CacheError``."""
    path = manifest_path(tmp_path / "step1")
    path.parent.mkdir(parents=True)
    path.write_text('{"operation": "opt_sp", "engine": "orca"}', encoding="utf-8")
    with pytest.raises(CacheError):
        load_manifest(tmp_path / "step1")

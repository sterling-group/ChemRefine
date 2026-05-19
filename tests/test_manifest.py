"""Tests for the per-step JSON manifest."""

from __future__ import annotations

from pathlib import Path

import pytest

from chemrefine.errors import CacheError
from chemrefine.manifest import load, manifest_path, save
from chemrefine.state import StepInputs


def _inputs(tmp_path: Path) -> StepInputs:
    return StepInputs(
        files=(
            (tmp_path / "in/step1_structure_0.inp", tmp_path / "out/step1_structure_0.out", "0"),
            (tmp_path / "in/step1_structure_1.inp", tmp_path / "out/step1_structure_1.out", "1"),
        )
    )


def test_save_writes_manifest_under_cache(tmp_path: Path):
    inputs = _inputs(tmp_path)
    path = save(inputs, tmp_path / "step1", operation="opt_sp", engine="fake")
    assert path == manifest_path(tmp_path / "step1")
    assert path.is_file()
    assert path.parent.name == "_cache"


def test_save_and_load_round_trip(tmp_path: Path):
    inputs = _inputs(tmp_path)
    save(inputs, tmp_path / "step1", operation="opt_sp", engine="fake")
    loaded = load(tmp_path / "step1")
    assert loaded is not None
    assert loaded.files == inputs.files


def test_load_missing_returns_none(tmp_path: Path):
    assert load(tmp_path / "step1") is None


def test_save_records_operation_and_engine(tmp_path: Path):
    import json

    inputs = _inputs(tmp_path)
    path = save(inputs, tmp_path / "step1", operation="goat", engine="orca")
    data = json.loads(path.read_text())
    assert data["operation"] == "goat"
    assert data["engine"] == "orca"


def test_load_corrupt_json_raises_cache_error(tmp_path: Path):
    path = manifest_path(tmp_path / "step1")
    path.parent.mkdir(parents=True)
    path.write_text("not valid json {", encoding="utf-8")
    with pytest.raises(CacheError):
        load(tmp_path / "step1")


def test_load_missing_files_key_raises_cache_error(tmp_path: Path):
    """A manifest written by an older format / different tool must surface as CacheError, not bare KeyError."""
    path = manifest_path(tmp_path / "step1")
    path.parent.mkdir(parents=True)
    path.write_text('{"operation": "opt_sp", "engine": "orca"}', encoding="utf-8")
    with pytest.raises(CacheError):
        load(tmp_path / "step1")

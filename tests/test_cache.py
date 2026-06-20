"""Tests for the per-step JSON cache, fingerprint, and manifest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from chemrefine.cache import (
    CACHE_FORMAT_VERSION,
    fingerprint,
    invalidate,
    is_valid,
    load,
    load_if_valid,
    load_manifest,
    manifest_path,
    parents_digest,
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


def test_fingerprint_changes_with_parents_digest():
    cfg = _cfg()
    assert fingerprint(cfg, ("0",), parents_digest="aaa") != fingerprint(
        cfg, ("0",), parents_digest="bbb"
    )


def _h2(spacing: float = 0.74, energy: float | None = None) -> Structure:
    atoms = Atoms(symbols=["H", "H"], positions=[[0, 0, 0], [spacing, 0, 0]])
    return Structure(id="0", atoms=atoms, energy_hartree=energy)


def test_parents_digest_stable_for_identical_content():
    assert parents_digest([_h2()]) == parents_digest([_h2()])


def test_parents_digest_changes_when_geometry_changes():
    """Same IDs, different coordinates — the digest is what catches an edited seed file."""
    assert parents_digest([_h2(spacing=0.74)]) != parents_digest([_h2(spacing=0.75)])


def test_parents_digest_changes_when_energy_changes():
    assert parents_digest([_h2(energy=-1.0)]) != parents_digest([_h2(energy=-1.1)])


def test_parents_digest_changes_when_id_changes():
    a = _h2()
    b = Structure(id="7", atoms=a.atoms, energy_hartree=a.energy_hartree)
    assert parents_digest([a]) != parents_digest([b])


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------


def test_save_creates_single_json_document(tmp_path: Path):
    """The cache is one JSON document — no pickle is ever written."""
    step_dir = tmp_path / "step1"
    save(
        step_cfg=_cfg(),
        parent_ids=("0", "1"),
        results=_results(),
        step_dir=step_dir,
        chemrefine_version="2.0.0",
    )
    assert (step_dir / "_cache" / "step.json").is_file()
    assert not (step_dir / "_cache" / "step.pkl").exists()


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


def test_round_trip_preserves_positions_forces_and_flags(tmp_path: Path):
    """Coordinates, forces, lineage, and status flags survive the JSON store."""
    forces = np.array([[0.1234567890123456, -2.5e-7, 3.0]], dtype=np.float64)
    struct = Structure(
        id="0-1",
        atoms=Atoms("H", positions=[[0.7414213562373095, 0.0, -1.5e-9]]),
        parent_id="0",
        energy_hartree=-1.0000000000000002,
        forces_ev_per_a=forces,
        converged=True,
        terminated=False,
    )
    step_dir = tmp_path / "step1"
    save(
        step_cfg=_cfg(),
        parent_ids=("0",),
        results=StepResults(structures=(struct,)),
        step_dir=step_dir,
        chemrefine_version="2.0.0",
    )
    loaded = load(step_dir).results.structures[0]
    assert loaded.parent_id == "0"
    assert loaded.energy_hartree == -1.0000000000000002
    assert loaded.converged is True and loaded.terminated is False
    np.testing.assert_array_equal(loaded.atoms.get_positions(), struct.atoms.get_positions())
    np.testing.assert_array_equal(loaded.forces_ev_per_a, forces)
    assert loaded.forces_ev_per_a.dtype == np.float64


def test_round_trip_keeps_parents_digest_stable(tmp_path: Path):
    """The load-bearing property: a JSON round-trip must not perturb the digest.

    Downstream steps fingerprint against ``parents_digest`` of *loaded*
    structures (exact float64 bytes); any drift would invalidate every
    downstream cache on resume.
    """
    struct = _h2(spacing=0.7414213562373095, energy=-1.1283791670955126)
    step_dir = tmp_path / "step1"
    save(
        step_cfg=_cfg(),
        parent_ids=("0",),
        results=StepResults(structures=(struct,)),
        step_dir=step_dir,
        chemrefine_version="2.0.0",
    )
    loaded = load(step_dir).results.structures
    assert parents_digest(loaded) == parents_digest([struct])


def test_load_missing_returns_none(tmp_path: Path):
    assert load(tmp_path / "step1") is None


def test_load_corrupt_json_raises(tmp_path: Path):
    cache_file = tmp_path / "step1" / "_cache" / "step.json"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_bytes(b"{not json")
    with pytest.raises(CacheError):
        load(tmp_path / "step1")


def test_load_rejects_old_cache_format(tmp_path: Path):
    import json

    step_dir = tmp_path / "step1"
    save(
        step_cfg=_cfg(),
        parent_ids=("0",),
        results=_results(),
        step_dir=step_dir,
        chemrefine_version="2.0.0",
    )
    cache_file = step_dir / "_cache" / "step.json"
    data = json.loads(cache_file.read_text(encoding="utf-8"))
    data["cache_format"] = "v9.9"
    cache_file.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(CacheError, match=r"format v9\.9"):
        load(step_dir)


def test_load_rejects_legacy_summary_sidecar(tmp_path: Path):
    """The pickle-era ``step.json`` was a summary without ``structures``.

    Loading one must raise (→ ``is_valid`` False → clean rebuild), never
    misread the summary as a complete cache.
    """
    step_dir = tmp_path / "step1"
    cache_file = step_dir / "_cache" / "step.json"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text(
        '{"cache_format": "v2.0", "chemrefine_version": "2.0.0", "fingerprint": "abc",'
        ' "reuse_fingerprint": "", "step": 1, "name": null, "engine": "fake",'
        ' "operation": "opt_sp", "structure_ids": ["0"], "parent_ids": [null],'
        ' "energies_hartree": [-1.0]}',
        encoding="utf-8",
    )
    with pytest.raises(CacheError, match="stale or corrupt"):
        load(step_dir)
    assert not is_valid(step_cfg=_cfg(), parent_ids=("0",), step_dir=step_dir)


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
    (cache_dir / "step.json").write_bytes(b"{not json")
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
# load_if_valid — single load + fingerprint check (the cache-hit fast path)
# ---------------------------------------------------------------------------


def test_load_if_valid_returns_cache_on_match(tmp_path: Path):
    """A matching fingerprint returns the cached StepCache (not just a bool)."""
    step_dir = tmp_path / "step1"
    cfg = _cfg()
    save(
        step_cfg=cfg,
        parent_ids=("0",),
        results=_results(),
        step_dir=step_dir,
        chemrefine_version="2.0.0",
    )
    cached = load_if_valid(step_cfg=cfg, parent_ids=("0",), step_dir=step_dir)
    assert cached is not None
    assert cached.fingerprint == fingerprint(cfg, ("0",))


def test_load_if_valid_returns_none_on_mismatch(tmp_path: Path):
    """A changed config returns None (re-run), never a stale cache."""
    step_dir = tmp_path / "step1"
    save(
        step_cfg=_cfg(charge=0),
        parent_ids=("0",),
        results=_results(),
        step_dir=step_dir,
        chemrefine_version="2.0.0",
    )
    assert load_if_valid(step_cfg=_cfg(charge=-1), parent_ids=("0",), step_dir=step_dir) is None


# ---------------------------------------------------------------------------
# invalidate
# ---------------------------------------------------------------------------


def test_invalidate_removes_cache_and_legacy_pickle(tmp_path: Path):
    step_dir = tmp_path / "step1"
    save(
        step_cfg=_cfg(),
        parent_ids=("0",),
        results=_results(),
        step_dir=step_dir,
        chemrefine_version="2.0.0",
    )
    # A pre-JSON run may have left a step.pkl behind — invalidate sweeps it too.
    (step_dir / "_cache" / "step.pkl").write_bytes(b"legacy")
    invalidate(step_dir)
    assert not (step_dir / "_cache" / "step.json").exists()
    assert not (step_dir / "_cache" / "step.pkl").exists()


def test_invalidate_missing_is_noop(tmp_path: Path):
    invalidate(tmp_path / "missing")  # must not raise


def test_cache_format_version_constant():
    """Bumping CACHE_FORMAT_VERSION is a public ABI break we want to notice."""
    # v2.0: a JSON document (pickle removed) whose fingerprint excludes the
    # sample config (filtering re-runs on every load, so a filter-only edit
    # must be a cache hit). Pickle-era summary sidecars are rejected
    # structurally (no `structures` key), so no bump was needed pre-release.
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

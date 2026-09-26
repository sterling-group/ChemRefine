"""Nested input files are resolved once, content-pinned and portable across run trees."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

import pytest
from ase import Atoms
from fake_engine import FakeEngine
from pydantic import Field

from chemrefine.config import Config, StepConfig, load_config, resolve_relative_paths
from chemrefine.engines._options import EngineOptions
from chemrefine.engines.api import ENGINES, preflight_steps
from chemrefine.errors import ConfigError
from chemrefine.input_files import (
    declared_input_files,
    normalized_input_options,
    option_pointer,
    resolve_input_file_options,
    validate_input_files,
)
from chemrefine.state import PipelineState, Structure
from chemrefine.step import build_context, derive_step_key, run_step
from chemrefine.validate import validate_config_text


class _Options(EngineOptions):
    """A nested scientific configuration that includes a list of input files."""

    problem: dict[str, Any] = Field(default_factory=dict)


class _FileEngine(FakeEngine):
    """A file consumer with a manifest grammar owned entirely by the engine."""

    options_cls: ClassVar[type[_Options]] = _Options
    locations = (("problem", "sources", 0, "path"),)
    preflight_refuses = "invalid declared input filenames"
    seen_paths: ClassVar[list[Path]] = []

    def input_file_options(self, options):
        """Declare the one nested filename, including its list index."""
        return self.locations

    def input_file_dependencies(self, options, files):
        """Read a manifest and enumerate its payload, tolerating not-yet-produced inputs."""
        result = {}
        for name, path in files.items():
            payload = json.loads(path.read_text()).get("payload")
            if payload:
                result[f"{name.lstrip('/')}/payload"] = (path.parent / payload).resolve()
        return result

    def check_step(self, step_cfg, *, charge, multiplicity):
        """Observe the exact filename a worker will receive without requiring it to exist."""
        self.seen_paths.append(Path(step_cfg.options["problem"]["sources"][0]["path"]))


def _config(base: Path, filename: Any = "./input.json") -> Config:
    """Construct the same source-relative config as the file and text loaders."""
    return resolve_relative_paths(
        Config(
            steps=[
                StepConfig(
                    step=1,
                    engine="fake",
                    options={"problem": {"sources": [{"path": filename}]}},
                )
            ]
        ),
        base=base,
    )


def _seeds() -> PipelineState:
    """One real parent supplies the per-structure calculation identity."""
    return PipelineState(structures=(Structure(id="0", atoms=Atoms("H")),))


def _key(config: Config):
    """Derive the production key through the same context used for job preparation."""
    engine = _FileEngine()
    ctx = build_context(config, config.steps[0], _seeds(), engine)
    return derive_step_key(ctx, config.steps[0], engine)


def test_source_provenance_is_private_and_resolution_is_idempotent(tmp_path):
    """Resolving nested inputs must not mutate YAML options or leak machine paths into YAML."""
    config = _config(tmp_path)
    original = config.steps[0]
    resolved = resolve_input_file_options(original, _FileEngine())
    assert original.options["problem"]["sources"][0]["path"] == "./input.json"
    assert resolved.options["problem"]["sources"][0]["path"] == str(tmp_path / "input.json")
    assert resolve_input_file_options(resolved, _FileEngine()) == resolved
    assert normalized_input_options(resolved, _FileEngine(), resolved.options) == original.options
    assert original.source_dir == tmp_path
    assert "source_dir" not in original.model_dump()
    assert "input_file_spellings" not in original.model_dump()
    assert original.model_copy().source_dir == tmp_path


def test_edited_or_removed_file_options_do_not_reuse_stale_provenance(tmp_path):
    """A model_copy edit is new user intent, even if its source carried resolved metadata."""
    original = resolve_input_file_options(_config(tmp_path).steps[0], _FileEngine())
    edited = original.model_copy(
        update={"options": {"problem": {"sources": [{"path": "new.json"}]}}}
    )
    resolved = resolve_input_file_options(edited, _FileEngine())
    assert resolved.input_file_spellings[0][1] == "new.json"
    no_files = _FileEngine()
    no_files.locations = ()
    assert not resolve_input_file_options(resolved, no_files).input_file_spellings
    assert (
        resolve_input_file_options(_config(tmp_path).steps[0], FakeEngine()).input_file_spellings
        == ()
    )


def test_files_and_manifest_payloads_change_identity_and_survive_relocation(tmp_path):
    """Same bytes in a moved tree match; edited payload bytes under one manifest do not."""
    keys = []
    for name in ("original", "relocated"):
        base = tmp_path / name
        base.mkdir()
        (base / "input.json").write_text('{"payload": "arrays.bin"}')
        (base / "arrays.bin").write_bytes(b"first data")
        keys.append(_key(_config(base)))
    assert keys[0] == keys[1]
    payload = tmp_path / "relocated" / "arrays.bin"
    payload.write_bytes(b"changed data")
    assert _key(_config(payload.parent)).row_keys != keys[1].row_keys
    payload.unlink()
    assert _key(_config(payload.parent)).fingerprint != keys[1].fingerprint
    (payload.parent / "input.json").write_text("{}")
    assert _key(_config(payload.parent)).fingerprint != keys[1].fingerprint


def test_absolute_references_are_keyed_by_basename_and_content(tmp_path):
    """Absolute paths supplied programmatically retain the existing model_path convention."""
    keys = []
    for name in ("a", "b"):
        base = tmp_path / name
        base.mkdir()
        path = base / "input.json"
        path.write_text("{}")
        keys.append(_key(_config(base, str(path))))
    assert keys[0] == keys[1]


def test_missing_upstream_files_defer_existence_checks_until_preparation(tmp_path):
    """Preflight can precede a producer step; submission cannot precede its product."""
    config = _config(tmp_path)
    with patch("chemrefine.engines.api.get_engine", return_value=_FileEngine()):
        preflight_steps(config.steps, charge=0, multiplicity=1)
    assert _key(config).fingerprint
    with pytest.raises(ConfigError, match=r"input file .* unavailable"):
        run_step(config, config.steps[0], _seeds(), engine=_FileEngine())
    (tmp_path / "input.json").write_text('{"payload": "arrays.bin"}')
    with pytest.raises(ConfigError, match=r"dependency:.* unavailable"):
        validate_input_files(config.steps[0], _FileEngine())
    (tmp_path / "arrays.bin").write_bytes(b"data")
    assert run_step(config, config.steps[0], _seeds(), engine=_FileEngine()).state.structures


def test_file_and_text_validation_preflight_and_context_share_the_config_base(
    tmp_path, monkeypatch
):
    """Launching from another directory must not change declared input resolution."""
    project = tmp_path / "project"
    project.mkdir()
    text = (
        "steps:\n  - step: 1\n    engine: fake\n    options:\n"
        "      problem:\n        sources:\n          - path: inputs/model.json\n"
    )
    source = project / "input.yaml"
    source.write_text(text)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(ENGINES, "fake", _FileEngine)
    _FileEngine.seen_paths.clear()
    loaded = load_config(source)
    report = validate_config_text(text, base_dir=project)
    assert report.ok
    preflight_steps(loaded.steps, charge=0, multiplicity=1)
    ctx = build_context(loaded, loaded.steps[0], _seeds(), _FileEngine())
    expected = project / "inputs" / "model.json"
    assert _FileEngine.seen_paths == [expected, expected]
    assert Path(ctx.step_cfg.options["problem"]["sources"][0]["path"]) == expected


@pytest.mark.parametrize("location", [(), (True,), (-1,), (None,), ["problem"]])
def test_invalid_engine_option_location_declarations_are_refused(tmp_path, location):
    """Malformed declarations fail at a public configuration boundary."""
    engine = _FileEngine()
    engine.locations = (location,)
    with pytest.raises(ConfigError, match="invalid input option path"):
        resolve_input_file_options(_config(tmp_path).steps[0], engine)


def test_duplicate_option_declarations_are_refused(tmp_path):
    """One leaf cannot acquire two different identities through duplicate declarations."""
    engine = _FileEngine()
    engine.locations *= 2
    with pytest.raises(ConfigError, match="duplicate input file option"):
        resolve_input_file_options(_config(tmp_path).steps[0], engine)


@pytest.mark.parametrize("filename", [None, 2, "", "  "])
def test_declared_leaf_must_be_a_nonempty_filename(tmp_path, filename):
    """A declared filename is neither an output toggle nor an arbitrary option value."""
    with pytest.raises(ConfigError, match="nonempty string"):
        resolve_input_file_options(_config(tmp_path, filename).steps[0], _FileEngine())


@pytest.mark.parametrize(
    "location",
    [("missing",), ("problem", "sources", 1), ("problem", 0), ("problem", "sources", "0")],
)
def test_absent_or_wrongly_typed_locations_are_refused(tmp_path, location):
    """List indices and dictionary keys are distinct, and must actually exist."""
    engine = _FileEngine()
    engine.locations = (location,)
    with pytest.raises(ConfigError, match="does not exist"):
        resolve_input_file_options(_config(tmp_path).steps[0], engine)


@pytest.mark.parametrize(
    "name,path",
    [
        ("", Path("/a")),
        ("/absolute", Path("/a")),
        (1, Path("/a")),
        ("ok", Path("relative")),
        ("ok", "/a"),
    ],
)
def test_dependency_contract_refuses_unstable_names_and_ambiguous_paths(tmp_path, name, path):
    """Manifest readers return portable identities and already-anchored filesystem paths."""
    engine = _FileEngine()
    engine.input_file_dependencies = lambda options, files: {name: path}
    with pytest.raises(ConfigError, match="input dependency"):
        declared_input_files(_config(tmp_path).steps[0], engine)


def test_json_pointer_escaping_and_consumer_without_manifest_hook(tmp_path):
    """Option names cannot collide merely because they contain slash or tilde."""
    assert option_pointer(("a/b", "~name", 2)) == "/a~1b/~0name/2"
    engine = FakeEngine()
    engine.input_file_options = lambda options: (("problem", "sources", 0, "path"),)
    files = declared_input_files(_config(tmp_path).steps[0], engine)
    assert files == {"option:/problem/sources/0/path:./input.json": tmp_path / "input.json"}


def test_invalid_declared_file_is_reported_by_text_validation(tmp_path, monkeypatch):
    """The GUI/CLI report translates the same resolution refusal, without raising."""
    monkeypatch.setitem(ENGINES, "fake", _FileEngine)
    report = validate_config_text("steps: [{step: 1, engine: fake}]", base_dir=tmp_path)
    assert not report.ok
    assert any("input file option" in issue.message for issue in report.issues)

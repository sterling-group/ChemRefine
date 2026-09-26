"""Typed nested references and selected manifest formats are discovered without SDKs."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest
from ase import Atoms
from pydantic import BaseModel, Field

from chemrefine.config import Config, StepConfig, resolve_relative_paths
from chemrefine.engines._input_files import typed_input_references
from chemrefine.engines.qiskit.bundles import write_bundle
from chemrefine.engines.qiskit.experiment import EXPERIMENTS, QiskitExperimentEngine
from chemrefine.errors import ConfigError
from chemrefine.input_files import (
    declared_input_files,
    normalized_input_options,
    resolve_input_file_options,
)
from chemrefine.state import PipelineState, Structure
from chemrefine.step import build_context, derive_step_key


class _Source(BaseModel):
    """A named bundle inside a structured source collection."""

    bundle: str = Field(json_schema_extra={"input_file": True, "file_format": "quantum_bundle"})
    label: str = "metadata is not a filename"


class _Plain(BaseModel):
    """A discriminated input which is deliberately not parsed as a bundle."""

    kind: Literal["plain"] = "plain"
    path: str = Field(json_schema_extra={"input_file": True})


class _Bundle(BaseModel):
    """The same logical option name has a different declared parser in this branch."""

    kind: Literal["bundle"] = "bundle"
    path: str = Field(json_schema_extra={"input_file": True, "file_format": "quantum_bundle"})


class _Nested(BaseModel):
    """Nested models, container filenames and a selected union branch."""

    sources: list[dict[str, _Source]] = Field(default_factory=list)
    selection: _Plain | _Bundle = Field(discriminator="kind")
    checkpoints: list[str | None] = Field(
        default_factory=list, json_schema_extra={"input_file": True}
    )
    optional: _Source | None = None


@pytest.fixture
def nested_engine(monkeypatch):
    """Register declarations only; dependency inspection never invokes a worker."""
    monkeypatch.setattr(EXPERIMENTS, "_specs", dict(EXPERIMENTS._specs))

    @EXPERIMENTS.register("nested_files", _Nested)
    def builder(**kwargs):
        pytest.fail("input discovery cannot execute scientific code")

    return QiskitExperimentEngine()


def _step(base, selection, **patch):
    """Keep configuration provenance separate from semantic scientific options."""
    config = resolve_relative_paths(
        Config(
            steps=[
                StepConfig(
                    step=1,
                    engine="qiskit-experiment",
                    options={
                        "experiment": {
                            "name": "nested_files",
                            "options": {"selection": selection, **patch},
                        }
                    },
                )
            ]
        ),
        base=base,
    )
    return config, config.steps[0]


def test_recursive_refs_keep_container_locations_and_only_selected_union_formats():
    model = _Nested(
        sources=[{"a/b~c": {"bundle": "arrays.json"}}],
        selection={"kind": "plain", "path": "plain.txt"},
        checkpoints=["state.qpy", None],
    )
    actual = typed_input_references(model, prefix=("options",))
    assert [(ref.location, ref.file_format) for ref in actual] == [
        (("options", "sources", 0, "a/b~c", "bundle"), "quantum_bundle"),
        (("options", "selection", "path"), None),
        (("options", "checkpoints", 0), None),
    ]
    assert typed_input_references(_Bundle(path="bundle.json"))[0].file_format == "quantum_bundle"


def test_nested_bundle_payloads_are_hashed_and_identity_survives_relocation(
    tmp_path, nested_engine
):
    root = tmp_path / "original"
    root.mkdir()
    arrays = write_bundle(
        root / "arrays.json", kind="dataset", arrays={"a": np.ones(2)}, metadata={}
    )
    (root / "plain.txt").write_text("This is not a manifest.")
    (root / "state.qpy").write_bytes(b"circuit bytes are hashed without importing Qiskit")
    config, step = _step(
        root,
        {"kind": "plain", "path": "plain.txt"},
        sources=[{"a/b~c": {"bundle": arrays.name}}],
        checkpoints=["state.qpy"],
    )
    files = declared_input_files(step, nested_engine)
    assert len(files) == 4
    assert any(
        "a~1b~0c/bundle/payload" in key and path.suffix == ".npz" for key, path in files.items()
    )
    resolved = resolve_input_file_options(step, nested_engine)
    assert resolved.options["experiment"]["options"]["sources"][0]["a/b~c"]["bundle"] == str(arrays)
    assert normalized_input_options(resolved, nested_engine, resolved.options) == step.options
    seeds = PipelineState(structures=(Structure(id="0", atoms=Atoms("H")),))
    key = derive_step_key(build_context(config, step, seeds, nested_engine), step, nested_engine)
    relocated = tmp_path / "relocated"
    shutil.copytree(root, relocated)
    copied, copied_step = _step(
        relocated,
        {"kind": "plain", "path": "plain.txt"},
        sources=[{"a/b~c": {"bundle": arrays.name}}],
        checkpoints=["state.qpy"],
    )
    other_key = derive_step_key(
        build_context(copied, copied_step, seeds, nested_engine), copied_step, nested_engine
    )
    assert key == other_key
    payload = next(relocated.glob("*.npz"))
    payload.write_bytes(payload.read_bytes() + b"edited")
    changed = derive_step_key(
        build_context(copied, copied_step, seeds, nested_engine), copied_step, nested_engine
    )
    assert changed.fingerprint != key.fingerprint


def test_selected_format_prevents_unrelated_components_from_parsing_plain_inputs(
    tmp_path, nested_engine, monkeypatch
):
    @EXPERIMENTS.register("unrelated_bundle", _Bundle)
    def unrelated(**kwargs):
        pytest.fail("unselected component must not execute")

    @EXPERIMENTS.register("flat_plain", _Plain)
    def flat_plain(**kwargs):
        pytest.fail("dependency discovery cannot execute a flat input consumer")

    plain = tmp_path / "plain.json"
    plain.write_text("not even JSON")
    flat_options = {"experiment": {"name": "flat_plain", "options": {"path": str(plain)}}}
    assert (
        nested_engine.input_file_dependencies(flat_options, {"/experiment/options/path": plain})
        == {}
    )
    _, step = _step(tmp_path, {"kind": "plain", "path": plain.name})
    files = declared_input_files(step, nested_engine)
    assert len(files) == 1
    bundle = write_bundle(tmp_path / "bundle.json", kind="test", arrays={}, metadata={})
    _, selected = _step(tmp_path, {"kind": "bundle", "path": bundle.name})
    assert len(declared_input_files(selected, nested_engine)) == 2


def test_typed_reference_discovery_imports_no_quantum_sdk():
    code = """import sys
from chemrefine.engines._input_files import typed_input_references
from chemrefine.engines.qiskit.experiment import QiskitExperimentEngine
engine = QiskitExperimentEngine()
assert engine.input_file_options({"experiment": {
    "name": "fermionic_shadows",
    "options": {"circuit_path": "state.qpy", "shadows": {"num_particles": 1}}
}})
assert not any(
    name == prefix or name.startswith(prefix + ".")
    for name in sys.modules
    for prefix in ("qiskit", "qiskit_aer", "qiskit_nature", "ffsim")
)
"""
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("value", ["", 1, {1: "file"}])
def test_declared_refs_reject_invalid_filenames_and_container_keys(value):
    class Raw(BaseModel):
        source: Any = Field(json_schema_extra={"input_file": True})

    with pytest.raises(ConfigError, match=r"filenames|string keys"):
        typed_input_references(Raw(source=value))


def test_declared_path_values_optional_refs_and_nonfile_scalars():
    class Raw(BaseModel):
        source: Any = Field(json_schema_extra={"input_file": True})
        scalar: int = 7
        bytes_value: bytes = b"unmarked"

    refs = typed_input_references(Raw(source={"a": (Path("source.dat"), None)}))
    assert refs[0].location == ("source", "a", 0)
    assert typed_input_references(Raw(source=[])) == ()


def test_invalid_format_marks_cycles_and_excess_nesting_fail_cleanly():
    class Format(BaseModel):
        path: str = Field(json_schema_extra={"input_file": True, "file_format": 1})

    with pytest.raises(ConfigError, match="formats"):
        typed_input_references(Format(path="x"))

    class BadModel(BaseModel):
        nested: _Source = Field(json_schema_extra={"input_file": True})

    with pytest.raises(ConfigError, match="inside nested"):
        typed_input_references(BadModel(nested=_Source(bundle="x")))

    class Values(BaseModel):
        data: Any

    cycle: list[Any] = []
    cycle.append(cycle)
    with pytest.raises(ConfigError, match="cyclic"):
        typed_input_references(Values(data=cycle))
    deep: list[Any] = []
    for _ in range(66):
        deep = [deep]
    with pytest.raises(ConfigError, match="64 option levels"):
        typed_input_references(Values(data=deep))


def test_file_bearing_defaults_and_aliases_are_refused_before_resolution():
    from pydantic import AliasChoices

    class Defaulted(BaseModel):
        path: str = Field("default.dat", json_schema_extra={"input_file": True})

    with pytest.raises(ConfigError, match="supplied explicitly"):
        typed_input_references(Defaulted())
    assert typed_input_references(Defaulted(path="explicit.dat"))[0].location == ("path",)

    class DefaultContainer(BaseModel):
        source: _Source = Field(default_factory=lambda: _Source(bundle="default.json"))

    with pytest.raises(ConfigError, match="supplied explicitly"):
        typed_input_references(DefaultContainer())

    for alias in (
        {"alias": "input"},
        {"serialization_alias": "input"},
        {"validation_alias": AliasChoices("path", "input")},
    ):

        class Aliased(BaseModel):
            path: str = Field(json_schema_extra={"input_file": True}, **alias)

        model = Aliased.model_validate({"input" if "alias" in alias else "path": "file.dat"})
        with pytest.raises(ConfigError, match="canonical names"):
            typed_input_references(model)

    class AliasContainer(BaseModel):
        source: _Source = Field(alias="input")

    with pytest.raises(ConfigError, match="canonical names"):
        typed_input_references(AliasContainer.model_validate({"input": {"bundle": "file.json"}}))

    class SameAlias(BaseModel):
        path: str = Field(alias="path", json_schema_extra={"input_file": True})

    assert typed_input_references(SameAlias(path="file.dat"))[0].location == ("path",)

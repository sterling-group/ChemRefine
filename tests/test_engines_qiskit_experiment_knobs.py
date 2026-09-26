"""Explicit example/guard/internal verdicts for every artifact component schema field."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml

from chemrefine.engines.api import get_engine

REPO = Path(__file__).resolve().parent.parent
VERDICTS = REPO / "tests/data/engines/qiskit-experiment/component_knobs.json"


def _schema_fields(schema: dict[str, Any]) -> set[str]:
    """Enumerate declared fields, including selected unions and object arrays, without SDKs."""

    def visit(node: dict[str, Any], prefix: str, references: tuple[str, ...]) -> set[str]:
        """Resolve local schema references without treating arbitrary dictionary keys as knobs."""
        fields = set()
        if "$ref" in node:
            reference = node["$ref"]
            assert reference.startswith("#/$defs/") and reference not in references
            fields |= visit(
                schema["$defs"][reference.split("/")[-1]], prefix, (*references, reference)
            )
        for name, value in node.get("properties", {}).items():
            location = f"{prefix}.{name}" if prefix else name
            fields.add(location)
            fields |= visit(value, location, references)
        for keyword in ("anyOf", "oneOf", "allOf"):
            for variant in node.get(keyword, []):
                fields |= visit(variant, prefix, references)
        if isinstance(node.get("items"), dict):
            fields |= visit(node["items"], prefix + "[]", references)
        for item in node.get("prefixItems", []):
            fields |= visit(item, prefix + "[]", references)
        if isinstance(node.get("additionalProperties"), dict):
            fields |= visit(node["additionalProperties"], prefix + ".*", references)
        return fields

    return visit(schema, "", ())


def _present(options: Any, field: str) -> bool:
    """Require a raw YAML declaration, following arrays and explicit selector shorthand."""
    first, separator, tail = field.partition(".")
    if first == "*":
        return isinstance(options, dict) and any(
            _present(value, tail) for value in options.values()
        )
    if isinstance(options, str):
        return field == "name"
    if not isinstance(options, dict):
        return False
    name = first.removesuffix("[]")
    if name not in options:
        return False
    value = options[name]
    if first.endswith("[]"):
        return isinstance(value, list) and any(_present(item, tail) for item in value)
    return not separator or _present(value, tail)


def _node_source(reference: str) -> str:
    """Validate a real pytest node and include its local helper/decorator evidence."""
    filename, separator, name = reference.partition("::")
    path = REPO / filename
    assert separator and name.startswith("test_") and path.is_file(), reference
    source = path.read_text(encoding="utf-8")
    module = ast.parse(source)
    functions = {
        node.name: node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert name in functions, f"missing guard test: {reference}"
    test = functions[name]
    assert any(
        isinstance(node, ast.Assert)
        or (
            isinstance(node, ast.Attribute)
            and node.attr in {"raises", "assert_allclose", "assert_array_equal", "fail"}
        )
        for node in ast.walk(test)
    ), f"guard has no assertion: {reference}"
    visited: set[str] = set()
    pieces = []

    def collect(current: str) -> None:
        """Retain local fixture/helper configuration used by the named behavioral test."""
        if current in visited:
            return
        visited.add(current)
        node = functions[current]
        pieces.append(ast.get_source_segment(source, node) or "")
        pieces.extend(
            ast.get_source_segment(source, decorator) or "" for decorator in node.decorator_list
        )
        for item in ast.walk(node):
            if isinstance(item, ast.Name) and item.id in functions:
                collect(item.id)

    collect(name)
    return "\n".join(pieces)


def _validate_filing(name: str, fields: set[str], filing: dict[str, Any]) -> None:
    """Every field receives one explicit verdict with auditable behavioral evidence."""
    assert set(filing) == {"examples", "guards", "internal"}, name
    counts = Counter(field for kind in filing for item in filing[kind] for field in item["fields"])
    assert set(counts) == fields, f"{name}: unfiled/stale fields {sorted(set(counts) ^ fields)}"
    assert all(count == 1 for count in counts.values()), f"{name}: duplicate verdicts {counts}"
    for kind in ("guards", "internal"):
        for item in filing[kind]:
            assert item["fields"] and len(item["reason"].strip()) >= 20, (name, item)
            evidence = "\n".join(_node_source(reference) for reference in item["tests"])
            assert evidence, (name, item)
            for field in item["fields"]:
                token = item.get("evidence", {}).get(
                    field, field.rsplit(".", 1)[-1].removesuffix("[]")
                )
                assert token in evidence, f"{name}.{field}: missing test evidence {token!r}"
            assert set(item.get("evidence", {})) <= set(item["fields"])
    for item in filing["examples"]:
        path = REPO / item["path"]
        assert path.is_file() and item["fields"], (name, item)
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        matches = [step for step in document["steps"] if step["step"] == item["step"]]
        assert len(matches) == 1 and matches[0]["engine"] == "qiskit-experiment", item
        options = matches[0]["options"]
        if name != "engine":
            assert options["experiment"]["name"] == name, item
            options = options["experiment"]["options"]
        for field in item["fields"]:
            assert _present(options, field), f"{name}.{field}: absent from raw example {path}"


def test_every_experiment_schema_field_has_one_explicit_knob_verdict():
    """New components, nested fields and stale verdicts fail without importing compute SDKs."""
    filed = json.loads(VERDICTS.read_text(encoding="utf-8"))
    assert set(filed) == {"version", "engine", "components", "delegates"}
    assert filed["version"] == 1
    engine = get_engine("qiskit-experiment")
    components = engine.component_catalog()["experiment"].components
    assert set(filed["components"]) == set(components)
    for name, component in components.items():
        _validate_filing(name, _schema_fields(component.options_schema), filed["components"][name])
    _validate_filing("engine", set(engine.options_cls.model_fields), filed["engine"])
    # Keep the generic engine fixture authoritative for the outer option model.
    outer = json.loads((VERDICTS.parent / "knobs.json").read_text(encoding="utf-8"))
    assert outer["model"] == engine.options_cls.__name__
    assert set(outer["examples"]) == {
        field for row in filed["engine"]["examples"] for field in row["fields"]
    }
    assert set(outer["tests_only"]) == {
        field
        for kind in ("guards", "internal")
        for row in filed["engine"][kind]
        for field in row["fields"]
    }
    assert set(filed["delegates"]) == {"sampler.options", "estimator.options"}
    for reference in filed["delegates"].values():
        _node_source(reference)


def test_experiment_knob_discovery_blocks_all_optional_provider_imports():
    """The artifact catalog and its schemas remain available in a plain ChemRefine wheel."""
    script = """
import importlib.abc
import sys
class NoProviders(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'qiskit', 'qiskit_nature', 'qiskit_algorithms', 'qiskit_aer', 'ffsim',
            'qiskit_fermions', 'pyscf', 'cvxpy', 'scs', 'qiskit_ibm_runtime',
            'qiskit_addon_cutting', 'qualtran', 'openfermion', 'samplomatic', 'qiskit_mitigation',
        }:
            raise AssertionError('optional provider import: ' + fullname)
sys.meta_path.insert(0, NoProviders())
from chemrefine.engines.api import get_engine
catalog = get_engine('qiskit-experiment').component_catalog()['experiment'].components
assert all(component.options_schema['properties'] for component in catalog.values())
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_knob_auditor_rejects_missing_duplicate_and_unrelated_evidence(tmp_path):
    """The gate cannot be satisfied by an empty or unrelated blanket tests-only verdict."""
    valid = {
        "examples": [],
        "guards": [
            {
                "fields": ["device"],
                "reason": "CUDA capability preflight is exercised before any worker publication.",
                "tests": [
                    "tests/test_engines_qiskit_experiment.py::test_invalid_experiments_fail_before_publication"
                ],
            }
        ],
        "internal": [],
    }
    _validate_filing("fixture", {"device"}, valid)
    with pytest.raises(AssertionError, match="unfiled/stale"):
        _validate_filing("fixture", {"device", "new_field"}, valid)
    with pytest.raises(AssertionError, match="duplicate"):
        _validate_filing("fixture", {"device"}, {**valid, "internal": valid["guards"]})
    invalid = {**valid, "guards": [{**valid["guards"][0], "fields": ["unrelated"]}]}
    with pytest.raises(AssertionError, match="missing test evidence"):
        _validate_filing("fixture", {"unrelated"}, invalid)
    with pytest.raises(AssertionError, match="missing guard test"):
        _node_source("tests/test_engines_qiskit_experiment.py::test_missing_node")


def test_nested_schema_and_raw_examples_do_not_confuse_dynamic_options_with_declared_fields():
    """Declared union/list/map fields stay distinct from arbitrary provider options."""
    schema = {
        "$defs": {"Leaf": {"properties": {"width": {"type": "integer"}}}},
        "properties": {
            "items": {"type": "array", "items": {"$ref": "#/$defs/Leaf"}},
            "optional": {"anyOf": [{"$ref": "#/$defs/Leaf"}, {"type": "null"}]},
            "options": {"type": "object", "additionalProperties": True},
            "mapping": {"additionalProperties": {"$ref": "#/$defs/Leaf"}},
        },
    }
    assert _schema_fields(schema) == {
        "items",
        "items[].width",
        "optional",
        "optional.width",
        "options",
        "mapping",
        "mapping.*.width",
    }
    assert _present({"items": [{"width": 3}]}, "items[].width")
    assert not _present({"items": []}, "items[].width")
    assert _present({"sampler": "statevector"}, "sampler.name")
    assert not _present({"sampler": "statevector"}, "sampler.options")
    assert not _present({"optional": None}, "optional.width")
    assert _present({"mapping": {"named": {"width": 3}}}, "mapping.*.width")

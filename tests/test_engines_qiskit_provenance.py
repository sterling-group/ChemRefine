"""Consumed component provenance is reproducible without SDK imports or sensitive paths."""

from __future__ import annotations

import json
import subprocess
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace

import pytest

from chemrefine.engines.api import BackendRequirement
from chemrefine.engines.qiskit import provenance
from chemrefine.engines.qiskit.options import ComponentSelection, QiskitOptions
from chemrefine.engines.qiskit.registry import ComponentSpec, NoComponentOptions


@pytest.fixture(autouse=True)
def clear_source_snapshot():
    """Keep source-identity simulations from contaminating other worker tests."""
    provenance.source_identity.cache_clear()
    yield
    provenance.source_identity.cache_clear()


def test_provider_inventory_follows_consumed_components_and_transitive_fidelity(monkeypatch):
    """Unused configured Runtime is excluded; QNSPSA's consumed sampler brings its stack."""
    monkeypatch.setattr(provenance, "version", lambda package: f"installed:{package}")
    configured = {
        "algorithm": "exact",
        "estimator": {"name": "ibm_runtime", "options": {"backend_name": "ibm_example"}},
    }
    exact = provenance.molecular_provenance(QiskitOptions.from_raw(configured))
    assert set(exact["consumed_components"]) == {"algorithm", "mapper"}
    assert "qiskit-ibm-runtime" not in exact["package_versions"]
    assert "pyscf" not in exact["package_versions"]
    options = QiskitOptions.from_raw(
        {
            **configured,
            "algorithm": "vqe",
            "optimizer": "qnspsa",
            "sampler": {"name": "ibm_runtime", "options": {"backend_name": "ibm_example"}},
        }
    )
    runtime = provenance.molecular_provenance(options, preparation_source="pyscf")
    assert runtime["consumed_components"]["sampler"]["name"] == "ibm_runtime"
    assert runtime["provider_profile"] == "qiskit-runtime"
    assert {"pyscf", "qiskit-ibm-runtime", "qiskit-mitigation", "samplomatic"} <= runtime[
        "package_versions"
    ].keys()
    serialized = json.dumps(runtime)
    assert "ibm_example" not in serialized and str(Path.home()) not in serialized
    assert runtime["environment"]["python_version"]
    assert runtime["environment"]["chemrefine_version"]


def test_supplied_count_graph_excludes_unexecuted_sampler(monkeypatch):
    """A supplied-count solve does not claim to have consumed the configured Aer sampler."""
    monkeypatch.setattr(provenance, "version", lambda package: f"installed:{package}")
    options = QiskitOptions.from_raw(
        {"algorithm": {"name": "sqd", "options": {"counts": {"11": 5}}}, "sampler": "aer"}
    )
    metadata = provenance.molecular_provenance(options)
    assert "sampler" not in metadata["consumed_components"]
    assert metadata["provider_profile"] == "qiskit-fermionic"
    assert "dependency families" in metadata["package_versions_scope"]


def test_experiment_provenance_follows_selected_sampler(monkeypatch):
    """Artifact reports use the same environment schema and consumed-resource selection."""
    monkeypatch.setattr(provenance, "version", lambda package: "1.0")
    metadata = provenance.experiment_provenance(
        ComponentSelection(
            name="pauli_measurement",
            options={"circuit_path": "bound.qpy", "observable": {"Z": 1}, "sampler": "aer"},
        )
    )
    assert set(metadata["consumed_components"]) == {"experiment", "sampler"}
    assert metadata["provider_profile"] == "qiskit-aer"
    assert "qiskit-aer" in metadata["package_versions"]


def test_custom_provider_and_missing_distribution_metadata_are_explicit(monkeypatch):
    """Unknown local distribution metadata is recorded, never invented or loaded as a module."""

    def absent(_name):
        raise PackageNotFoundError

    monkeypatch.setattr(provenance, "distribution", absent)
    monkeypatch.setattr(provenance, "version", absent)
    spec = ComponentSpec(
        NoComponentOptions,
        lambda: None,
        backend_requirement=BackendRequirement(extra="custom-provider", import_name="custom_sdk"),
    )
    metadata = provenance.execution_provenance(
        {"sampler": (ComponentSelection.named("custom"), spec)}
    )
    assert metadata["package_versions"] == {}
    assert not metadata["dependency_metadata_available"]
    assert "custom-sdk" in metadata["missing_distribution_metadata"]
    assert metadata["provider_profile"] == "custom-provider"


def test_metadata_extra_closure_respects_markers_and_cycles(monkeypatch):
    """Installed metadata is authoritative; unselected extras and base GUI packages stay out."""
    monkeypatch.setattr(
        provenance,
        "distribution",
        lambda _: SimpleNamespace(
            requires=[
                "base-package",
                'unused; extra == "other"',
                'wanted; extra == "one"',
                'chemrefine[two]; extra == "one"',
                'chemrefine[one]; extra == "two"',
                'child; extra == "two"',
            ]
        ),
    )
    assert provenance._profile_packages({"one"}) == ({"wanted", "child"}, True)
    monkeypatch.setattr(provenance, "distribution", lambda _: SimpleNamespace(requires=None))
    assert provenance._profile_packages({"one"}) == (set(), True)


@pytest.mark.parametrize("dirty", [False, True])
def test_checkout_identity_is_cached_and_excludes_worktree_paths(monkeypatch, dirty):
    """Capture a commit and dirty flag once, discarding filenames from git status."""
    calls = []
    monkeypatch.setattr(Path, "exists", lambda _: True)

    def run(arguments, **kwargs):
        calls.append(arguments)
        assert kwargs["timeout"] == 5 and kwargs["capture_output"]
        return SimpleNamespace(
            stdout="a" * 40 if "rev-parse" in arguments else " M secret/path" if dirty else ""
        )

    monkeypatch.setattr(provenance.subprocess, "run", run)
    first = provenance.source_identity()
    assert first == {"kind": "git_checkout", "commit": "a" * 40, "dirty": dirty}
    assert provenance.source_identity() == first and len(calls) == 2
    assert "secret" not in json.dumps(first)


@pytest.mark.parametrize(
    "direct,expected",
    [
        (
            {"url": "https://secret@example.invalid/repo", "vcs_info": {"commit_id": "a" * 40}},
            {"kind": "vcs_install", "commit": "a" * 40, "dirty": None},
        ),
        (
            {"archive_info": {"hashes": {"sha256": "b" * 64}}},
            {"kind": "archive_install", "sha256": "b" * 64},
        ),
        (
            {"url": "file:///secret/path", "dir_info": {"editable": True}},
            {"kind": "editable_install", "commit": None, "dirty": None},
        ),
        ({"vcs_info": [], "archive_info": []}, {"kind": "installed_distribution"}),
        ({"archive_info": {"hashes": []}}, {"kind": "installed_distribution"}),
        ([], {"kind": "installed_distribution"}),
        (None, {"kind": "installed_distribution"}),
    ],
)
def test_installed_source_metadata_never_copies_urls_or_local_paths(monkeypatch, direct, expected):
    """Installed wheels preserve trustworthy commit/archive identifiers and explicit unknowns."""
    monkeypatch.setattr(Path, "exists", lambda _: False)
    monkeypatch.setattr(
        provenance,
        "distribution",
        lambda _: SimpleNamespace(
            read_text=lambda _: json.dumps(direct) if direct is not None else None
        ),
    )
    result = provenance.source_identity()
    assert result.items() >= expected.items()
    assert "secret" not in json.dumps(result)


@pytest.mark.parametrize("failure", [OSError("missing git"), subprocess.TimeoutExpired("git", 5)])
@pytest.mark.parametrize("metadata", ["absent", "malformed"])
def test_source_identity_survives_missing_git_or_metadata(monkeypatch, failure, metadata):
    """Provenance availability must not turn a finished numerical result into a failure."""
    monkeypatch.setattr(Path, "exists", lambda _: True)

    def fail(*_args, **_kwargs):
        raise failure

    def installed(_name):
        if metadata == "absent":
            raise PackageNotFoundError
        return SimpleNamespace(read_text=lambda _: "{invalid json")

    monkeypatch.setattr(provenance.subprocess, "run", fail)
    monkeypatch.setattr(provenance, "distribution", installed)
    assert provenance.source_identity()["kind"] == "installed_distribution"

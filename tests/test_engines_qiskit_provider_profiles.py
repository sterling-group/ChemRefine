"""Managed quantum profiles must match their recursively installed project extras."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from chemrefine.engines.api import BackendRequirement
from chemrefine.engines.qiskit.profiles import BACKEND_PROFILES

PROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
METADATA = tomllib.loads(PROJECT.read_text(encoding="utf-8"))["project"]
EXTRAS = METADATA["optional-dependencies"]
PROFILES = tuple(sorted(BACKEND_PROFILES.names()))


def installed_closure(extra: str, python_version: str = "3.12") -> tuple[set[str], set[str]]:
    """Follow marked self-references exactly as an extras-aware installer would."""
    environment = default_environment()
    environment.update(python_version=python_version, python_full_version=f"{python_version}.0")
    visited: set[str] = set()
    distributions: set[str] = set()

    def expand(name: str, ancestors: frozenset[str]) -> None:
        """Reject cycles and recursively account for all project extras and dependencies."""
        assert name not in ancestors, f"Cyclic packaging extras: {sorted(ancestors)} -> {name}"
        assert name in EXTRAS, f"Undefined project extra {name!r}"
        visited.add(name)
        for text in EXTRAS[name]:
            requirement = Requirement(text)
            if requirement.marker is not None and not requirement.marker.evaluate(environment):
                continue
            if canonicalize_name(requirement.name) == canonicalize_name(METADATA["name"]):
                for child in requirement.extras:
                    expand(child, ancestors | {name})
            else:
                distributions.add(canonicalize_name(requirement.name))

    expand(extra, frozenset())
    return visited, distributions


@pytest.mark.parametrize("extra", PROFILES)
def test_profile_providers_equal_recursive_packaging_closure(extra):
    """A worker resolver may promise only the provider profiles installed by that extra."""
    installed_extras, _distributions = installed_closure(extra)
    assert BACKEND_PROFILES.provided_extras(extra) == installed_extras & set(PROFILES)


@pytest.mark.parametrize("extra", PROFILES)
def test_profile_import_probe_is_installed_by_its_extra(extra):
    """Each chosen worker's declared availability probe corresponds to a packaged dependency."""
    _extras, distributions = installed_closure(extra)
    requirement = BACKEND_PROFILES.resolve([BackendRequirement(extra=extra, import_name="unused")])
    assert requirement.extra == extra
    # All built-in probe imports differ from their wheel names only by underscores.
    assert canonicalize_name(requirement.import_name) in distributions


def test_toolkit_does_not_claim_python312_resource_worker_dependencies():
    """The ordinary toolkit and separately provisioned resource worker retain distinct domains."""
    included, distributions = installed_closure("qiskit-toolkit", python_version="3.11")
    assert "qiskit-resources" not in included
    assert not {"openfermion", "qualtran"} & distributions
    included, distributions = installed_closure("qiskit-resources", python_version="3.12")
    assert "qiskit" in included
    assert {"openfermion", "qualtran", "pyscf", "qiskit-nature"} <= distributions

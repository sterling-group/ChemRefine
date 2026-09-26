"""Shared execution provenance derived from the consumed quantum component graph.

Provider versions describe the selected dependency families, not an assertion that
every installed dependency executed. No optional SDK is imported for discovery.
Source identity is captured once per process without persisting paths or URLs.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
from collections.abc import Mapping
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from chemrefine import __version__
from chemrefine.engines.qiskit.options import ComponentSelection, QiskitOptions
from chemrefine.engines.qiskit.profiles import BACKEND_PROFILES
from chemrefine.engines.qiskit.registry import (
    REGISTRIES,
    ComponentSpec,
    consumed_component_categories,
)


@lru_cache(maxsize=1)
def source_identity() -> dict[str, Any]:
    """Capture a checkout or sanitized installed-source identifier once per process."""
    root = Path(__file__).resolve().parents[4]
    if (root / ".git").exists():
        try:
            # Fixed read-only commands; the directory comes from this module, not user input.
            commit = subprocess.run(  # noqa: S603
                ["git", "-C", str(root), "rev-parse", "HEAD"],  # noqa: S607
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            status = subprocess.run(  # noqa: S603
                ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"],  # noqa: S607
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
            return {"kind": "git_checkout", "commit": commit, "dirty": bool(status)}
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        raw = distribution("chemrefine").read_text("direct_url.json")
        direct = json.loads(raw or "{}")
    except (PackageNotFoundError, ValueError):
        direct = {}
    if isinstance(direct, dict):
        vcs = direct.get("vcs_info", {})
        if isinstance(vcs, dict) and re.fullmatch(r"[0-9a-fA-F]{40,64}", str(vcs.get("commit_id"))):
            return {"kind": "vcs_install", "commit": vcs["commit_id"], "dirty": None}
        archive = direct.get("archive_info", {})
        hashes = archive.get("hashes", {}) if isinstance(archive, dict) else {}
        if isinstance(hashes, dict) and re.fullmatch(r"[0-9a-fA-F]{64}", str(hashes.get("sha256"))):
            return {"kind": "archive_install", "sha256": hashes["sha256"]}
        if direct.get("dir_info") == {"editable": True}:
            return {"kind": "editable_install", "commit": None, "dirty": None}
    return {"kind": "installed_distribution", "version": __version__}


def _profile_packages(extras: set[str]) -> tuple[set[str], bool]:
    """Follow selected self-referencing extras in installed metadata, without SDK imports."""
    try:
        requirements = [Requirement(item) for item in distribution("chemrefine").requires or ()]
    except PackageNotFoundError:
        return set(), False
    packages: set[str] = set()
    visited: set[str] = set()
    pending = set(extras)
    while pending:
        extra = pending.pop()
        visited.add(extra)
        for requirement in requirements:
            # Ordinary orchestration requirements are not optional provider families.
            if requirement.marker is None or not requirement.marker.evaluate({"extra": extra}):
                continue
            if canonicalize_name(requirement.name) == "chemrefine":
                pending.update(requirement.extras - visited)
            else:
                packages.add(canonicalize_name(requirement.name))
    return packages, True


def execution_provenance(
    components: Mapping[str, tuple[ComponentSelection, ComponentSpec]],
    *,
    additional_packages: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Record the consumed graph, its selected provider families and the executing environment."""
    requirements = [
        spec.backend_requirement
        for _selection, spec in components.values()
        if spec.backend_requirement is not None
    ]
    extras = {requirement.extra for requirement in requirements} | {"qiskit-core"}
    packages, metadata_available = _profile_packages(extras)
    packages.update({"numpy", "scipy", *additional_packages})
    # A custom provider may not appear in ChemRefine's installed extras metadata.
    packages.update(canonicalize_name(requirement.import_name) for requirement in requirements)
    versions = {}
    missing = []
    for package in sorted(packages):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            missing.append(package)
    return {
        "package_versions": versions,
        "package_versions_scope": "selected component dependency families; not execution tracing",
        "missing_distribution_metadata": missing,
        "dependency_metadata_available": metadata_available,
        "consumed_components": {
            category: {
                "name": selection.name,
                "requires": sorted(spec.requires),
                "provider_extra": spec.backend_requirement.extra
                if spec.backend_requirement
                else None,
            }
            for category, (selection, spec) in sorted(components.items())
        },
        "provider_profile": BACKEND_PROFILES.resolve(requirements).extra,
        "environment": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "system": platform.system(),
            "machine": platform.machine(),
            "chemrefine_version": __version__,
            "source": dict(source_identity()),
            "source_capture": "first provenance collection in this process",
        },
    }


def molecular_provenance(
    options: QiskitOptions, *, preparation_source: str | None = None
) -> dict[str, Any]:
    """Exclude configured but unconsumed molecular components from provider provenance."""
    selections = options.component_selections()
    return execution_provenance(
        {
            category: (selections[category], REGISTRIES[category].spec(selections[category].name))
            for category in consumed_component_categories(options)
        },
        additional_packages=("pyscf",) if preparation_source == "pyscf" else (),
    )


def experiment_provenance(selection: ComponentSelection) -> dict[str, Any]:
    """Collect the selected experiment and the execution resources its registry consumes."""
    from chemrefine.engines.qiskit.experiment import EXPERIMENTS

    spec = EXPERIMENTS.spec(selection.name)
    options = EXPERIMENTS.options_for(selection)
    components = {"experiment": (selection, spec)}
    for category in spec.requires & REGISTRIES.keys():
        selected = getattr(options, category)
        components[category] = (selected, REGISTRIES[category].spec(selected.name))
    return execution_provenance(components)

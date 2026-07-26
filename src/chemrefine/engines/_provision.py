"""Managed per-backend environments + launcher resolution (a generic building block).

One Python process can host only one MLIP dependency family (their torch/e3nn trees conflict),
so ChemRefine keeps its core install light and runs each backend from an **isolated managed
environment**, resolved **by name** — never a path in the YAML. This module is fully generic:
what a step needs arrives as a :class:`~chemrefine.engines.api.BackendRequirement` from the
engine's ``backend_requirement`` hook (capability by Protocol, like NMS), and everything here
works off that DTO — no backend names, tasks, or tables.

* :func:`preflight_backends` — validate every step **before any job submits**: a backend is
  runnable if an explicit ``backend_python`` override is set, a managed env exists, or the
  backend is importable in the current env (the single-env case). Otherwise it raises
  :class:`~chemrefine.errors.ConfigError` naming the ``chemrefine backends install`` fix.
* :func:`resolve_launcher` / :func:`launcher_for` — the interpreter that runs a step's
  backend (server or direct script): override → managed-env python → ``sys.executable``.
* :func:`build_backend_env` — create a managed env under :func:`chemrefine_home`, using the
  **same tool that created the current env** (:func:`detect_env_tool`: conda / uv / venv), and
  install ``chemrefine[<extra>]`` into it matched to the orchestrator's own install (same
  index version, or the same local/git source for direct installs).

A managed env is a plain venv/conda env; *running* it needs only ``<env>/bin/python`` — the
provisioning tool is needed only at build time (e.g. once on a login node; offline compute
nodes just run the resolved interpreter).
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit
from urllib.request import url2pathname

from chemrefine import __version__
from chemrefine.config import StepConfig
from chemrefine.engines.api import (
    ENGINES,
    BackendRequirement,
    ProvisionableEngine,
    get_engine,
)
from chemrefine.errors import ConfigError

EnvTool = Literal["conda", "uv", "venv"]


def known_backend_extras() -> frozenset[str]:
    """Every backend extra the registered engines can require.

    The union of each provisionable engine's ``backend_extras()`` — registration-driven
    (the MLIP extras come from the backend specs, PySCF's from its engines), never a
    hardcoded list. Drives ``chemrefine backends`` listing + name validation.
    """
    extras: set[str] = set()
    for name in ENGINES:
        engine = get_engine(name)
        if isinstance(engine, ProvisionableEngine):
            extras.update(engine.backend_extras())
    return frozenset(extras)


def chemrefine_home() -> Path:
    """Root for ChemRefine-managed state (managed envs live under ``<home>/backends``).

    ``$CHEMREFINE_HOME`` if set; else alongside the install —
    ``<sys.prefix>/share/chemrefine`` — when that prefix is writable (removed with the env);
    else ``~/.chemrefine/<interpreter tag>`` (read-only / shared / system installs).

    The home-directory fallback is namespaced by interpreter tag (``cpython-312``) because
    ``$HOME`` is routinely shared across machines on HPC. Two clusters with different
    Python versions would otherwise resolve the same
    ``~/.chemrefine/backends/<extra>/bin/python`` and one would silently run the other's
    env. The ``sys.prefix`` branch needs no tag: it is already inside one interpreter.
    """
    env = os.environ.get("CHEMREFINE_HOME")
    if env:
        return Path(env)
    if os.access(Path(sys.prefix), os.W_OK):
        return Path(sys.prefix) / "share" / "chemrefine"
    return Path.home() / ".chemrefine" / sys.implementation.cache_tag


def backend_env_path(extra: str) -> Path:
    """The managed env directory for ``extra`` (``<chemrefine_home>/backends/<extra>``)."""
    return chemrefine_home() / "backends" / extra


def backend_env_python(extra: str) -> Path:
    """``<managed env>/bin/python`` for ``extra`` (whether or not it exists yet)."""
    return backend_env_path(extra) / "bin" / "python"


def resolve_launcher(requirement: BackendRequirement, override: str | None = None) -> str:
    """The interpreter that runs this backend: override → managed env → this interpreter.

    Availability is guaranteed earlier by :func:`preflight_backends`, so this never errors —
    the ``sys.executable`` fallback is the single-env case, where the backend is importable
    alongside the orchestrator and the orchestrator's own interpreter is the right launcher
    (a bare ``python`` may not exist on a compute node's PATH).
    """
    if override:
        return override
    env_python = backend_env_python(requirement.extra)
    if env_python.is_file():
        return str(env_python)
    return sys.executable


def launcher_for(engine: object, options: dict[str, Any] | None) -> str:
    """The interpreter that runs ``engine``'s backend for a step with ``options``.

    The one home for the "is this engine provisionable, and what does it need?"
    dance shared by the script engines and the ExtOpt server launch. Engines
    without a backend run with the orchestrator's own interpreter.
    """
    if isinstance(engine, ProvisionableEngine):
        raw = options or {}
        return resolve_launcher(engine.backend_requirement(raw), raw.get("backend_python"))
    return sys.executable


def require_backend(requirement: BackendRequirement, override: str | None = None) -> None:
    """Raise :class:`ConfigError` unless this backend is runnable here.

    Runnable = an explicit ``backend_python`` override, a provisioned managed env, or the
    backend importable in the current env (a cheap :func:`importlib.util.find_spec` probe —
    no heavy import). The error names both fixes.
    """
    if override:
        return
    if backend_env_python(requirement.extra).is_file():
        return
    if importlib.util.find_spec(requirement.import_name) is not None:
        return
    raise ConfigError(
        f"backend '{requirement.extra}' is not available: '{requirement.import_name}' is not "
        f"importable here and no managed env exists. Provision it once with "
        f"`chemrefine backends install {requirement.extra}` (or install "
        f"`chemrefine[{requirement.extra}]` into this environment)."
    )


def preflight_backends(steps: Sequence[StepConfig]) -> None:
    """Validate every step's backend availability before any job submits.

    Called at the top of :func:`chemrefine.pipeline.run` so a missing backend fails the run
    up front (fail-fast) instead of at its step. Engines that aren't
    :class:`~chemrefine.engines.api.ProvisionableEngine` have nothing to check.
    """
    for step_cfg in steps:
        engine = get_engine(step_cfg.engine)
        if isinstance(engine, ProvisionableEngine):
            raw = step_cfg.options or {}
            require_backend(engine.backend_requirement(raw), raw.get("backend_python"))


def detect_env_tool() -> EnvTool:
    """How the current env was created: ``conda`` / ``uv`` / plain ``venv``.

    ``conda`` when ``$CONDA_PREFIX`` is the active prefix; ``uv`` when a ``uv`` binary is on
    PATH and the env's ``pyvenv.cfg`` carries uv's marker; else ``venv``. Managed envs are
    built with the same tool, so they live in the ecosystem already in use.
    """
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and Path(conda_prefix) == Path(sys.prefix):
        return "conda"
    cfg = Path(sys.prefix) / "pyvenv.cfg"
    if shutil.which("uv") and cfg.is_file() and "uv = " in cfg.read_text(encoding="utf-8"):
        return "uv"
    return "venv"


def _direct_url() -> dict[str, Any] | None:
    """Parsed PEP 610 ``direct_url.json`` for the installed ChemRefine dist, or ``None``.

    ``None`` means a normal index (PyPI) install — or no/corrupt metadata — so the
    caller falls back to the version-pinned spec.
    """
    try:
        raw = importlib.metadata.distribution("ChemRefine").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _install_target(extra: str) -> str:
    """The pip requirement that reproduces this orchestrator's install with ``extra``.

    Keeps the managed env matched to the ChemRefine that drives it — by *source*, not
    just version: an index install pins ``==__version__``; a PEP 610 direct install
    (``pip install -e .``, a local path, a git URL) reinstalls from the same source,
    since that version is typically not published on any index.
    """
    direct = _direct_url()
    if direct is None or "url" not in direct:
        return f"chemrefine[{extra}]=={__version__}"
    url = direct["url"]
    vcs = direct.get("vcs_info")
    if vcs:
        spec = f"{vcs['vcs']}+{url}"
        ref = vcs.get("commit_id") or vcs.get("requested_revision")
        if ref:
            spec += f"@{ref}"
        if direct.get("subdirectory"):
            spec += f"#subdirectory={direct['subdirectory']}"
        return f"chemrefine[{extra}] @ {spec}"
    if url.startswith("file://"):
        path = Path(url2pathname(urlsplit(url).path))
        if not path.exists():
            raise ConfigError(
                f"ChemRefine was installed from {path}, which no longer exists; the "
                f"managed env for '{extra}' must be built from the same source. "
                f"Reinstall ChemRefine, then re-run `chemrefine backends install {extra}`."
            )
    return f"chemrefine[{extra}] @ {url}"


def _build_commands(tool: EnvTool, path: Path, extra: str) -> list[list[str]]:
    """The argv sequence that creates the env at ``path`` and installs ``chemrefine[extra]``.

    The install target is matched to the orchestrator's own install — see
    :func:`_install_target` — so a managed env always matches the ChemRefine
    that drives it.
    """
    target = _install_target(extra)
    env_python = str(path / "bin" / "python")
    if tool == "uv":
        return [
            ["uv", "venv", str(path)],
            ["uv", "pip", "install", "--python", env_python, target],
        ]
    if tool == "conda":
        pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
        return [
            ["conda", "create", "-y", "-p", str(path), f"python={pyver}"],
            [env_python, "-m", "pip", "install", target],
        ]
    return [
        [sys.executable, "-m", "venv", str(path)],
        [env_python, "-m", "pip", "install", target],
    ]


def build_backend_env(extra: str, *, tool: EnvTool | None = None) -> Path:
    """Create (or reuse) the managed env for ``extra``; return its ``python``.

    Idempotent — returns immediately when the env's ``python`` already exists. Builds with
    ``tool`` (default: :func:`detect_env_tool`); every subprocess must succeed.
    """
    python = backend_env_python(extra)
    if python.is_file():
        return python
    backend_env_path(extra).parent.mkdir(parents=True, exist_ok=True)
    for argv in _build_commands(tool or detect_env_tool(), backend_env_path(extra), extra):
        subprocess.run(argv, check=True)
    return python

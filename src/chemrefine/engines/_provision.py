"""Managed per-backend environments + launcher resolution (a generic building block).

One Python process can host only one MLIP dependency family (their torch/e3nn trees conflict),
so ChemRefine keeps its core install light and runs each backend from an **isolated managed
environment**, resolved **by name** — never a path in the YAML. This module is fully generic:
what a step needs arrives as a :class:`~chemrefine.engines.api.BackendRequirement` from the
engine's ``backend_requirement`` hook (capability by Protocol, like NMS), and everything here
works off that DTO — no backend names, tasks, or tables.

* :func:`preflight_backends` — validate every step **before any job submits**: a backend is
  runnable if an explicit ``backend_python`` override is set, a managed env exists (and, where
  its name does not determine its contents, can import it), or the backend is importable in
  the current env (the single-env case). Otherwise it raises
  :class:`~chemrefine.errors.ConfigError` naming the ``chemrefine backends install`` fix.
* :func:`resolve_launcher` / :func:`launcher_for` — the interpreter that runs a step's
  backend (server or direct script): override → managed-env python → ``sys.executable``.
* :func:`build_backend_env` — create a managed env under :func:`chemrefine_home`, using the
  **same tool that created the current env** (:func:`detect_env_tool`: conda / uv / venv), and
  install ``chemrefine[<extra>]`` into it matched to the orchestrator's own install (same
  index version, or the same local/git source for direct installs).
* :func:`resolve_base_python` — **which Python that env is created on**. A backend's stack is
  isolated precisely so it never has to match anyone else's, and the interpreter is part of
  that stack: where ``chemrefine[<extra>]`` declares (through markers) that it does not
  install on the orchestrator's Python, the env is built on the newest one it does declare
  rather than handed to pip to discover by compiling.

A managed env is a plain venv/conda env; *running* it needs only ``<env>/bin/python`` — the
provisioning tool is needed only at build time (e.g. once on a login node; offline compute
nodes just run the resolved interpreter).
"""

from __future__ import annotations

import functools
import importlib.metadata
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit
from urllib.request import url2pathname

from chemrefine import __version__
from chemrefine.config import StepConfig
from chemrefine.engines._backend_server.base import ExtOptServed
from chemrefine.engines._options import EngineOptions
from chemrefine.engines.api import (
    ENGINES,
    BackendRequirement,
    CalculationEngine,
    OptionsDeclaring,
    ProvisionableEngine,
    get_engine,
)
from chemrefine.errors import BackendProvisionError, ConfigError

if TYPE_CHECKING:  # `packaging` is imported lazily, inside the functions that evaluate markers
    from packaging.requirements import Requirement

EnvTool = Literal["conda", "uv", "venv"]

_VERSION = re.compile(r"3\.\d+")
"""A bare ``X.Y`` — the one ``--python`` form that names a version rather than a file."""

_PYTHON_CLASSIFIER = re.compile(r"Programming Language :: Python :: (3\.\d+)")
"""One ``Programming Language :: Python :: 3.12`` trove classifier.

Anchored on ``3.`` so the bare ``:: 3`` classifier and the implementation ones
(``:: Implementation :: CPython``) are not read as versions."""


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
    """The managed env directory for ``extra`` (``<chemrefine_home>/backends/<extra>``).

    A ``-gpu`` extra shares the env of the extra it extends. ``[pyscf-gpu]`` *is* ``[pyscf]``
    plus gpu4pyscf and cutensor — a superset, not a rival — so both belong in one directory,
    and ``chemrefine backends install pyscf-gpu`` adds the accelerator packages to the env
    ``… install pyscf`` already built. The one-env-per-extra rule everywhere else exists
    because the MLIP stacks genuinely *conflict* (MACE pins ``e3nn==0.4.4``, FAIRChem needs
    ``>=0.5``); where nothing conflicts it would only duplicate a large PySCF tree.

    Keyed to the suffix rather than to a table, for the same reason
    :func:`chemrefine.config.reject_shell_unsafe` is keyed to a property: a second
    accelerator extra added later is covered by existing, and no name that is not an
    accelerator variant can be caught by it.
    """
    return chemrefine_home() / "backends" / extra.removesuffix("-gpu")


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


def launcher_for(engine: CalculationEngine, options: dict[str, Any] | None) -> str:
    """The interpreter that runs ``engine``'s backend for a step with ``options``.

    The one home for the "is this engine provisionable, and what does it need?"
    dance shared by the script engines and the ExtOpt server launch. Engines
    without a backend run with the orchestrator's own interpreter.
    """
    if isinstance(engine, ProvisionableEngine):
        raw = options or {}
        return resolve_launcher(engine.backend_requirement(raw), _backend_python(engine, raw))
    return sys.executable


def _backend_python(engine: CalculationEngine, options: dict[str, Any]) -> str | None:
    """The step's ``backend_python`` override, read through the engine's own model.

    ``backend_python`` is an :class:`~chemrefine.engines._options.EngineOptions` field, so it
    is read through the model that declares it rather than off the raw dict: a second reader
    of a declared knob is free to disagree with the first about defaults and aliases. The
    engine's ``options_cls`` is used when it declares one, exactly as
    :func:`chemrefine.engines._job.gpus_from_options` does.

    Detected with :class:`~chemrefine.engines.api.OptionsDeclaring`, like every other
    capability in ``engines/``. It was the subsystem's last ``getattr`` duck-probe — and the
    annotation on it was an assertion nothing checked, since ``getattr`` with a default is
    ``Any``. :mod:`chemrefine.engines._backend_server.registry` records having removed the
    other one for the same reason.
    """
    options_cls = engine.options_cls if isinstance(engine, OptionsDeclaring) else EngineOptions
    return options_cls.from_raw_lenient(options).backend_python


def _importable_by(python: Path, import_name: str) -> bool:
    """Whether ``import_name`` can be found by *another* interpreter.

    The env's own answer, not this process's. Asking the path whether the directory exists
    cannot tell a ``pyscf`` env from a ``pyscf-gpu`` one — they are the same directory —
    so a GPU step would pass against a CPU-only env and PySCF would fall back to CPU
    mid-run, which is the failure this exists to prevent: it is logged to the backend
    server's log and nowhere the user looks, so the run *succeeds* with the wrong hardware.

    ``find_spec`` rather than an import: it answers the same question without paying for
    torch or libcint, and this runs once per provisionable step at preflight.
    """
    probe = "import importlib.util,sys;sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)"
    try:
        # No shell. The interpreter is a path this module composed, the program is the
        # literal above, and `import_name` is an engine's own ClassVar — nothing here comes
        # from the YAML. Passed as argv rather than interpolated so it cannot be code.
        return (
            subprocess.run(  # noqa: S603
                [str(python), "-c", probe, import_name], check=False
            ).returncode
            == 0
        )
    except OSError:
        return False


def require_backend(requirement: BackendRequirement, override: str | None = None) -> None:
    """Raise :class:`ConfigError` unless this backend is runnable here.

    Runnable = an explicit ``backend_python`` override, a provisioned managed env that can
    actually import the backend, or the backend importable in the current env (a cheap
    :func:`importlib.util.find_spec` probe — no heavy import).

    A managed env is proved by its **name** wherever the name determines its contents, and
    that is everywhere except the shared-env case: ``[pyscf]`` and ``[pyscf-gpu]`` live in
    one directory (see :func:`backend_env_path`), so ``backends/pyscf`` existing says
    nothing about whether gpu4pyscf is in it. Only there is the env asked, and asking costs
    one subprocess for one kind of step.

    Probing *everywhere* was the obvious alternative and is wrong: it would make a faked env
    — CI's `provisioned-backend` job, which symlinks a bare interpreter precisely to prove
    the suite does not depend on real backends — need MACE and FAIRChem actually installed
    before it could pass.

    The override is taken on trust: it is the documented escape hatch for an environment the
    provisioner does not manage, it may name a bare command rather than a path, and
    second-guessing what the user pointed at explicitly is not this function's business.
    """
    if override:
        return
    env_python = backend_env_python(requirement.extra)
    if env_python.is_file():
        shares_env = backend_env_path(requirement.extra).name != requirement.extra
        if shares_env and not _importable_by(env_python, requirement.import_name):
            raise ConfigError(
                f"the managed env for '{requirement.extra}' cannot import "
                f"'{requirement.import_name}' — it holds the base stack, not this one. Run "
                f"`chemrefine backends install {requirement.extra}`, which installs into "
                f"the env that is already there rather than building a second one."
            )
        return
    if importlib.util.find_spec(requirement.import_name) is not None:
        return
    raise ConfigError(
        f"backend '{requirement.extra}' is not available: '{requirement.import_name}' is not "
        f"importable here and no managed env exists. {_install_advice(requirement.extra)}"
    )


def _install_advice(extra: str) -> str:
    """How to get this backend — and, where it matters, why one of the two ways cannot work.

    "Install ``chemrefine[<extra>]`` into this environment" is the wrong half of the advice
    on a Python the extra does not support: pip would report success having installed
    nothing, because every requirement the extra declares is marker-excluded there. Only the
    managed env can help, and it can — :func:`resolve_base_python` builds it on a Python the
    backend supports.
    """
    here = this_python()
    supported = _supported_pythons(extra)
    if supported and here not in supported:
        return (
            f"`chemrefine[{extra}]` does not install on Python {here} — it needs "
            f"{' or '.join(supported)} — so install it into a managed env of its own with "
            f"`chemrefine backends install {extra}`, which builds one on a Python it supports."
        )
    return (
        f"Provision it once with `chemrefine backends install {extra}` (or install "
        f"`chemrefine[{extra}]` into this environment)."
    )


def _require_server_deps(requirement: BackendRequirement) -> None:
    """Raise :class:`ConfigError` unless flask and waitress are importable here.

    The server half of an ExtOpt backend, held to the same fail-fast rule as the backend
    import itself. ``require_backend`` accepting "the backend is importable in the current
    env" is what admits the single-env case — and in that case the gradient server runs
    under this same interpreter, so a `pip install -e .` without extras beside a
    hand-installed backend passed preflight and died at server startup *inside the job*,
    after every upstream step had already been paid for. The error names both fixes,
    like the backend one does.
    """
    missing = [name for name in ("flask", "waitress") if importlib.util.find_spec(name) is None]
    if missing:
        raise ConfigError(
            f"backend '{requirement.extra}' serves gradients over HTTP, which needs "
            f"{' and '.join(missing)} — not importable in this environment. Install "
            f"`chemrefine[server]` into it, or provision the backend in its own managed "
            f"env with `chemrefine backends install {requirement.extra}`."
        )


def preflight_backends(steps: Sequence[StepConfig]) -> None:
    """Validate every step's backend availability before any job submits.

    Called at the top of :func:`chemrefine.pipeline.run` so a missing backend fails the run
    up front (fail-fast) instead of at its step. Engines that aren't
    :class:`~chemrefine.engines.api.ProvisionableEngine` have nothing to check.

    An :class:`~chemrefine.engines._backend_server.base.ExtOptServed` engine is also held
    to the *server* half of its contract, but only when the launcher resolves to this very
    interpreter (:func:`_require_server_deps`): a managed env carries ``[server]`` by
    construction — every backend extra cross-references it — and an explicit
    ``backend_python`` is the documented escape hatch, trusted like everything else about
    it.
    """
    for step_cfg in steps:
        engine = get_engine(step_cfg.engine)
        if isinstance(engine, ProvisionableEngine):
            raw = step_cfg.options or {}
            requirement = engine.backend_requirement(raw)
            override = _backend_python(engine, raw)
            require_backend(requirement, override)
            if (
                isinstance(engine, ExtOptServed)
                and resolve_launcher(requirement, override) == sys.executable
            ):
                _require_server_deps(requirement)


@functools.lru_cache(maxsize=1)
def _candidate_pythons() -> tuple[str, ...]:
    """Every Python this ChemRefine is declared to support, newest first.

    Read off our own installed metadata — the ``Programming Language :: Python :: 3.x``
    classifiers, filtered by ``Requires-Python`` — rather than written down here, because
    that pair is already the declaration CI's matrix is built from. A second list would be
    one more thing to move when the matrix moves.

    Falls back to the running interpreter alone when the metadata says nothing (a tree run
    without an installed dist): the honest answer there is "the Python I am", which is also
    exactly the behaviour this module had before it could choose.
    """
    from packaging.specifiers import SpecifierSet

    here = f"{sys.version_info.major}.{sys.version_info.minor}"
    try:
        meta = importlib.metadata.metadata("ChemRefine")
    except importlib.metadata.PackageNotFoundError:
        return (here,)
    supported = SpecifierSet(meta.get("Requires-Python") or "")
    versions = [
        match.group(1)
        for classifier in meta.get_all("Classifier") or []
        if (match := _PYTHON_CLASSIFIER.fullmatch(str(classifier)))
        and supported.contains(f"{match.group(1)}.0")
    ]
    if not versions:
        return (here,)
    return tuple(sorted(versions, key=lambda v: int(v.split(".")[1]), reverse=True))


@functools.cache
def _supported_pythons(extra: str) -> tuple[str, ...]:
    """The candidate Pythons on which ``chemrefine[extra]`` installs everything it declares.

    Read off this project's own ``Requires-Dist`` metadata, so the claim the provisioner
    acts on is the same string pip resolves — see the ``[project.optional-dependencies]``
    comment for what a ``python_version`` marker on a backend extra means.
    """
    from packaging.requirements import Requirement

    requirements = [Requirement(raw) for raw in importlib.metadata.requires("ChemRefine") or []]
    return _supported_from(requirements, extra, _candidate_pythons())


def _supported_from(
    requirements: Sequence[Requirement], extra: str, candidates: Sequence[str]
) -> tuple[str, ...]:
    """Which of ``candidates`` ``extra`` contributes *all* of its requirements on.

    For each candidate, the set of requirement names the extra contributes with markers
    evaluated at that version; a version is supported when that set is the full one —
    nothing marker-excluded there. An extra that names no Python is supported everywhere,
    which is the answer for every backend whose stack ships wheels across the matrix.

    "The extra installs *something*" would be the obvious test and is the wrong one:
    hatchling flattens the ``chemrefine[server]`` cross-reference every backend extra
    carries into ``flask``/``waitress``, so every extra contributes something on every
    Python, capped or not.

    Takes the requirements rather than reading them, so the same rule can be applied to
    ``pyproject.toml`` directly — which is what the drift test does, and what keeps the
    claim checkable without a fresh install of the metadata that carries it.
    """
    named = {version: _names_for(requirements, extra, version) for version in candidates}
    full: frozenset[str] = frozenset().union(*named.values())
    return tuple(version for version in candidates if named[version] == full)


def _names_for(requirements: Sequence[Requirement], extra: str, version: str) -> frozenset[str]:
    """The requirement names ``extra`` contributes on Python ``version``.

    Only a marked requirement can belong to an extra (``extra == "…"`` is itself a marker),
    so an unmarked core dependency is never counted — and one carrying an unrelated marker
    would be counted identically for every candidate, which cannot change the comparison
    :func:`_supported_from` makes.
    """
    env = {"extra": extra, "python_version": version, "python_full_version": f"{version}.0"}
    return frozenset(
        r.name for r in requirements if r.marker is not None and r.marker.evaluate(env)
    )


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


@dataclass(frozen=True)
class BasePython:
    """How a managed env gets created: which Python, by which tool, from which interpreter.

    ``interpreter`` is ``None`` when the tool supplies the Python itself — conda resolves
    ``python=3.12`` from its channels and uv downloads a managed build — which is why the
    version, not a path, is the field that is always set.

    ``tool`` is carried here rather than passed alongside because obtaining a Python can
    *change* it: on a plain-venv orchestrator with no ``python3.12`` on PATH, a ``uv`` binary
    is the one thing that can still produce one, and an env uv creates must be installed into
    by uv (it has no ``pip`` of its own).
    """

    version: str
    tool: EnvTool
    interpreter: str | None = None


def this_python() -> str:
    """The running interpreter's ``X.Y`` — the version markers are evaluated against."""
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def resolve_base_python(
    extra: str, tool: EnvTool | None = None, override: str | None = None
) -> BasePython:
    """The Python a managed env for ``extra`` is built on, and how it is obtained.

    The orchestrator's own interpreter whenever ``chemrefine[<extra>]`` installs on it — the
    common case, and what this module did unconditionally before it could choose. Otherwise
    the newest Python the extra claims (:func:`_supported_pythons`), because a backend whose
    stack has no wheels here is not a slow install but a source build of torch or a vendored
    abseil, and the whole point of a *managed* env is that a backend's dependencies never had
    to match anyone else's. The interpreter is the last part of that stack that did.

    Raises
    ------
    BackendProvisionError
        If a Python the extra supports cannot be obtained, or ``override`` names an
        interpreter this machine does not have.
    """
    tool = tool or detect_env_tool()
    if override is not None:
        return _named_base(override, tool)
    here = this_python()
    supported = _supported_pythons(extra)
    if not supported or here in supported:
        return BasePython(here, tool, sys.executable)
    return _obtain(supported, tool, extra, here)


def _obtain(supported: Sequence[str], tool: EnvTool, extra: str, here: str) -> BasePython:
    """Where an interpreter for the newest supported version comes from.

    conda and uv are asked for a *version* and produce it themselves — conda from its
    channels, uv by downloading a managed build. A plain venv can only be created by an
    interpreter that already exists, so the version is looked for under its canonical name
    (which is what deadsnakes, Homebrew, pyenv shims and a loaded HPC module all provide),
    and a ``uv`` binary is the fallback even where uv did not create the current env: it is
    the one tool on the machine that can still supply the missing Python.
    """
    version = supported[0]
    if tool in ("conda", "uv"):
        return BasePython(version, tool)
    found = shutil.which(f"python{version}")
    if found is not None:
        return BasePython(version, tool, found)
    if shutil.which("uv") is not None:
        return BasePython(version, "uv")
    raise BackendProvisionError(
        f"`chemrefine[{extra}]` installs on Python {' or '.join(supported)}, not on this "
        f"interpreter's {here} — everything the extra declares is excluded there, so its "
        f"managed env has to be built on one of them. Neither `python{version}` nor `uv` is "
        f"on PATH: install a Python {version} (`pip install uv`, `conda create`, or your "
        f"distribution's package), or name one with `chemrefine backends install {extra} "
        f"--python /path/to/python{version}`."
    )


def _named_base(override: str, tool: EnvTool) -> BasePython:
    """A ``--python`` the user named: ``3.12``, a command name, or a path to an interpreter.

    A bare version is handed to the tool that can produce it and looked up on PATH for the
    one that cannot; anything else must resolve to an interpreter here and now, and its
    version is read back from it — conda takes a version, never a path, so ``--python
    /opt/py312/bin/python3`` still has to become ``python=3.12``.
    """
    if _VERSION.fullmatch(override):
        found = shutil.which(f"python{override}")
        if found is None and tool == "venv":
            raise BackendProvisionError(
                f"`--python {override}` names no interpreter on PATH (looked for "
                f"`python{override}`) and this environment was made by venv, which can only "
                f"create an env from an interpreter that exists. Give the path to one, or "
                f"install `uv`, which downloads the Python it is asked for."
            )
        return BasePython(override, tool, found)
    interpreter = shutil.which(override)
    if interpreter is None:
        raise BackendProvisionError(
            f"`--python {override}` is neither an `X.Y` version nor an interpreter this "
            f"machine has (nothing on PATH, and no such executable file)."
        )
    return BasePython(_version_of(interpreter), tool, interpreter)


def _version_of(interpreter: str) -> str:
    """Ask an interpreter for its own ``X.Y``."""
    probe = "import sys;print('%d.%d' % sys.version_info[:2])"
    try:
        # No shell. The program is the literal above, and the interpreter is either a path
        # `shutil.which` resolved from `--python` or one this module built itself.
        done = subprocess.run(  # noqa: S603
            [interpreter, "-c", probe], check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as e:
        raise BackendProvisionError(
            f"`{interpreter}` could not be run to ask for its version ({e})."
        ) from e
    return done.stdout.strip()


def _env_python_version(env_path: Path) -> str | None:
    """The ``X.Y`` a managed env was built on, read off its own layout.

    ``<env>/lib/python3.12/`` is there in a conda prefix and a venv alike, so this costs a
    ``glob`` rather than a subprocess — which matters on a path that runs before every
    install. ``None`` when the layout says nothing (a half-made or faked env): unknown is
    not evidence of *wrong*, and this must not refuse an env it simply cannot read.
    """
    for lib in sorted((env_path / "lib").glob("python3.*")):
        return lib.name.removeprefix("python")
    return None


def _require_supported_env(extra: str, env_path: Path) -> None:
    """Refuse to install into a managed env whose Python the extra does not install on.

    pip *succeeds* against such an env, having installed nothing at all: every requirement
    the extra declares is marker-excluded there, so what comes back is the ``[server]`` half
    and a zero exit code. The env then passes :func:`require_backend` — a managed env is
    proved by its name — and the step fails on the backend import inside the job, which is
    the one place nobody is watching.

    That is what an env built before the extra's claim narrowed looks like, and it cannot be
    repaired in place: the interpreter is settled when the directory is created.
    """
    supported = _supported_pythons(extra)
    version = _env_python_version(env_path)
    if version is not None and supported and version not in supported:
        raise BackendProvisionError(
            f"the managed env for {extra!r} at {env_path} was built on Python {version}, "
            f"which `chemrefine[{extra}]` does not install on — it needs "
            f"{' or '.join(supported)}, and installing into this one would report success "
            f"having installed nothing. Remove that directory and re-run `chemrefine "
            f"backends install {extra}`, which builds it on a Python the backend supports."
        )


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


def _build_commands(base: BasePython, path: Path, extra: str) -> tuple[list[str], list[str]]:
    """The argv that creates the env at ``path``, and the one that installs into it.

    The install target is matched to the orchestrator's own install — see
    :func:`_install_target` — so a managed env always matches the ChemRefine
    that drives it. Which *Python* it is created on comes from ``base``: conda and uv are
    asked for a version and produce it, a venv is created by an interpreter that exists.
    """
    target = _install_target(extra)
    env_python = str(path / "bin" / "python")
    if base.tool == "uv":
        return (
            # uv takes a version or a path here, so an explicit `--python` is passed through
            # exactly as given rather than reduced to its version.
            ["uv", "venv", "--python", base.interpreter or base.version, str(path)],
            ["uv", "pip", "install", "--python", env_python, target],
        )
    if base.tool == "conda":
        return (
            ["conda", "create", "-y", "-p", str(path), f"python={base.version}"],
            [env_python, "-m", "pip", "install", target],
        )
    return (
        [base.interpreter or sys.executable, "-m", "venv", str(path)],
        [env_python, "-m", "pip", "install", target],
    )


def _existing_env_tool(env_path: Path, tool: EnvTool) -> EnvTool:
    """The tool that installs into an env that is already there.

    An env uv created carries uv's marker in its own ``pyvenv.cfg`` — the same marker
    :func:`detect_env_tool` reads for the current env — and has no ``pip`` inside it. Reading
    it back is what keeps a second ``backends install`` into that env working when the
    orchestrator's own env was not made by uv, which is exactly the case uv gets used for
    here: a plain-venv machine with no interpreter of the version a backend needs.
    """
    cfg = env_path / "pyvenv.cfg"
    if cfg.is_file() and "uv = " in cfg.read_text(encoding="utf-8"):
        return "uv"
    return tool


def build_backend_env(
    extra: str, *, tool: EnvTool | None = None, python: str | None = None
) -> Path:
    """Create the managed env for ``extra`` if it is absent, then install into it.

    The env is created on the Python the extra supports (:func:`resolve_base_python`), which
    is the orchestrator's own interpreter unless the extra's requirements are excluded there.
    ``python`` overrides that choice with a version, a command name, or a path — and is
    refused against an env that already exists, where only the install runs and the
    interpreter is settled: silently ignoring it would report success for a rebuild that
    never happened. An existing env whose Python the extra does *not* install on is refused
    outright (:func:`_require_supported_env`) rather than installed into, for the same
    reason: pip succeeds there having installed nothing.

    **The install always runs**, where this used to return early whenever the env's
    ``python`` existed. That early return made ``backends install pyscf-gpu`` a silent no-op
    on a machine that already had a ``pyscf`` env — the two share a directory (see
    :func:`backend_env_path`), so the command reported success having installed nothing and
    the GPU step then ran on CPU. pip is idempotent and cheap when the requirement is
    already satisfied, and this is an explicit user command rather than a per-run path, so
    letting pip decide costs a few seconds and removes a whole class of wrong.

    Builds with ``tool`` (default: :func:`detect_env_tool`); every subprocess must succeed.

    A failing subprocess becomes a :class:`~chemrefine.errors.BackendProvisionError` rather
    than escaping as ``CalledProcessError`` / ``FileNotFoundError``. :mod:`chemrefine.errors`
    promises every exception carries an ``exit_code`` the CLI maps to a deterministic
    status, and ``cli.backends_install`` catches only
    :class:`~chemrefine.errors.ChemRefineError` — so a bare one leaves that contract and
    reaches the user as a traceback. This is the likeliest failure the command has: it is documented
    as "run once on a login node with internet", so running it without one is the first
    mistake anybody makes. ``FileNotFoundError`` is the second: :func:`detect_env_tool`
    reports ``conda`` from ``$CONDA_PREFIX``, but the build shells out to a ``conda``
    *binary*, which on many clusters is only a shell function.

    A half-built env is removed on the way out, because leaving it is worse than not
    building it: an env whose *create* succeeded and whose *install* failed would be served
    to every later run as though it were provisioned, and the step would fail on the import
    instead. Only an env **this call created** is removed — tearing down a working ``pyscf``
    env because a later ``pyscf-gpu`` install hit a resolver wall would lose work the user
    already has.
    """
    env_python = backend_env_python(extra)
    env_path = backend_env_path(extra)
    fresh = not env_python.is_file()
    detected = tool or detect_env_tool()
    if python is not None and not fresh:
        raise BackendProvisionError(
            f"the managed env for {extra!r} already exists at {env_path}, and `--python` only "
            f"applies where one is created — this call would install into the env that is "
            f"there, on the Python it already has. Remove that directory to rebuild it."
        )
    if fresh:
        base = resolve_base_python(extra, detected, python)
    else:
        _require_supported_env(extra, env_path)
        base = BasePython(this_python(), _existing_env_tool(env_path, detected))
    create, install = _build_commands(base, env_path, extra)
    env_path.parent.mkdir(parents=True, exist_ok=True)
    for argv in [create, install] if fresh else [install]:
        try:
            # this install's own metadata; no shell, no user-supplied string.
            subprocess.run(argv, check=True)  # noqa: S603
        except (OSError, subprocess.CalledProcessError) as e:
            if fresh:
                shutil.rmtree(env_path, ignore_errors=True)
            raise BackendProvisionError(
                f"could not build the managed env for {extra!r}: `{shlex.join(argv)}` "
                f"failed ({e}). Re-run `chemrefine backends install {extra}` once the "
                f"cause is fixed, or install `chemrefine[{extra}]` into this environment."
            ) from e
    return env_python

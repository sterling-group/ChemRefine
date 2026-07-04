"""Engine plugin subsystem.

Importing this package self-registers every bundled engine: plugins are **auto-discovered** —
every bare-named subpackage under ``engines/`` is a plugin and is imported here, which runs its
``@register`` decorators and populates the registry (:data:`chemrefine.engines.api.ENGINES`).
The pipeline then looks engines up by name with :func:`get_engine` and never imports a concrete
engine module directly.

The contract lives in :mod:`chemrefine.engines.api`; the reusable building blocks an engine
is built from are the underscored modules (:mod:`._job`, :mod:`._execution`, :mod:`._script`,
:mod:`._backend_server`, :mod:`._provision`) — the underscore is what excludes them from
discovery. Adding a new engine is **fully self-contained**: drop a bare-named
``engines/<name>/`` package in and it is discovered; nothing here (or anywhere else) changes.

The provisioning entry points (:func:`preflight_backends`, :func:`build_backend_env`,
:func:`backend_env_path`, :func:`known_backend_extras`) are re-exported here (with
:func:`get_engine` / :func:`register`) as part of the subsystem's public face, so the flat
pipeline + CLI never import a building block directly.
"""

import importlib
import pkgutil

from chemrefine.engines._provision import (
    backend_env_path,
    build_backend_env,
    known_backend_extras,
    preflight_backends,
)
from chemrefine.engines.api import get_engine, register

__all__ = [
    "backend_env_path",
    "build_backend_env",
    "get_engine",
    "known_backend_extras",
    "preflight_backends",
    "register",
]


def _load_plugins() -> None:
    """Import every bare-named subpackage — each self-registers its engines.

    The naming convention is the discovery rule: a bare-named sub**package** is a plugin;
    underscored packages are building blocks and plain modules (``api``, ``_job``, …) are
    never plugins. Idempotent (``sys.modules`` short-circuits re-imports), and separate from
    the import below so tests can point it at a temporary path to prove drop-in additions.
    """
    for mod in pkgutil.iter_modules(__path__):
        if mod.ispkg and not mod.name.startswith("_"):
            importlib.import_module(f"{__name__}.{mod.name}")


_load_plugins()

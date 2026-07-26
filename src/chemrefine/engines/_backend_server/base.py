"""``ComputeBackend`` ABC + the dataclass it consumes.

Every ExtOpt-served backend implements one method — :meth:`calc` — and
returns ``(energy_hartree, gradient_hartree_per_bohr)``. The shared
Flask server (see :mod:`.server`) handles request parsing, calculator
caching, and JSON marshalling around that single contract.

This module also owns the package-local networking defaults
(``DEFAULT_BIND_HOST``, ``SERVER_URL_FILENAME``). They live here —
not in :mod:`chemrefine.quantities` — because that module is reserved
for physical constants. Networking defaults are configuration, not
chemistry.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

DEFAULT_BIND_HOST: str = "127.0.0.1"
"""Loopback host the ExtOpt server binds to by default."""

DEFAULT_BIND_PORT: int = 0
"""``0`` asks the kernel for any free ephemeral port (collision-safe)."""

SERVER_URL_FILENAME: str = "server.url"
"""Filename of the sidecar that records ``host:port`` once the server is ready."""

SERVER_TOKEN_FILENAME: str = "server.token"  # noqa: S105 — a filename, not a secret
"""Filename of the sidecar holding the per-run bearer token (written ``0600``).

The server binds loopback, but on a multi-tenant HPC node any same-host
user can reach loopback ports — the token (readable only by the job owner)
is what makes ``/calculate`` usable by the owning run alone."""


@dataclass(frozen=True)
class CalculationData:
    """Geometry + per-call settings extracted from a single ``.extinp.tmp``.

    Attributes
    ----------
    symbols
        Atomic element symbols (``H``, ``C``, ``O``, …) in input order.
    positions_angstrom
        ``(n_atoms, 3)`` array of Ångström-unit coordinates.
    charge
        Total system charge.
    multiplicity
        Spin multiplicity (``2S + 1``).
    nthreads
        Worker threads available for this call (passed by ORCA).
    dograd
        Whether ORCA requested gradients on this call. ``False`` means
        "energy only" — the backend may still return a gradient list,
        but the client will write only the energy to the ``.engrad``.
    settings
        Backend-specific knobs forwarded from the wrapper-script POST
        payload (e.g. ``{"method": "dft", "xc": "pbe"}`` for PySCF).
    """

    symbols: tuple[str, ...]
    positions_angstrom: NDArray[np.float64]
    charge: int
    multiplicity: int
    nthreads: int
    dograd: bool
    settings: dict[str, Any]


@runtime_checkable
class ComputeBackend(Protocol):
    """Contract every ExtOpt-served backend implements.

    Concrete backends own their CLI surface (no backend literals in the
    shared :mod:`server` / :mod:`client`):

    * :meth:`add_cli_args` registers backend-specific argparse flags on
      the shared server / client parsers — the shared layer iterates
      over the registry and asks each backend to contribute.
    * :meth:`settings_from_args` packs the parsed flags into the
      ``settings`` block of each ``/calculate`` POST so the server can
      forward them to the backend per call.
    * :meth:`server_cli_from_options` translates a validated YAML
      ``step.options`` dict into the matching ``--flag value`` tokens
      the engine's SLURM ``run_block`` invokes the server with.
    * :meth:`from_args` builds a per-process calculator instance from
      the parsed server CLI namespace.
    * :meth:`calc` answers one ``/calculate`` request.
    """

    name: str

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Register this backend's argparse flags on a shared parser."""
        ...

    @classmethod
    def settings_from_args(cls, args: argparse.Namespace) -> dict[str, Any]:
        """Pack per-call backend knobs into a settings dict (POST payload).

        Default: ``{}`` — the shipped backends are single-channel (the calculator is built
        once on the server from the step's YAML options, so the per-call POST carries nothing).
        A backend with genuine per-geometry knobs overrides this.
        """
        return {}

    @classmethod
    def server_cli_from_options(cls, options: dict[str, Any]) -> list[str]:
        """Translate validated YAML options into ``--flag value`` tokens."""
        ...

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> ComputeBackend:
        """Build a calculator instance from the shared server CLI namespace."""
        ...

    def calc(self, data: CalculationData) -> tuple[float, list[list[float]]]:
        """Return ``(energy_hartree, gradient_hartree_per_bohr)``.

        ``gradient_hartree_per_bohr`` may be an empty list when
        ``data.dograd`` is ``False``. Otherwise it is a length-``n_atoms``
        list of three-component ``[gx, gy, gz]`` rows.
        """
        ...


def tokens_from_options(
    options: dict[str, Any],
    *,
    value_flags: tuple[tuple[str, str], ...] = (),
    bool_flags: tuple[str, ...] = (),
) -> list[str]:
    """Turn a validated options dict into ``--flag value`` / ``--flag`` CLI tokens.

    The shared ``server_cli_from_options`` body for every backend: ``value_flags`` are
    ``(option_key, cli_flag)`` pairs emitted as ``[cli_flag, str(value)]`` when the value is
    truthy (covering MLIP's kebab mapping ``model_name → --model`` and PySCF's plain
    ``method → --method`` alike); ``bool_flags`` are option keys emitted as ``--{key}`` when
    truthy. Falsy values are omitted so the engine's ``run_block`` emits only flags the user set.
    """
    tokens: list[str] = []
    for key, flag in value_flags:
        value = options.get(key)
        if value:
            tokens.extend([flag, str(value)])
    for key in bool_flags:
        if options.get(key):
            tokens.append(f"--{key}")
    return tokens

"""``BaseExtOptCalculator`` ABC + the dataclass it consumes.

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

SERVER_URL_FILENAME: str = "server.url"
"""Filename of the sidecar that records ``host:port`` once the server is ready."""


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
class BaseExtOptCalculator(Protocol):
    """Contract every ExtOpt-served backend implements.

    Concrete backends construct themselves from the parsed server CLI
    via :meth:`from_args` and answer requests via :meth:`calc`.
    """

    name: str

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> BaseExtOptCalculator:
        """Build a calculator instance from the shared server CLI namespace."""
        ...

    def calc(
        self, data: CalculationData
    ) -> tuple[float, list[list[float]]]:
        """Return ``(energy_hartree, gradient_hartree_per_bohr)``.

        ``gradient_hartree_per_bohr`` may be an empty list when
        ``data.dograd`` is ``False``. Otherwise it is a length-``n_atoms``
        list of three-component ``[gx, gy, gz]`` rows.
        """
        ...

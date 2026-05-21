"""Shared infrastructure for ORCA External-Optimizer (ProgExt) backends.

Every MLFF / PySCF backend implements one small class —
:class:`BaseExtOptCalculator` — and registers itself in
:data:`registry.CALCULATORS`. The shared Flask server, the
``.extinp.tmp`` / ``.engrad`` protocol helpers, and the wrapper-script
client all live here so adding a third backend means writing one
``extopt_calc.py`` plus a registry entry, not duplicating ~150 LOC of
HTTP glue.

Architecture mirrors the upstream ``faccts/orca-external-tools``
project, validated against multi-backend HPC deployments.
"""

from __future__ import annotations

from chemrefine.engines._extopt.base import (
    DEFAULT_BIND_HOST,
    SERVER_URL_FILENAME,
    BaseExtOptCalculator,
    CalculationData,
)

__all__ = [
    "DEFAULT_BIND_HOST",
    "SERVER_URL_FILENAME",
    "BaseExtOptCalculator",
    "CalculationData",
]

"""Unit-conversion helpers built on :mod:`chemrefine.constants`.

Two layers of API:

* :func:`convert` — a generic ``(value, from_unit, to_unit) → value``
  function backed by a lookup table. Supports both scalar floats and
  array-likes (returns ``ndarray`` for the latter). Aliases (e.g.
  ``"ha"`` → ``"hartree"``, ``"Å"`` → ``"angstrom"``) keep YAML / ORCA
  text matchable case-insensitively.
* :func:`boltzmann_weights` — a domain-specific helper that returns
  normalized Boltzmann probabilities for an array of relative
  energies in kcal/mol.

Every numeric factor below is sourced from :mod:`chemrefine.constants`.
There are no raw magic numbers in this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from chemrefine.constants import (
    BOHR_TO_ANGSTROM,
    DEFAULT_TEMPERATURE_K,
    HARTREE_PER_BOHR_TO_EV_PER_A,
    HARTREE_TO_EV,
    HARTREE_TO_KCALMOL,
    HARTREE_TO_KJMOL,
    KCALMOL_TO_HARTREE,
    KJMOL_TO_HARTREE,
    KJMOL_TO_KCALMOL,
    R_KCALMOL_K,
)

# Re-exported so callers can ``from chemrefine.units import DEFAULT_TEMPERATURE_K``
# without learning where the canonical scalar lives.
__all__ = [
    "DEFAULT_TEMPERATURE_K",
    "R_KCALMOL_K",
    "boltzmann_weights",
    "convert",
]

T = TypeVar("T", float, int, ArrayLike)

# ---------------------------------------------------------------------------
# Conversion table
# ---------------------------------------------------------------------------

_CONVERSIONS: Mapping[tuple[str, str], float] = {
    # Energy
    ("hartree", "kcal/mol"): HARTREE_TO_KCALMOL,
    ("kcal/mol", "hartree"): KCALMOL_TO_HARTREE,
    ("hartree", "kj/mol"): HARTREE_TO_KJMOL,
    ("kj/mol", "hartree"): KJMOL_TO_HARTREE,
    ("kj/mol", "kcal/mol"): KJMOL_TO_KCALMOL,
    ("kcal/mol", "kj/mol"): 1.0 / KJMOL_TO_KCALMOL,
    ("hartree", "ev"): HARTREE_TO_EV,
    ("ev", "hartree"): 1.0 / HARTREE_TO_EV,
    # Length
    ("bohr", "angstrom"): BOHR_TO_ANGSTROM,
    ("angstrom", "bohr"): 1.0 / BOHR_TO_ANGSTROM,
    # Gradient / force
    ("hartree/bohr", "ev/angstrom"): HARTREE_PER_BOHR_TO_EV_PER_A,
    ("ev/angstrom", "hartree/bohr"): 1.0 / HARTREE_PER_BOHR_TO_EV_PER_A,
}

_ALIASES: Mapping[str, str] = {
    "ha": "hartree",
    "a.u.": "hartree",
    "kcal": "kcal/mol",
    "kj": "kj/mol",
    "a": "angstrom",
    "å": "angstrom",
    "ang": "angstrom",
}


def _normalize(unit: str) -> str:
    """Return the canonical lowercase form for a unit string."""
    key = unit.strip().lower()
    return _ALIASES.get(key, key)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def convert(value: T, from_unit: str, to_unit: str) -> T:
    """Convert ``value`` from ``from_unit`` to ``to_unit``.

    Scalar inputs return scalars; array-likes return :class:`numpy.ndarray`
    of ``float64``. Raises :class:`ValueError` when the requested pair
    is not in the lookup table.
    """
    src = _normalize(from_unit)
    dst = _normalize(to_unit)
    if src == dst:
        return value
    factor = _CONVERSIONS.get((src, dst))
    if factor is None:
        raise ValueError(f"unknown unit conversion: {from_unit!r} → {to_unit!r}")
    if isinstance(value, (int, float)):
        return float(value) * factor  # type: ignore[return-value]
    return np.asarray(value, dtype=np.float64) * factor  # type: ignore[return-value]


def boltzmann_weights(
    energies_kcal: ArrayLike,
    temperature_k: float = DEFAULT_TEMPERATURE_K,
) -> NDArray[np.float64]:
    """Return normalized Boltzmann weights for energies in kcal/mol.

    The input is used as-is — callers that want *relative* energies
    should subtract the minimum first. Output sums to ``1.0`` (or to
    ``0.0`` if every weight underflows to zero).
    """
    arr = np.asarray(energies_kcal, dtype=np.float64)
    weights = np.exp(-arr / (R_KCALMOL_K * temperature_k))
    total = weights.sum()
    if total == 0.0:
        return weights
    return weights / total

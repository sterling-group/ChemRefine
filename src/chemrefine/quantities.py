"""Physical constants, derived conversions, and unit helpers.

ChemRefine's numerical API surface. Every CODATA-anchored scalar,
every derived energy / length / gradient factor, the :func:`convert`
function with its alias-aware lookup, and :func:`boltzmann_weights`
all live here so callers have one place to look and contributors
have one place to extend.

Adding a new constant is a single line at module scope; the public
surface is discovered by Python's default module visibility (no
``__all__`` ceremony required). Every constant is annotated
:class:`typing.Final` so a type checker flags accidental
reassignment.

Values follow CODATA 2022 unless noted; ``HARTREE_TO_EV`` and
``BOHR_TO_ANGSTROM`` match the precision ORCA itself uses (CODATA
2018) so back-and-forth conversions round-trip cleanly with ORCA's
output.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, TypeVar, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

# ---------------------------------------------------------------------------
# Physical anchors (CODATA 2022 / SI base; exact since the 2019 redefinition)
# ---------------------------------------------------------------------------

AVOGADRO: Final[float] = 6.02214076e23
"""Avogadro number, mol⁻¹."""

BOLTZMANN: Final[float] = 1.380649e-23
"""Boltzmann constant, J/K."""

MOLAR_GAS_CONSTANT: Final[float] = AVOGADRO * BOLTZMANN
"""Molar gas constant ``R``, J/(mol·K)."""

# ---------------------------------------------------------------------------
# Atomic-unit anchors
# ---------------------------------------------------------------------------

HARTREE_TO_J: Final[float] = 4.3597447222060e-18
"""1 Hartree in Joules (CODATA 2022)."""

HARTREE_TO_EV: Final[float] = 27.211386245988
"""1 Hartree in electron-volts (matches ORCA's internal value)."""

BOHR_TO_ANGSTROM: Final[float] = 0.529177210903
"""1 Bohr in Ångström (matches ORCA's internal value)."""

# ---------------------------------------------------------------------------
# Derived energy conversions
# ---------------------------------------------------------------------------

HARTREE_TO_KJ: Final[float] = HARTREE_TO_J * 1.0e-3
HARTREE_TO_KJMOL: Final[float] = HARTREE_TO_KJ * AVOGADRO
"""1 Hartree in kJ/mol (≈ 2625.5311584660003)."""

KJMOL_TO_KCALMOL: Final[float] = 1.0 / 4.184
"""Thermochemical-calorie conversion (IUPAC ``cal_th = 4.184 J``)."""

HARTREE_TO_KCALMOL: Final[float] = HARTREE_TO_KJMOL * KJMOL_TO_KCALMOL
"""1 Hartree in kcal/mol (≈ 627.5094740631)."""

KJMOL_TO_HARTREE: Final[float] = 1.0 / HARTREE_TO_KJMOL
KCALMOL_TO_HARTREE: Final[float] = 1.0 / HARTREE_TO_KCALMOL

# ---------------------------------------------------------------------------
# Derived gradient unit
# ---------------------------------------------------------------------------

HARTREE_PER_BOHR_TO_EV_PER_A: Final[float] = HARTREE_TO_EV / BOHR_TO_ANGSTROM
"""ORCA gradients are in Hartree/Bohr; ASE forces are in eV/Å.

Multiply ``-∂E/∂x`` by this factor to get an ASE-compatible force.
"""

# ---------------------------------------------------------------------------
# Convenience: R in kcal/(mol·K) for Boltzmann statistics on kcal energies
# ---------------------------------------------------------------------------

R_KCALMOL_K: Final[float] = MOLAR_GAS_CONSTANT * 1.0e-3 * KJMOL_TO_KCALMOL
"""Gas constant, kcal/(mol·K) — equals ≈ 1.98720425e-3."""

# ---------------------------------------------------------------------------
# Standard reference temperature
# ---------------------------------------------------------------------------

DEFAULT_TEMPERATURE_K: Final[float] = 298.15
"""IUPAC standard ambient temperature (25 °C). Used as the default for
Boltzmann statistics and as the default value for ``temperature_k`` in
:class:`chemrefine.config._SampleBase`."""


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

T = TypeVar("T", float, int, ArrayLike)


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
        # An int input converts to a float — the T=int constraint can't hold.
        return float(value) * factor  # type: ignore[return-value]
    return np.asarray(value, dtype=np.float64) * factor


def boltzmann_weights(
    energies_kcal: ArrayLike,
    temperature_k: float = DEFAULT_TEMPERATURE_K,
) -> NDArray[np.float64]:
    """Return normalized Boltzmann weights for energies in kcal/mol.

    Absolute or relative energies both work: the minimum is subtracted
    internally before exponentiating. That shift cancels in the normalization,
    so it never changes the result — it only keeps ``exp`` in range. Passing
    raw (large negative) absolute energies would otherwise overflow to ``inf``
    and normalize to ``nan``.

    Output sums to ``1.0`` (or to ``0.0`` if every weight underflows to zero).
    """
    arr = np.asarray(energies_kcal, dtype=np.float64)
    if arr.size:
        arr = arr - arr.min()
    weights = np.exp(-arr / (R_KCALMOL_K * temperature_k))
    total = weights.sum()
    if total == 0.0:
        return weights
    return cast(NDArray[np.float64], weights / total)

"""Energy-unit conversions and physical constants used across ChemRefine.

ChemRefine stores energies internally in Hartree (engine native) and
converts to kcal/mol for user-facing reports. Boltzmann statistics use the
gas constant in kcal/(mol·K). Default temperature is the standard 298.15 K
(25 °C) — overridable per call.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

HARTREE_TO_KCAL_MOL: float = 2625.49964 / 4.184
R_KCAL_MOL_K: float = 8.314462618e-3 / 4.184
DEFAULT_TEMPERATURE_K: float = 298.15


def hartree_to_kcal(energies: ArrayLike) -> NDArray[np.float64]:
    """Convert energies in Hartree to kcal/mol."""
    return np.asarray(energies, dtype=np.float64) * HARTREE_TO_KCAL_MOL


def kcal_to_hartree(energies: ArrayLike) -> NDArray[np.float64]:
    """Convert energies in kcal/mol to Hartree."""
    return np.asarray(energies, dtype=np.float64) / HARTREE_TO_KCAL_MOL


def boltzmann_weights(
    energies_kcal: ArrayLike,
    temperature_k: float = DEFAULT_TEMPERATURE_K,
) -> NDArray[np.float64]:
    """Return normalized Boltzmann weights for an array of relative energies (kcal/mol).

    The input is taken as-is (no auto-shift): callers that want relative
    energies should subtract the minimum first. Returns weights that sum
    to 1.0.
    """
    arr = np.asarray(energies_kcal, dtype=np.float64)
    weights = np.exp(-arr / (R_KCAL_MOL_K * temperature_k))
    total = weights.sum()
    if total == 0.0:
        return weights
    return weights / total

"""Physical constants and conversion factors used across ChemRefine.

Single source of truth for every number that isn't a runtime input.
Engines and helpers must import from here rather than re-define a
local copy — duplicated values drift, this module does not.

Values are CODATA 2022 unless noted; ``HARTREE_TO_EV`` and
``BOHR_TO_ANGSTROM`` match the precision ORCA itself uses (CODATA
2018) so back-and-forth conversions round-trip cleanly with ORCA's
output.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Avogadro / Boltzmann (SI base, exact since the 2019 redefinition)
# ---------------------------------------------------------------------------

AVOGADRO: float = 6.02214076e23
"""Avogadro number, mol⁻¹."""

BOLTZMANN: float = 1.380649e-23
"""Boltzmann constant, J/K."""

MOLAR_GAS_CONSTANT: float = AVOGADRO * BOLTZMANN
"""Molar gas constant ``R``, J/(mol·K)."""

# ---------------------------------------------------------------------------
# Atomic-unit anchors
# ---------------------------------------------------------------------------

HARTREE_TO_J: float = 4.3597447222060e-18
"""1 Hartree in Joules (CODATA 2022)."""

HARTREE_TO_EV: float = 27.211386245988
"""1 Hartree in electron-volts (matches ORCA's internal value)."""

BOHR_TO_ANGSTROM: float = 0.529177210903
"""1 Bohr in Ångström (matches ORCA's internal value)."""

# ---------------------------------------------------------------------------
# Derived energy conversions
# ---------------------------------------------------------------------------

HARTREE_TO_KJ: float = HARTREE_TO_J * 1.0e-3
HARTREE_TO_KJMOL: float = HARTREE_TO_KJ * AVOGADRO
"""1 Hartree in kJ/mol (≈ 2625.5311584660003)."""

KJMOL_TO_KCALMOL: float = 1.0 / 4.184
"""Thermochemical-calorie conversion (IUPAC ``cal_th = 4.184 J``)."""

HARTREE_TO_KCALMOL: float = HARTREE_TO_KJMOL * KJMOL_TO_KCALMOL
"""1 Hartree in kcal/mol (≈ 627.5094740631)."""

KJMOL_TO_HARTREE: float = 1.0 / HARTREE_TO_KJMOL
KCALMOL_TO_HARTREE: float = 1.0 / HARTREE_TO_KCALMOL

# ---------------------------------------------------------------------------
# Derived gradient unit
# ---------------------------------------------------------------------------

HARTREE_PER_BOHR_TO_EV_PER_A: float = HARTREE_TO_EV / BOHR_TO_ANGSTROM
"""ORCA gradients are in Hartree/Bohr; ASE forces are in eV/Å.

Multiply ``-∂E/∂x`` by this factor to get an ASE-compatible force.
"""

# ---------------------------------------------------------------------------
# Convenience: R in kcal/(mol·K) for Boltzmann statistics on kcal energies
# ---------------------------------------------------------------------------

R_KCALMOL_K: float = MOLAR_GAS_CONSTANT * 1.0e-3 * KJMOL_TO_KCALMOL
"""Gas constant, kcal/(mol·K) — equals ≈ 1.98720425e-3."""

# ---------------------------------------------------------------------------
# Standard reference temperature
# ---------------------------------------------------------------------------

DEFAULT_TEMPERATURE_K: float = 298.15
"""IUPAC standard ambient temperature (25 °C). Used as the default for
Boltzmann statistics in :mod:`chemrefine.units` and as the default
value for ``temperature_k`` in
:class:`chemrefine.config._SampleBase`."""

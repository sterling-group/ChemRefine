"""Parse energies from an ORCA ``.out``: the electronic energy + thermochemistry.

One section, one module. ``FINAL SINGLE POINT ENERGY`` and the ``THERMOCHEMISTRY`` block
(Gibbs / enthalpy / electronic+ZPE) are all *energies*, so they live together here (the
thermochemistry moved out of :mod:`chemrefine.engines.orca.frequencies`, which now owns only
the vibrational table + normal modes).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_FINAL_ENERGY_RE = re.compile(
    r"FINAL SINGLE POINT ENERGY(?:\s*\(From external program\))?\s+(-?\d+\.\d+)"
)

_THERMO_MARKER = "THERMOCHEMISTRY"
# ORCA prints "<label>   ...   <value> Eh"; tolerate the dotted padding.
_GIBBS_RE = re.compile(r"Final Gibbs free energy\s*\.*\s*(-?\d+\.\d+)")
_ENTHALPY_RE = re.compile(r"Total Enthalpy\s*\.*\s*(-?\d+\.\d+)")
_ZPE_RE = re.compile(r"Zero point energy\s*\.*\s*(-?\d+\.\d+)")


def parse_final_energy_from_text(text: str) -> float | None:
    """Return the **last** ``FINAL SINGLE POINT ENERGY`` (Hartree), or ``None`` if absent.

    The last is taken because a geometry optimisation re-prints it as it iterates (and a
    PES segment ends with the converged point's energy).
    """
    matches = _FINAL_ENERGY_RE.findall(text)
    return float(matches[-1]) if matches else None


@dataclass(frozen=True)
class Thermochemistry:
    """Absolute thermochemistry (Hartree) extracted from an ORCA freq output.

    Any quantity whose line is absent from the block is ``None``.
    """

    gibbs_hartree: float | None
    enthalpy_hartree: float | None
    energy_zpe_hartree: float | None


def parse_thermochemistry_from_text(
    text: str, *, electronic_hartree: float
) -> Thermochemistry | None:
    """Return Gibbs / enthalpy / electronic+ZPE (Hartree), or ``None`` if no block.

    ORCA's ``THERMOCHEMISTRY`` section prints an absolute ``Final Gibbs free energy`` and
    ``Total Enthalpy``; ``Zero point energy`` is the (positive) ZPE *correction*, so
    electronic+ZPE = ``electronic_hartree`` + that correction. Returns ``None`` when the output
    has no thermochemistry block at all (e.g. a plain ``opt_sp`` with no frequencies).
    """
    if _THERMO_MARKER not in text:
        return None
    # Take the last of each (a compound job may print thermochemistry more than once; the final
    # block is the one we want), matching the energy parser.
    gibbs = _GIBBS_RE.findall(text)
    enthalpy = _ENTHALPY_RE.findall(text)
    zpe = _ZPE_RE.findall(text)
    return Thermochemistry(
        gibbs_hartree=float(gibbs[-1]) if gibbs else None,
        enthalpy_hartree=float(enthalpy[-1]) if enthalpy else None,
        energy_zpe_hartree=(electronic_hartree + float(zpe[-1])) if zpe else None,
    )

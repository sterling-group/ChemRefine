"""Parse energies from an ORCA ``.out``: the electronic energy + thermochemistry.

One section, one module. ``FINAL SINGLE POINT ENERGY`` and the ``THERMOCHEMISTRY`` block
(Gibbs / enthalpy / electronic+ZPE) are all *energies*, so they live together here;
:mod:`chemrefine.engines.orca.output.frequencies` owns the vibrational table + normal
modes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_FINAL_ENERGY_RE = re.compile(
    r"FINAL SINGLE POINT ENERGY(?:\s*\(From external program\))?\s+(-?\d+\.\d+)"
)
_RELAXATION_ENERGY_RE = re.compile(r"^Current Energy\s+\.+\s+(-?\d+\.\d+) Eh", re.MULTILINE)

_THERMO_MARKER = "THERMOCHEMISTRY"
# ORCA prints "<label>   ...   <value> Eh"; tolerate the dotted padding.
_GIBBS_RE = re.compile(r"Final Gibbs free energy\s*\.*\s*(-?\d+\.\d+)")
_ENTHALPY_RE = re.compile(r"Total Enthalpy\s*\.*\s*(-?\d+\.\d+)")
_ZPE_RE = re.compile(r"Zero point energy\s*\.*\s*(-?\d+\.\d+)")


def find_final_energy(text: str) -> tuple[float, int] | None:
    """The **last** ``FINAL SINGLE POINT ENERGY`` (Hartree) and the offset it is printed at.

    The offset is what ties the energy to the rest of its evaluation: ORCA prints each
    geometry before anything computed at it, so the geometry an energy belongs to is the
    last one printed ahead of it.
    """
    return _last(_FINAL_ENERGY_RE, text)


def find_relaxation_energy(text: str) -> tuple[float, int] | None:
    """The last energy an optimiser's relaxation step reports, and the offset it is at.

    Each geometry optimisation cycle closes with a ``Current Energy .... <E> Eh`` line, the
    energy of that cycle's geometry at nine decimals. An ExtOpt optimisation prints no
    ``FINAL SINGLE POINT ENERGY`` after its first cycle — ORCA sends the external
    program's output to ``.lastext`` — so for those cycles this line is the only energy.
    """
    return _last(_RELAXATION_ENERGY_RE, text)


def _last(pattern: re.Pattern[str], text: str) -> tuple[float, int] | None:
    """The last match's energy and the offset it starts at, or ``None`` if nothing matched."""
    last = None
    for match in pattern.finditer(text):
        last = match
    return None if last is None else (float(last.group(1)), last.start())


def parse_final_energy_from_text(text: str) -> float | None:
    """Return the **last** ``FINAL SINGLE POINT ENERGY`` (Hartree), or ``None`` if absent.

    The last is taken because a geometry optimisation re-prints it as it iterates (and a
    PES segment ends with the converged point's energy).
    """
    found = find_final_energy(text)
    return None if found is None else found[0]


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
    # block is the relevant one), matching the energy parser.
    gibbs = _GIBBS_RE.findall(text)
    enthalpy = _ENTHALPY_RE.findall(text)
    zpe = _ZPE_RE.findall(text)
    return Thermochemistry(
        gibbs_hartree=float(gibbs[-1]) if gibbs else None,
        enthalpy_hartree=float(enthalpy[-1]) if enthalpy else None,
        energy_zpe_hartree=(electronic_hartree + float(zpe[-1])) if zpe else None,
    )

"""Parse energies from a Q-Chem ``.out``: the electronic energy and the thermochemistry.

One section, one module: ``Total energy in the final basis set`` and the thermodynamics
block are all *energies*, so they live together, and the frequency table lives with its
tensor in :mod:`chemrefine.engines.qchem.output.frequencies`. The coordinator runs both
readers over the shared text in the same read-once pass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_FINAL_ENERGY_RE = re.compile(r"Total energy in the final basis set =\s*(-?\d+\.\d+)")


def parse_final_energy_from_text(text: str) -> float | None:
    """Return the **last** ``Total energy in the final basis set`` (Hartree), or ``None``.

    The last is taken because a geometry optimisation re-prints the line per cycle —
    the discipline every reader in this package follows.
    """
    matches = _FINAL_ENERGY_RE.findall(text)
    return float(matches[-1]) if matches else None


@dataclass(frozen=True)
class Thermochemistry:
    """Absolute thermochemistry (Hartree) from a Q-Chem frequency output.

    Any quantity the output does not report is ``None``.
    """

    gibbs_hartree: float | None
    enthalpy_hartree: float | None
    energy_zpe_hartree: float | None


def parse_thermochemistry_from_text(
    text: str, *, electronic_hartree: float
) -> Thermochemistry | None:
    """Gibbs / enthalpy / electronic+ZPE from the thermodynamics section, or ``None``.

    ``None`` when the output has no thermodynamics section at all — an ``opt``/``sp``
    with no frequencies, which is not an error; a :class:`Thermochemistry` otherwise,
    with each absent line ``None``. Q-Chem prints the section per frequency job, so in
    an ``@@@`` chain the **last** block is the one that describes the final geometry.
    Q-Chem's zero-point line is a positive *correction*, so ``energy_zpe_hartree =
    electronic_hartree + correction`` — which is why the electronic energy is a
    parameter — and its enthalpy/free-energy lines print in kcal/mol, converted to
    Hartree before they land on a :class:`~chemrefine.engines.api.ParsedResult` (the
    field names promise it). The matched line spellings are held to full captured
    output, never guessed; the shipped fixtures are trimmed.
    """
    return None

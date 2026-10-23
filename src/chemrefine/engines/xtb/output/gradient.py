"""Read xtb's Turbomole-format ``gradient`` file: the energy, geometry and gradient of one cycle.

``--grad`` writes it in the run's working directory, as ``<namespace>.gradient``::

    $grad
      cycle =      1    SCF energy =    -5.08487182319   |dE/dxyz| =  0.006329
        0.00000000000000      0.00000000000000      0.22676713495445      O
        ...                                                                    (N rows, bohr)
       1.3174938250541E-17  -1.8094238989079E-17   3.5163861354515E-03
        ...                                                                    (N rows, Eh/bohr)
    $end

xtb appends a cycle to a file that is already there, so the **last** cycle is read. The
file carries its own energy and coordinates, which is what lets the coordinator check that
the gradient belongs to the structure it is attached to.

Chosen over the ``.engrad`` file xtb writes beside it because its name does not depend on
the geometry file's, and its numbers carry more digits (the gradient to fourteen
significant figures, the coordinates to fourteen decimals).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

GRADIENT_FILE = "gradient"
"""The name ``--grad`` gives the file in the job's working directory, after the run's
``--namespace`` and a dot (``step1_0.gradient``)."""

_CYCLE_RE = re.compile(r"^\s*cycle\s*=\s*\d+\s+SCF energy\s*=\s*(\S+)", re.MULTILINE)


@dataclass(frozen=True)
class GradientCycle:
    """One cycle of a ``gradient`` file, in its own units (Hartree, bohr, Eh/bohr)."""

    energy_hartree: float
    symbols: tuple[str, ...]
    positions_bohr: NDArray[np.float64]
    gradient_hartree_per_bohr: NDArray[np.float64]


def _floats(tokens: list[str], line: str) -> list[float]:
    """The tokens as finite floats — Fortran ``D`` exponents read as ``E`` — or ``ValueError``."""
    values = [float(token.replace("D", "E").replace("d", "e")) for token in tokens]
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite value in {line.strip()!r}")
    return values


def parse_gradient_text(text: str) -> GradientCycle:
    """The last cycle of a ``gradient`` file's text.

    Raises :class:`ValueError` when there is no cycle, a row does not read, a value is not
    finite (``float()`` accepts ``nan``), or the gradient rows do not number the atoms: a
    file cut off mid-write would otherwise hand on a gradient for fewer atoms than the
    structure has. The caller files the error as the output's.
    """
    cycles = list(_CYCLE_RE.finditer(text))
    if not cycles:
        raise ValueError("no `cycle =` block")
    last = cycles[-1]
    energy = _floats([last.group(1)], last.group(0))[0]
    symbols: list[str] = []
    positions: list[list[float]] = []
    gradient: list[list[float]] = []
    for line in text[last.end() :].splitlines()[1:]:
        parts = line.split()
        if not parts or parts[0].startswith("$") or _CYCLE_RE.match(line):
            break
        if len(parts) == 4 and not gradient:
            positions.append(_floats(parts[:3], line))
            symbols.append(parts[3].capitalize())
        elif len(parts) == 3:
            gradient.append(_floats(parts, line))
        else:
            raise ValueError(f"unreadable row {line.strip()!r}")
    if not symbols or len(gradient) != len(symbols):
        raise ValueError(f"{len(gradient)} gradient rows for {len(symbols)} atoms")
    return GradientCycle(
        energy_hartree=energy,
        symbols=tuple(symbols),
        positions_bohr=np.array(positions, dtype=np.float64),
        gradient_hartree_per_bohr=np.array(gradient, dtype=np.float64),
    )

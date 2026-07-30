"""Parse the ``CARTESIAN GRADIENT`` block from an ORCA ``.out`` into forces (eV/Å).

One section, one module. ORCA prints the gradient (``∂E/∂x``, Hartree/Bohr); forces are
``F = -∂E/∂x`` converted to ASE-native eV/Å.
"""

from __future__ import annotations

import re

import numpy as np
from numpy.typing import NDArray

from chemrefine.quantities import HARTREE_PER_BOHR_TO_EV_PER_A

_GRAD_BLOCK_RE = re.compile(
    r"CARTESIAN GRADIENT\s*\n-+\n((?:.*?\n)+?)-+\n",
    re.DOTALL,
)
_GRAD_LINE_RE = re.compile(
    r"^\s*(\d+)\s+[A-Za-z]{1,3}\s*:\s*"
    r"([+-]?\d*\.?\d+(?:[EeDd][+-]?\d+)?)\s+"
    r"([+-]?\d*\.?\d+(?:[EeDd][+-]?\d+)?)\s+"
    r"([+-]?\d*\.?\d+(?:[EeDd][+-]?\d+)?)\s*$"
)


def parse_forces_from_text(text: str, *, to_ev_per_A: bool = True) -> NDArray[np.float64] | None:
    """Return the **last** ``CARTESIAN GRADIENT`` block as forces, or ``None``.

    ``F = -∂E/∂x``, converted to eV/Å unless ``to_ev_per_A`` is ``False``.
    """
    blocks = _GRAD_BLOCK_RE.findall(text)
    if not blocks:
        return None
    rows: list[list[float]] = []
    for line in blocks[-1].strip().splitlines():
        m = _GRAD_LINE_RE.match(line)
        if not m:
            continue
        # The regex admits a Fortran `D` exponent in either case, so the rewrite has to
        # cover both — accepting a spelling the conversion then cannot parse would turn a
        # gradient row into a bare ValueError.
        dx, dy, dz = (float(m.group(i).replace("D", "E").replace("d", "e")) for i in (2, 3, 4))
        fx, fy, fz = -dx, -dy, -dz
        if to_ev_per_A:
            fx *= HARTREE_PER_BOHR_TO_EV_PER_A
            fy *= HARTREE_PER_BOHR_TO_EV_PER_A
            fz *= HARTREE_PER_BOHR_TO_EV_PER_A
        rows.append([fx, fy, fz])
    if not rows:
        return None
    return np.array(rows, dtype=np.float64)

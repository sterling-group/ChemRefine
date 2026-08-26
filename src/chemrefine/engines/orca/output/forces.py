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


def parse_forces_from_text(
    text: str, *, n_atoms: int, to_ev_per_A: bool = True
) -> NDArray[np.float64] | None:
    """Return the **last** ``CARTESIAN GRADIENT`` block as forces, or ``None``.

    ``F = -∂E/∂x``, converted to eV/Å unless ``to_ev_per_A`` is ``False``. ``None`` means
    the output has no gradient block at all — a plain single point, which is not an error.

    Held to the two rules every other reader of numbers in this package already obeys, and
    for the reasons they give. Both raise :class:`ValueError`, which
    :func:`~chemrefine.engines.orca.output.coordinator.parse_dft_from_text` turns into an
    :class:`~chemrefine.errors.OutputParseError` — so a bad gradient becomes *this
    structure's* ledgered failure rather than something that surfaces a step later:

    * **Finite.** ``float()`` accepts an overflowing exponent and yields ``inf``, exactly as
      it accepts ``nan``. A non-finite force is stored unexamined by the ``arrays.npz``
      sidecar and is what an ``mlip-train`` step would go on to fit
      (:func:`chemrefine.engines._script.output._require_finite` states the rule;
      :func:`chemrefine.cache._require_finite_arrays` is the backstop, and reaching it costs
      the whole step's results rather than one structure).
    * **One row per atom.** A row the pattern cannot read is *skipped*, because it has to
      be: ORCA closes the block with its own summary lines (``Difference to translation
      invariance``, ``Norm of the Cartesian gradient``), which are not atom rows and appear
      in all 227 recorded blocks. That makes the count the only thing that can tell a
      summary line from a lost atom — an unreadable row (a ``*****`` field overflow) would
      otherwise yield a short array against a full geometry, which nothing downstream
      re-checks: :attr:`~chemrefine.state.Structure.forces_ev_per_a` declares no shape, the
      finiteness backstop passes it, and FAIRChem's dataset writer stores it.
      :func:`chemrefine.engines._script.contract.forces_from_gradient` holds the other forces
      reader to the same count, and is where this wording comes from.
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
        if not np.isfinite([dx, dy, dz]).all():
            raise ValueError(f"non-finite gradient component in {line.strip()!r}")
        fx, fy, fz = -dx, -dy, -dz
        if to_ev_per_A:
            fx *= HARTREE_PER_BOHR_TO_EV_PER_A
            fy *= HARTREE_PER_BOHR_TO_EV_PER_A
            fz *= HARTREE_PER_BOHR_TO_EV_PER_A
        rows.append([fx, fy, fz])
    if len(rows) != n_atoms:
        raise ValueError(
            f"read {len(rows)} gradient row(s) for a {n_atoms}-atom structure — a row the "
            f"pattern could not read (a `*****` field overflow is the usual cause) is "
            f"indistinguishable from a missing atom"
        )
    return np.array(rows, dtype=np.float64)

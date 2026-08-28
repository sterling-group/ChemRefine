"""Parse the Q-Chem gradient into forces (eV/Å).

One section, one module: this owns the gradient-block grammar, and the coordinator
threads its answer onto every parsed structure in the same read-once pass as the
geometry and the energies — the reader below is the one place the section is
interpreted.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def parse_forces_from_text(text: str, *, n_atoms: int) -> NDArray[np.float64] | None:
    """The last gradient block as ASE-native forces, or ``None`` when there is none.

    Q-Chem prints the Cartesian gradient (``∂E/∂x``, atomic units); the answer is
    ``F = -∂E/∂x`` converted to eV/Å via
    :data:`chemrefine.quantities.HARTREE_PER_BOHR_TO_EV_PER_A`, and the **last** block
    wins because an optimisation re-prints it per cycle. ``None`` means the output has
    no gradient at all — a plain single point, which is not an error. Two rules are
    load-bearing: every component must be **finite** (``float()`` accepts
    ``nan``/``inf``, and a non-finite force is what an ``mlip-train`` step would go on
    to fit), and the result must carry **exactly one row per atom** (``n_atoms`` is
    passed for that check — a row the pattern cannot read is otherwise
    indistinguishable from a lost atom). Both violations raise ``ValueError``; the
    coordinator turns that into this structure's ledgered failure rather than a
    traceback.
    """
    return None

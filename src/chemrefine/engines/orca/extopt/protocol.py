"""ExtOpt file-format helpers.

TODO: implement against the v3 :file:`utils_extopt.py` on ``main``. The
canonical layout is small but format-sensitive — needs a real ORCA run
to capture the exact line endings, decimal precision, and atom-block
shape ORCA expects.
"""

from __future__ import annotations

from pathlib import Path

from numpy.typing import NDArray


def read_extinp(path: str | Path) -> tuple[tuple[str, ...], NDArray]:
    """Read an ``.extinp.tmp`` file written by ORCA.

    Should return ``(symbols, positions_angstrom)``. Placeholder.
    """
    raise NotImplementedError(
        "read_extinp not yet ported — see TODO in engines/orca/extopt/protocol.py"
    )


def write_engrad(
    *,
    path: str | Path,
    n_atoms: int,
    energy_hartree: float,
    gradients_hartree_per_bohr: NDArray,
) -> Path:
    """Write an ``.engrad`` file that ORCA can read back. Placeholder."""
    raise NotImplementedError(
        "write_engrad not yet ported — see TODO in engines/orca/extopt/protocol.py"
    )


def write_wrapper_script(
    *,
    path: str | Path,
    server_url: str,
) -> Path:
    """Write the ``ProgExt`` wrapper script ORCA invokes per optimisation step.

    The script POSTs the ``.extinp.tmp`` to the external server and
    receives an ``.engrad`` back. Placeholder.
    """
    raise NotImplementedError(
        "write_wrapper_script not yet ported — see TODO in engines/orca/extopt/protocol.py"
    )

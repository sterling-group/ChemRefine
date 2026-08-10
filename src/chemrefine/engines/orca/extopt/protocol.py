"""ExtOpt file-format helpers.

ORCA's ``ProgExt`` model invokes the wrapper script once per
optimization step. The wrapper:

1. Reads the ``.extinp.tmp`` ORCA wrote (xyz filename, charge, mult,
   ncores, dograd flag).
2. Reads the referenced ``.xyz``.
3. Sends both to the long-running ExtOpt server.
4. Writes the returned energy + gradient to ``.engrad`` so ORCA can
   take its next step.

Unit conversions go through :mod:`chemrefine.quantities` so the values
match the rest of ChemRefine.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chemrefine.engines._backend_server.base import CalculationData
from chemrefine.errors import JobFailureError

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

# ORCA ``ProgExt`` file-format suffixes: ORCA hands the wrapper ``X.extinp.tmp``
# and reads ``X.engrad`` back. Single source for the protocol's filenames.
EXTINP_SUFFIX = ".extinp.tmp"
ENGRAD_SUFFIX = ".engrad"

_EXTINP_LINES = 5
"""Lines ORCA writes to a ``.extinp.tmp``: xyz name, charge, mult, ncores, dograd."""


def read_extinp(
    inpfile: str | Path,
    *,
    settings: dict[str, Any] | None = None,
) -> CalculationData:
    """Parse an ORCA-written ``.extinp.tmp`` and its referenced ``.xyz``.

    The ``.extinp.tmp`` layout (one value per line, optional ``#``
    comments) is::

        struct.xyz   # XYZ filename
        0            # charge
        1            # multiplicity
        4            # ncores
        1            # dograd (1 = yes, 0 = no)

    ``settings`` (backend knobs from the wrapper-script CLI) are
    threaded through unchanged.
    """
    inpfile = Path(inpfile)
    lines = inpfile.read_text(encoding="utf-8").splitlines()
    # A truncated file — ORCA killed mid-write, a full disk — would otherwise raise
    # IndexError inside the wrapper script, so ORCA gets no `.engrad` and the step
    # fails with a bare traceback in the runlog instead of a classified failure.
    if len(lines) < _EXTINP_LINES:
        raise JobFailureError(
            f"truncated ExtOpt input {inpfile}: expected {_EXTINP_LINES} lines, got {len(lines)}"
        )
    xyz_name = lines[0].split("#")[0].strip()
    charge = int(lines[1].split("#")[0].strip())
    mult = int(lines[2].split("#")[0].strip())
    nthreads = int(lines[3].split("#")[0].strip())
    dograd = bool(int(lines[4].split("#")[0].strip()))
    xyz_path = inpfile.parent / xyz_name if not Path(xyz_name).is_absolute() else Path(xyz_name)
    symbols, positions = _read_xyz(xyz_path)
    return CalculationData(
        symbols=tuple(symbols),
        positions_angstrom=positions,
        charge=charge,
        multiplicity=mult,
        nthreads=nthreads,
        dograd=dograd,
        settings=dict(settings) if settings else {},
    )


def _read_xyz(xyz_path: Path) -> tuple[list[str], NDArray[np.float64]]:
    """Return ``(symbols, positions)`` from a plain-format ``.xyz``.

    Held to the same rule as the ``.extinp.tmp`` header above, because it is the same
    file event: ORCA writes both per ProgExt call, so the kill or full disk that
    truncates one truncates the other. Unguarded, a short file surfaced as a bare
    ``IndexError`` in the runlog — a message about a list, three frames from the file
    that caused it — where the header's guard names the file and the cause.
    """
    import numpy as np

    lines = xyz_path.read_text(encoding="utf-8").splitlines()
    try:
        natoms = int(lines[0].strip())
    except (IndexError, ValueError) as e:
        raise JobFailureError(
            f"malformed ExtOpt geometry {xyz_path}: the first line is not an atom count"
        ) from e
    if len(lines) < 2 + natoms:
        raise JobFailureError(
            f"truncated ExtOpt geometry {xyz_path}: expected {natoms} atom row(s), "
            f"got {max(0, len(lines) - 2)}"
        )
    symbols: list[str] = []
    coords: list[list[float]] = []
    for line in lines[2 : 2 + natoms]:
        parts = line.split()
        if len(parts) < 4:
            raise JobFailureError(f"malformed ExtOpt geometry {xyz_path}: bad atom row {line!r}")
        try:
            row = [float(x) for x in parts[1:4]]
        except ValueError as e:
            raise JobFailureError(
                f"malformed ExtOpt geometry {xyz_path}: bad atom row {line!r}"
            ) from e
        symbols.append(parts[0])
        coords.append(row)
    return symbols, np.asarray(coords, dtype=np.float64)


def write_engrad(
    *,
    path: str | Path,
    n_atoms: int,
    energy_hartree: float,
    gradients_hartree_per_bohr: list[list[float]] | None,
    dograd: bool = True,
) -> Path:
    """Write an ``.engrad`` file ORCA can read back.

    The format is fixed by ORCA and looks like::

        #
        # Number of atoms
        #
        3
        #
        # Total energy [Eh]
        #
        -76.123456789012e+00
        #
        # Gradient [Eh/Bohr] A1X, A1Y, A1Z, ...
        #
        1.234567890123e-04
        ...

    The gradient block is omitted when ``dograd`` is false (ORCA's
    energy-only mode).
    """
    path = Path(path)
    lines = [
        "#",
        "# Number of atoms",
        "#",
        f"{n_atoms}",
        "#",
        "# Total energy [Eh]",
        "#",
        f"{energy_hartree:.12e}",
    ]
    if dograd:
        if gradients_hartree_per_bohr is None:
            raise ValueError("dograd=True but no gradient provided")
        lines.extend(["#", "# Gradient [Eh/Bohr] A1X, A1Y, A1Z, ...", "#"])
        for row in gradients_hartree_per_bohr:
            lines.extend(f"{component:.12e}" for component in row)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_wrapper_script(
    *,
    path: str | Path,
    backend: str,
    url_file: str | Path,
    extra_args: str = "",
) -> Path:
    """Write the ``ProgExt`` wrapper bash script ORCA invokes per step.

    The script reads the sidecar URL file (so the wrapper picks up the
    kernel-assigned port the server bound to), then execs the shared
    ``orca.extopt.bridge`` module to relay the call. The bridge runs with the
    orchestrator's own interpreter (``sys.executable``) — it needs chemrefine
    but no backend stack, and a bare ``python`` on PATH may be a system
    interpreter without chemrefine installed.
    """
    path = Path(path)
    extra = f" {extra_args}" if extra_args else ""
    script = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'URL_FILE="{url_file}"\n'
        'if [ ! -s "$URL_FILE" ]; then\n'
        '  echo "ChemRefine extopt: $URL_FILE missing or empty" >&2\n'
        "  exit 1\n"
        "fi\n"
        'SERVER_URL=$(cat "$URL_FILE")\n'
        f"exec {shlex.quote(sys.executable)} -m chemrefine.engines.orca.extopt.bridge "
        f'--backend {backend} --bind "$SERVER_URL"{extra} "$1"\n'
    )
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)
    return path

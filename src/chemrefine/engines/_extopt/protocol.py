"""ExtOpt file-format helpers.

ORCA's ``ProgExt`` model invokes the wrapper script once per
optimization step. The wrapper:

1. Reads the ``.extinp.tmp`` ORCA wrote (xyz filename, charge, mult,
   ncores, dograd flag).
2. Reads the referenced ``.xyz``.
3. Sends both to the long-running ExtOpt server.
4. Writes the returned energy + gradient to ``.engrad`` so ORCA can
   take its next step.

Unit conversions go through :mod:`chemrefine.constants` so the values
match the rest of ChemRefine.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from chemrefine.constants import BOHR_TO_ANGSTROM, HARTREE_TO_EV
from chemrefine.engines._extopt.base import CalculationData


def read_extinp(
    inpfile: str | Path,
    *,
    settings: dict | None = None,
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


def _read_xyz(xyz_path: Path) -> tuple[list[str], list[list[float]]]:
    """Return ``(symbols, positions)`` from a plain-format ``.xyz``."""
    import numpy as np

    with xyz_path.open(encoding="utf-8") as fh:
        natoms = int(fh.readline().strip())
        fh.readline()  # comment line
        symbols: list[str] = []
        coords: list[list[float]] = []
        for _ in range(natoms):
            parts = fh.readline().split()
            symbols.append(parts[0])
            coords.append([float(x) for x in parts[1:4]])
    return symbols, np.asarray(coords, dtype=float)


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
    ``_extopt.client`` module to relay the call.
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
        "exec python -m chemrefine.engines._extopt.client "
        f'--backend {backend} --bind "$SERVER_URL"{extra} "$1"\n'
    )
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)
    return path


# ---------------------------------------------------------------------------
# Sidecar URL file (kernel-assigned port handoff)
# ---------------------------------------------------------------------------


def write_server_url(url_file: str | Path, url: str) -> Path:
    """Atomically write ``host:port`` to ``url_file`` (tempfile + rename)."""
    target = Path(url_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".url.", dir=target.parent)
    try:
        with open(fd, "w", encoding="utf-8") as fh:
            fh.write(url)
        Path(tmp_name).replace(target)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return target


def read_server_url(url_file: str | Path) -> str:
    """Return the ``host:port`` previously written by :func:`write_server_url`."""
    return Path(url_file).read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# Convenience: ASE Atoms → (energy_hartree, gradient_hartree_per_bohr)
# ---------------------------------------------------------------------------


def atoms_to_payload(atoms) -> tuple[float, list[list[float]]]:
    """Convert an ASE-evaluated ``atoms`` into ChemRefine units.

    ASE reports energies in eV and forces in eV/Å; ORCA's
    ``.engrad`` wants Hartree and Hartree/Bohr. The negative sign on
    forces flips them into ``-∂E/∂x`` gradient convention.
    """
    energy_ev = atoms.get_potential_energy()
    forces_ev_per_a = atoms.get_forces()
    energy_hartree = energy_ev / HARTREE_TO_EV
    gradient = (-forces_ev_per_a * BOHR_TO_ANGSTROM / HARTREE_TO_EV).tolist()
    return energy_hartree, gradient

"""Parsing for ORCA-produced output files.

The only operation with a verified regex set is the standard DFT
``opt_sp`` flavour — those patterns are tested against the real
``tests/data/orca.out`` fixture. The remaining operations (``goat``,
``pes``, ``docker``, ``solvator``) ship as placeholders until a real
example is captured for each one. Each placeholder raises
:class:`NotImplementedError` so callers fail loudly rather than silently
returning empty data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from chemrefine.errors import OutputParseError

# Physical constants (CODATA 2018).
_EV_PER_HARTREE = 27.211386245988
_BOHR_TO_ANGSTROM = 0.529177210903
_HARTREE_PER_BOHR_TO_EV_PER_A = _EV_PER_HARTREE / _BOHR_TO_ANGSTROM

# Last block, since geometry optimisation re-prints these as it iterates.
_COORD_BLOCK_RE = re.compile(
    r"CARTESIAN COORDINATES\s+\(ANGSTROEM\)\s*\n-+\n((?:.*?\n)+?)-+\n",
    re.DOTALL,
)
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
_FINAL_ENERGY_RE = re.compile(
    r"FINAL SINGLE POINT ENERGY(?:\s*\(From external program\))?\s+(-?\d+\.\d+)"
)


@dataclass(frozen=True)
class ParsedStructure:
    """One structure extracted from an ORCA output file."""

    symbols: tuple[str, ...]
    positions: NDArray[np.float64]
    energy_hartree: float
    forces_eV_per_A: NDArray[np.float64] | None


# ---------------------------------------------------------------------------
# DFT (opt_sp) — verified against the fixture
# ---------------------------------------------------------------------------


def parse_dft(path: str | Path) -> list[ParsedStructure]:
    """Parse a DFT ``opt_sp`` output. Returns a single-element list.

    Raises :class:`OutputParseError` if the energy or coordinates block
    is missing.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")

    energy_matches = _FINAL_ENERGY_RE.findall(text)
    if not energy_matches:
        raise OutputParseError(f"no FINAL SINGLE POINT ENERGY in {path}")

    coord_blocks = _COORD_BLOCK_RE.findall(text)
    if not coord_blocks:
        raise OutputParseError(f"no CARTESIAN COORDINATES block in {path}")

    symbols, positions = _parse_coord_block(coord_blocks[-1])
    return [
        ParsedStructure(
            symbols=symbols,
            positions=positions,
            energy_hartree=float(energy_matches[-1]),
            forces_eV_per_A=parse_forces(text),
        )
    ]


def parse_forces(text: str, *, to_ev_per_A: bool = True) -> NDArray[np.float64] | None:
    """Return the **last** ``CARTESIAN GRADIENT`` block as forces, or ``None``.

    Gradients in ORCA are ``∂E/∂x`` in Hartree/Bohr; this function
    returns ``F = -∂E/∂x`` converted to eV/Å unless ``to_ev_per_A`` is
    False.
    """
    blocks = _GRAD_BLOCK_RE.findall(text)
    if not blocks:
        return None
    rows: list[list[float]] = []
    for line in blocks[-1].strip().splitlines():
        m = _GRAD_LINE_RE.match(line)
        if not m:
            continue
        dx = float(m.group(2).replace("D", "E"))
        dy = float(m.group(3).replace("D", "E"))
        dz = float(m.group(4).replace("D", "E"))
        fx, fy, fz = -dx, -dy, -dz
        if to_ev_per_A:
            scale = _HARTREE_PER_BOHR_TO_EV_PER_A
            fx, fy, fz = fx * scale, fy * scale, fz * scale
        rows.append([fx, fy, fz])
    if not rows:
        return None
    return np.array(rows, dtype=np.float64)


def _parse_coord_block(block: str) -> tuple[tuple[str, ...], NDArray[np.float64]]:
    """Parse the body of a ``CARTESIAN COORDINATES (ANGSTROEM)`` block."""
    symbols: list[str] = []
    positions: list[list[float]] = []
    for line in block.strip().splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        symbols.append(parts[0])
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return tuple(symbols), np.array(positions, dtype=np.float64)


# ---------------------------------------------------------------------------
# Placeholders — implement with a real fixture in hand
# ---------------------------------------------------------------------------


def parse_goat_ensemble(path: str | Path) -> list[ParsedStructure]:
    """Parse a GOAT ``.finalensemble.xyz`` file into one ParsedStructure per frame.

    TODO: implement once a real GOAT ``.finalensemble.xyz`` sample is
    available. The v3 parser in :file:`src/chemrefine/orca_interface.py`
    on ``main`` (``parse_goat_finalensemble``) is the reference.
    """
    raise NotImplementedError(
        "GOAT output parser not yet ported — see TODO in engines/orca/output.py"
    )


def parse_pes(path: str | Path) -> list[ParsedStructure]:
    """Parse an ORCA PES-scan output (one frame per converged scan point).

    TODO: implement once a real PES scan output is available. The v3
    parser is :func:`parse_pes_output` in ``orca_interface.py`` on
    ``main`` — it splits on ``"*** OPTIMIZATION RUN DONE ***"`` and
    extracts one geometry + energy per fragment.
    """
    raise NotImplementedError(
        "PES output parser not yet ported — see TODO in engines/orca/output.py"
    )


def parse_docker(path: str | Path) -> list[ParsedStructure]:
    """Parse an ORCA Docker ``.docker.struc1.all.optimized.xyz`` file.

    TODO: implement once a real Docker output is available. The v3
    parser is :func:`parse_docker_xyz` in ``orca_interface.py``. Note
    that the v3 code drops the last structure (deemed non-sensible by
    upstream) — preserve that behaviour.
    """
    raise NotImplementedError(
        "Docker output parser not yet ported — see TODO in engines/orca/output.py"
    )


def parse_solvator(path: str | Path) -> list[ParsedStructure]:
    """Parse an ORCA Solvator ``.solventbuild.xyz`` file.

    TODO: implement once a real Solvator output is available. The v3
    parser is :func:`parse_solvator_ensemble` in ``orca_interface.py``.
    """
    raise NotImplementedError(
        "Solvator output parser not yet ported — see TODO in engines/orca/output.py"
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def parse_output(path: str | Path, operation: str) -> list[ParsedStructure]:
    """Pick the right parser based on the YAML ``operation`` string."""
    op = operation.lower().replace("+", "_")
    if op in {"opt_sp", "dft", "sp"}:
        return parse_dft(path)
    if op == "goat":
        return parse_goat_ensemble(path)
    if op == "pes":
        return parse_pes(path)
    if op == "docker":
        return parse_docker(path)
    if op == "solvator":
        return parse_solvator(path)
    raise OutputParseError(f"unknown ORCA operation: {operation!r}")

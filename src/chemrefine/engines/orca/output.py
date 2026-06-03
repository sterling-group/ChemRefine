"""Parsing for ORCA-produced output files.

Verified against real fixtures under ``tests/data/``:

* ``opt_sp`` — final geometry + energy from the main ``.out``
  (``orca.out`` fixture)
* ``goat`` — multi-frame ensemble (``goat_finalensemble.xyz`` fixture)
* ``docker`` — multi-frame docked structures
  (``docker_allopt.xyz`` fixture)
* ``solvator`` — solvent-build ensemble
  (``solvator_solventbuild.xyz`` fixture)
* ``pes`` — one frame per converged scan point, split on
  ``*** OPTIMIZATION RUN DONE ***`` markers.

ExtOpt ``.extinp.tmp`` / ``.engrad`` round-trip helpers live in
:mod:`chemrefine.engines.orca.extopt.protocol`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from chemrefine.errors import OutputParseError
from chemrefine.quantities import HARTREE_PER_BOHR_TO_EV_PER_A

# Last block, since geometry optimisation re-prints these as it iterates.
_COORD_BLOCK_RE = re.compile(
    r"CARTESIAN COORDINATES\s+\(ANGSTROEM\)\s*\n-+\n((?:.*?\n)+?)-+\n",
    re.DOTALL,
)
_GOAT_HEADER_RE = re.compile(r"^\s*(-?\d+\.\d+)")
_DOCKER_HEADER_RE = re.compile(r"Eopt\s*=\s*(-?\d+\.\d+)\s*\(Eh\)", re.IGNORECASE)
_SOLVATOR_HEADER_RE = re.compile(r"Energy\s+(-?\d+\.\d+)", re.IGNORECASE)
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
# Run-status markers. ORCA prints "****ORCA TERMINATED NORMALLY****" only on a
# clean exit; any "... NOT CONVERGED ..." (SCF or geometry/MaxIter) marks a
# failed stationary point. Read in the same single pass as energy/coords.
_TERMINATED_RE = re.compile(r"ORCA TERMINATED NORMALLY")
_NOT_CONVERGED_RE = re.compile(r"NOT CONVERGED", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedStructure:
    """One structure extracted from an ORCA output file.

    ``terminated`` / ``converged`` are the run-status flags read in the same
    pass (``None`` when not applicable, e.g. sidecar ensemble frames); a
    structure is a *failure* only when one is explicitly ``False``.
    """

    symbols: tuple[str, ...]
    positions: NDArray[np.float64]
    energy_hartree: float
    forces_ev_per_a: NDArray[np.float64] | None
    converged: bool | None = None
    terminated: bool | None = None


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
            forces_ev_per_a=parse_forces(text),
            terminated=bool(_TERMINATED_RE.search(text)),
            converged=not bool(_NOT_CONVERGED_RE.search(text)),
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
            fx *= HARTREE_PER_BOHR_TO_EV_PER_A
            fy *= HARTREE_PER_BOHR_TO_EV_PER_A
            fz *= HARTREE_PER_BOHR_TO_EV_PER_A
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
# Multi-frame XYZ ensembles (GOAT / Docker / Solvator)
# ---------------------------------------------------------------------------


def _parse_xyz_ensemble(
    path: str | Path,
    header_re: re.Pattern[str],
    *,
    fmt_name: str,
) -> list[ParsedStructure]:
    """Walk a multi-frame XYZ file, extracting one structure per frame.

    Each frame is shaped as::

        <n_atoms>
        <header containing an energy float>
        <symbol> <x> <y> <z>     # n_atoms rows

    ``header_re`` must capture the per-frame Hartree energy in group 1.
    Frames whose header doesn't match are skipped with a warning rather
    than aborting — some ORCA writers occasionally emit a stray blank
    frame at the end of an ensemble.
    """
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()

    structures: list[ParsedStructure] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.isdigit():
            i += 1
            continue
        n_atoms = int(line)
        if i + 1 + n_atoms >= len(lines):
            break
        header = lines[i + 1]
        m = header_re.search(header)
        if m is None:
            i += 2 + n_atoms
            continue
        energy = float(m.group(1))

        symbols: list[str] = []
        positions: list[list[float]] = []
        ok = True
        for offset in range(n_atoms):
            parts = lines[i + 2 + offset].split()
            if len(parts) < 4:
                ok = False
                break
            symbols.append(parts[0])
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if not ok:
            i += 2 + n_atoms
            continue

        structures.append(
            ParsedStructure(
                symbols=tuple(symbols),
                positions=np.array(positions, dtype=np.float64),
                energy_hartree=energy,
                forces_ev_per_a=None,
            )
        )
        i += 2 + n_atoms

    if not structures:
        raise OutputParseError(f"no {fmt_name} frames found in {path}")
    return structures


def parse_goat_ensemble(path: str | Path) -> list[ParsedStructure]:
    """Parse a GOAT ``.finalensemble.xyz`` file.

    Header layout: ``<energy_hartree> converged=<bool>``. Returns one
    :class:`ParsedStructure` per frame. Forces are not available in
    this format (set to ``None``).
    """
    return _parse_xyz_ensemble(path, _GOAT_HEADER_RE, fmt_name="GOAT")


def parse_docker(path: str | Path) -> list[ParsedStructure]:
    """Parse an ORCA Docker ``.docker.struc1.allopt.xyz`` ensemble.

    Header layout: ``<idx> Eopt=<energy_hartree> (Eh) Einter=<inter> (kcal/mol)``.
    Returns one :class:`ParsedStructure` per frame, **dropping the
    final frame** because the upstream tool's last structure is
    flagged as non-sensible there.
    """
    structures = _parse_xyz_ensemble(path, _DOCKER_HEADER_RE, fmt_name="Docker")
    if len(structures) <= 1:
        raise OutputParseError(
            f"Docker output {path} has {len(structures)} frame(s); "
            "expected at least 2 (the last is dropped as non-sensible)"
        )
    return structures[:-1]


def parse_solvator(path: str | Path) -> list[ParsedStructure]:
    """Parse an ORCA Solvator ``.solventbuild.xyz`` ensemble.

    Header layout: ``Energy <energy_hartree>``. Returns one
    :class:`ParsedStructure` per frame.
    """
    return _parse_xyz_ensemble(path, _SOLVATOR_HEADER_RE, fmt_name="Solvator")


# ---------------------------------------------------------------------------
# PES scan
# ---------------------------------------------------------------------------


_PES_SEGMENT_RE = re.compile(r"\*{3}\s*OPTIMIZATION RUN DONE\s*\*{3}")
_PES_COORD_HEADER_RE = re.compile(
    r"^\s*CARTESIAN COORDINATES\s*\(ANGSTROEM\)\s*$", re.MULTILINE
)
_PES_DASH_RE = re.compile(r"^\s*-{3,}\s*$")


def parse_pes(path: str | Path) -> list[ParsedStructure]:
    """Parse an ORCA PES-scan output (one frame per converged scan point).

    The file is split on ``*** OPTIMIZATION RUN DONE ***``. For each
    completed segment the parser takes the **last** coordinate block and
    the **last** ``FINAL SINGLE POINT ENERGY`` line — that's the
    converged geometry for that scan point.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    segments = _PES_SEGMENT_RE.split(text)[:-1]  # last fragment has no DONE marker
    terminated = bool(_TERMINATED_RE.search(text))

    structures: list[ParsedStructure] = []
    for seg in segments:
        atoms = _parse_last_pes_coord_block(seg)
        if not atoms:
            continue
        energy = _parse_last_pes_energy(seg)
        if energy is None:
            continue
        symbols = tuple(sym for sym, *_ in atoms)
        positions = np.array([[x, y, z] for _, x, y, z in atoms], dtype=np.float64)
        structures.append(
            ParsedStructure(
                symbols=symbols,
                positions=positions,
                energy_hartree=energy,
                forces_ev_per_a=None,
                # Each frame is a converged scan point (split on RUN DONE).
                converged=True,
                terminated=terminated,
            )
        )
    if not structures:
        raise OutputParseError(f"no PES scan frames found in {path}")
    return structures


def _parse_last_pes_coord_block(segment: str) -> list[tuple[str, float, float, float]]:
    """Return ``(symbol, x, y, z)`` rows from the last coord block in ``segment``."""
    matches = list(_PES_COORD_HEADER_RE.finditer(segment))
    if not matches:
        return []
    # Skip ahead past the header line itself, then past any dashed
    # separator and / or blank lines, then read atom rows until a blank
    # line ends the block.
    lines = segment.splitlines()
    start_line = segment.count("\n", 0, matches[-1].end()) + 1
    idx = start_line
    while idx < len(lines) and (
        not lines[idx].strip() or _PES_DASH_RE.match(lines[idx])
    ):
        idx += 1
    atoms: list[tuple[str, float, float, float]] = []
    while idx < len(lines):
        ln = lines[idx]
        if not ln.strip():
            break
        parts = ln.split()
        # ORCA prints either "C  x y z" (4 tokens) or "1  C  x y z" (5 tokens).
        if len(parts) == 4 and _is_float_triplet(parts[1:]):
            sym = parts[0]
            x, y, z = (float(p) for p in parts[1:])
            atoms.append((sym, x, y, z))
        elif len(parts) == 5 and _is_float_triplet(parts[2:]):
            sym = parts[1]
            x, y, z = (float(p) for p in parts[2:])
            atoms.append((sym, x, y, z))
        idx += 1
    return atoms


def _parse_last_pes_energy(segment: str) -> float | None:
    """Return the last ``FINAL SINGLE POINT ENERGY`` in ``segment``, or ``None``."""
    matches = _FINAL_ENERGY_RE.findall(segment)
    return float(matches[-1]) if matches else None


def _is_float_triplet(tokens: list[str]) -> bool:
    """Return ``True`` if every token in a 3-element list parses as a float."""
    try:
        for tok in tokens:
            float(tok)
    except ValueError:
        return False
    return True


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

# ORCA-defined ensemble sidecar suffixes (``<base>.<suffix>``), one per
# multi-structure operation. Single source for these external-contract names.
_GOAT_SUFFIX = "finalensemble.xyz"
_DOCKER_SUFFIX = "docker.struc1.allopt.xyz"
_SOLVATOR_SUFFIX = "solventbuild.xyz"


def _ensemble_sidecar(out_path: str | Path, suffix: str) -> Path:
    """Return the ORCA ensemble sidecar ``<base>.<suffix>`` next to the ``.out``.

    Multi-structure operations (GOAT / Docker / Solvator) write their ensemble
    to a file named after ORCA's ``%base`` (the ``.out`` stem), not into the
    ``.out`` log itself — e.g. ``step1_structure_0.finalensemble.xyz``. The
    SLURM ``*.xyz`` glob copies it back beside the ``.out``. Raises
    :class:`OutputParseError` if it is absent.
    """
    out_path = Path(out_path)
    sidecar = out_path.with_name(f"{out_path.stem}.{suffix}")
    if not sidecar.is_file():
        raise OutputParseError(
            f"expected ORCA ensemble file {sidecar.name} next to {out_path.name}; "
            "not found (did the run produce it?)"
        )
    return sidecar


def parse_output(path: str | Path, operation: str) -> list[ParsedStructure]:
    """Pick the right parser based on the YAML ``operation`` string.

    ``opt_sp`` / ``sp`` / ``pes`` read the ``.out`` directly; the multi-frame
    ensemble operations read their ``<base>.<suffix>`` sidecar instead.
    """
    op = operation.lower().replace("+", "_")
    if op in {"opt_sp", "dft", "sp"}:
        return parse_dft(path)
    if op == "pes":
        return parse_pes(path)
    if op == "goat":
        return parse_goat_ensemble(_ensemble_sidecar(path, _GOAT_SUFFIX))
    if op == "docker":
        return parse_docker(_ensemble_sidecar(path, _DOCKER_SUFFIX))
    if op == "solvator":
        return parse_solvator(_ensemble_sidecar(path, _SOLVATOR_SUFFIX))
    raise OutputParseError(f"unknown ORCA operation: {operation!r}")

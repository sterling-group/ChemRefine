"""Multi-structure ORCA outputs: GOAT / Docker / Solvator ensembles + PES scans.

These parsers yield a **list** of structures (vs the single-structure ``.out`` assembled by
:mod:`chemrefine.engines.orca.output.coordinator`). GOAT / Docker / Solvator are multi-frame
``.xyz`` sidecars (``<base>.<suffix>`` next to the ``.out``); PES is a scan whose converged
points are segments of the ``.out`` text itself. PES reuses the shared energy reader
(:mod:`chemrefine.engines.orca.output.energy`) + the run-status reader
(:mod:`chemrefine.engines.orca.output.status`); the ``.xyz`` walkers are self-contained.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from chemrefine.engines.api import ParsedResult
from chemrefine.engines.orca.output import energy, status
from chemrefine.errors import OutputParseError

# Per-frame energy-header regex + the ``<base>.<suffix>`` sidecar filename ORCA writes the
# ensemble to — the single source for these external-contract names.
_GOAT_HEADER_RE = re.compile(r"^\s*(-?\d+\.\d+)")
_DOCKER_HEADER_RE = re.compile(r"Eopt\s*=\s*(-?\d+\.\d+)\s*\(Eh\)", re.IGNORECASE)
_SOLVATOR_HEADER_RE = re.compile(r"Energy\s+(-?\d+\.\d+)", re.IGNORECASE)
GOAT_SUFFIX = "finalensemble.xyz"
DOCKER_SUFFIX = "docker.struc1.allopt.xyz"
DOCKER_SUFFIX_611 = "docker.struc1.all.optimized.xyz"
SOLVATOR_SUFFIX = "solventbuild.xyz"
SOLVATOR_SUFFIX_611 = "solvator.solventbuild.xyz"


def ensemble_sidecar(out_path: str | Path, suffix: str, *fallbacks: str) -> Path:
    """Return the ORCA ensemble sidecar ``<base>.<suffix>`` next to the ``.out``.

    Multi-structure operations write their ensemble to a file named after ORCA's ``%base``
    (the ``.out`` stem), not into the ``.out`` log — e.g. ``step1_0.finalensemble.xyz``. The
    SLURM ``*.xyz`` glob copies it back beside the ``.out``. ``fallbacks`` are alternative
    suffixes for operations whose filename changed across ORCA releases (e.g. Docker's
    ``allopt`` → ``all.optimized`` in 6.1.1); the first existing candidate wins. Raises
    :class:`OutputParseError` naming every candidate if none exists.
    """
    out_path = Path(out_path)
    candidates = [out_path.with_name(f"{out_path.stem}.{s}") for s in (suffix, *fallbacks)]
    for sidecar in candidates:
        if sidecar.is_file():
            return sidecar
    names = " or ".join(c.name for c in candidates)
    raise OutputParseError(
        f"expected ORCA ensemble file {names} next to {out_path.name}; "
        "not found (did the run produce it?)"
    )


# ---------------------------------------------------------------------------
# Multi-frame XYZ ensembles (GOAT / Docker / Solvator)
# ---------------------------------------------------------------------------


def _parse_xyz_ensemble(
    path: str | Path, header_re: re.Pattern[str], *, fmt_name: str
) -> list[ParsedResult]:
    """Walk a multi-frame XYZ file, extracting one structure per frame.

    Each frame is ``<n_atoms>`` / ``<header containing an energy float>`` / ``n_atoms`` rows.
    ``header_re`` must capture the per-frame Hartree energy in group 1; frames whose header
    doesn't match are skipped (some ORCA writers emit a stray blank frame at the end).
    """
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    structures: list[ParsedResult] = []
    i = 0
    while i < len(lines):
        structure, i = _parse_xyz_frame(lines, i, header_re)
        if structure is not None:
            structures.append(structure)
    if not structures:
        raise OutputParseError(f"no {fmt_name} frames found in {path}")
    return structures


def _parse_xyz_frame(
    lines: list[str], i: int, header_re: re.Pattern[str]
) -> tuple[ParsedResult | None, int]:
    """Parse one XYZ frame at ``lines[i]``; return ``(structure_or_None, next_index)``.

    ``None`` means "nothing here, advance past it": a non-count line, a header that doesn't
    match, a short/corrupt atom row (past the frame), or a truncated trailing frame.
    """
    line = lines[i].strip()
    if not line.isdigit():
        return None, i + 1
    n_atoms = int(line)
    if i + 1 + n_atoms >= len(lines):
        return None, len(lines)  # truncated trailing frame → stop
    m = header_re.search(lines[i + 1])
    if m is None:
        return None, i + 2 + n_atoms
    symbols: list[str] = []
    positions: list[list[float]] = []
    for offset in range(n_atoms):
        parts = lines[i + 2 + offset].split()
        if len(parts) < 4:
            return None, i + 2 + n_atoms
        try:
            row = [float(parts[1]), float(parts[2]), float(parts[3])]
        except ValueError:
            # Corrupt coordinate token (``*****`` overflow etc.) — skip the frame.
            return None, i + 2 + n_atoms
        symbols.append(parts[0])
        positions.append(row)
    structure = ParsedResult(
        symbols=tuple(symbols),
        positions=np.array(positions, dtype=np.float64),
        energy_hartree=float(m.group(1)),
        forces_ev_per_a=None,
    )
    return structure, i + 2 + n_atoms


def parse_goat_ensemble(path: str | Path) -> list[ParsedResult]:
    """Parse a GOAT ``.finalensemble.xyz`` file (header ``<energy_hartree> converged=<bool>``)."""
    return _parse_xyz_ensemble(path, _GOAT_HEADER_RE, fmt_name="GOAT")


def parse_docker(path: str | Path) -> list[ParsedResult]:
    """Parse an ORCA Docker pose ensemble (``allopt`` or 6.1.1's ``all.optimized``).

    Header layout: ``<idx> Eopt=<energy_hartree> (Eh) …``. Returns one structure per frame.
    Only the legacy ``allopt.xyz`` layout **drops the final frame** (that writer flagged its
    last structure as non-sensible); 6.1.1's ``all.optimized.xyz`` holds only real poses.
    """
    path = Path(path)
    structures = _parse_xyz_ensemble(path, _DOCKER_HEADER_RE, fmt_name="Docker")
    if not path.name.endswith(DOCKER_SUFFIX):
        return structures
    if len(structures) <= 1:
        raise OutputParseError(
            f"Docker output {path} has {len(structures)} frame(s); "
            "expected at least 2 (the last is dropped as non-sensible)"
        )
    return structures[:-1]


def parse_solvator(path: str | Path) -> list[ParsedResult]:
    """Parse an ORCA Solvator ensemble (header ``Energy <e_hartree>``).

    Same frame layout under both filenames — ``solventbuild.xyz`` (legacy) and
    6.1.1's ``solvator.solventbuild.xyz``.
    """
    return _parse_xyz_ensemble(path, _SOLVATOR_HEADER_RE, fmt_name="Solvator")


# ---------------------------------------------------------------------------
# PES scan (segments of the ``.out`` text)
# ---------------------------------------------------------------------------


_PES_SEGMENT_RE = re.compile(r"\*{3}\s*OPTIMIZATION RUN DONE\s*\*{3}")
_PES_COORD_HEADER_RE = re.compile(r"^\s*CARTESIAN COORDINATES\s*\(ANGSTROEM\)\s*$", re.MULTILINE)
_PES_DASH_RE = re.compile(r"^\s*-{3,}\s*$")


def parse_pes(path: str | Path) -> list[ParsedResult]:
    """Parse an ORCA PES-scan output (one frame per converged scan point)."""
    return parse_pes_from_text(
        Path(path).read_text(encoding="utf-8", errors="replace"), src=str(path)
    )


def parse_pes_from_text(text: str, *, src: str = "<text>") -> list[ParsedResult]:
    """Parse a PES-scan output from already-read text (one structure per converged point)."""
    segments = _PES_SEGMENT_RE.split(text)[:-1]  # last fragment has no DONE marker
    terminated = status.parse_terminated_normally(text)

    structures: list[ParsedResult] = []
    for seg in segments:
        atoms = _parse_last_pes_coord_block(seg)
        if not atoms:
            continue
        seg_energy = energy.parse_final_energy_from_text(seg)
        if seg_energy is None:
            continue
        symbols = tuple(sym for sym, *_ in atoms)
        positions = np.array([[x, y, z] for _, x, y, z in atoms], dtype=np.float64)
        structures.append(
            ParsedResult(
                symbols=symbols,
                positions=positions,
                energy_hartree=seg_energy,
                forces_ev_per_a=None,
                # Each frame is a converged scan point (split on RUN DONE).
                converged=True,
                terminated_normally=terminated,
            )
        )
    if not structures:
        raise OutputParseError(f"no PES scan frames found in {src}")
    return structures


def _parse_last_pes_coord_block(segment: str) -> list[tuple[str, float, float, float]]:
    """Return ``(symbol, x, y, z)`` rows from the last coord block in ``segment``."""
    matches = list(_PES_COORD_HEADER_RE.finditer(segment))
    if not matches:
        return []
    lines = segment.splitlines()
    start_line = segment.count("\n", 0, matches[-1].end()) + 1
    idx = start_line
    while idx < len(lines) and (not lines[idx].strip() or _PES_DASH_RE.match(lines[idx])):
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


def _is_float_triplet(tokens: list[str]) -> bool:
    """Return ``True`` if every token in a 3-element list parses as a float."""
    try:
        for tok in tokens:
            float(tok)
    except ValueError:
        return False
    return True

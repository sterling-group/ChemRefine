"""Filesystem I/O helpers — XYZ read/write, SMILES → 3D, CSV reporting.

Keeps every concrete file format in one place so individual engine
modules don't grow their own XYZ-handling code. ``write_xyz`` accepts
either :class:`ase.Atoms` objects or ``(symbol, x, y, z)`` tuples for
back-compatibility with parsers that produce raw coordinate lists.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from ase import Atoms

from chemrefine.ids import structure_artifact_path
from chemrefine.quantities import (
    DEFAULT_TEMPERATURE_K,
    HARTREE_TO_KCALMOL,
    boltzmann_weights,
)

_CSV_PRECISION = 8
_NATURAL_PART = re.compile(r"(\d+)")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------


def natural_key(name: str | Path) -> list[object]:
    """Return a list suitable for ``sorted(..., key=natural_key)`` natural ordering.

    ``"step10.out"`` sorts after ``"step2.out"`` instead of before it.
    """
    return [int(p) if p.isdigit() else p.lower() for p in _NATURAL_PART.split(str(name))]


# ---------------------------------------------------------------------------
# XYZ
# ---------------------------------------------------------------------------


CoordList = Sequence[tuple[str, float, float, float]]


def write_xyz(
    structures: Sequence[Atoms | CoordList],
    structure_ids: Sequence[str],
    step_number: int,
    output_dir: str | Path,
) -> list[Path]:
    """Write each structure to ``output_dir/step{N}_structure_{ID}.xyz``.

    Each ``structures`` entry may be an :class:`ase.Atoms` or a list of
    ``(symbol, x, y, z)`` tuples — the loop coerces tuple form on the
    fly. Returns the list of written paths in input order. Raises
    :class:`ValueError` if ``structures`` and ``structure_ids`` differ
    in length.
    """
    if len(structures) != len(structure_ids):
        raise ValueError(
            f"structures ({len(structures)}) and structure_ids ({len(structure_ids)}) "
            "must have the same length"
        )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for geometry, sid in zip(structures, structure_ids, strict=True):
        if isinstance(geometry, Atoms):
            atoms = geometry
        else:
            atoms = Atoms(
                symbols=[row[0] for row in geometry],
                positions=np.array([row[1:] for row in geometry], dtype=float),
            )
        path = structure_artifact_path(out, step_number, sid, "xyz")
        lines = [str(len(atoms)), f"step {step_number} structure {sid}"]
        for symbol, (x, y, z) in zip(
            atoms.get_chemical_symbols(), atoms.get_positions(), strict=True
        ):
            lines.append(f"{symbol:2s} {x:.6f} {y:.6f} {z:.6f}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(path)
    return written


def gather_output_files(directory: str | Path, pattern: str) -> list[Path]:
    """Return files in ``directory`` matching ``pattern``, sorted naturally."""
    d = Path(directory)
    if not d.exists():
        return []
    matches = sorted(d.glob(pattern), key=natural_key)
    return list(matches)


# ---------------------------------------------------------------------------
# SMILES → 3D
# ---------------------------------------------------------------------------


def _conformer_to_xyz_lines(mol, comment: str) -> list[str]:
    """Render an embedded RDKit ``mol``'s conformer as XYZ-format text lines."""
    conf = mol.GetConformer()
    natoms = mol.GetNumAtoms()
    lines = [str(natoms), comment]
    for atom_idx in range(natoms):
        atom = mol.GetAtomWithIdx(atom_idx)
        pos = conf.GetAtomPosition(atom_idx)
        lines.append(f"{atom.GetSymbol():2s} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")
    return lines


def smiles_to_xyz(
    csv_file: str | Path,
    output_dir: str | Path,
    *,
    smiles_column: str = "smiles",
    max_attempts: int = 10,
    random_seed: int = 42,
) -> list[Path]:
    """Convert a CSV column of SMILES into individual 3D XYZ files.

    Each successful conversion writes ``output_dir/structure_{row}.xyz``.
    Invalid SMILES are logged and skipped — they do not abort the run.

    ``random_seed`` seeds RDKit's conformer embedding (which is otherwise
    non-deterministic), so repeated runs — and ``resume``, which
    re-bootstraps the seeds — regenerate identical 3D geometries.
    """
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import AllChem

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_file)
    if smiles_column not in df.columns:
        raise ValueError(f"column {smiles_column!r} not found in {csv_file}")

    written: list[Path] = []
    for idx, raw in enumerate(df[smiles_column]):
        if not isinstance(raw, str) or not raw.strip():
            continue
        mol = Chem.MolFromSmiles(raw)
        if mol is None:
            logger.warning("invalid SMILES at row %d: %s", idx, raw)
            continue
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, maxAttempts=max_attempts, randomSeed=random_seed) != 0:
            logger.warning("failed 3D embedding for SMILES: %s", raw)
            continue
        AllChem.UFFOptimizeMolecule(mol)

        lines = _conformer_to_xyz_lines(mol, f"SMILES: {raw}")
        xyz_path = out / f"structure_{idx}.xyz"
        xyz_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(xyz_path)
    return written


# ---------------------------------------------------------------------------
# Per-step CSV report
# ---------------------------------------------------------------------------


def _boltzmann_columns(energy_kcal: np.ndarray, temperature_k: float) -> dict[str, np.ndarray]:
    """Return the four Boltzmann-derived report columns for a sorted energy array."""
    dE = energy_kcal - energy_kcal.min()
    weights = boltzmann_weights(dE, temperature_k)
    pct_total = weights * 100.0
    return {
        "dE (kcal/mol)": dE,
        "Boltzmann Weight": weights,
        "% Total": pct_total,
        "% Cumulative": np.cumsum(pct_total),
    }


def save_step_csv(
    energies_hartree: Iterable[float],
    structure_ids: Iterable[str],
    step_number: int,
    output_dir: str | Path,
    *,
    filename: str = "steps.csv",
    temperature_k: float = DEFAULT_TEMPERATURE_K,
) -> Path:
    """Append a per-structure summary row for ``step_number`` to a cumulative CSV.

    Columns: ``Step, Conformer, Energy (Hartree), Energy (kcal/mol),
    dE (kcal/mol), Boltzmann Weight, % Total, % Cumulative``.
    Sorted by energy ascending. Step 1 writes the header; later steps
    append without a header.
    """
    import pandas as pd

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename

    df = pd.DataFrame(
        {
            "Step": step_number,
            "Conformer": list(structure_ids),
            "Energy (Hartree)": list(energies_hartree),
        }
    )
    df["Energy (kcal/mol)"] = pd.to_numeric(
        df["Energy (Hartree)"] * HARTREE_TO_KCALMOL, errors="coerce"
    )
    df = df.dropna(subset=["Energy (kcal/mol)"])
    if df.empty:
        # Nothing finite to summarise (all energies None/NaN) — skip the row
        # rather than crash computing Boltzmann columns on an empty frame.
        logger.warning("step %d: no finite energies to summarise; skipping CSV", step_number)
        return path
    df = df.sort_values("Energy (kcal/mol)").reset_index(drop=True)

    for column, values in _boltzmann_columns(
        df["Energy (kcal/mol)"].to_numpy(dtype=float), temperature_k
    ).items():
        df[column] = values
    df = df.round(
        {
            "Energy (kcal/mol)": _CSV_PRECISION,
            "dE (kcal/mol)": _CSV_PRECISION,
            "Boltzmann Weight": _CSV_PRECISION,
            "% Total": _CSV_PRECISION,
            "% Cumulative": _CSV_PRECISION,
        }
    )

    mode = "w" if step_number == 1 else "a"
    header = step_number == 1
    df.to_csv(path, mode=mode, index=False, header=header)
    logger.info("saved step %d summary to %s", step_number, path)
    return path

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
from typing import Any, cast

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from numpy.typing import NDArray

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


def write_single_xyz(geometry: Atoms | CoordList, path: str | Path, *, comment: str = "") -> Path:
    """Write one geometry to ``path`` as plain XYZ; return ``path``.

    ``geometry`` may be an :class:`ase.Atoms` or a list of ``(symbol, x, y, z)``
    tuples (coerced on the fly). The parent directory is created if needed — this
    is how a structure's per-id directory comes into being (engines call this with
    :func:`chemrefine.ids.input_geometry_path`).
    """
    if isinstance(geometry, Atoms):
        atoms = geometry
    else:
        atoms = Atoms(
            symbols=[row[0] for row in geometry],
            positions=np.array([row[1:] for row in geometry], dtype=float),
        )
    lines = [str(len(atoms)), comment]
    for symbol, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.get_positions(), strict=True):
        lines.append(f"{symbol:2s} {x:.6f} {y:.6f} {z:.6f}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def write_xyz(
    structures: Sequence[Atoms | CoordList],
    structure_ids: Sequence[str],
    step_number: int,
    output_dir: str | Path,
) -> list[Path]:
    """Write each structure **flat** to ``output_dir/step{N}_{ID}.xyz``.

    A flat, directly-globbable layout for *seed* sets (a directory of seeds is
    read back via :func:`gather_output_files` with ``*.xyz``). Per-step engine
    inputs do **not** use this — they live in per-structure directories via
    :func:`write_single_xyz` + :func:`chemrefine.ids.input_geometry_path`.
    Raises :class:`ValueError` if ``structures`` and ``structure_ids`` differ
    in length.
    """
    if len(structures) != len(structure_ids):
        raise ValueError(
            f"structures ({len(structures)}) and structure_ids ({len(structure_ids)}) "
            "must have the same length"
        )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return [
        write_single_xyz(
            geometry, out / f"step{step_number}_{sid}.xyz", comment=f"step {step_number} {sid}"
        )
        for geometry, sid in zip(structures, structure_ids, strict=True)
    ]


def gather_output_files(directory: str | Path, pattern: str) -> list[Path]:
    """Return files in ``directory`` matching ``pattern``, sorted naturally."""
    d = Path(directory)
    if not d.exists():
        return []
    matches = sorted(d.glob(pattern), key=natural_key)
    return list(matches)


def read_xyz_frames(path: str | Path) -> list[Atoms]:
    """Read every frame of an XYZ file into ASE ``Atoms`` — the one XYZ reader.

    ``format="extxyz"`` selects ASE's robust reader: the naive ``"xyz"`` parser
    loops ``int(lines.pop(0))`` over every line and dies (``invalid literal for
    int(): '\\n'``) on the trailing blank lines that editors and ORCA routinely
    leave on real files. ``index=":"`` returns all frames; a single-frame file
    yields a one-element list, so callers never special-case frame count.
    """
    return cast("list[Atoms]", ase_read(str(path), index=":", format="extxyz"))


# ---------------------------------------------------------------------------
# SMILES → 3D
# ---------------------------------------------------------------------------


def _conformer_to_xyz_lines(mol: Any, comment: str) -> list[str]:
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
        # rdkit builds AllChem's surface at import, so its members exist at runtime but not
        # in any stub — these two are real functions mypy cannot see.
        if (
            AllChem.EmbedMolecule(  # type: ignore[attr-defined]
                mol, maxAttempts=max_attempts, randomSeed=random_seed
            )
            != 0
        ):
            logger.warning("failed 3D embedding for SMILES: %s", raw)
            continue
        AllChem.UFFOptimizeMolecule(mol)  # type: ignore[attr-defined]

        lines = _conformer_to_xyz_lines(mol, f"SMILES: {raw}")
        xyz_path = out / f"structure_{idx}.xyz"
        xyz_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(xyz_path)
    return written


# ---------------------------------------------------------------------------
# Per-step CSV report
# ---------------------------------------------------------------------------


def _boltzmann_columns(
    energy_kcal: NDArray[np.float64], temperature_k: float
) -> dict[str, NDArray[np.float64]]:
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
    energies_hartree: Iterable[float | None],
    structure_ids: Iterable[str],
    step_number: int,
    output_dir: str | Path,
    *,
    filename: str = "steps.csv",
    temperature_k: float = DEFAULT_TEMPERATURE_K,
    energy_type: str = "electronic",
) -> Path:
    """Append a per-structure summary row for ``step_number`` to a cumulative CSV.

    Columns: ``Step, Conformer, Energy (Hartree), Energy (kcal/mol),
    dE (kcal/mol), Boltzmann Weight, % Total, % Cumulative, Energy type``.
    Sorted by energy ascending. Step 1 writes the header; later steps
    append without a header.

    ``energy_type`` names which energy the caller passed (``electronic`` / ``gibbs`` /
    ``enthalpy`` / ``electronic_zero_point``) and is recorded verbatim in the last
    column. It is a column rather than a rename of the energy headers because
    ``steps.csv`` is cumulative across steps that may filter on different energies —
    one header has to serve them all.

    Structures whose energy is ``None`` are dropped from the report (nothing to
    summarise), which is how a backfilled ``on_failure: best`` structure with no
    computed energy passes through without breaking the Boltzmann columns.
    """
    import pandas as pd

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    if step_number == 1:
        # Step 1 starts a fresh report each run; later steps append to it.
        # Truncate up front so a step 1 with no finite energies (the early
        # return below) can't leave a prior run's stale rows for step 2 to
        # append onto.
        path.unlink(missing_ok=True)

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
    # Last column, so the historic header prefix is unchanged for existing tooling.
    df["Energy type"] = energy_type

    # The header follows the *file*, not the step number. Keyed off `step_number == 1`, a
    # step 1 that summarises nothing (every energy None — which `on_failure: best` produces
    # on step 1, where the backfills are seeds with no energy yet) leaves step 2 appending
    # header-less to a file that does not exist, so the report opens with a data row and
    # every `read_csv` takes it for the column names.
    df.to_csv(path, mode="a", index=False, header=not path.exists())
    logger.info("saved step %d summary to %s", step_number, path)
    return path

"""Filesystem I/O helpers — XYZ read/write, SMILES → 3D, CSV reporting.

Keeps every concrete file format in one place so individual engine
modules don't grow their own XYZ-handling code. ``write_xyz`` accepts
either :class:`ase.Atoms` objects or ``(symbol, x, y, z)`` tuples for
back-compatibility with parsers that produce raw coordinate lists.

Two engine-side XYZ *readers* stay deliberately outside this module, each for a
stated price — the rule for a new engine is "use :func:`read_xyz_frames` /
:func:`write_single_xyz` unless you can name your price too":

* the ExtOpt wrapper's plain-format reader
  (``chemrefine.engines.orca.extopt.protocol._read_xyz``) runs in a fresh process
  once per ORCA optimizer step, and importing this module costs ~0.5 s of
  ``ase.io`` before it reads a thing;
* the ORCA ensemble walkers (:mod:`chemrefine.engines.orca.output.ensembles`)
  parse per-format energy headers and skip corrupt frames mid-file — neither of
  which ASE's reader can express.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from ase import Atoms
from ase.io import read as ase_read
from numpy.typing import NDArray

from chemrefine.quantities import (
    DEFAULT_TEMPERATURE_K,
    HARTREE_TO_KCALMOL,
    boltzmann_weights,
)

if TYPE_CHECKING:
    # Annotation-only: the ensemble writer reads `Structure` fields via getattr, so the
    # runtime import graph stays as it is — this module keeps importing no pydantic.
    from chemrefine.state import Structure

_CSV_PRECISION = 8
_NATURAL_PART = re.compile(r"(\d+)")

STEPS_CSV_COLUMNS = (
    "Step",
    "Conformer",
    "Energy (Hartree)",
    "Energy (kcal/mol)",
    "dE (kcal/mol)",
    "Boltzmann Weight",
    "% Total",
    "% Cumulative",
    "Energy type",
)
"""The columns of ``steps.csv``, in order — the report's schema, addressable.

Public and load-bearing rather than a line of prose in a docstring, because three things
outside this function depend on these exact strings: :func:`chemrefine.agent_tools.
get_results` hands whole rows to agents, the GUI's results table names four of them in
``index.html``, and whatever a user reads the file with names them too.

:func:`save_step_csv` selects by this tuple before writing, so the frame it assembled and
the schema it promises cannot drift apart; ``tests/test_gui_assets.py`` compares the
page's hardcoded names against it, so the page cannot drift from either. Renaming a column
is still allowed — it just now has to be done here, where every reader is looking."""

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


def _xyz_frame_lines(atoms: Atoms, comment: str) -> list[str]:
    """One plain-XYZ frame: the count line, the comment line, one row per atom.

    The single spelling of the frame format, shared by the one-geometry writer and the
    multi-frame ensemble writer — two writers with their own row formatting are
    byte-identical only until one of them is edited.
    """
    lines = [str(len(atoms)), comment]
    for symbol, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.get_positions(), strict=True):
        lines.append(f"{symbol:2s} {x:.6f} {y:.6f} {z:.6f}")
    return lines


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
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(_xyz_frame_lines(atoms, comment)) + "\n", encoding="utf-8")
    return target


def write_ensemble_xyz(
    structures: Sequence[Structure],
    path: str | Path,
    *,
    step: int,
    energy_attr: str = "energy_hartree",
    energy_label: str = "E",
) -> Path:
    """Write ``structures`` as one multi-frame XYZ at ``path``; return ``path``.

    The human-facing ensemble a step leaves behind — every frame is a final geometry, so a
    user opens one file in Avogadro/VMD instead of walking per-structure directories.
    Written whole, never appended: completion order is nondeterministic, and a file built
    as jobs land would differ between two runs of the same step, where this one is
    byte-identical on a re-run, a resume, and a rebuild.

    Frames are sorted ascending by ``energy_attr`` — the step's own ranking energy
    (:func:`chemrefine.filtering.ranking_energy`), so the file leads with the conformer
    the step's filter would keep first. The sort is stable and sends energy-less
    structures (``on_failure: best`` backfills, missing thermochemistry) to the end in
    their incoming results order, which is the cache's stable manifest order.

    Each comment line carries ``step{N} id={id} {label}={value:.8f} Eh`` (``n/a`` when the
    energy is absent): the id is what ties a frame back to its structure directory and its
    ``steps.csv`` row. ase's extxyz reader parses these frames, so
    :func:`read_xyz_frames` round-trips the file.

    An **empty** ``structures`` removes any stale file rather than writing a zero-frame
    one: a step whose filter kept nothing must not leave the previous run's survivors
    lying about — the same reasoning as :func:`save_step_csv`'s step-1 truncation.
    """
    target = Path(path)
    if not structures:
        target.unlink(missing_ok=True)
        return target

    def _key(s: Structure) -> tuple[bool, float]:
        value = getattr(s, energy_attr)
        return (value is None, 0.0 if value is None else float(value))

    lines: list[str] = []
    for s in sorted(structures, key=_key):
        value = getattr(s, energy_attr)
        caption = f"{energy_label}=n/a" if value is None else f"{energy_label}={value:.8f} Eh"
        lines.extend(_xyz_frame_lines(s.atoms, f"step{step} id={s.id} {caption}"))
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


def _conformer_rows(mol: Any) -> CoordList:
    """An embedded RDKit ``mol``'s conformer as ``(symbol, x, y, z)`` rows.

    Rows rather than rendered text, so the writing goes through
    :func:`write_single_xyz` like every other geometry this module emits. Its
    predecessor rendered the lines itself — a second XYZ writer one function away from
    the canonical one, byte-identical only for as long as nobody edited either.
    """
    conf = mol.GetConformer()
    rows: list[tuple[str, float, float, float]] = []
    for atom_idx in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(atom_idx)
        rows.append((mol.GetAtomWithIdx(atom_idx).GetSymbol(), pos.x, pos.y, pos.z))
    return rows


def embed_smiles(smiles: str, *, max_attempts: int = 10, random_seed: int = 42) -> CoordList:
    """One SMILES → embedded, UFF-relaxed 3D coordinate rows; raises on failure.

    The single-molecule seam under :func:`smiles_to_xyz`'s CSV loop, public because the
    two callers want opposite failure behaviour: a CSV sweep logs a bad row and keeps
    going, while a caller embedding one *explicit* SMILES (the agent's structure-building
    tool) wants the :class:`ValueError` raised at the molecule it names.

    ``random_seed`` seeds RDKit's conformer embedding (which is otherwise
    non-deterministic), so repeated calls regenerate identical 3D geometries.
    """
    from rdkit import Chem

    # Imported from the modules that define them rather than from ``rdkit.Chem.AllChem``,
    # which collects them with ``import *`` — a star import re-exports nothing, so reading
    # them off ``AllChem`` means reaching for names its stubs do not carry.
    from rdkit.Chem.rdDistGeom import EmbedMolecule
    from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    if EmbedMolecule(mol, maxAttempts=max_attempts, randomSeed=random_seed) != 0:
        raise ValueError(f"failed 3D embedding for SMILES: {smiles}")
    UFFOptimizeMolecule(mol)
    return _conformer_rows(mol)


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
    Embedding itself is :func:`embed_smiles`, per row.
    """
    import pandas as pd

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_file)
    if smiles_column not in df.columns:
        raise ValueError(f"column {smiles_column!r} not found in {csv_file}")

    written: list[Path] = []
    for idx, raw in enumerate(df[smiles_column]):
        if not isinstance(raw, str) or not raw.strip():
            continue
        try:
            rows = embed_smiles(raw, max_attempts=max_attempts, random_seed=random_seed)
        except ValueError as e:
            logger.warning("row %d: %s", idx, e)
            continue
        written.append(
            write_single_xyz(rows, out / f"structure_{idx}.xyz", comment=f"SMILES: {raw}")
        )
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

    Columns are :data:`STEPS_CSV_COLUMNS`, in that order — selected by it just before the
    write, so this function cannot quietly emit a shape the constant does not describe.
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
    # Last column, so the leading header columns stay stable for existing tooling.
    df["Energy type"] = energy_type

    # Order and completeness in one line: a column this function forgot to build raises
    # KeyError here rather than shipping a report missing it, and the write order is the
    # documented one by construction instead of by the order the frame happened to grow.
    df = df[list(STEPS_CSV_COLUMNS)]

    # The header follows the *file*, not the step number. Keyed off `step_number == 1`, a
    # step 1 that summarises nothing (every energy None — which `on_failure: best` produces
    # on step 1, where the backfills are seeds with no energy yet) leaves step 2 appending
    # header-less to a file that does not exist, so the report opens with a data row and
    # every `read_csv` takes it for the column names.
    df.to_csv(path, mode="a", index=False, header=not path.exists())
    logger.info("saved step %d summary to %s", step_number, path)
    return path

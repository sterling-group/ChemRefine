"""Tests for filesystem I/O helpers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from ase.io import read as ase_read

from chemrefine.io import (
    gather_output_files,
    natural_key,
    save_step_csv,
    write_xyz,
)


def test_importing_io_does_not_pull_pandas():
    """pandas loads only when CSV is actually read/written, not on `import chemrefine.io`."""
    out = subprocess.run(
        [sys.executable, "-c", "import sys, chemrefine.io; print('pandas' in sys.modules)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False"


# ---------------------------------------------------------------------------
# natural_key
# ---------------------------------------------------------------------------


def test_natural_key_orders_step10_after_step2():
    files = ["step10.out", "step2.out", "step1.out"]
    assert sorted(files, key=natural_key) == ["step1.out", "step2.out", "step10.out"]


def test_natural_key_handles_paths():
    assert natural_key("step1_structure_0.out") < natural_key("step1_structure_10.out")


def test_natural_key_orders_three_digit_ids_numerically():
    """Regression: >100 structures must not sort lexicographically.

    Plain string sorting puts ``structure_100`` before ``structure_11``
    before ``structure_2``; natural ordering compares the digit runs as
    integers so the order is 2 < 11 < 100.
    """
    names = [
        "step1_structure_100.out",
        "step1_structure_2.out",
        "step1_structure_11.out",
    ]
    assert sorted(names, key=natural_key) == [
        "step1_structure_2.out",
        "step1_structure_11.out",
        "step1_structure_100.out",
    ]


# ---------------------------------------------------------------------------
# write_xyz round-trip
# ---------------------------------------------------------------------------


def _h2o() -> Atoms:
    return Atoms(
        symbols=["O", "H", "H"],
        positions=[[0.0, 0.0, 0.117], [0.0, 0.757, -0.467], [0.0, -0.757, -0.467]],
    )


def test_write_xyz_writes_one_file_per_structure(tmp_path: Path):
    # write_xyz is the flat seed writer (directly globbable by bootstrap).
    paths = write_xyz([_h2o(), _h2o()], ["0", "1"], step_number=3, output_dir=tmp_path)
    assert len(paths) == 2
    assert paths[0].name == "step3_0.xyz"
    assert paths[1].name == "step3_1.xyz"
    assert all(p.exists() for p in paths)
    assert all(p.parent == tmp_path for p in paths)  # flat, not nested


def test_write_single_xyz_to_input_geometry_path(tmp_path: Path):
    from chemrefine.ids import input_geometry_path
    from chemrefine.io import write_single_xyz

    path = write_single_xyz(_h2o(), input_geometry_path(tmp_path, 2, "0-1"))
    assert path == tmp_path / "0-1" / "step2_0-1_inp.xyz"
    assert path.is_file()
    back = ase_read(str(path), format="xyz")
    assert list(back.get_chemical_symbols()) == ["O", "H", "H"]


def test_write_xyz_roundtrips(tmp_path: Path):
    original = _h2o()
    [path] = write_xyz([original], ["0"], step_number=1, output_dir=tmp_path)
    back = ase_read(str(path), format="xyz")
    assert list(back.get_chemical_symbols()) == ["O", "H", "H"]
    np.testing.assert_allclose(back.get_positions(), original.get_positions(), atol=1e-6)


def test_write_xyz_accepts_tuple_form(tmp_path: Path):
    tuples = [("H", 0.0, 0.0, 0.0), ("H", 0.74, 0.0, 0.0)]
    [path] = write_xyz([tuples], ["0"], step_number=1, output_dir=tmp_path)
    atoms = ase_read(str(path), format="xyz")
    assert list(atoms.get_chemical_symbols()) == ["H", "H"]


def test_write_xyz_length_mismatch_raises(tmp_path: Path):
    with pytest.raises(ValueError):
        write_xyz([_h2o()], ["0", "1"], step_number=1, output_dir=tmp_path)


# ---------------------------------------------------------------------------
# gather_output_files
# ---------------------------------------------------------------------------


def test_gather_output_files_returns_natural_order(tmp_path: Path):
    for name in ["step10.out", "step1.out", "step2.out"]:
        (tmp_path / name).touch()
    found = gather_output_files(tmp_path, "*.out")
    assert [p.name for p in found] == ["step1.out", "step2.out", "step10.out"]


def test_gather_output_files_orders_more_than_100_structures(tmp_path: Path):
    """Regression: a seed dir with >100 structures must glob in numeric order."""
    for sid in (100, 2, 11, 1, 99):
        (tmp_path / f"step1_structure_{sid}.xyz").touch()
    found = gather_output_files(tmp_path, "*.xyz")
    assert [p.name for p in found] == [
        "step1_structure_1.xyz",
        "step1_structure_2.xyz",
        "step1_structure_11.xyz",
        "step1_structure_99.xyz",
        "step1_structure_100.xyz",
    ]


def test_gather_output_files_missing_dir_returns_empty(tmp_path: Path):
    assert gather_output_files(tmp_path / "missing", "*.out") == []


# ---------------------------------------------------------------------------
# save_step_csv
# ---------------------------------------------------------------------------


def test_save_step_csv_writes_header_on_step_one(tmp_path: Path):
    path = save_step_csv([-1.0, -1.001], ["0", "1"], step_number=1, output_dir=tmp_path)
    text = path.read_text()
    assert "Step,Conformer,Energy (Hartree)" in text.splitlines()[0]
    assert "% Cumulative" in text.splitlines()[0]


def test_save_step_csv_appends_without_header_on_later_steps(tmp_path: Path):
    save_step_csv([-1.0], ["0"], step_number=1, output_dir=tmp_path)
    save_step_csv([-2.0], ["1"], step_number=2, output_dir=tmp_path)
    text = (tmp_path / "steps.csv").read_text()
    # header appears exactly once
    assert text.count("Step,Conformer,Energy (Hartree)") == 1


def test_save_step_csv_sorts_by_energy(tmp_path: Path):
    path = save_step_csv([-1.0, -2.0, -0.5], ["a", "b", "c"], step_number=1, output_dir=tmp_path)
    rows = path.read_text().strip().splitlines()[1:]  # drop header
    conformers = [row.split(",")[1] for row in rows]
    assert conformers == ["b", "a", "c"]  # ascending by absolute energy


def test_save_step_csv_all_nan_energies_skips_write(tmp_path: Path):
    """No finite energies → skip the summary instead of crashing on an empty frame."""
    path = save_step_csv(
        [float("nan"), float("nan")], ["0", "1"], step_number=1, output_dir=tmp_path
    )
    assert not path.exists()


def test_save_step_csv_step_one_truncates_stale_file(tmp_path: Path):
    """A step 1 with no finite energies removes a prior run's report.

    Otherwise step 2 would append onto the stale step-1 rows from the
    previous run instead of starting a fresh report.
    """
    stale = save_step_csv([-1.0], ["0"], step_number=1, output_dir=tmp_path)
    assert stale.exists()
    path = save_step_csv([float("nan")], ["0"], step_number=1, output_dir=tmp_path)
    assert not path.exists()


def test_save_step_csv_writes_a_header_when_step_one_wrote_nothing(tmp_path: Path):
    """The report must never start with a data row.

    Truncation and the header were two decisions keyed off the same `step_number == 1`, so a
    step 1 that summarised nothing left step 2 appending `header=False` onto a file that did
    not exist -- producing a CSV whose first line is data, which every `read_csv` then eats as
    the column names.

    Reachable without contrivance: `on_failure: best` on step 1 with no `sample:` keeps the
    backfilled seeds, and those carry no energy yet.
    """
    assert not save_step_csv([None, None], ["0", "1"], step_number=1, output_dir=tmp_path).exists()

    path = save_step_csv([-1.0, -1.5], ["0", "1"], step_number=2, output_dir=tmp_path)

    assert path.read_text(encoding="utf-8").splitlines()[0].startswith("Step,Conformer,")
    frame = pd.read_csv(path)
    assert list(frame["Step"]) == [2, 2]


# ---------------------------------------------------------------------------
# smiles_to_xyz error paths
# ---------------------------------------------------------------------------


def test_smiles_to_xyz_missing_column_raises(tmp_path: Path):
    from chemrefine.io import smiles_to_xyz

    csv = tmp_path / "no_smiles.csv"
    csv.write_text("other_column\nC\n", encoding="utf-8")
    with pytest.raises(ValueError, match="smiles"):
        smiles_to_xyz(csv, tmp_path / "out")


def test_smiles_to_xyz_skips_blank_and_invalid_smiles(tmp_path: Path):
    """Empty / non-string / unparseable SMILES are warned and skipped, never abort the loop."""
    from chemrefine.io import smiles_to_xyz

    csv = tmp_path / "mixed.csv"
    # Row 0: blank string  → skipped (line 144 branch)
    # Row 1: invalid SMILES → skipped (line 147-148 branch)
    # Row 2: valid SMILES   → produces a file
    csv.write_text("smiles\n\n!!!nonsense!!!\nC\n", encoding="utf-8")
    written = smiles_to_xyz(csv, tmp_path / "out")
    assert len(written) == 1  # only the valid one


def test_smiles_to_xyz_skips_nan_and_whitespace_rows(tmp_path: Path):
    """NaN cells (pandas turns empty cells into float('nan')) and whitespace-only
    strings hit the ``not isinstance(raw, str) or not raw.strip()`` skip branch.

    We construct the dataframe via :mod:`pandas` directly to make the NaN row
    explicit — relying on CSV-parsing heuristics for "empty cell" is fragile
    across pandas versions.
    """
    import pandas as pd

    from chemrefine.io import smiles_to_xyz

    df = pd.DataFrame({"smiles": [float("nan"), "   ", "C"]})
    # pandas is imported lazily inside smiles_to_xyz, so patch it at the source.
    with patch("pandas.read_csv", return_value=df):
        written = smiles_to_xyz(tmp_path / "ignored.csv", tmp_path / "out")
    assert len(written) == 1


def test_smiles_to_xyz_logs_when_embed_fails(tmp_path: Path):
    """A non-zero return from ``AllChem.EmbedMolecule`` triggers a warning + skip."""
    from rdkit.Chem import AllChem

    from chemrefine.io import smiles_to_xyz

    csv = tmp_path / "one.csv"
    csv.write_text("smiles\nC\n", encoding="utf-8")
    with patch.object(AllChem, "EmbedMolecule", return_value=1):
        written = smiles_to_xyz(csv, tmp_path / "out")
    assert written == []


def test_smiles_to_xyz_is_reproducible(tmp_path: Path):
    """Two conversions of the same CSV must produce identical 3D geometries.

    RDKit's embedding is non-deterministic unless seeded; the pipeline
    re-bootstraps its seeds on every ``run``/``resume``, so unseeded
    embedding would make runs irreproducible (and churn the seed-content
    cache fingerprint).
    """
    from chemrefine.io import smiles_to_xyz

    csv = tmp_path / "mols.csv"
    csv.write_text("smiles\nCCO\nc1ccccc1\n", encoding="utf-8")
    first = smiles_to_xyz(csv, tmp_path / "out_a")
    second = smiles_to_xyz(csv, tmp_path / "out_b")
    assert [p.read_text() for p in first] == [p.read_text() for p in second]

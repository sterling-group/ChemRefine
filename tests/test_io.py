"""Tests for filesystem I/O helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from chemrefine.io import (
    gather_output_files,
    natural_key,
    read_xyz,
    save_step_csv,
    write_xyz,
)

# ---------------------------------------------------------------------------
# natural_key
# ---------------------------------------------------------------------------


def test_natural_key_orders_step10_after_step2():
    files = ["step10.out", "step2.out", "step1.out"]
    assert sorted(files, key=natural_key) == ["step1.out", "step2.out", "step10.out"]


def test_natural_key_handles_paths():
    assert natural_key("step1_structure_0.out") < natural_key("step1_structure_10.out")


# ---------------------------------------------------------------------------
# write_xyz / read_xyz round-trip
# ---------------------------------------------------------------------------


def _h2o() -> Atoms:
    return Atoms(
        symbols=["O", "H", "H"],
        positions=[[0.0, 0.0, 0.117], [0.0, 0.757, -0.467], [0.0, -0.757, -0.467]],
    )


def test_write_xyz_writes_one_file_per_structure(tmp_path: Path):
    paths = write_xyz([_h2o(), _h2o()], ["0", "1"], step_number=3, output_dir=tmp_path)
    assert len(paths) == 2
    assert paths[0].name == "step3_structure_0.xyz"
    assert paths[1].name == "step3_structure_1.xyz"
    assert all(p.exists() for p in paths)


def test_write_xyz_roundtrips(tmp_path: Path):
    original = _h2o()
    [path] = write_xyz([original], ["0"], step_number=1, output_dir=tmp_path)
    back = read_xyz(path)
    assert list(back.get_chemical_symbols()) == ["O", "H", "H"]
    np.testing.assert_allclose(back.get_positions(), original.get_positions(), atol=1e-6)


def test_write_xyz_accepts_tuple_form(tmp_path: Path):
    tuples = [("H", 0.0, 0.0, 0.0), ("H", 0.74, 0.0, 0.0)]
    [path] = write_xyz([tuples], ["0"], step_number=1, output_dir=tmp_path)
    atoms = read_xyz(path)
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
    path = save_step_csv(
        [-1.0, -2.0, -0.5], ["a", "b", "c"], step_number=1, output_dir=tmp_path
    )
    rows = path.read_text().strip().splitlines()[1:]  # drop header
    conformers = [row.split(",")[1] for row in rows]
    assert conformers == ["b", "a", "c"]  # ascending by absolute energy


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

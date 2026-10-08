"""Tests for filesystem I/O helpers."""

from __future__ import annotations

import csv
import errno
import os
import shutil
import sqlite3
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from ase.io import read as ase_read

import chemrefine.io as chemrefine_io
from chemrefine.errors import OutputTreeError
from chemrefine.io import (
    gather_output_files,
    natural_key,
    read_xyz_frames,
    save_step_csv,
    write_ensemble_xyz,
    write_single_xyz,
)
from chemrefine.quantities import DEFAULT_TEMPERATURE_K
from chemrefine.state import Structure


def test_importing_io_does_not_pull_pandas():
    """pandas loads only when CSV is actually read/written, not on `import chemrefine.io`."""
    out = subprocess.run(
        [sys.executable, "-c", "import sys, chemrefine.io; print('pandas' in sys.modules)"],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert out.stdout.strip() == "False"


def test_the_net_names_a_refusal_in_a_process_that_never_loaded_sqlite(tmp_path: Path):
    """sqlite3 is never imported for the output tree's net, which works without it.

    Importing :mod:`chemrefine.io` is how every process in the package starts — the ExtOpt
    wrapper spawned once per optimizer step among them — and most never touch a database,
    so the net reads sqlite's error class off the loaded module rather than importing it.
    Run where nothing has loaded sqlite: the refused write is named, and sqlite3 is still
    not loaded after it. The child starts without the variables a coverage plugin uses to
    load itself into child interpreters, since coverage brings sqlite3 with it.
    """
    target = tmp_path / "missing" / "step1_0.inp"
    probe = (
        "import sys\n"
        "from pathlib import Path\n"
        "from chemrefine.errors import OutputTreeError\n"
        "from chemrefine.io import tree_write\n"
        f"target = Path({str(target)!r})\n"
        "try:\n"
        "    with tree_write(target):\n"
        "        target.write_text('x', encoding='utf-8')\n"
        "except OutputTreeError as e:\n"
        "    print(e)\n"
        "print('sqlite3' in sys.modules)\n"
    )
    plain = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("COV_CORE_", "COVERAGE_PROCESS_"))
    }
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
        env=plain,
    )
    assert out.stdout.splitlines() == [f"cannot write {target}: No such file or directory", "False"]


# ---------------------------------------------------------------------------
# natural_key
# ---------------------------------------------------------------------------


def test_natural_key_orders_step10_after_step2():
    files = ["step10.out", "step2.out", "step1.out"]
    assert sorted(files, key=natural_key) == ["step1.out", "step2.out", "step10.out"]


def test_natural_key_handles_paths():
    assert natural_key("step1_structure_0.out") < natural_key("step1_structure_10.out")


def test_natural_key_survives_a_superscript_digit():
    """``"²".isdigit()`` is True and ``int("²")`` raises — the guard must be isdecimal.

    The class was already fixed at ``StepConfig.matches`` and the GUI's step selector;
    this was the third ``int(isdigit)`` site, reachable from a seed directory holding a
    file with a superscript between decimal digits — which ended the run in a bare
    ``ValueError`` from the sort instead of anything naming the file.
    """
    assert natural_key("v2²3.xyz") == ["v", 2, "²", 3, ".xyz"]
    assert sorted(["v2²3.xyz", "v2²2.xyz"], key=natural_key) == ["v2²2.xyz", "v2²3.xyz"]


def test_natural_key_orders_three_digit_ids_numerically():
    """>100 structures must not sort lexicographically.

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
# write_single_xyz round-trip
# ---------------------------------------------------------------------------


def _h2o() -> Atoms:
    return Atoms(
        symbols=["O", "H", "H"],
        positions=[[0.0, 0.0, 0.117], [0.0, 0.757, -0.467], [0.0, -0.757, -0.467]],
    )


def test_write_single_xyz_to_input_geometry_path(tmp_path: Path):
    from chemrefine.ids import input_geometry_path

    path = write_single_xyz(_h2o(), input_geometry_path(tmp_path, 2, "0-1"))
    assert path == tmp_path / "0-1" / "step2_0-1_inp.xyz"
    assert path.is_file()
    back = ase_read(str(path), format="xyz")
    assert list(back.get_chemical_symbols()) == ["O", "H", "H"]


def test_write_single_xyz_roundtrips(tmp_path: Path):
    original = _h2o()
    path = write_single_xyz(original, tmp_path / "step1_0.xyz")
    back = ase_read(str(path), format="xyz")
    assert list(back.get_chemical_symbols()) == ["O", "H", "H"]
    np.testing.assert_allclose(back.get_positions(), original.get_positions(), atol=1e-6)


def test_write_single_xyz_accepts_tuple_form(tmp_path: Path):
    tuples = [("H", 0.0, 0.0, 0.0), ("H", 0.74, 0.0, 0.0)]
    path = write_single_xyz(tuples, tmp_path / "step1_0.xyz")
    atoms = ase_read(str(path), format="xyz")
    assert list(atoms.get_chemical_symbols()) == ["H", "H"]


# ---------------------------------------------------------------------------
# write_ensemble_xyz — the per-step multi-frame files
# ---------------------------------------------------------------------------


def _ensemble_structure(sid: str, *, energy: float | None, gibbs: float | None = None) -> Structure:
    return Structure(
        id=sid,
        atoms=Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]),
        energy_hartree=energy,
        gibbs_hartree=gibbs,
    )


def test_write_ensemble_xyz_round_trips_through_the_one_reader(tmp_path: Path):
    """Every frame comes back through ``read_xyz_frames`` — ragged atom counts included.

    The file exists to be opened by external viewers, but the round-trip that has to hold
    by contract is our own extxyz reader: seeding a next project from a step's ensemble is
    the natural workflow, and it must not lose frames.
    """
    structures = (
        Structure(id="0", atoms=Atoms("H", positions=[[0, 0, 0]]), energy_hartree=-1.0),
        Structure(
            id="1",
            atoms=Atoms("OH2", positions=[[0, 0, 0.117], [0, 0.757, -0.467], [0, -0.757, -0.467]]),
            energy_hartree=-76.4,
        ),
    )
    path = write_ensemble_xyz(
        structures,
        tmp_path / "step1_ensemble.xyz",
        step=1,
        energy_attr="energy_hartree",
        energy_label="E",
    )

    frames = read_xyz_frames(path)

    assert [len(f) for f in frames] == [3, 1]  # energy ascending: water first
    assert frames[0].get_chemical_symbols() == ["O", "H", "H"]
    np.testing.assert_allclose(
        frames[0].get_positions(), structures[1].atoms.get_positions(), atol=1e-6
    )


def test_write_ensemble_xyz_orders_by_energy_with_energyless_last(tmp_path: Path):
    """Ascending by the ranking energy; frames without one keep their incoming order, last.

    The incoming order is the cache's stable manifest order, so the tail is deterministic
    too — an ``on_failure: best`` backfill (no energy yet) lands after every ranked frame
    without shuffling against its fellow backfills.
    """
    structures = (
        _ensemble_structure("no-e-first", energy=None),
        _ensemble_structure("high", energy=-1.0),
        _ensemble_structure("no-e-second", energy=None),
        _ensemble_structure("low", energy=-2.0),
    )
    path = write_ensemble_xyz(
        structures,
        tmp_path / "step1_ensemble.xyz",
        step=1,
        energy_attr="energy_hartree",
        energy_label="E",
    )

    ids = [
        line.split()[1]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("step1 ")
    ]

    assert ids == ["id=low", "id=high", "id=no-e-first", "id=no-e-second"]


def test_write_ensemble_xyz_comment_carries_step_id_and_energy(tmp_path: Path):
    """The exact caption contract: id ties the frame to its directory and steps.csv row."""
    path = write_ensemble_xyz(
        (_ensemble_structure("0-3", energy=-153.123456789),),
        tmp_path / "step1_ensemble.xyz",
        step=1,
        energy_attr="energy_hartree",
        energy_label="E",
    )
    assert path.read_text(encoding="utf-8").splitlines()[1] == "step1 id=0-3 E=-153.12345679 Eh"


def test_write_ensemble_xyz_captions_the_steps_own_ranking_energy(tmp_path: Path):
    """A Gibbs-ranked step sorts and captions by Gibbs — the same energy its filter uses.

    The two structures disagree on purpose: electronic order is 0 < 1, Gibbs order is
    1 < 0, so a writer still reading electronic energy would fail on both the order and
    the caption.
    """
    structures = (
        _ensemble_structure("0", energy=-2.0, gibbs=-1.0),
        _ensemble_structure("1", energy=-1.0, gibbs=-2.0),
    )
    path = write_ensemble_xyz(
        structures,
        tmp_path / "step2_ensemble.xyz",
        step=2,
        energy_attr="gibbs_hartree",
        energy_label="G",
    )
    comments = [
        line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("step2 ")
    ]
    assert comments == ["step2 id=1 G=-2.00000000 Eh", "step2 id=0 G=-1.00000000 Eh"]


def test_write_ensemble_xyz_marks_a_missing_energy_as_na(tmp_path: Path):
    """No value is a stated ``n/a``, never a fabricated number the frame would rank by."""
    path = write_ensemble_xyz(
        (_ensemble_structure("0", energy=None),),
        tmp_path / "step1_ensemble.xyz",
        step=1,
        energy_attr="energy_hartree",
        energy_label="E",
    )
    assert path.read_text(encoding="utf-8").splitlines()[1] == "step1 id=0 E=n/a"


def test_write_ensemble_xyz_empty_input_removes_a_stale_file(tmp_path: Path):
    """A step that kept nothing must not leave the previous run's survivors lying about.

    Same reasoning as ``save_step_csv``'s step-1 truncation: the file describes *this*
    run's outcome, and an empty outcome is an absent file, not a zero-frame one ase would
    refuse to read anyway.
    """
    path = tmp_path / "step1_survivors.xyz"
    path.write_text("stale\n", encoding="utf-8")

    write_ensemble_xyz((), path, step=1, energy_attr="energy_hartree", energy_label="E")

    assert not path.exists()
    write_ensemble_xyz(
        (), path, step=1, energy_attr="energy_hartree", energy_label="E"
    )  # absent stays absent, no raise


# ---------------------------------------------------------------------------
# gather_output_files
# ---------------------------------------------------------------------------


def test_gather_output_files_returns_natural_order(tmp_path: Path):
    for name in ["step10.out", "step1.out", "step2.out"]:
        (tmp_path / name).touch()
    found = gather_output_files(tmp_path, "*.out")
    assert [p.name for p in found] == ["step1.out", "step2.out", "step10.out"]


def test_gather_output_files_orders_more_than_100_structures(tmp_path: Path):
    """A seed dir with >100 structures must glob in numeric order."""
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
    path = save_step_csv(
        [-1.0, -1.001],
        ["0", "1"],
        step_number=1,
        path=tmp_path / "steps.csv",
        temperature_k=DEFAULT_TEMPERATURE_K,
        energy_type="electronic",
    )
    text = path.read_text(encoding="utf-8")
    assert "Step,Conformer,Energy (Hartree)" in text.splitlines()[0]
    assert "% Cumulative" in text.splitlines()[0]


def test_save_step_csv_appends_without_header_on_later_steps(tmp_path: Path):
    save_step_csv(
        [-1.0],
        ["0"],
        step_number=1,
        path=tmp_path / "steps.csv",
        temperature_k=DEFAULT_TEMPERATURE_K,
        energy_type="electronic",
    )
    save_step_csv(
        [-2.0],
        ["1"],
        step_number=2,
        path=tmp_path / "steps.csv",
        temperature_k=DEFAULT_TEMPERATURE_K,
        energy_type="electronic",
    )
    text = (tmp_path / "steps.csv").read_text(encoding="utf-8")
    # header appears exactly once
    assert text.count("Step,Conformer,Energy (Hartree)") == 1


def test_save_step_csv_sorts_by_energy(tmp_path: Path):
    path = save_step_csv(
        [-1.0, -2.0, -0.5],
        ["a", "b", "c"],
        step_number=1,
        path=tmp_path / "steps.csv",
        temperature_k=DEFAULT_TEMPERATURE_K,
        energy_type="electronic",
    )
    rows = path.read_text(encoding="utf-8").strip().splitlines()[1:]  # drop header
    conformers = [row.split(",")[1] for row in rows]
    assert conformers == ["b", "a", "c"]  # ascending by absolute energy


def test_save_step_csv_derived_columns_carry_the_right_numbers(tmp_path: Path):
    """The report's numbers are pinned by literals, not by shape.

    Every other assertion on this file checks headers, order or truncation — all of
    which survive a wrong unit factor or a wrong percentage scale, because both are
    order-preserving. This is the user-facing artifact of the whole run, so the values
    are held to independently computed literals: two energies exactly 1 kcal/mol apart
    (the second is ``-1 Eh + 1/627.5094740629``), whose Boltzmann split at the default
    298.15 K is 84.39/15.61 — numbers a mutation of ``HARTREE_TO_KCALMOL``'s use, the
    ``* 100.0`` percentage scale, or the default temperature cannot reproduce.
    """
    path = save_step_csv(
        [-1.0, -1.0 + 1.0 / 627.5094740629],
        ["low", "high"],
        step_number=1,
        path=tmp_path / "steps.csv",
        temperature_k=DEFAULT_TEMPERATURE_K,
        energy_type="electronic",
    )
    with path.open(encoding="utf-8", newline="") as handle:
        low, high = list(csv.DictReader(handle))
    assert low["Conformer"] == "low" and high["Conformer"] == "high"
    assert float(low["Energy (kcal/mol)"]) == pytest.approx(-627.50947406, abs=1e-6)
    assert float(high["Energy (kcal/mol)"]) == pytest.approx(-626.50947406, abs=1e-6)
    assert float(low["dE (kcal/mol)"]) == 0.0
    assert float(high["dE (kcal/mol)"]) == pytest.approx(1.0, abs=1e-8)
    assert float(low["Boltzmann Weight"]) == pytest.approx(0.84393549, abs=1e-6)
    assert float(high["Boltzmann Weight"]) == pytest.approx(0.15606451, abs=1e-6)
    assert float(low["% Total"]) == pytest.approx(84.39354867, abs=1e-4)
    assert float(high["% Total"]) == pytest.approx(15.60645133, abs=1e-4)
    assert float(low["% Cumulative"]) == pytest.approx(84.39354867, abs=1e-4)
    assert float(high["% Cumulative"]) == pytest.approx(100.0, abs=1e-6)


def test_save_step_csv_all_nan_energies_skips_write(tmp_path: Path):
    """No finite energies → skip the summary instead of crashing on an empty frame."""
    path = save_step_csv(
        [float("nan"), float("nan")],
        ["0", "1"],
        step_number=1,
        path=tmp_path / "steps.csv",
        temperature_k=DEFAULT_TEMPERATURE_K,
        energy_type="electronic",
    )
    assert not path.exists()


def test_save_step_csv_step_one_truncates_stale_file(tmp_path: Path):
    """A step 1 with no finite energies removes a prior run's report.

    Otherwise step 2 would append onto the stale step-1 rows from the
    previous run instead of starting a fresh report.
    """
    stale = save_step_csv(
        [-1.0],
        ["0"],
        step_number=1,
        path=tmp_path / "steps.csv",
        temperature_k=DEFAULT_TEMPERATURE_K,
        energy_type="electronic",
    )
    assert stale.exists()
    path = save_step_csv(
        [float("nan")],
        ["0"],
        step_number=1,
        path=tmp_path / "steps.csv",
        temperature_k=DEFAULT_TEMPERATURE_K,
        energy_type="electronic",
    )
    assert not path.exists()


def test_save_step_csv_writes_a_header_when_step_one_wrote_nothing(tmp_path: Path):
    """The report must never start with a data row.

    Truncation and the header keyed off the same `step_number == 1` leave a step 1 that
    summarises nothing followed by a step 2 appending `header=False` onto a file that does not
    exist -- producing a CSV whose first line is data, which every `read_csv` then eats as the
    column names.

    Reachable without contrivance: `on_failure: best` on step 1 with no `sample:` keeps the
    backfilled seeds, and those carry no energy yet.
    """
    assert not save_step_csv(
        [None, None],
        ["0", "1"],
        step_number=1,
        path=tmp_path / "steps.csv",
        temperature_k=DEFAULT_TEMPERATURE_K,
        energy_type="electronic",
    ).exists()

    path = save_step_csv(
        [-1.0, -1.5],
        ["0", "1"],
        step_number=2,
        path=tmp_path / "steps.csv",
        temperature_k=DEFAULT_TEMPERATURE_K,
        energy_type="electronic",
    )

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
    # The blank line never reaches the loop: pandas' default skip_blank_lines=True drops
    # it at read time, so this file exercises the invalid-SMILES skip and the valid row
    # only. The blank/NaN skip branch is covered by the next test, which patches the
    # DataFrame in precisely because a CSV cannot deliver those rows — do not delete it
    # on the strength of this file's shape.
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
    """A non-zero return from ``EmbedMolecule`` triggers a warning + skip.

    Patched on ``rdkit.Chem.rdDistGeom``, the module that defines it and the one
    :func:`~chemrefine.io.smiles_to_xyz` imports it from.
    """
    from rdkit.Chem import rdDistGeom

    from chemrefine.io import smiles_to_xyz

    csv = tmp_path / "one.csv"
    csv.write_text("smiles\nC\n", encoding="utf-8")
    with patch.object(rdDistGeom, "EmbedMolecule", return_value=1):
        written = smiles_to_xyz(csv, tmp_path / "out")
    assert written == []


def test_embed_smiles_raises_at_the_molecule_that_would_not_embed():
    """The single-molecule seam raises where the CSV sweep above logs and skips.

    That opposite failure behaviour is why :func:`embed_smiles` is public: the agent's
    structure-building tool embeds one explicit SMILES and wants the refusal at the
    molecule it names, not a warning in a log it is not reading.
    """
    from rdkit.Chem import rdDistGeom

    from chemrefine.io import embed_smiles

    with pytest.raises(ValueError, match="invalid SMILES: not-a-molecule"):
        embed_smiles("not-a-molecule")
    with (
        patch.object(rdDistGeom, "EmbedMolecule", return_value=1),
        pytest.raises(ValueError, match="failed 3D embedding for SMILES: CCO"),
    ):
        embed_smiles("CCO")


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
    assert [p.read_text(encoding="utf-8") for p in first] == [
        p.read_text(encoding="utf-8") for p in second
    ]


# ---------------------------------------------------------------------------
# atomic_write — the one writer for every file this package replaces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", [0o600, 0o755])
def test_a_requested_mode_is_on_the_file_before_it_appears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: int
):
    """The mode is set on the temp file, so ``path`` never exists with any other.

    A token readable for an instant, or a wrapper ORCA finds but cannot execute, is what
    a chmod after the rename would allow; the rename is where the check has to look.
    """
    import stat

    at_rename: list[int] = []
    real_replace = Path.replace

    def watched(self: Path, target: Path) -> Path:
        at_rename.append(stat.S_IMODE(self.stat().st_mode))
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", watched)
    target = tmp_path / "file"
    chemrefine_io.atomic_write(target, b"x", mode=mode)
    assert at_rename == [mode]
    assert stat.S_IMODE(target.stat().st_mode) == mode


def test_a_write_that_cannot_re_mode_leaves_nothing_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A failed re-mode must strand neither the temp file nor its descriptor.

    ``fchmod`` is the one step between ``mkstemp`` and ``fdopen`` that genuinely fails —
    shared filesystems and FUSE/CIFS mounts refuse it. Done before the ``fdopen`` it took
    both halves with it: nothing owned the descriptor yet, and the ``finally`` had not been
    entered, so a ``.tmp_*.part`` stayed in the directory the user was writing to. That is a
    nuisance once and an accumulation on the GUI's waitress threads, which write the user's
    config through this same writer.
    """
    monkeypatch.setattr(os, "fchmod", lambda *a: (_ for _ in ()).throw(OSError("EPERM")))
    target = tmp_path / "doc.json"
    with pytest.raises(OSError, match="EPERM"):
        chemrefine_io.atomic_write(target, b'{"a": 1}')

    assert not target.exists(), "a failed write must not leave a partial document"
    assert list(tmp_path.iterdir()) == [], (
        f"temp residue survived a failed re-mode: {[p.name for p in tmp_path.iterdir()]}"
    )


@pytest.mark.skipif(
    not chemrefine_io._PROC_STATUS.is_file(),
    reason="this kernel does not publish Umask; only the fallback probe exists here",
)
def test_the_umask_is_read_without_ever_being_written(monkeypatch: pytest.MonkeyPatch):
    """Reading the umask by setting it is a process-global write other threads see.

    ``os.umask`` reads by writing, so between the clear and the restore anything another
    thread creates is made with no mask at all and a ``mkdir`` lands 0777 — world-writable
    directories under an output tree that is routinely shared. The window is reachable:
    the GUI runs on four waitress threads whose handlers both write through this module
    and call ``Path.mkdir``. Linux publishes the value instead (``umask(2)``), so nothing
    here may call ``os.umask`` at all.
    """
    expected = next(
        int(line.split()[1], 8)
        for line in chemrefine_io._PROC_STATUS.read_text(encoding="utf-8").splitlines()
        if line.startswith("Umask:")
    )
    monkeypatch.setattr(os, "umask", lambda _mask: pytest.fail("read the umask by setting it"))

    assert chemrefine_io._umask() == expected


@pytest.mark.parametrize("published", ["no status file at all", "a status file without the field"])
def test_the_umask_falls_back_when_the_kernel_does_not_publish_it(
    published: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A kernel older than 4.7 has no ``Umask`` field, and must still get the right mode.

    Both ways the field can be absent take the probe, and the probe has to put back what
    it set — otherwise the first cache write of the run would leave the process at umask
    0 and every file after it world-writable, which is worse than the race it replaces.
    Asking twice is what proves the restore happened.
    """
    status = tmp_path / "status"
    if published == "a status file without the field":
        status.write_text("Name:\tpython\nPid:\t1\n", encoding="utf-8")
    monkeypatch.setattr(chemrefine_io, "_PROC_STATUS", status)

    previous = os.umask(0o027)
    try:
        assert chemrefine_io._umask() == 0o027
        assert chemrefine_io._umask() == 0o027
    finally:
        os.umask(previous)


# ---------------------------------------------------------------------------
# The output tree's write seam — tree_write, its whole-file forms, cleanup
# ---------------------------------------------------------------------------

_AS_ROOT = os.geteuid() == 0


def _refuse(code: int) -> Callable[..., NoReturn]:
    """A stand-in for a call the filesystem refuses with ``code``, in its own words."""

    def refuse(*_args: object, **_kwargs: object) -> NoReturn:
        raise OSError(code, os.strerror(code))

    return refuse


@pytest.mark.parametrize(
    "in_the_way",
    [
        pytest.param(
            False,
            id="unwritable-directory",
            marks=pytest.mark.skipif(_AS_ROOT, reason="root writes through any mode"),
        ),
        pytest.param(True, id="directory-in-the-way"),
    ],
)
def test_atomic_write_names_the_file_it_was_asked_for(tmp_path: Path, in_the_way: bool):
    """A refusal names ``path``, never the temp file the data was staged in.

    Whoever could not save a config or a template reads the error's filename, and a
    ``.tmp_*.part`` is a name they never saw and cannot find. Checked where the temp file is
    created, in a directory this account cannot write, and where it is renamed, over a
    directory standing where the file should be.
    """
    folder = tmp_path / "templates"
    folder.mkdir()
    target = folder / "step1.inp"
    if in_the_way:
        (target / "kept").mkdir(parents=True)
    else:
        folder.chmod(0o555)
    try:
        with pytest.raises(OSError) as excinfo:
            chemrefine_io.atomic_write(target, b"! Opt\n")
        left = sorted(p.name for p in folder.iterdir())
    finally:
        folder.chmod(0o755)
    assert excinfo.value.filename == str(target)
    assert ".tmp_" not in str(excinfo.value)
    assert left == (["step1.inp"] if in_the_way else [])


@pytest.mark.skipif(_AS_ROOT, reason="root writes through any mode")
@pytest.mark.parametrize("atomic", [True, False], ids=["atomic", "plain"])
def test_a_directory_the_run_cannot_write_is_the_output_trees_refusal(tmp_path: Path, atomic: bool):
    """A write into a directory this account cannot write is ``OutputTreeError``.

    One class and one sentence whichever way the file is written: the file the run asked
    for, then the filesystem's reason, with the ``PermissionError`` on ``__cause__`` and
    nothing left behind.
    """
    sealed = tmp_path / "_cache"
    sealed.mkdir()
    sealed.chmod(0o555)
    target = sealed / "step.json"
    try:
        with pytest.raises(OutputTreeError) as excinfo:
            chemrefine_io.write_tree_file(target, "{}", atomic=atomic)
        left = list(sealed.iterdir())
    finally:
        sealed.chmod(0o755)
    assert str(excinfo.value) == f"cannot write {target}: Permission denied"
    assert isinstance(excinfo.value.__cause__, PermissionError)
    assert left == []


@pytest.mark.parametrize("atomic", [True, False], ids=["atomic", "plain"])
def test_a_full_disk_is_named_at_the_file_being_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, atomic: bool
):
    """A disk that fills mid-write reads as the file the run was writing and the reason."""
    monkeypatch.setattr(os, "fsync", _refuse(errno.ENOSPC))
    monkeypatch.setattr(Path, "write_bytes", _refuse(errno.ENOSPC))
    target = tmp_path / "step1" / "step1_ensemble.xyz"
    with pytest.raises(OutputTreeError) as excinfo:
        chemrefine_io.write_tree_file(target, "frame", atomic=atomic)
    assert str(excinfo.value) == f"cannot write {target}: No space left on device"


def test_an_atomic_tree_write_refused_mid_write_keeps_the_file_it_replaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A refused atomic write leaves the previous file whole and no temp file beside it."""
    target = tmp_path / "_cache" / "step.json"
    target.parent.mkdir()
    target.write_text("previous", encoding="utf-8")
    monkeypatch.setattr(os, "fsync", _refuse(errno.ENOSPC))
    with pytest.raises(OutputTreeError):
        chemrefine_io.write_tree_file(target, "next", atomic=True)
    assert target.read_text(encoding="utf-8") == "previous"
    assert [p.name for p in target.parent.iterdir()] == ["step.json"]


@pytest.mark.skipif(_AS_ROOT, reason="root writes and reads through any mode")
def test_a_refusal_names_what_refused_it_when_that_is_not_the_file(tmp_path: Path):
    """The file the filesystem named comes before its reason when it is not the one written.

    A write whose directory cannot be created names that directory; a copy whose source
    cannot be read names the source. Each is the file to look at, and the sentence still
    opens with the file the run was writing.
    """
    sealed = tmp_path / "step1"
    sealed.mkdir()
    source = tmp_path / "attempt1" / "step1_3.out"
    source.parent.mkdir()
    source.write_text("x", encoding="utf-8")
    sealed.chmod(0o555)
    source.chmod(0)
    target = sealed / "0" / "step1_0_inp.xyz"
    destination = tmp_path / "step1_0.out"
    try:
        with pytest.raises(OutputTreeError) as made:
            chemrefine_io.write_tree_file(target, "x")
        with pytest.raises(OutputTreeError) as copied, chemrefine_io.tree_write(destination):
            shutil.copy2(source, destination)
    finally:
        sealed.chmod(0o755)
        source.chmod(0o644)
    assert str(made.value) == f"cannot write {target}: {target.parent}: Permission denied"
    assert str(copied.value) == f"cannot write {destination}: {source}: Permission denied"


def test_a_refusal_with_no_reason_of_its_own_is_given_as_its_text(tmp_path: Path):
    """An error the filesystem's wording is missing from reads as the text it carries."""
    target = tmp_path / "step1" / "steps.csv"
    with pytest.raises(OutputTreeError) as excinfo, chemrefine_io.tree_write(target):
        raise OSError("the writer gave up")
    assert str(excinfo.value) == f"cannot write {target}: the writer gave up"


@pytest.mark.parametrize("atomic", [True, False], ids=["atomic", "plain"])
def test_write_tree_file_creates_the_directory_and_writes_text_as_utf8(
    tmp_path: Path, atomic: bool
):
    """Missing parents are created; text lands as its UTF-8 bytes, bytes as they are."""
    target = tmp_path / "step1" / "0" / "step1_0_inp.xyz"
    chemrefine_io.write_tree_file(target, "Å 0.0", atomic=atomic)
    assert target.read_bytes() == "Å 0.0".encode()
    chemrefine_io.write_tree_file(target, b"NUMPY", atomic=atomic)
    assert target.read_bytes() == b"NUMPY"


def _report(step: int) -> Callable[[Path], object]:
    """``save_step_csv`` for one structure at ``step``, aimed at the path it is handed."""
    return lambda target: save_step_csv(
        [-1.0], ["0"], step, target, temperature_k=DEFAULT_TEMPERATURE_K, energy_type="electronic"
    )


def _ensemble(*structures: Structure) -> Callable[[Path], object]:
    """``write_ensemble_xyz`` of ``structures`` as step 1, aimed at the path it is handed."""
    return lambda target: write_ensemble_xyz(
        structures, target, step=1, energy_attr="energy_hartree", energy_label="E"
    )


@pytest.mark.skipif(_AS_ROOT, reason="root writes through any mode")
@pytest.mark.parametrize(
    ("relative", "stale", "write"),
    [
        ("step1_0_inp.xyz", False, lambda target: write_single_xyz(_h2o(), target)),
        ("step1_ensemble.xyz", False, _ensemble(_ensemble_structure("0", energy=-1.0))),
        ("step1_survivors.xyz", True, _ensemble()),
        ("steps.csv", False, _report(2)),
        ("steps.csv", True, _report(1)),
        ("reports/steps.csv", False, _report(2)),
    ],
    ids=["geometry", "ensemble", "stale-ensemble", "report", "stale-report", "report-directory"],
)
def test_io_writers_refuse_as_the_output_tree_at_the_file_they_write(
    tmp_path: Path, relative: str, stale: bool, write: Callable[[Path], object]
):
    """Each of io's writers refuses as the output tree: the file it writes, then the reason.

    A sealed directory refuses all three things these writers do — create a file, remove a
    stale one, create the directory a report goes in — and each refusal is
    ``OutputTreeError`` naming the file the writer was handed, then the directory when that
    is what was refused, with a stale file left as it was.
    """
    sealed = tmp_path / "step1"
    sealed.mkdir()
    target = sealed / relative
    if stale:
        target.write_text("stale\n", encoding="utf-8")
    sealed.chmod(0o555)
    try:
        with pytest.raises(OutputTreeError) as excinfo:
            write(target)
    finally:
        sealed.chmod(0o755)
    refused = "" if target.parent == sealed else f"{target.parent}: "
    assert str(excinfo.value) == f"cannot write {target}: {refused}Permission denied"
    assert isinstance(excinfo.value.__cause__, PermissionError)
    assert not stale or target.read_text(encoding="utf-8") == "stale\n"


_CLEANUP_REFUSALS = [
    pytest.param(
        lambda: OSError(errno.EROFS, os.strerror(errno.EROFS), "/out/.tmp_x.part"), id="raw"
    ),
    pytest.param(
        lambda: OutputTreeError("cannot write /out/.chemrefine.lock: Read-only file system"),
        id="tree-write",
    ),
]
"""What a cleanup can meet: a temp file's raw removal, or a write through the tree's net."""


@pytest.mark.parametrize("refusal", _CLEANUP_REFUSALS)
@pytest.mark.parametrize(
    "stopping",
    [
        lambda: SystemExit(143),
        KeyboardInterrupt,
        lambda: OutputTreeError("cannot write /out/step1/_cache/step.json: No space left"),
    ],
    ids=["sigterm", "interrupt", "refused-write"],
)
def test_cleanup_during_an_unwind_keeps_the_exception_in_flight(
    stopping: Callable[[], BaseException],
    refusal: Callable[[], Exception],
    caplog: pytest.LogCaptureFixture,
):
    """A cleanup the filesystem refuses while a run stops is logged; the stop goes on.

    Raised, the refusal would take the place of what was stopping the run — a SIGTERM's
    143, a Ctrl-C, a refused write — and the run would exit with the cleanup's code and
    sentence. The very exception in flight is the one that leaves, whether the cleanup met
    the raw ``OSError`` of a temp file's removal or the ``OutputTreeError`` of a lock or a
    lease ledger released through the output tree's net.
    """
    in_flight = stopping()
    with pytest.raises(type(in_flight)) as excinfo:
        try:
            raise in_flight
        finally:
            with chemrefine_io.cleanup():
                raise refusal()
    assert excinfo.value is in_flight
    assert f"could not clean up during {type(in_flight).__name__}" in caplog.text
    assert "Read-only file system" in caplog.text


@pytest.mark.parametrize("refusal", _CLEANUP_REFUSALS)
def test_cleanup_with_nothing_in_flight_raises_its_refusal(refusal: Callable[[], Exception]):
    """With no exception in flight, a refused cleanup is the caller's failure, raised."""
    refused = refusal()
    with pytest.raises(type(refused)) as excinfo, chemrefine_io.cleanup():
        raise refused
    assert excinfo.value is refused


def test_cleanup_lets_a_bug_through():
    """Only a refusal is dropped while an exception is in flight; a bug surfaces."""
    with pytest.raises(TypeError, match="not a refusal"):
        try:
            raise SystemExit(143)
        finally:
            with chemrefine_io.cleanup():
                raise TypeError("not a refusal")


def test_a_temp_file_that_will_not_go_does_not_hide_why_the_write_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """A temp file that cannot be removed after a failed write leaves the write's own error.

    The full disk is why the write failed and what the caller words; the removal's I/O
    error is logged beside it rather than raised in its place.
    """
    monkeypatch.setattr(os, "fsync", _refuse(errno.ENOSPC))
    monkeypatch.setattr(Path, "unlink", _refuse(errno.EIO))
    with pytest.raises(OSError) as excinfo:
        chemrefine_io.atomic_write(tmp_path / "doc.json", b"{}")
    assert excinfo.value.errno == errno.ENOSPC
    assert "could not clean up during OSError" in caplog.text
    assert "Input/output error" in caplog.text


def test_a_copy_that_gathers_its_refusals_is_named_by_the_first(tmp_path: Path):
    """``copytree`` tries every file, then raises what it could not copy all together.

    Given as its own text that is a list's repr; the refusal reads as the first file the
    copy could not write, in the filesystem's words, and how many more there were.
    """
    destination = tmp_path / "step1" / "0" / "tensors"
    rows = [
        (
            str(tmp_path / f"{name}.npz"),
            str(destination / f"{name}.npz"),
            f"[Errno 28] No space left on device: '{destination / name}.npz'",
        )
        for name in ("hessian", "gradient")
    ]
    with pytest.raises(OutputTreeError) as two, chemrefine_io.tree_write(destination):
        raise shutil.Error(rows)
    with pytest.raises(OutputTreeError) as one, chemrefine_io.tree_write(destination):
        raise shutil.Error(rows[:1])
    assert str(two.value) == f"cannot write {destination}: {rows[0][2]} (and 1 more)"
    assert str(one.value) == f"cannot write {destination}: {rows[0][2]}"


@pytest.mark.parametrize(
    ("code", "sentence"),
    [
        (sqlite3.SQLITE_CANTOPEN, "unable to open database file"),
        (sqlite3.SQLITE_FULL, "database or disk is full"),
        (sqlite3.SQLITE_IOERR_WRITE, "disk I/O error"),
        (sqlite3.SQLITE_READONLY_DIRECTORY, "attempt to write a readonly database"),
        (sqlite3.SQLITE_PERM, "access permission denied"),
    ],
    ids=["cantopen", "full", "ioerr-write", "readonly-directory", "perm"],
)
def test_a_sqlite_refusal_is_the_output_trees(tmp_path: Path, code: int, sentence: str):
    """sqlite refuses a write in an exception of its own, and the net names it all the same.

    A database in a full or read-only tree says so through :class:`sqlite3.OperationalError`,
    no ``OSError``: named by its code's family — a write cut short and a read-only directory
    arrive as extended codes — with sqlite's own sentence as the reason.
    """
    refusal = sqlite3.OperationalError(sentence)
    refusal.sqlite_errorcode = code
    target = tmp_path / "train.db"
    with pytest.raises(OutputTreeError) as named, chemrefine_io.tree_write(target):
        raise refusal
    assert str(named.value) == f"cannot write {target}: {sentence}"
    assert named.value.__cause__ is refusal


@pytest.mark.parametrize(
    "code", [sqlite3.SQLITE_ERROR, sqlite3.SQLITE_BUSY, None], ids=["sql", "busy", "no-code"]
)
def test_a_sqlite_error_that_refuses_no_write_leaves_as_itself(tmp_path: Path, code: int | None):
    """A SQL mistake, a busy lock, an error with no code: none is the filesystem's refusal."""
    error = sqlite3.OperationalError("no such table: systems")
    if code is not None:
        error.sqlite_errorcode = code
    with (
        pytest.raises(sqlite3.OperationalError) as passed,
        chemrefine_io.tree_write(tmp_path / "train.db"),
    ):
        raise error
    assert passed.value is error


# ---------------------------------------------------------------------------
# The output tree's questions — absent
# ---------------------------------------------------------------------------


def test_only_no_such_file_proves_that_nothing_is_there(tmp_path: Path):
    """``absent`` is the filesystem's own "no such file or directory" and nothing else.

    A path through a regular file is a tree out of shape, not an output that was never
    written: the read that follows names it.
    """
    held = tmp_path / "step1_0.out"
    held.write_text("", encoding="utf-8")
    assert chemrefine_io.absent(tmp_path / "step1_1.out")
    assert chemrefine_io.absent(tmp_path / "1" / "step1_1.out")
    assert not chemrefine_io.absent(held)
    assert not chemrefine_io.absent(tmp_path)
    assert not chemrefine_io.absent(held / "step1_0.out")


@pytest.mark.skipif(_AS_ROOT, reason="root reads through any mode")
def test_a_file_this_account_cannot_reach_is_not_called_absent(tmp_path: Path):
    """Under a directory it cannot search the run cannot tell, and says so on every Python.

    ``Path.is_file`` raises there on Python 3.11 to 3.13 and answers ``False`` on 3.14;
    both a file that is there and a name that is not are "not provably absent" here.
    """
    locked = tmp_path / "0"
    locked.mkdir()
    (locked / "step1_0.out").write_text("", encoding="utf-8")
    locked.chmod(0)
    try:
        answers = [chemrefine_io.absent(locked / name) for name in ("step1_0.out", "never.out")]
    finally:
        locked.chmod(0o755)
    assert answers == [False, False]

"""Tests for ``chemrefine.attempts`` — sealing a structure's state away and installing a new one.

The two callers compose these operations differently — the convergence retry seals and re-runs,
NMS seals around work it has already done inside the attempt — so the primitives are tested
here directly rather than through either.
"""

from __future__ import annotations

from pathlib import Path

from chemrefine import attempts


def test_archive_numbers_attempts_sequentially(tmp_path: Path):
    """Next-free K, never blocked by an existing or oddly-named one."""
    sid_dir = tmp_path / "0"
    sid_dir.mkdir()
    (sid_dir / "step1_0.out").write_text("fail", encoding="utf-8")
    dest = attempts.archive(sid_dir)
    assert dest.name == "attempt1"
    assert (dest / "step1_0.out").is_file()  # the loose file moved in
    assert not (sid_dir / "step1_0.out").exists()

    (sid_dir / "step1_0.out").write_text("fail2", encoding="utf-8")
    assert attempts.archive(sid_dir).name == "attempt2"  # next free

    # A manually-added higher attempt + a non-matching 'attempt*' dir: K = max+1,
    # the odd dir is ignored, and existing attempt dirs are left in place.
    (sid_dir / "attempt5").mkdir()
    (sid_dir / "attemptX").mkdir()  # matches the glob but not attempt<digits>
    (sid_dir / "step1_0.out").write_text("fail3", encoding="utf-8")
    assert attempts.archive(sid_dir).name == "attempt6"
    assert (sid_dir / "attempt1").is_dir() and (sid_dir / "attempt5").is_dir()


def test_begin_creates_the_directory_seal_fills_it(tmp_path: Path):
    """The two halves compose to what ``archive`` does in one call.

    NMS needs them apart: it runs its children inside the attempt before there is anything
    to seal into it.
    """
    sid_dir = tmp_path / "0"
    sid_dir.mkdir()
    attempt = attempts.begin(sid_dir)
    assert attempt.is_dir() and attempt.name == "attempt1"

    (sid_dir / "step1_0.out").write_text("round 1", encoding="utf-8")
    attempts.seal(sid_dir, attempt)
    assert (attempt / "step1_0.out").read_text() == "round 1"
    assert not (sid_dir / "step1_0.out").exists()


def test_seal_takes_the_engine_written_directory_with_it(tmp_path: Path):
    """``tensors/`` belongs to the attempt that produced it, not to the path it sits at.

    ``pyscf-extopt``'s ``save_tensors`` writes a whole directory into the structure dir. Left
    behind, the next calculation at that path writes its own files beside the stale ones and
    the directory describes two runs at once. Only ``attempt*/`` is exempt — folding one
    attempt into another would lose a run's history.
    """
    structure_dir = tmp_path / "0"
    (structure_dir / "tensors").mkdir(parents=True)
    (structure_dir / "tensors" / "active.npz").write_text("round 1")
    (structure_dir / "step1_0.out").write_text("round 1")
    (structure_dir / "attempt1").mkdir()
    (structure_dir / "attempt1" / "step1_0.out").write_text("round 0")

    dest = attempts.archive(structure_dir)

    assert dest.name == "attempt2"
    assert (dest / "tensors" / "active.npz").read_text() == "round 1"
    assert (dest / "step1_0.out").read_text() == "round 1"
    assert sorted(p.name for p in structure_dir.iterdir()) == ["attempt1", "attempt2"]
    assert (structure_dir / "attempt1" / "step1_0.out").read_text() == "round 0"


def test_promote_rewrites_the_id_prefix_not_the_suffix(tmp_path: Path):
    """Artifact names are not all single-extension, so promotion is a prefix swap.

    ``step1_0_m6_pos_trj.xyz`` and ``step1_0_m6_pos.property.json`` both have to land as the
    parent's, which ``with_suffix`` would mangle. Anything not named after the source is left
    alone — copying it up under an unchanged name would attribute it to the parent falsely.

    An engine-written directory (``pyscf-extopt``'s ``tensors/``) is named by the engine and
    not after the structure, so it is promoted under its own name. Leaving it behind would
    reintroduce the disagreement promotion exists to remove, one level down.
    """
    attempt = tmp_path / "0" / "attempt1"
    child = attempt / "0_m6_pos"
    child.mkdir(parents=True)
    for name in (
        "step1_0_m6_pos.out",
        "step1_0_m6_pos_trj.xyz",
        "step1_0_m6_pos.property.json",
        "unrelated.log",
    ):
        (child / name).write_text(name)
    (child / "tensors").mkdir()
    (child / "tensors" / "active.npz").write_text("winner tensors")

    attempts.promote(attempt, step=1, source_id="0_m6_pos", target_id="0")

    assert sorted(p.name for p in (tmp_path / "0").iterdir()) == [
        "attempt1",
        "step1_0.out",
        "step1_0.property.json",
        "step1_0_trj.xyz",
        "tensors",
    ]
    assert (tmp_path / "0" / "step1_0.out").read_text() == "step1_0_m6_pos.out"
    assert (tmp_path / "0" / "tensors" / "active.npz").read_text() == "winner tensors"
    assert (child / "step1_0_m6_pos.out").is_file(), "the source dir is copied from, not emptied"


def test_promoting_a_retried_source_leaves_its_own_attempt_behind(tmp_path: Path):
    """A source that was itself retried carries an ``attemptK/``; that must not travel with it.

    The retry runs against the child's own directory, so an unconverged one gains
    ``attemptK/<child>/attempt1/`` holding the run that failed. Promotion must copy the
    calculation that won, not the one it discarded — and the destination for a directory named
    ``attempt1`` is the parent's ``attempt1``, which is the very directory being read from.
    """
    attempt = tmp_path / "0" / "attempt1"
    child = attempt / "0_m5_pos"
    (child / "attempt1").mkdir(parents=True)
    (child / "attempt1" / "step1_0_m5_pos.out").write_text("the child's discarded run")
    (child / "step1_0_m5_pos.out").write_text("the run that won")
    (attempt / "step1_0.out").write_text("round 1, archived")

    attempts.promote(attempt, step=1, source_id="0_m5_pos", target_id="0")

    assert (tmp_path / "0" / "step1_0.out").read_text() == "the run that won"
    assert (attempt / "step1_0.out").read_text() == "round 1, archived", (
        "the discarded run must not overwrite the archived round 1"
    )
    assert sorted(p.name for p in attempt.iterdir()) == ["0_m5_pos", "step1_0.out"]


def test_archive_previous_skips_a_structure_with_no_loose_files(tmp_path: Path):
    """A first run has nothing to archive, so it is a no-op rather than an empty attempt."""
    step_dir = tmp_path / "step1"
    (step_dir / "0").mkdir(parents=True)
    (step_dir / "1").mkdir()
    (step_dir / "1" / "step1_1.out").write_text("previous run")

    archived = attempts.archive_previous(step_dir, ["0", "1", "missing"])

    assert [p.parent.name for p in archived] == ["1"]
    assert not (step_dir / "0" / "attempt1").exists()
    assert (step_dir / "1" / "attempt1" / "step1_1.out").read_text() == "previous run"


def test_archive_previous_archives_an_engine_directory_left_alone(tmp_path: Path):
    """A canonical holding only an engine-written directory is prior work, not a first run.

    A crash mid-seal moves the loose files and dies before the ``tensors/`` — the trigger
    must see what ``seal`` moves, or the re-run's copy-back merges into the leftover and
    the directory describes two calculations. Attempts alone still mean nothing to archive.
    """
    step_dir = tmp_path / "step1"
    (step_dir / "0" / "tensors").mkdir(parents=True)
    (step_dir / "0" / "tensors" / "hessian.npy").write_bytes(b"\x00")
    (step_dir / "1" / "attempt1").mkdir(parents=True)
    (step_dir / "1" / "attempt1" / "step1_1.out").write_text("sealed already")

    archived = attempts.archive_previous(step_dir, ["0", "1"])

    assert [p.parent.name for p in archived] == ["0"]
    assert (step_dir / "0" / "attempt1" / "tensors" / "hessian.npy").is_file()
    assert not (step_dir / "0" / "tensors").exists()
    assert not (step_dir / "1" / "attempt2").exists()

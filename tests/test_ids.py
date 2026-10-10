"""Tests for hierarchical structure-ID allocation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from chemrefine import ids
from chemrefine.errors import CacheError, ConfigError
from chemrefine.ids import (
    StepTemplate,
    allocate_child_ids,
    require_template,
    step_template_path,
)

# ---------------------------------------------------------------------------
# allocate_child_ids
# ---------------------------------------------------------------------------


def test_allocate_child_ids_fanout_one_preserves_parent():
    assert allocate_child_ids(["0", "1"], [1, 1]) == ["0", "1"]


def test_allocate_child_ids_fanout_zero_drops_parent():
    assert allocate_child_ids(["0", "1"], [0, 2]) == ["1-0", "1-1"]


def test_allocate_child_ids_branches():
    assert allocate_child_ids(["0"], [3]) == ["0-0", "0-1", "0-2"]


def test_allocate_child_ids_mixed():
    assert allocate_child_ids(["0", "1"], [2, 1]) == ["0-0", "0-1", "1"]


def test_allocate_child_ids_length_mismatch_raises():
    # The explicit refusal's wording; `zip(strict=True)` below it raises the same type,
    # so an unmatched raise let the documented message be deleted unnoticed.
    with pytest.raises(ValueError, match="same length"):
        allocate_child_ids(["0"], [1, 2])


def test_allocate_child_ids_negative_fanout_raises():
    with pytest.raises(ValueError):
        allocate_child_ids(["0"], [-1])


# ---------------------------------------------------------------------------
# step_template_path / require_template — naming and refusal are separate jobs
# ---------------------------------------------------------------------------


def test_step_template_path_does_not_require_the_file_to_exist(tmp_path):
    """Naming only. `build_context` calls this for every step, including ones whose
    template is missing — a step must still be able to compute its cache key, or it could
    never be re-fingerprinted at all."""
    assert step_template_path(tmp_path, 3, template=None, suffix="inp") == tmp_path / "step3.inp"
    assert step_template_path(tmp_path, 3, template="custom.py", suffix="inp") == (
        tmp_path / "custom.py"
    )


def test_require_template_refuses_an_engine_that_declares_none(tmp_path):
    """`ctx.template is None` means the engine is not `TemplateDriven`.

    Distinct from a template that was named and is missing: this one is a wiring mistake in
    the engine, not a missing file the user can create, and the message has to say so.
    """
    with pytest.raises(ConfigError, match="declares none"):
        require_template(None, label="ORCA")


def test_require_template_refuses_a_named_template_that_is_absent(tmp_path):
    """The likeliest error of a first run — the message names the path to create."""
    missing = tmp_path / "step1.inp"
    with pytest.raises(ConfigError, match=str(missing)):
        require_template(missing, label="ORCA")


def test_require_template_returns_the_template_as_it_read_it(tmp_path):
    """Where the template is and its text, together: a consumer reads neither again."""
    present = tmp_path / "step1.inp"
    present.write_text("! Opt\n", encoding="utf-8")
    assert require_template(present, label="ORCA") == StepTemplate(present, "! Opt\n")


def test_attempt_dirs_are_ordered_by_number_not_by_name_or_clock(tmp_path):
    """``attempt10`` is newer than ``attempt9``, and the clock has no say.

    The one ordering every reader of ``attemptK/`` shares — ``rebuild-cache`` through
    ``latest_attempt_dir``, the mode viewer through its walk. A lexical sort puts
    ``attempt10`` before ``attempt2``; an mtime sort answers with copy order on a tree
    copied without timestamps. Both wrong pickers are set up here to lose.
    """
    import os
    import time

    from chemrefine.ids import attempt_dirs_newest_first, latest_attempt_dir

    now = time.time()
    for name, age in (("attempt2", 0), ("attempt10", 7200), ("attempt9", 3600)):
        (tmp_path / name).mkdir()
        os.utime(tmp_path / name, (now - age, now - age))  # attempt2 is the newest by clock
    (tmp_path / "attemptX").mkdir()  # not an attempt: no number
    (tmp_path / "attempt3.bak").write_text("", encoding="utf-8")

    assert [p.name for p in attempt_dirs_newest_first(tmp_path)] == [
        "attempt10",
        "attempt9",
        "attempt2",
    ]
    assert latest_attempt_dir(tmp_path) == tmp_path / "attempt10"
    assert attempt_dirs_newest_first(tmp_path / "nothing-here") == []


def test_a_structures_canonical_state_is_everything_but_its_attempts(tmp_path):
    """One membership test splits a structure's directory, so the halves cannot overlap.

    An ``attempt<n>`` directory is an attempt; a file of that name, a directory with no
    number and an engine-written directory are the calculation's own and move with it.
    """
    for name in ("attempt1", "attempt12", "attemptX", "tensors"):
        (tmp_path / name).mkdir()
    for name in ("attempt7", "step1_0.out"):
        (tmp_path / name).write_text("", encoding="utf-8")

    assert ids.canonical_entries(tmp_path) == [
        (tmp_path / "attempt7", False),
        (tmp_path / "attemptX", True),
        (tmp_path / "step1_0.out", False),
        (tmp_path / "tensors", True),
    ]
    assert [p.name for p in ids.attempt_dirs_newest_first(tmp_path)] == ["attempt12", "attempt1"]
    assert ids.canonical_entries(tmp_path / "nothing-here") == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads through any mode")
@pytest.mark.parametrize("mode", [0o000, 0o111], ids=["no-access", "search-only"])
def test_a_structure_directory_that_cannot_be_listed_is_not_one_with_no_attempts(tmp_path, mode):
    """An unlistable directory is refused by name; it does not number its next attempt ``1``.

    Read as empty, the next attempt would be numbered over the first, a rebuild would find
    no children, and a promoted winner would lose the record that names it.
    """
    structure_dir = tmp_path / "0"
    (structure_dir / "attempt1").mkdir(parents=True)
    structure_dir.chmod(mode)
    try:
        for ask in (ids.next_attempt_dir, ids.latest_attempt_dir, ids.canonical_entries):
            with pytest.raises(CacheError) as excinfo:
                ask(structure_dir)
            assert str(excinfo.value) == f"cannot read {structure_dir}: Permission denied"
    finally:
        structure_dir.chmod(0o755)


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads through any mode")
def test_a_directory_listed_but_not_searched_still_numbers_its_attempts(tmp_path, list_only):
    """The same answer on every Python where ``Path.is_dir`` gives two and neither is it."""
    structure_dir = tmp_path / "0"
    (structure_dir / "attempt1").mkdir(parents=True)
    (structure_dir / "attempt2").mkdir()
    list_only(structure_dir)
    answers = (ids.next_attempt_dir(structure_dir), ids.latest_attempt_dir(structure_dir))
    assert answers == (structure_dir / "attempt3", structure_dir / "attempt2")


def test_the_output_trees_bookkeeping_names():
    """Spelled once, here: a reader cannot look for a name its writer no longer uses."""
    assert ids.cache_dir(Path("s")) == Path("s/_cache")
    assert ids.steps_report_path(Path("o")) == Path("o/steps.csv")
    assert ids.agent_runs_dir(Path("o")) == Path("o/agent_runs")

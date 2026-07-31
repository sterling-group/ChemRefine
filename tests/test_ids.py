"""Tests for hierarchical structure-ID allocation."""

from __future__ import annotations

import pytest

from chemrefine.errors import ConfigError
from chemrefine.ids import allocate_child_ids, require_template, step_template_path

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
    with pytest.raises(ValueError):
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


def test_require_template_returns_the_path_when_it_is_there(tmp_path):
    present = tmp_path / "step1.inp"
    present.write_text("! Opt\n", encoding="utf-8")
    assert require_template(present, label="ORCA") == present

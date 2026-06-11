"""Tests for hierarchical structure-ID allocation."""

from __future__ import annotations

import pytest

from chemrefine.ids import (
    allocate_child_ids,
    extract_structure_id,
    resolve_persistent_ids,
    validate_structure_ids,
)

# ---------------------------------------------------------------------------
# extract_structure_id
# ---------------------------------------------------------------------------


def test_extract_structure_id_plain():
    assert extract_structure_id("step1_structure_0.inp") == "0"
    assert extract_structure_id("step3_structure_42.out") == "42"


def test_extract_structure_id_hierarchical():
    assert extract_structure_id("step2_structure_0-1.inp") == "0-1"
    assert extract_structure_id("step5_structure_0-1-2.xyz") == "0-1-2"


def test_extract_structure_id_unrecognized_returns_none():
    assert extract_structure_id("random.txt") is None


# ---------------------------------------------------------------------------
# validate_structure_ids
# ---------------------------------------------------------------------------


def test_validate_structure_ids_accepts_ints_and_strs():
    assert validate_structure_ids([0, "1", 2], step_id=1) == ["0", "1", "2"]


def test_validate_structure_ids_rejects_none():
    with pytest.raises(ValueError):
        validate_structure_ids(None, step_id=1)  # type: ignore[arg-type]


def test_validate_structure_ids_rejects_empty():
    with pytest.raises(ValueError):
        validate_structure_ids([], step_id=1)


def test_validate_structure_ids_rejects_negative_int():
    with pytest.raises(ValueError):
        validate_structure_ids([-1], step_id=1)


def test_validate_structure_ids_rejects_invalid_str():
    with pytest.raises(ValueError):
        validate_structure_ids(["-1"], step_id=1)


def test_validate_structure_ids_rejects_unsupported_type():
    with pytest.raises(TypeError):
        validate_structure_ids([3.14], step_id=1)


def test_validate_structure_ids_rejects_bare_string():
    """A bare string isn't a sequence of IDs — it's a single ID; must reject."""
    with pytest.raises(TypeError):
        validate_structure_ids("not-a-sequence", step_id=1)  # type: ignore[arg-type]


def test_validate_structure_ids_rejects_bytes():
    with pytest.raises(TypeError):
        validate_structure_ids(b"also-not-a-sequence", step_id=1)  # type: ignore[arg-type]


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
# resolve_persistent_ids
# ---------------------------------------------------------------------------


def test_resolve_persistent_ids_step1_bootstrap():
    assert resolve_persistent_ids(step_number=1, parent_ids=None, child_count=3) == [
        "0",
        "1",
        "2",
    ]


def test_resolve_persistent_ids_no_parents_acts_as_bootstrap():
    assert resolve_persistent_ids(step_number=2, parent_ids=[], child_count=2) == ["0", "1"]


def test_resolve_persistent_ids_one_to_one_preserves():
    assert resolve_persistent_ids(step_number=2, parent_ids=["0", "1", "2"], child_count=3) == [
        "0",
        "1",
        "2",
    ]


def test_resolve_persistent_ids_single_parent_fanout():
    assert resolve_persistent_ids(step_number=2, parent_ids=["7"], child_count=3) == [
        "7-0",
        "7-1",
        "7-2",
    ]


def test_resolve_persistent_ids_even_fanout():
    assert resolve_persistent_ids(step_number=2, parent_ids=["0", "1"], child_count=4) == [
        "0-0",
        "0-1",
        "1-0",
        "1-1",
    ]


def test_resolve_persistent_ids_fewer_children_than_parents():
    # 3 parents, 2 children → first 2 children inherit, last parent drops
    assert resolve_persistent_ids(step_number=2, parent_ids=["0", "1", "2"], child_count=2) == [
        "0",
        "1",
    ]


def test_resolve_persistent_ids_uneven_extra_goes_to_first_parent():
    # 2 parents, 3 children → parent 0 fans out to 2 children, parent 1 keeps
    assert resolve_persistent_ids(step_number=2, parent_ids=["0", "1"], child_count=3) == [
        "0-0",
        "0-1",
        "1",
    ]

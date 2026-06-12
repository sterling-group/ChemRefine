"""Tests for hierarchical structure-ID allocation."""

from __future__ import annotations

import pytest

from chemrefine.ids import allocate_child_ids

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

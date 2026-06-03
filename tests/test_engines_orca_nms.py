"""Tests for ``chemrefine.engines.orca.nms`` (pure target-aware displacement)
and the engine's two-round orchestration + step resolution."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from chemrefine.engines.orca import nms
from chemrefine.state import Structure


def _struct(positions, sid: str = "0") -> Structure:
    return Structure(id=sid, atoms=Atoms("H2", positions=positions))


def _modes(n_modes: int) -> np.ndarray:
    """(2 atoms, 3, n_modes): mode k moves atom0 +0.1·(k+1) x, atom1 -0.1·(k+1) x."""
    t = np.zeros((2, 3, n_modes))
    for k in range(n_modes):
        t[0, 0, k] = 0.1 * (k + 1)
        t[1, 0, k] = -0.1 * (k + 1)
    return t


# ---------------------------------------------------------------------------
# NmsOptions
# ---------------------------------------------------------------------------


def test_nms_options_defaults():
    o = nms.NmsOptions()
    assert o.target == "minimum"
    assert o.displacement_value == 1.0
    assert o.num_random_displacements == 1
    assert o.ts_mode_index is None
    assert o.seed == 42


def test_nms_options_from_raw_filters_unknown_keys():
    o = nms.NmsOptions.from_raw({"target": "ts", "ts_mode_index": 7, "basis": "ignored"})
    assert o.target == "ts"
    assert o.ts_mode_index == 7


def test_nms_options_rejects_bad_target():
    with pytest.raises(ValueError):
        nms.NmsOptions(target="saddle")  # type: ignore[arg-type]


def test_target_imaginary_count():
    assert nms.target_imaginary_count(nms.NmsOptions(target="minimum")) == 0
    assert nms.target_imaginary_count(nms.NmsOptions(target="ts")) == 1
    assert nms.target_imaginary_count(nms.NmsOptions(target="random")) is None


# ---------------------------------------------------------------------------
# displace_along_mode (pure)
# ---------------------------------------------------------------------------


def test_displace_along_mode_returns_pos_and_neg():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mode = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    pos, neg = nms.displace_along_mode(positions, mode, displacement=2.0)
    np.testing.assert_allclose(pos, [[0.2, 0.0, 0.0], [0.8, 0.0, 0.0]])
    np.testing.assert_allclose(neg, [[-0.2, 0.0, 0.0], [1.2, 0.0, 0.0]])


def test_displace_along_mode_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        nms.displace_along_mode(np.zeros((2, 3)), np.zeros((3, 3)), displacement=1.0)


# ---------------------------------------------------------------------------
# select_displacements — target-aware
# ---------------------------------------------------------------------------


def _rng():
    return np.random.default_rng(0)


def test_select_minimum_displaces_every_imaginary_mode():
    sel = nms.select_displacements(
        _struct([[0, 0, 0], [1, 0, 0]]), {5: -42.0, 6: -100.0}, _modes(8),
        nms.NmsOptions(target="minimum"), _rng(),
    )
    assert [label for label, _ in sel] == ["m5_pos", "m5_neg", "m6_pos", "m6_neg"]


def test_select_ts_keeps_largest_imaginary_removes_spurious():
    # mode 6 is most-imaginary → reaction coordinate (kept); mode 5 displaced.
    sel = nms.select_displacements(
        _struct([[0, 0, 0], [1, 0, 0]]), {5: -42.0, 6: -200.0}, _modes(8),
        nms.NmsOptions(target="ts"), _rng(),
    )
    assert [label for label, _ in sel] == ["m5_pos", "m5_neg"]


def test_select_ts_honors_explicit_mode_index():
    sel = nms.select_displacements(
        _struct([[0, 0, 0], [1, 0, 0]]), {5: -42.0, 6: -200.0}, _modes(8),
        nms.NmsOptions(target="ts", ts_mode_index=5), _rng(),
    )
    # Keep mode 5 (explicit RC) → displace the other imaginary mode 6.
    assert [label for label, _ in sel] == ["m6_pos", "m6_neg"]


def test_select_random_count_and_determinism():
    opts = nms.NmsOptions(target="random", num_random_displacements=2, seed=42)
    a = nms.select_displacements(_struct([[0, 0, 0], [1, 0, 0]]), {}, _modes(12), opts,
                                 np.random.default_rng(42))
    b = nms.select_displacements(_struct([[0, 0, 0], [1, 0, 0]]), {}, _modes(12), opts,
                                 np.random.default_rng(42))
    assert len(a) == 4  # 2 modes x (pos, neg)
    assert [label for label, _ in a] == [label for label, _ in b]


def test_select_skips_modes_outside_the_tensor():
    sel = nms.select_displacements(
        _struct([[0, 0, 0], [1, 0, 0]]), {37: -100.0}, _modes(6),
        nms.NmsOptions(target="minimum"), _rng(),
    )
    assert sel == []


def test_select_applies_displacement_value():
    sel = nms.select_displacements(
        _struct([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), {5: -42.0}, _modes(6),
        nms.NmsOptions(displacement_value=2.0), _rng(),
    )
    _, pos = sel[0]  # m5_pos; mode 5 moves atom0 by 0.1*6 = 0.6, x2.0 = 1.2
    np.testing.assert_allclose(pos[0], [1.2, 0.0, 0.0])
    np.testing.assert_allclose(pos[1], [1.0 - 1.2, 0.0, 0.0])

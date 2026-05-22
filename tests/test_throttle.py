"""Tests for the PAL-budget throttler."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from chemrefine.throttle import Throttler


def _never_finished(_job_id: str) -> bool:
    return False


def _always_finished(_job_id: str) -> bool:
    return True


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_throttler_rejects_zero_max_cores():
    with pytest.raises(ValueError):
        Throttler(max_cores=0)


def test_throttler_initial_state_is_empty():
    t = Throttler(max_cores=64)
    assert t.cores_in_use == 0
    assert t.active_jobs == ()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_register_charges_cores():
    t = Throttler(max_cores=64)
    t.register("1", 16)
    t.register("2", 8)
    assert t.cores_in_use == 24
    assert set(t.active_jobs) == {"1", "2"}


def test_register_rejects_zero_pal():
    t = Throttler(max_cores=64)
    with pytest.raises(ValueError):
        t.register("1", 0)


# ---------------------------------------------------------------------------
# wait_for_room
# ---------------------------------------------------------------------------


def test_wait_for_room_no_wait_when_budget_available():
    t = Throttler(max_cores=64)
    with patch("time.sleep") as sleeper:
        t.wait_for_room(16, is_finished=_never_finished)
    sleeper.assert_not_called()


def test_wait_for_room_reaps_finished_jobs_to_free_budget():
    t = Throttler(max_cores=32, poll_interval=0)
    t.register("done", 16)
    t.register("running", 8)
    state = {"poll_count": 0}

    def is_finished(jid: str) -> bool:
        state["poll_count"] += 1
        return jid == "done"

    with patch("time.sleep"):
        t.wait_for_room(16, is_finished=is_finished)
    assert t.cores_in_use == 8
    assert t.active_jobs == ("running",)


def test_wait_for_room_blocks_until_jobs_finish():
    t = Throttler(max_cores=8, poll_interval=0)
    t.register("a", 4)
    t.register("b", 4)
    state = {"calls": 0}

    def is_finished(jid: str) -> bool:
        state["calls"] += 1
        # Only after the 3rd poll do we report 'a' done.
        return state["calls"] >= 3 and jid == "a"

    with patch("time.sleep"):
        t.wait_for_room(4, is_finished=is_finished)
    assert t.active_jobs == ("b",)


def test_reap_logs_freed_cores(caplog):
    """The reap log line announcing freed cores must fire when a job completes."""
    import logging as _logging

    t = Throttler(max_cores=32)
    t.register("done", 8)
    with caplog.at_level(_logging.INFO, logger="chemrefine.throttle"):
        t._reap(lambda _: True)
    assert any("freed 8 cores" in record.message for record in caplog.records)


def test_wait_for_room_rejects_request_above_budget():
    t = Throttler(max_cores=16)
    with pytest.raises(ValueError):
        t.wait_for_room(32, is_finished=_always_finished)


# ---------------------------------------------------------------------------
# wait_all
# ---------------------------------------------------------------------------


def test_wait_all_empties_active_set():
    t = Throttler(max_cores=64, poll_interval=0)
    t.register("a", 8)
    t.register("b", 16)
    with patch("time.sleep"):
        t.wait_all(is_finished=_always_finished)
    assert t.cores_in_use == 0
    assert t.active_jobs == ()


def test_wait_all_returns_immediately_when_no_jobs():
    t = Throttler(max_cores=64)
    with patch("time.sleep") as sleeper:
        t.wait_all(is_finished=_never_finished)
    sleeper.assert_not_called()


def test_wait_all_sleeps_until_jobs_finish():
    """wait_all must sleep at least once when an active job is still running."""
    t = Throttler(max_cores=64, poll_interval=0)
    t.register("running", 8)
    state = {"calls": 0}

    def is_finished(_jid: str) -> bool:
        state["calls"] += 1
        return state["calls"] >= 2  # finishes on the second poll

    with patch("time.sleep") as sleeper:
        t.wait_all(is_finished=is_finished)
    assert t.active_jobs == ()
    sleeper.assert_called()


# ---------------------------------------------------------------------------
# timeout support
# ---------------------------------------------------------------------------


def test_wait_for_room_raises_on_timeout():
    """wait_for_room raises ThrottleTimeoutError when deadline expires."""
    from chemrefine.errors import ThrottleTimeoutError

    t = Throttler(max_cores=8, poll_interval=0)
    t.register("blocker", 8)
    with (
        patch("time.monotonic", side_effect=[0.0, 0.0, 99.0]),
        pytest.raises(ThrottleTimeoutError),
    ):
        t.wait_for_room(4, is_finished=_never_finished, max_wait_seconds=5.0)


def test_wait_all_raises_on_timeout():
    """wait_all raises ThrottleTimeoutError when deadline expires."""
    from chemrefine.errors import ThrottleTimeoutError

    t = Throttler(max_cores=8, poll_interval=0)
    t.register("blocker", 8)
    with (
        patch("time.monotonic", side_effect=[0.0, 0.0, 99.0]),
        pytest.raises(ThrottleTimeoutError),
    ):
        t.wait_all(is_finished=_never_finished, max_wait_seconds=5.0)

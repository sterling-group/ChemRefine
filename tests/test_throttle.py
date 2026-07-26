"""Tests for the PAL-budget throttler."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from chemrefine.throttle import Throttler


def _never_finished(_job_ids):
    """Nothing ever completes — one poll returns the empty set."""
    return set()


def _always_finished(job_ids):
    """Everything the throttler asks about is already done."""
    return set(job_ids)


def _finishes(*done: str):
    """A poll that reports exactly ``done`` (intersected with what is active)."""
    return lambda job_ids: {j for j in job_ids if j in done}


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_throttler_rejects_zero_max_cores():
    with pytest.raises(ValueError):
        Throttler(max_cores=0)


def test_throttler_initial_state_is_empty():
    t = Throttler(max_cores=64)
    assert t.cores_in_use == 0
    assert t.gpus_in_use == 0
    assert t.active_jobs == ()


def test_throttler_rejects_negative_max_gpus():
    with pytest.raises(ValueError):
        Throttler(max_cores=8, max_gpus=-1)


# ---------------------------------------------------------------------------
# GPU budget + device assignment
# ---------------------------------------------------------------------------


def test_gpus_in_use_tracks_gpu_demand():
    t = Throttler(max_cores=64, max_gpus=4)
    t.register("a", 8, gpus=1)
    t.register("b", 8, gpus=2)
    assert t.gpus_in_use == 3
    assert t.cores_in_use == 16


def test_wait_for_room_blocks_on_gpu_budget_even_when_cores_free():
    """One GPU, two GPU jobs: the second waits even though cores are plentiful."""
    t = Throttler(max_cores=64, max_gpus=1, poll_interval=0)
    t.register("g0", 1, gpus=1, device=0)
    state = {"calls": 0}

    def finished(job_ids):
        state["calls"] += 1
        return {j for j in job_ids if j == "g0"} if state["calls"] >= 2 else set()

    with patch("time.sleep") as sleeper:
        t.wait_for_room(1, finished=finished, gpus_needed=1)
    sleeper.assert_called()  # had to wait for the GPU to free
    assert t.active_jobs == ()


def test_wait_for_room_admits_cpu_job_while_gpu_saturated():
    """A gpus=0 job isn't blocked by a saturated GPU budget (CPU jobs keep flowing)."""
    t = Throttler(max_cores=64, max_gpus=1)
    t.register("g0", 1, gpus=1, device=0)
    with patch("time.sleep") as sleeper:
        t.wait_for_room(8, finished=_never_finished, gpus_needed=0)
    sleeper.assert_not_called()


def test_wait_for_room_rejects_gpu_request_above_budget():
    t = Throttler(max_cores=16, max_gpus=1)
    with pytest.raises(ValueError):
        t.wait_for_room(1, finished=_always_finished, gpus_needed=2)


def test_assign_device_returns_lowest_free_index_and_reuses_freed():
    t = Throttler(max_cores=64, max_gpus=2)
    assert t.assign_device() == 0
    t.register("a", 1, gpus=1, device=0)
    assert t.assign_device() == 1
    t.register("b", 1, gpus=1, device=1)
    t._reap(_finishes("a"))  # frees device 0
    assert t.assign_device() == 0


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
        t.wait_for_room(16, finished=_never_finished)
    sleeper.assert_not_called()


def test_wait_for_room_reaps_finished_jobs_to_free_budget():
    t = Throttler(max_cores=32, poll_interval=0)
    t.register("done", 16)
    t.register("running", 8)
    state = {"poll_count": 0}

    def finished(job_ids):
        state["poll_count"] += 1
        return {j for j in job_ids if j == "done"}

    with patch("time.sleep"):
        t.wait_for_room(16, finished=finished)
    # One poll answers the whole active batch, however many jobs there are.
    assert state["poll_count"] == 1
    assert t.cores_in_use == 8
    assert t.active_jobs == ("running",)


def test_wait_for_room_blocks_until_jobs_finish():
    t = Throttler(max_cores=8, poll_interval=0)
    t.register("a", 4)
    t.register("b", 4)
    state = {"calls": 0}

    def finished(job_ids):
        state["calls"] += 1
        # Only on the 3rd poll do we report 'a' done.
        return {j for j in job_ids if j == "a"} if state["calls"] >= 3 else set()

    with patch("time.sleep"):
        t.wait_for_room(4, finished=finished)
    assert t.active_jobs == ("b",)


def test_reap_logs_freed_cores(caplog):
    """The reap log line announcing freed cores must fire when a job completes."""
    import logging as _logging

    t = Throttler(max_cores=32)
    t.register("done", 8)
    with caplog.at_level(_logging.INFO, logger="chemrefine.throttle"):
        t._reap(_always_finished)
    assert any("freed 8 cores" in record.message for record in caplog.records)


def test_wait_for_room_rejects_request_above_budget():
    t = Throttler(max_cores=16)
    with pytest.raises(ValueError):
        t.wait_for_room(32, finished=_always_finished)


# ---------------------------------------------------------------------------
# wait_all
# ---------------------------------------------------------------------------


def test_wait_all_empties_active_set():
    t = Throttler(max_cores=64, poll_interval=0)
    t.register("a", 8)
    t.register("b", 16)
    with patch("time.sleep"):
        t.wait_all(finished=_always_finished)
    assert t.cores_in_use == 0
    assert t.active_jobs == ()


def test_wait_all_returns_immediately_when_no_jobs():
    t = Throttler(max_cores=64)
    with patch("time.sleep") as sleeper:
        t.wait_all(finished=_never_finished)
    sleeper.assert_not_called()


def test_wait_all_sleeps_until_jobs_finish():
    """wait_all must sleep at least once when an active job is still running."""
    t = Throttler(max_cores=64, poll_interval=0)
    t.register("running", 8)
    state = {"calls": 0}

    def finished(job_ids):
        state["calls"] += 1
        return set(job_ids) if state["calls"] >= 2 else set()  # done on the second poll

    with patch("time.sleep") as sleeper:
        t.wait_all(finished=finished)
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
        t.wait_for_room(4, finished=_never_finished, max_wait_seconds=5.0)


def test_wait_all_raises_on_timeout():
    """wait_all raises ThrottleTimeoutError when deadline expires."""
    from chemrefine.errors import ThrottleTimeoutError

    t = Throttler(max_cores=8, poll_interval=0)
    t.register("blocker", 8)
    with (
        patch("time.monotonic", side_effect=[0.0, 0.0, 99.0]),
        pytest.raises(ThrottleTimeoutError),
    ):
        t.wait_all(finished=_never_finished, max_wait_seconds=5.0)

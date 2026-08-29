"""Tests for the PAL-budget throttler."""

from __future__ import annotations

from itertools import chain, repeat
from unittest.mock import patch

import pytest

from chemrefine.throttle import GpuBudget, Throttler


def _clock_that_expires(*before: float, then: float):
    """A ``time.monotonic`` stand-in: ``before`` readings, then ``then`` forever.

    ``repeat`` rather than a fixed-length list because a finite ``side_effect`` pins the
    *number of times* the code under test reads the clock, not what it does with the
    readings — so adding or removing one ``monotonic()`` call makes the test fail with
    ``StopIteration`` instead of on its actual assertion.
    """
    return chain(before, repeat(then))


def _never_finished(_job_ids):
    """Nothing ever completes — one poll returns the empty set."""
    return set()


def _always_finished(job_ids):
    """Everything the throttler asks about is already done."""
    return set(job_ids)


def _finishes(*done: str):
    """A poll that reports exactly ``done`` (intersected with what is active)."""
    return lambda job_ids: {j for j in job_ids if j in done}


def _never_called(_job_ids):
    """Asserts it is not reached — for waits that must not poll at all."""
    raise AssertionError("polled the scheduler with no active jobs")


def _idx(count: int) -> tuple[str, ...]:
    """The device tokens of a plain N-GPU host — what `nvidia-smi` detection yields.

    Spelled out here rather than imported from `slurm.dispatch` so these tests describe a
    budget directly, without depending on how one is resolved from a host.
    """
    return tuple(str(i) for i in range(count))


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
        Throttler(max_cores=8, gpus=GpuBudget(-1))


def test_throttler_rejects_a_negative_poll_interval():
    """Checked with the other two, not left to `time.sleep`.

    Unvalidated, a bad cadence raises from inside a wait — after the batch is submitted,
    which is the expensive moment to learn the number was wrong.
    """
    with pytest.raises(ValueError, match="poll_interval"):
        Throttler(max_cores=8, poll_interval=-1)


# ---------------------------------------------------------------------------
# GPU budget + device assignment
# ---------------------------------------------------------------------------


def test_gpus_in_use_tracks_gpu_demand():
    t = Throttler(max_cores=64, gpus=GpuBudget(4, _idx(4)))
    t.register("a", 8, gpus=1)
    t.register("b", 8, gpus=2)
    assert t.gpus_in_use == 3
    assert t.cores_in_use == 16


def test_the_gpu_budget_blocks_a_gpu_job_even_when_cores_are_free():
    """One GPU, two GPU jobs: the second cannot start though cores are plentiful."""
    t = Throttler(max_cores=64, gpus=GpuBudget(1, _idx(1)))
    t.register("g0", 1, gpus=1, device="0")
    assert t.has_room(1, gpus_needed=1) is False


def test_a_cpu_job_is_admitted_while_the_gpu_budget_is_saturated():
    """A gpus=0 job isn't blocked by a saturated GPU budget (CPU jobs keep flowing)."""
    t = Throttler(max_cores=64, gpus=GpuBudget(1, _idx(1)))
    t.register("g0", 1, gpus=1, device="0")
    assert t.has_room(8, gpus_needed=0) is True


def test_a_gpu_request_above_the_budget_never_has_room():
    """No guard here any more: an unsatisfiable request is simply never admitted.

    `run_batch` rejects it as a `ConfigError` before a throttler is built
    (`test_run_batch_rejects_a_gpu_step_over_the_budget`), which is where a config mistake
    belongs — with an exit code, naming the step and the setting to change.
    """
    t = Throttler(max_cores=16, gpus=GpuBudget(1, _idx(1)))
    assert t.has_room(1, gpus_needed=2) is False


def test_assign_device_returns_the_first_free_token_and_reuses_freed():
    t = Throttler(max_cores=64, gpus=GpuBudget(2, _idx(2)))
    assert t.assign_device() == "0"
    t.register("a", 1, gpus=1, device="0")
    assert t.assign_device() == "1"
    t.register("b", 1, gpus=1, device="1")
    t._reap(_finishes("a"))  # frees device 0
    assert t.assign_device() == "0"


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
# waiting
# ---------------------------------------------------------------------------


def test_one_poll_answers_the_whole_active_batch():
    """However many jobs are active, a wait costs one scheduler query per tick.

    Asking per job is what turned a step with N concurrent jobs into N `squeue` subprocesses
    every poll interval — see `slurm.finished_jobs`.
    """
    t = Throttler(max_cores=32, poll_interval=0)
    t.register("done", 16)
    t.register("running", 8)
    state = {"poll_count": 0}

    def finished(job_ids):
        state["poll_count"] += 1
        return {j for j in job_ids if j == "done"}

    with patch("time.sleep"):
        assert t.wait_for_completion(finished=finished) == ("done",)
    assert state["poll_count"] == 1
    assert t.cores_in_use == 8
    assert t.active_jobs == ("running",)


def test_a_wait_sleeps_until_something_actually_finishes():
    t = Throttler(max_cores=8, poll_interval=0)
    t.register("a", 4)
    t.register("b", 4)
    state = {"calls": 0}

    def finished(job_ids):
        state["calls"] += 1
        # Only on the 3rd poll do we report 'a' done.
        return {j for j in job_ids if j == "a"} if state["calls"] >= 3 else set()

    with patch("time.sleep") as sleeper:
        assert t.wait_for_completion(finished=finished) == ("a",)
    sleeper.assert_called()
    assert t.active_jobs == ("b",)


def test_reap_logs_freed_cores(caplog):
    """The reap log line announcing freed cores must fire when a job completes."""
    import logging as _logging

    t = Throttler(max_cores=32)
    t.register("done", 8)
    with caplog.at_level(_logging.INFO, logger="chemrefine.throttle"):
        t._reap(_always_finished)
    assert any("freed 8 cores" in record.message for record in caplog.records)


def test_draining_is_the_callers_loop():
    """There is no `wait_all`: a drain is `wait_for_completion` until nothing is active.

    Two lines at the call site, and the same two lines the streaming queue already runs — so
    a scheduler that also wants to submit as slots free does not need a second kind of wait.
    """
    t = Throttler(max_cores=64, poll_interval=0)
    t.register("a", 8)
    t.register("b", 16)
    with patch("time.sleep"):
        while t.active_jobs:
            t.wait_for_completion(finished=_always_finished)
    assert t.cores_in_use == 0
    assert t.active_jobs == ()


# ---------------------------------------------------------------------------
# timeout support
# ---------------------------------------------------------------------------


def test_a_stalled_wait_raises_on_timeout():
    """One wait, so one timeout message."""
    from chemrefine.errors import ThrottleTimeoutError

    t = Throttler(max_cores=8, poll_interval=0)
    t.register("blocker", 8)
    with (
        patch("time.monotonic", side_effect=_clock_that_expires(0.0, then=99.0)),
        pytest.raises(
            ThrottleTimeoutError, match=r"timed out after 5\.0s waiting for a job to finish"
        ),
    ):
        t.wait_for_completion(finished=_never_finished, max_wait_seconds=5.0)


# --- throttle ---------------------------------------------------------------


def test_throttler_register_rejects_negative_gpus():
    from chemrefine.throttle import GpuBudget, Throttler

    with pytest.raises(ValueError, match="gpus must be >= 0"):
        Throttler(max_cores=8, gpus=GpuBudget(1, _idx(1))).register("g", 1, gpus=-1)


def test_throttler_assign_device_raises_when_all_taken():
    from chemrefine.throttle import GpuBudget, Throttler

    # Unreachable on the real call path (`has_room` admits first), so assign_device
    # fails loud rather than silently colliding two GPU jobs on one device.
    t = Throttler(max_cores=8, gpus=GpuBudget(1, _idx(1)))
    t.register("g", 1, gpus=1, device="0")
    with pytest.raises(RuntimeError, match="no free GPU device"):
        t.assign_device()


def test_assign_device_hands_out_the_granted_tokens_not_indices():
    """The devices the run owns, not `range(count)` — the whole point of carrying tokens.

    Given `CUDA_VISIBLE_DEVICES=2,3` the free *indices* are 0 and 1, which are somebody
    else's GPUs. Tokens are also the only way to name a MIG instance.
    """
    t = Throttler(max_cores=64, gpus=GpuBudget(2, ("2", "MIG-8a2f")))
    assert t.assign_device() == "2"
    t.register("a", 1, gpus=1, device="2")
    assert t.assign_device() == "MIG-8a2f"


# --- the two primitives -----------------------------------------------------
#
# The whole scheduling surface: `has_room` says whether a job can start, `wait_for_completion`
# blocks until one ends and says which. Everything else — filling slots, draining, retrying —
# is a caller's loop over those two.


def test_reap_returns_the_ids_it_reaped():
    """`_reap` reports *which* jobs finished, not just that the budget freed up.

    The streaming queue needs the identity to know whose output to parse; the older
    callers only ever needed the freed cores, which is why it used to return None.
    """
    t = Throttler(max_cores=8)
    t.register("a", 4)
    t.register("b", 4)

    assert sorted(t._reap(_finishes("a"))) == ["a"]
    assert t.active_jobs == ("b",)
    assert t._reap(_never_finished) == ()


def test_has_room_polls_nothing():
    """`has_room` is pure — it must not reap, or a caller cannot ask before waiting."""
    t = Throttler(max_cores=8)
    t.register("a", 8)

    assert t.has_room(4) is False
    assert t.active_jobs == ("a",), "has_room reaped; it is supposed to be a pure predicate"
    assert t.has_room(0) is True


def test_wait_for_completion_returns_the_first_completion():
    t = Throttler(max_cores=8, poll_interval=0)
    t.register("a", 4)
    t.register("b", 4)

    assert sorted(t.wait_for_completion(finished=_finishes("b"))) == ["b"]


def test_wait_for_completion_with_nothing_active_returns_empty():
    """No jobs means no completions — not an error, and not a wait."""
    assert Throttler(max_cores=8).wait_for_completion(finished=_never_called) == ()


def test_timeout_bounds_the_stall_not_the_whole_drain():
    """`job_timeout_seconds` means "nothing has finished for this long", everywhere.

    Both halves must be observable, on one clock cadence (4s per reading, a 5s bound).
    A wait in which nothing completes trips it: the deadline check runs on every empty
    poll and raises once the stall exceeds the bound. And a completion re-anchors it:
    each call builds a fresh deadline, so 12s of total elapsed time never trips while
    every stretch without progress stays under 5s — bounding the *total* is what would
    make a healthy multi-hour step trip a timeout set to catch a stuck one. An earlier
    version's `finished` returned a job on every poll, so `deadline.check` never
    executed and the bound, the patch and the ticking clock were all inert.
    """
    from chemrefine.errors import ThrottleTimeoutError

    starved = Throttler(max_cores=8, poll_interval=0)
    starved.register("a", 1)
    ticking = (float(i) * 4.0 for i in range(1000))
    with (
        patch("time.monotonic", side_effect=lambda: next(ticking)),
        pytest.raises(ThrottleTimeoutError, match="waiting for a job to finish"),
    ):
        starved.wait_for_completion(finished=lambda ids: set(), max_wait_seconds=5.0)

    t = Throttler(max_cores=8, poll_interval=0)
    t.register("a", 1)
    t.register("b", 1)
    # Per call: one empty poll (the check runs, under the bound), then a completion.
    answers = iter([set(), {"a"}, set(), {"b"}])
    ticking = (float(i) * 4.0 for i in range(1000))
    with patch("time.monotonic", side_effect=lambda: next(ticking)):
        assert t.wait_for_completion(finished=lambda ids: next(answers), max_wait_seconds=5.0) == (
            "a",
        )
        # 8s have elapsed — past the bound in total, and a fresh deadline says so is fine.
        assert t.wait_for_completion(finished=lambda ids: next(answers), max_wait_seconds=5.0) == (
            "b",
        )
    assert t.active_jobs == ()

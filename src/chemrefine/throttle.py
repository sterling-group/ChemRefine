"""Two-resource (CPU + GPU) budget throttler for batch job submission.

A pipeline step typically submits many jobs at once, but a host has a
fixed CPU budget (``max_cores``) and — locally — a fixed number of GPUs
(``max_gpus``). The throttler tracks each active job's ``(cores, gpus)``
demand and answers two questions about it: :meth:`~Throttler.has_room`
(can this job start now?) and :meth:`~Throttler.wait_for_completion`
(block until one finishes, and say which). A scheduler is those two in a
loop; the throttler owns neither the queue nor what a finished job means.
Polling delegates to a caller-supplied ``finished`` callable — set-shaped, so
one scheduler query answers the whole active batch — which also lets tests
substitute a fake without monkeypatching :mod:`subprocess`.

:class:`StallDeadline` lives here too, because a stall bound is the same concept as the
budget it protects: it is what ``job_timeout_seconds`` means, and every wait in the
codebase — this one and :func:`chemrefine.slurm.wait_for_jobs` — measures it the same way.

The GPU budget only bites locally: under SLURM the scheduler places GPUs
itself (``--gres=gpu``), so the batch engine sets ``max_gpus`` effectively
unlimited there and to the detected device count locally. For a local GPU
job the throttler also hands out a free device index via
:meth:`assign_device` so concurrent jobs land on distinct GPUs.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass

from chemrefine.errors import ThrottleTimeoutError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GpuBudget:
    """How many GPU jobs may run at once, and which devices they may be pinned to.

    One value because the two facts are one decision and have to agree. A count without the
    matching devices is what let :meth:`Throttler.assign_device` hand out index ``0`` on a
    host where the run had been granted devices ``2`` and ``3``: the budget was the host's
    total and the pin was the lowest free *index*, so jobs landed on hardware the run did
    not own.

    ``devices`` is empty exactly when chemrefine is not the one placing GPUs — under SLURM,
    where ``--gres`` does it and :attr:`count` is effectively unlimited. Locally the two
    always match (``count == len(devices)``), an invariant
    :func:`chemrefine.slurm.resolve_gpu_budget` — the only producer — maintains.

    The tokens are **strings**, not indices: ``CUDA_VISIBLE_DEVICES`` accepts GPU UUIDs and
    ``MIG-…`` handles as readily as indices, and a MIG instance can be named no other way.
    An ``int`` would have made an inherited allocation unrepresentable.
    """

    count: int
    devices: tuple[str, ...] = ()


NO_GPUS = GpuBudget(count=0)
"""The budget of a batch that asks for no GPU — the default, and the whole of the CPU path."""

FinishedFn = Callable[[Collection[str]], set[str]]
"""Poll the scheduler once for a whole set of job ids; return those that are done.

Set-shaped rather than per-job on purpose: the throttler asks about every active job
on every tick, and a per-job callable turned that into one scheduler subprocess per
job per tick (see :func:`chemrefine.slurm.finished_jobs`)."""


class StallDeadline:
    """How long a wait may go **without progress** — what ``job_timeout_seconds`` means.

    One definition for every wait in the codebase. Both polling loops
    (:meth:`Throttler.wait_for_completion` and :func:`chemrefine.slurm.wait_for_jobs`)
    used to carry their own arithmetic and raise their own
    :class:`~chemrefine.errors.ThrottleTimeoutError`, and they drifted: one restarted the
    clock whenever a job completed, the other only when a whole *array* did, so the same
    number meant a stall bound on one path and a total-runtime bound on the other. A shared
    object cannot drift — what remains per-caller is what counts as progress, which is
    genuinely different (a completion here, any scheduler activity there).

    ``None`` waits forever, which is the default: under SLURM the partition's own time limit
    already bounds a job, and a deadline chemrefine invents on top of that can only fire
    early.
    """

    def __init__(self, max_wait_seconds: float | None) -> None:
        self._budget = max_wait_seconds
        self._deadline: float | None = None
        self.progress()

    def progress(self) -> None:
        """Restart the clock — the caller saw something move."""
        self._deadline = time.monotonic() + self._budget if self._budget is not None else None

    def check(self, waiting_for: str) -> None:
        """Raise :class:`~chemrefine.errors.ThrottleTimeoutError` if the stall has run long enough.

        ``waiting_for`` completes "timed out after Ns waiting for …", so a caller says what
        its own wait was about without owning the error or the wording around it.
        """
        if self._deadline is not None and time.monotonic() >= self._deadline:
            raise ThrottleTimeoutError(f"timed out after {self._budget}s waiting for {waiting_for}")


class Throttler:
    """Track active jobs against CPU-core and GPU budgets."""

    def __init__(self, *, max_cores: int, gpus: GpuBudget = NO_GPUS, poll_interval: float = 10.0):
        if max_cores < 1:
            raise ValueError(f"max_cores must be >= 1; got {max_cores}")
        if gpus.count < 0:
            raise ValueError(f"max_gpus must be >= 0; got {gpus.count}")
        if poll_interval < 0:
            # Validated here with the other two rather than left to `time.sleep`, which would
            # raise from inside a wait — after a batch has been submitted, which is the
            # expensive moment to learn about a bad number.
            raise ValueError(f"poll_interval must be >= 0; got {poll_interval}")
        self.max_cores = max_cores
        self._gpus = gpus
        self.poll_interval = poll_interval
        # job_id -> (cores, gpus, device_token|None)
        self._active: dict[str, tuple[int, int, str | None]] = {}

    # -- introspection -----------------------------------------------------

    @property
    def max_gpus(self) -> int:
        """Concurrent-GPU ceiling — :attr:`GpuBudget.count` of the budget this was built on.

        A property rather than a field so the budget stays the single value, and read-only
        because the two halves must not drift: raising this without matching devices is the
        state :class:`GpuBudget` exists to make unrepresentable.
        """
        return self._gpus.count

    @property
    def cores_in_use(self) -> int:
        """Sum of core demand of all currently active jobs."""
        return sum(cores for cores, _gpus, _dev in self._active.values())

    @property
    def gpus_in_use(self) -> int:
        """Sum of GPU demand of all currently active jobs."""
        return sum(gpus for _cores, gpus, _dev in self._active.values())

    @property
    def active_jobs(self) -> tuple[str, ...]:
        """Snapshot of active job IDs."""
        return tuple(self._active)

    # -- mutation ----------------------------------------------------------

    def register(self, job_id: str, pal: int, *, gpus: int = 0, device: str | None = None) -> None:
        """Mark a newly-submitted job as active, charging ``pal`` cores + ``gpus`` GPUs.

        ``device`` is the local GPU token handed out by :meth:`assign_device`
        (``None`` for CPU jobs or under SLURM, where the scheduler pins the GPU).
        """
        if pal < 1:
            raise ValueError(f"pal must be >= 1; got {pal}")
        if gpus < 0:
            raise ValueError(f"gpus must be >= 0; got {gpus}")
        self._active[job_id] = (pal, gpus, device)

    def assign_device(self) -> str:
        """Return the first of the budget's devices not held by an active job.

        Call **after** :meth:`has_room` has said yes (so a slot is guaranteed free) and
        **before** :meth:`register`, then pass the result as the job's
        ``CUDA_VISIBLE_DEVICES`` so concurrent local GPU jobs don't all pile onto one device.

        Drawn from :attr:`GpuBudget.devices` rather than from ``range(max_gpus)``, because
        the lowest free *index* is not necessarily a device this run owns: given
        ``CUDA_VISIBLE_DEVICES=2,3`` the free indices are 0 and 1, which are somebody else's
        GPUs. The tokens are also what makes a MIG instance addressable at all.
        """
        used = {dev for _c, gpus, dev in self._active.values() if gpus > 0 and dev is not None}
        for device in self._gpus.devices:
            if device not in used:
                return device
        # Unreachable when `has_room` admitted this job first; fail loud rather
        # than silently colliding two local GPU jobs on one device if that breaks.
        raise RuntimeError(
            f"no free GPU device among {list(self._gpus.devices)} — "
            f"call has_room before assign_device"
        )

    def has_room(self, pal_needed: int, *, gpus_needed: int = 0) -> bool:
        """Whether both budgets can admit a job of this size **right now**.

        Pure — polls nothing and reaps nothing. The one expression of "does this fit", and the
        whole of the admission decision: a caller tests this, submits if it is true, and waits
        in :meth:`wait_for_completion` if it is not. Splitting those into a blocking
        "wait for room" would mean a second loop that has to agree with this one about the
        same two budgets.
        """
        return (
            self.cores_in_use + pal_needed <= self.max_cores
            and self.gpus_in_use + gpus_needed <= self.max_gpus
        )

    # -- waiting -----------------------------------------------------------

    def wait_for_completion(
        self,
        *,
        finished: FinishedFn,
        max_wait_seconds: float | None = None,
    ) -> tuple[str, ...]:
        """Block until at least one active job finishes; return the ids that did.

        **The class's only wait**, and the only loop that polls. A caller that wants room
        tests :meth:`has_room` and waits here; a caller that wants the batch drained waits
        here until :attr:`active_jobs` is empty. Both are two lines at the call site
        (:func:`chemrefine.engines._execution._run_queue` is both at once), and neither needs
        a wait of its own — which is what keeps the polling, the reaping and the deadline in
        one place instead of three.

        Returns ``()`` when nothing is active. The caller decides what that means: for a
        drain it is success, and for a queue with work left it is impossible.

        ``max_wait_seconds`` bounds *this* wait — how long to sit with nothing completing.
        That is what :attr:`~chemrefine.config.Config.job_timeout_seconds` means on every path
        (:func:`chemrefine.slurm.wait_for_jobs` re-anchors it the same way): a bound on the
        whole drain would make a healthy long batch trip a timeout set to catch a stuck one.
        The deadline is a fresh :class:`StallDeadline` per call, and this returns on the first
        completion — so the clock restarts on every completion without this loop tracking that
        itself.
        """
        deadline = StallDeadline(max_wait_seconds)
        while self._active:
            if done := self._reap(finished):
                return done
            deadline.check("a job to finish")
            time.sleep(self.poll_interval)
        return ()

    # -- internals ---------------------------------------------------------

    def _reap(self, finished: FinishedFn) -> tuple[str, ...]:
        """Drop every active job the scheduler reports done — one poll for all of them.

        Returns the ids it reaped, so a caller can act on each completion rather than only on
        the budget it freed.

        Its one caller polls only under ``while self._active``, so this does not re-test that:
        a guard here would be the loop condition written twice, and the second copy is the one
        that would be left behind if the loop ever changed.
        """
        done = tuple(finished(tuple(self._active)))
        for jid in done:
            cores, gpus, _dev = self._active.pop(jid)
            logger.info("job %s finished, freed %d cores + %d gpus", jid, cores, gpus)
        return done

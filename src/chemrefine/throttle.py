"""Two-resource (CPU + GPU) budget throttler for batch job submission.

A pipeline step typically submits many jobs at once, but a host has a
fixed CPU budget (``max_cores``) and — locally — a fixed number of GPUs
(``max_gpus``). The throttler tracks each active job's ``(cores, gpus)``
demand and blocks new submissions until **both** budgets allow it.
Polling delegates to a caller-supplied ``is_finished`` callable so tests
can substitute a fake without monkeypatching :mod:`subprocess`.

The GPU budget only bites locally: under SLURM the scheduler places GPUs
itself (``--gres=gpu``), so the batch engine sets ``max_gpus`` effectively
unlimited there and to the detected device count locally. For a local GPU
job the throttler also hands out a free device index via
:meth:`assign_device` so concurrent jobs land on distinct GPUs.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from chemrefine.errors import ThrottleTimeoutError

logger = logging.getLogger(__name__)

IsFinishedFn = Callable[[str], bool]


class Throttler:
    """Track active jobs against CPU-core and GPU budgets."""

    def __init__(self, *, max_cores: int, max_gpus: int = 0, poll_interval: float = 10.0):
        if max_cores < 1:
            raise ValueError(f"max_cores must be >= 1; got {max_cores}")
        if max_gpus < 0:
            raise ValueError(f"max_gpus must be >= 0; got {max_gpus}")
        self.max_cores = max_cores
        self.max_gpus = max_gpus
        self.poll_interval = poll_interval
        # job_id -> (cores, gpus, device_index|None)
        self._active: dict[str, tuple[int, int, int | None]] = {}

    # -- introspection -----------------------------------------------------

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

    def register(self, job_id: str, pal: int, *, gpus: int = 0, device: int | None = None) -> None:
        """Mark a newly-submitted job as active, charging ``pal`` cores + ``gpus`` GPUs.

        ``device`` is the local GPU index handed out by :meth:`assign_device`
        (``None`` for CPU jobs or under SLURM, where the scheduler pins the GPU).
        """
        if pal < 1:
            raise ValueError(f"pal must be >= 1; got {pal}")
        if gpus < 0:
            raise ValueError(f"gpus must be >= 0; got {gpus}")
        self._active[job_id] = (pal, gpus, device)

    def assign_device(self) -> int:
        """Return the lowest GPU index in ``[0, max_gpus)`` not held by an active job.

        Call **after** :meth:`wait_for_room` has admitted a GPU job (so a slot is
        guaranteed free) and **before** :meth:`register`, then pass the result as
        the job's ``CUDA_VISIBLE_DEVICES`` so concurrent local GPU jobs don't all
        pile onto device 0.
        """
        used = {dev for _c, gpus, dev in self._active.values() if gpus > 0 and dev is not None}
        for idx in range(self.max_gpus):
            if idx not in used:
                return idx
        # Unreachable when wait_for_room admitted this job first; fail loud rather
        # than silently colliding two local GPU jobs on device 0 if that breaks.
        raise RuntimeError(
            f"no free GPU device in [0, {self.max_gpus}) — call wait_for_room before assign_device"
        )

    # -- waiting -----------------------------------------------------------

    def wait_for_room(
        self,
        pal_needed: int,
        *,
        is_finished: IsFinishedFn,
        gpus_needed: int = 0,
        max_wait_seconds: float | None = None,
    ) -> None:
        """Block until ``pal_needed`` cores **and** ``gpus_needed`` GPUs can be allocated.

        Calls ``is_finished`` once per loop iteration to reap completed
        jobs; sleeps ``poll_interval`` seconds before re-checking when
        room is still insufficient. Returns as soon as both budgets
        allow the request.

        Parameters
        ----------
        max_wait_seconds:
            Optional deadline in seconds. Raises
            :class:`~chemrefine.errors.ThrottleTimeoutError` if the
            budget has not freed up within this many seconds. ``None``
            (default) waits indefinitely.
        """
        if pal_needed > self.max_cores:
            raise ValueError(
                f"requested {pal_needed} cores exceeds the total budget {self.max_cores}"
            )
        if gpus_needed > self.max_gpus:
            raise ValueError(
                f"requested {gpus_needed} gpus exceeds the total budget {self.max_gpus}"
            )
        deadline = time.monotonic() + max_wait_seconds if max_wait_seconds is not None else None
        while True:
            self._reap(is_finished)
            if (
                self.cores_in_use + pal_needed <= self.max_cores
                and self.gpus_in_use + gpus_needed <= self.max_gpus
            ):
                return
            if deadline is not None and time.monotonic() >= deadline:
                raise ThrottleTimeoutError(
                    f"timed out after {max_wait_seconds}s waiting for "
                    f"{pal_needed} cores + {gpus_needed} gpus"
                )
            logger.debug(
                "waiting on budget: cores %d+%d/%d, gpus %d+%d/%d",
                self.cores_in_use,
                pal_needed,
                self.max_cores,
                self.gpus_in_use,
                gpus_needed,
                self.max_gpus,
            )
            time.sleep(self.poll_interval)

    def wait_all(
        self,
        *,
        is_finished: IsFinishedFn,
        max_wait_seconds: float | None = None,
    ) -> None:
        """Block until every active job has finished.

        Parameters
        ----------
        max_wait_seconds:
            Optional deadline. Raises
            :class:`~chemrefine.errors.ThrottleTimeoutError` if any jobs
            are still active when the deadline expires.
        """
        deadline = time.monotonic() + max_wait_seconds if max_wait_seconds is not None else None
        while self._active:
            self._reap(is_finished)
            if not self._active:
                return
            if deadline is not None and time.monotonic() >= deadline:
                raise ThrottleTimeoutError(
                    f"timed out after {max_wait_seconds}s waiting for all jobs to finish"
                )
            time.sleep(self.poll_interval)

    # -- internals ---------------------------------------------------------

    def _reap(self, is_finished: IsFinishedFn) -> None:
        """Drop every active job that ``is_finished`` reports done."""
        for jid in list(self._active):
            if is_finished(jid):
                cores, gpus, _dev = self._active.pop(jid)
                logger.info("job %s finished, freed %d cores + %d gpus", jid, cores, gpus)

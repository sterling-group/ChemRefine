"""PAL-budget throttler for batch job submission.

A pipeline step typically submits many jobs at once, but the cluster has
a fixed total core budget (``max_cores``). The throttler tracks active
jobs and blocks new submissions until enough cores free up. Polling
delegates to a caller-supplied ``is_finished`` callable so tests can
substitute a fake without monkeypatching :mod:`subprocess`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from chemrefine.errors import ThrottleTimeoutError

logger = logging.getLogger(__name__)

IsFinishedFn = Callable[[str], bool]


class Throttler:
    """Track active jobs against a total PAL budget."""

    def __init__(self, *, max_cores: int, poll_interval: float = 10.0):
        if max_cores < 1:
            raise ValueError(f"max_cores must be >= 1; got {max_cores}")
        self.max_cores = max_cores
        self.poll_interval = poll_interval
        self._active: dict[str, int] = {}

    # -- introspection -----------------------------------------------------

    @property
    def cores_in_use(self) -> int:
        """Sum of PAL values of all currently active jobs."""
        return sum(self._active.values())

    @property
    def active_jobs(self) -> tuple[str, ...]:
        """Snapshot of active job IDs."""
        return tuple(self._active)

    # -- mutation ----------------------------------------------------------

    def register(self, job_id: str, pal: int) -> None:
        """Mark a newly-submitted job as active, charging ``pal`` cores."""
        if pal < 1:
            raise ValueError(f"pal must be >= 1; got {pal}")
        self._active[job_id] = pal

    # -- waiting -----------------------------------------------------------

    def wait_for_room(
        self,
        pal_needed: int,
        *,
        is_finished: IsFinishedFn,
        max_wait_seconds: float | None = None,
    ) -> None:
        """Block until ``pal_needed`` cores can be allocated.

        Calls ``is_finished`` once per loop iteration to reap completed
        jobs; sleeps ``poll_interval`` seconds before re-checking when
        room is still insufficient. Returns as soon as the budget
        allows the request.

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
        deadline = time.monotonic() + max_wait_seconds if max_wait_seconds is not None else None
        while True:
            self._reap(is_finished)
            if self.cores_in_use + pal_needed <= self.max_cores:
                return
            if deadline is not None and time.monotonic() >= deadline:
                raise ThrottleTimeoutError(
                    f"timed out after {max_wait_seconds}s waiting for {pal_needed} cores"
                )
            logger.debug(
                "waiting on cores: %d in use + %d needed > %d budget",
                self.cores_in_use,
                pal_needed,
                self.max_cores,
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
                cores = self._active.pop(jid)
                logger.info("job %s finished, freed %d cores", jid, cores)

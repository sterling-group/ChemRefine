"""Exception hierarchy for ChemRefine.

Every exception carries a non-zero ``exit_code`` so the CLI can translate
the failure mode into a deterministic process exit code. The root class is
:class:`ChemRefineError` — all package exceptions inherit from it so the
CLI can catch one type and dispatch on the code.

Currently allocated exit codes (assign the next free integer when adding
a new exception class):

============  ============================================
Exit code     Meaning
============  ============================================
``1``         :class:`ChemRefineError` — generic / catch-all,
              and :class:`NoUsableCacheError` for a non-submitting
              step with no cache the configuration can serve
              (same code; the subclass exists to be *caught*)
``2``         :class:`ConfigError` — YAML config invalid
``3``         :class:`EngineNotFoundError` — unknown engine
``4``         :class:`JobSubmissionError` — sbatch refused the job
``5``         :class:`JobFailureError` — job ran but indicated failure
``6``         :class:`OutputParseError` — could not parse engine output,
              and :class:`OutputTerminationError` for the case where the
              reason is that the program died
``7``         :class:`CacheError` — step cache corrupt / unwritable
``8``         :class:`ThrottleTimeoutError` — wait deadline expired
``9``         :class:`BackendProvisionError` — a managed backend env
              could not be built
``10``        :class:`RunLockError` — another run holds this output tree
============  ============================================
"""

from __future__ import annotations


class ChemRefineError(Exception):
    """Base class for every ChemRefine-raised exception."""

    exit_code: int = 1


class NoUsableCacheError(ChemRefineError):
    """A step that may not submit has no cache the current configuration can serve.

    Raised by :func:`chemrefine.step.run_step` when a ``cache-only`` step cannot be served
    from disk. A subclass sharing the generic code rather than a code of its own — like
    :class:`OutputTerminationError`, what it adds is not a different exit status but an
    *identity*: past a scoped rebuild's target, :func:`chemrefine.pipeline.run` catches
    exactly this to end the report quietly at the first step the configuration cannot
    vouch for, while the same failure anywhere else must still fail the run.
    """


class ConfigError(ChemRefineError):
    """The YAML config is malformed, missing required fields, or invalid."""

    exit_code = 2


class EngineNotFoundError(ChemRefineError):
    """A step requested an engine that is not in the registry."""

    exit_code = 3


class JobSubmissionError(ChemRefineError):
    """``sbatch`` (or the local runner) refused to accept a job."""

    exit_code = 4


class JobFailureError(ChemRefineError):
    """A submitted job finished but its output indicates failure."""

    exit_code = 5


class OutputParseError(ChemRefineError):
    """An engine output file could not be parsed."""

    exit_code = 6


class OutputTerminationError(OutputParseError):
    """An output could not be read because the program that wrote it did not finish.

    A subclass rather than a separate code: it *is* an unreadable output, so the exit
    status is the same. What it adds is the reason — the run died — which
    :func:`chemrefine.lifecycle._parse_job` files as
    :attr:`~chemrefine.state.FailureKind.NOT_TERMINATED_NORMALLY` rather than
    ``UNPARSEABLE``. The distinction is what a reader acts on: one says look at the
    parser, the other says look at the job.
    """


class CacheError(ChemRefineError):
    """A step cache is corrupt, version-mismatched, or unwritable."""

    exit_code = 7


class ThrottleTimeoutError(ChemRefineError):
    """A job did not finish within the configured wait deadline."""

    exit_code = 8


class BackendProvisionError(ChemRefineError):
    """Building a managed backend environment failed.

    Its own code rather than :class:`ConfigError`'s, because nothing is wrong with the
    config: the YAML named a backend this machine could not *build* — no network on the
    node, a resolver conflict, a full disk, or an env tool that is a shell function rather
    than a binary on ``PATH``. Those are all fixed by acting on the machine, not the file,
    and a script wrapping ``chemrefine backends install`` needs to tell the two apart.
    """

    exit_code = 9


class RunLockError(ChemRefineError):
    """Another driver holds this output tree's run lock.

    Its own code because the fix is unlike any other failure's: nothing is wrong with the
    config, the jobs, or this machine — a second ``chemrefine`` invocation was pointed at
    an output tree a first one is still working in, and the resume machinery cannot tell a
    *live* concurrent driver from a dead one by the files alone (a matching manifest
    fingerprint is exactly what both leave). The remedy is to wait, or to delete a lock
    whose holder is known dead — see :func:`chemrefine.pipeline.run_lock`.
    """

    exit_code = 10

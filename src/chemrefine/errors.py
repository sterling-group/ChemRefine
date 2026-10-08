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
``7``         :class:`CacheError` — step cache corrupt / unreadable
``8``         :class:`ThrottleTimeoutError` — wait deadline expired
``9``         :class:`BackendProvisionError` — a managed backend env
              could not be built
``10``        :class:`RunLockError` — another run holds this output tree
``11``        :class:`OutputTreeError` — the output tree refused a write
============  ============================================
"""

from __future__ import annotations

from http.client import HTTPException
from typing import TYPE_CHECKING
from urllib.error import HTTPError, URLError

if TYPE_CHECKING:
    from urllib.request import Request


class ChemRefineError(Exception):
    """Base class for every ChemRefine-raised exception."""

    exit_code: int = 1


def refusal_text(error: ChemRefineError) -> str:
    """A refusal as every surface shows it: the class that names it, then its message.

    ``ConfigError: …`` / ``RunLockError: …`` — the class is the part of the taxonomy a
    reader acts on (fix the config, or wait for the other driver), and the message says
    what to change. The CLI logs it, the terminal chat prints it, and an agent's model
    reads it as the tool's answer (:func:`chemrefine.agent_tools.answering`).
    """
    return f"{type(error).__name__}: {error}"


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
    """The step cache's verdict on its own state: what it cannot read, trust, prove or store.

    A ``_cache/`` document it cannot read or trust — unreadable, corrupt, written by another
    version — a directory of the run's tree it cannot list, outputs on disk it cannot prove
    are this configuration's, a structure it will not store. The filesystem refusing a
    write under ``_cache/`` is no verdict of the cache's: that is :class:`OutputTreeError`,
    as it is for every other file of the run.
    """

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

    A filesystem that cannot hard-link at all refuses with this code too, before anything
    runs: it can never hold the lock, and the remedy is to put ``output_dir`` on one that
    can.
    """

    exit_code = 10


class OutputTreeError(ChemRefineError):
    """The filesystem refused a write the run makes in its own output tree.

    A full disk or quota, a read-only mount, or a directory under ``output_dir`` this
    account cannot write. The message names the file the run was writing, and the one the
    filesystem refused when that was another — a directory it could not create, the source
    of a copy (:func:`chemrefine.io.tree_write`). Nothing is wrong with the config, the jobs
    or the cache. Its own code, for the reason :class:`BackendProvisionError` has one: the
    fix is the machine's, and a script wrapping a run must tell a full disk from a step
    that halted.

    Free space, remount read-write or fix the permissions, then run the command that
    stopped again. ``run`` and ``rerun`` begin by discarding caches, so ``resume`` in their
    place would serve what they were asked to discard; a ``run``, ``resume``, ``rerun`` or
    ``rerun-errors`` that stopped inside its steps continues with ``chemrefine resume``,
    which adopts what the jobs already finished — an artifact step's product among it. A
    message that also names a SLURM job the run could not cancel, or a live run whose lock
    it could not put back, says what to do about that first.

    Not an :class:`OSError`, though the filesystem's refusal is its ``__cause__``: every
    ``except OSError`` between the write and the CLI would take it for one of its own and
    word it again — the seed reader's net would call a full disk under ``_seed/`` a bad
    seed file.
    """

    exit_code = 11


EXIT_CODES: dict[str, int] = {
    name: obj.exit_code
    for name, obj in list(globals().items())
    if isinstance(obj, type) and issubclass(obj, ChemRefineError)
}
"""Every class above, mapped to the code it exits with — the taxonomy, derived once.

Scanned from this module's own namespace, which catches an indirect subclass:
``ChemRefineError.__subclasses__()`` lists direct subclasses only and would miss
:class:`OutputTerminationError`. Beside the classes, it is the one table every caller reads.
"""


ENDPOINT_UNREACHABLE: tuple[type[BaseException], ...] = (URLError, OSError, HTTPException)
"""What a :mod:`urllib` fetch raises when an endpoint gives no usable answer — one tuple.

Every fetch this package makes goes through :func:`fetch` and catches this, after any
narrower class it words for itself, rather than a tuple of its own: spelled per site,
``HTTPException`` — a reply that arrives and then breaks the protocol, ``IncompleteRead``
on a dropped connection — would reach one fetch and not the other. ``URLError`` is DNS
and a refused connection (and, as ``HTTPError``, an HTTP status a caller that wants the
code catches first); ``OSError`` is a socket that fails or times out (``TimeoutError`` is
one); ``HTTPException`` is the broken protocol. What a *body* that did arrive means —
undecodable, not JSON, empty — is each caller's own verdict and stays beside it.
"""


def fetch(request: Request, *, timeout: float) -> bytes:
    """The body of one HTTP exchange — every fetch this package makes goes through here.

    Raises what :mod:`urllib` raises (:data:`ENDPOINT_UNREACHABLE`, ``HTTPError`` among
    it), with one thing done first: an ``HTTPError`` is also the error *response*, a file
    over the body the server sent, and it is closed here the way the ``with`` closes a
    success. Its ``code`` and ``reason`` survive the close for the caller to word; left to
    the collector, the open file is a ``ResourceWarning`` on 3.14. The request's scheme is
    each caller's to vouch for.
    """
    from urllib.request import urlopen

    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 — each caller's scheme
            body: bytes = response.read()
    except HTTPError as e:
        e.close()
        raise
    return body

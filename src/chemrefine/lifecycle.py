"""The body of a step: run a set of structures, classify, apply the policy, persist.

Given an engine, a :class:`~chemrefine.state.StepContext` and some structures, this produces
:class:`~chemrefine.state.StepResults` honouring the step's ``on_failure`` setting. It is the
part :mod:`chemrefine.step` and :mod:`chemrefine.nms` both need — ``step`` wraps it in caching
and filtering, ``nms`` runs it for each round of displaced children — so it sits below both.

The phases:

* :func:`submit_and_parse` — prepare, submit, parse a set of structures in one directory.
* :func:`parse_with_failures` — parse outputs that already exist, classifying each
  (:func:`succeeded`, :func:`failure_kind`).
* :func:`retry_unconverged` — give the convergence failures one more try, together, from
  the best geometries they reached.
* :func:`resubmit_unusable` — re-run whatever left no usable result at all, for a step being
  resumed or repaired.
* :func:`run_with_retries` — run a prepared batch *and* everything it earns as one piece of
  work: a re-run, or a :class:`ChildRound` (NMS round 2), starts as soon as the job that
  earned it frees a slot rather than after the batch drains.
* :func:`run_child_rounds` — the same queue for a caller whose parents finished elsewhere.
* :func:`apply_failure_policy` — ``stop | skip | best``, and the ledger that records it.
* :func:`finalize` — the two of those a step always ends with, in the right order.

It does **not** own the order of a whole step. :mod:`chemrefine.step` interleaves phases with
work of its own — the manifest is written between ``prepare`` and ``submit``, so an interrupted
run leaves proof of what its outputs were computed for — and that composition belongs there.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

from chemrefine import __version__, attempts, cache
from chemrefine.config import StepConfig
from chemrefine.engines.api import CalculationEngine, StreamingSubmit, sweep
from chemrefine.errors import ChemRefineError, OutputParseError, OutputTerminationError
from chemrefine.state import (
    Failure,
    FailureKind,
    FailureRecord,
    JobTriple,
    PipelineState,
    StepContext,
    StepInputs,
    StepResults,
    Structure,
)

logger = logging.getLogger(__name__)


def succeeded(s: Structure) -> bool:
    """A parsed structure failed only when an engine success flag is explicitly False.

    ``None`` (engine doesn't report it) is treated as 'not a failure signal', so
    backends that don't set termination/convergence flags are never gated.
    """
    return s.terminated_normally is not False and s.converged is not False


def failure_kind(s: Structure) -> FailureKind:
    """Classify why a parsed structure counts as a failure.

    ``terminated_normally is False`` → the engine crashed / didn't finish cleanly;
    ``converged is False`` → it finished but the SCF/geometry didn't converge;
    otherwise :attr:`FailureKind.FAILED` (a flag the engine set with no name of its own here).
    """
    if s.terminated_normally is False:
        return FailureKind.NOT_TERMINATED_NORMALLY
    if s.converged is False:
        return FailureKind.NOT_CONVERGED
    return FailureKind.FAILED


def _parse_job(
    engine: CalculationEngine, triple: JobTriple, ctx: StepContext
) -> tuple[list[Structure], Failure | None]:
    """Parse one job: the structures it produced, and its failure if it has one.

    A job is a *failure* when its output is missing, unparseable, or parses to a structure
    the engine marks unconverged / not-terminated. A fan-out job (a GOAT ensemble, a PES
    scan) yields several structures and at most one failure, described by the best geometry
    among the bad frames — the reason has to describe the geometry carried forward, because
    :func:`retry_unconverged` routes on it.

    An output that could not be read *because the program died* is filed as
    :attr:`~chemrefine.state.FailureKind.NOT_TERMINATED_NORMALLY` rather than ``UNPARSEABLE``: both
    describe an unusable output, but only one of them points at the job. A ledger full of
    "unparseable" sends a reader to the parser for what is a cluster or input problem.
    """
    _inp, out, sid = triple
    if not out.is_file():
        return [], Failure(sid, FailureKind.MISSING_OUTPUT, None)
    try:
        parsed = list(engine.parse(StepInputs(files=(triple,)), ctx).structures)
    except OutputTerminationError as e:
        return [], Failure(sid, FailureKind.NOT_TERMINATED_NORMALLY, None, detail=str(e))
    except OutputParseError as e:
        return [], Failure(sid, FailureKind.UNPARSEABLE, None, detail=str(e))
    bad = [s for s in parsed if not succeeded(s)]
    if not bad:
        return parsed, None
    best = min(bad, key=lambda s: (s.energy_hartree is None, s.energy_hartree or 0.0))
    return parsed, Failure(sid, failure_kind(best), best)


def parse_with_failures(
    engine: CalculationEngine, inputs: StepInputs, ctx: StepContext
) -> tuple[list[Structure], list[Failure]]:
    """Parse a batch's outputs and classify them. Writes nothing.

    Use this to read a tree you must not modify — :func:`chemrefine.nms.rebuild_nms` re-reads
    round-2 children only to re-derive which one won. Everything that *owns* the results it
    parses wants :func:`parse_and_record`.

    Parsing is per job (see :meth:`_ResultLedger.absorb`) so one bad output never crashes the
    step: its failure is captured and the rest still parse.
    """
    ledger = _ResultLedger.read_only(inputs)
    ledger.absorb_all(engine, ctx, inputs)
    return ledger.emit()


def parse_and_record(
    engine: CalculationEngine, inputs: StepInputs, ctx: StepContext
) -> tuple[list[Structure], list[Failure]]:
    """:func:`parse_with_failures`, and drop each job's canonical result record beside it.

    The record is the engine-independent JSON every calculation leaves next to its native
    output — written for failures too, since an unconverged result is still a parsed one.
    """
    ledger = _ResultLedger(inputs)
    ledger.absorb_all(engine, ctx, inputs)
    return ledger.emit()


# ---------------------------------------------------------------------------
# Retry — re-run a structure once from its best geometry
# ---------------------------------------------------------------------------


def submit_and_parse(
    engine: CalculationEngine, ctx: StepContext, structures: Sequence[Structure]
) -> tuple[list[Structure], list[Failure]]:
    """Prepare, submit and parse ``structures`` in ``ctx.step_dir``.

    The caller supplies the context, so the same call runs a structure at its canonical place
    or a set of NMS children inside an attempt directory — the difference is ``ctx.step_dir``.
    """
    run_ctx = replace(ctx, prev_state=PipelineState(structures=tuple(structures)))
    inputs = engine.prepare(run_ctx)
    engine.submit(inputs, run_ctx)
    return parse_and_record(engine, inputs, run_ctx)


def resubmit_unusable(
    engine: CalculationEngine,
    ctx: StepContext,
    inputs: StepInputs,
    *,
    stale: Iterable[str] = (),
) -> tuple[list[Structure], list[Failure]]:
    """Parse ``inputs``, re-run every job that produced no usable result, report the merged lot.

    What resuming an interrupted step and ``rerun-errors`` both do. **One parse of the tree**
    answers both questions it has — which jobs have to run again, and what the finished
    structures are — so a 277-structure step does not read every output twice; only the jobs
    actually re-run are parsed a second time.

    *Unusable* means the job did not leave a result this step can stand behind: no output at
    all, an output that will not parse, or one whose engine says it did not terminate
    normally. A structure that merely failed to **converge** is excluded and left to
    :func:`retry_unconverged`, which restarts it from the geometry it did reach —
    resubmitting the identical input would only fail the same way.

    Judging by what parsed rather than by which files exist is what closes the window every
    re-run opens: preparing one archives the previous attempt and writes a fresh input, so a
    run killed in between leaves an output that is present but truncated.
    A caller that asked ``out.is_file()`` would call that finished and hand the structure to
    the failure policy with an unused attempt and a good geometry sitting in ``attemptK/``.

    Prior artifacts are archived into ``attemptK/`` and inputs are **regenerated** rather than
    reused: archiving is what stops a job that dies without producing output from re-reading
    the old result, and regenerating means a ``rerun-errors`` after a template edit actually
    runs the edited template — the on-disk input would otherwise contradict the fingerprint
    the cache is keyed on.

    ``stale`` names rows condemned by the caller's provenance verdict — an output on disk
    that was computed for a *different parent* (the incremental resume's row-key diff). A
    stale output can be perfectly usable and still be the wrong answer, which no parse can
    see — so whatever its first parse produced is discarded unread by the same
    :meth:`~_ResultLedger.restart` a resubmission always gets, and the row re-runs
    unconditionally. Condemned rows arrive **already archived by the caller** — moved
    aside *before* the manifest was stamped with the keys that condemn them, which is the
    ordering that closes the stamp-then-crash window (a manifest vouching for outputs
    still at canonical). Archiving them again here would only seal the freshly rendered
    input into an attempt of its own, so the archive below skips them.
    """
    condemned = frozenset(stale)
    ledger = _ResultLedger(inputs)
    ledger.absorb_all(engine, ctx, inputs)
    _successes, failures = ledger.emit()

    unusable = {f.sid for f in failures if f.kind is not FailureKind.NOT_CONVERGED} | condemned
    seeds = tuple(s for s in ctx.prev_state.structures if s.id in unusable)
    if not seeds:
        return ledger.emit()

    logger.info(
        "step %d: resubmitting %d job(s) with no usable result: %s",
        ctx.step_cfg.step,
        len(seeds),
        ", ".join(s.id for s in seeds),
    )
    for s in seeds:
        ledger.restart(s.id)
    attempts.archive_previous(ctx.step_dir, (s.id for s in seeds if s.id not in condemned))
    retry_ctx = replace(ctx, prev_state=PipelineState(structures=seeds))
    redone = engine.prepare(retry_ctx)
    engine.submit(redone, retry_ctx)
    ledger.absorb_all(engine, retry_ctx, redone)
    return ledger.emit()


def retryable_best(failure: Failure) -> Structure | None:
    """The geometry a failure should be re-run from, or ``None`` if it should not be.

    **What counts as retryable**, in one place. Convergence-only: a crashed or missing-output
    job would fail the same way a second time, so it is left to the ``on_failure`` policy —
    and a convergence failure with no geometry to restart from has nothing to offer either.

    Both retry paths ask this and only this. How a retry is then *scheduled* differs between
    them for a real reason — :func:`rerun_from_best` submits a whole batch and parses it
    against one combined ``prev_state``, while :class:`_QueueSink` handles one job at a time
    and carries each structure separately — so that part is deliberately not shared.
    """
    if failure.kind is not FailureKind.NOT_CONVERGED or failure.best is None:
        return None
    return failure.best


def retry_unconverged(
    engine: CalculationEngine,
    ctx_for_prepare: StepContext,
    successes: list[Structure],
    failures: list[Failure],
) -> tuple[list[Structure], list[Failure]]:
    """Retry the *unconverged* failures once, from their best geometries, in one batch.

    Convergence-only — a crashed / missing-output failure (or one with no best geometry) is
    left untouched for the ``on_failure`` policy. A single inline pass (no recursion): each
    structure is retried at most once per run; a later ``resume`` archives into the next
    ``attemptK/``. ``ctx_for_prepare.step_dir`` is what nests the structures, so the caller
    passes the context whose directory the re-runs belong under.

    For jobs that have **already run** — a step being resumed or repaired. Work still in
    flight is retried by the queue instead (:class:`_QueueSink`), which puts a re-run in the
    slot its own failed job just freed; that path and this one share the predicate
    (:func:`retryable_best`) and nothing else, because scheduling a batch after the fact and
    scheduling one job mid-drain are genuinely different problems.

    **One batch, not one call per structure.** Submission is budgeted by
    :class:`chemrefine.throttle.Throttler`, which is built per
    :func:`~chemrefine.engines._execution.run_batch` call and does not return until that
    call's jobs are done. Retrying in a loop would therefore hand the throttler one job at a
    time: each retry waiting for the previous one to *finish* before the next is submitted,
    so a 128-core budget runs a single 16-core job and idles the other 112 — and the
    wall-clock cost is the sum of the retries rather than the longest of them. Passing them
    together is the whole of what lets the existing budget apply.

    **The structures are resubmitted whole, not rebuilt from their parts.** Reconstructing
    one as ``Structure(id=..., atoms=...)`` would drop every field not named — ``parent_id``
    above all, which :func:`chemrefine.engines._job.build_structures` reads back off
    ``prev_state``, so a structure that merely needed a second attempt would leave the step
    an orphan, permanently, into the cache. That is not cosmetic downstream:
    :func:`chemrefine.filtering._filter_by_parent` groups on ``parent_id or id``, so an
    orphan is indistinguishable from a seed, forms its own singleton group, and survives a
    filter that should have discarded it. They are frozen and nothing here mutates them, so
    there is nothing to copy — their stale result fields are overwritten by the re-parse, and
    passing them whole is what carries a field added to
    :class:`~chemrefine.state.Structure` later through the retry without anyone remembering to.

    The partition is one pass rather than two comprehensions because a
    :class:`~chemrefine.state.Failure` carries a :class:`~chemrefine.state.Structure`, whose
    ``atoms`` make ``==`` unreliable — so "the failures that were not retried" has to be
    accumulated as it is decided, not recovered afterwards by comparing against the
    retried ones.
    """
    bests: list[Structure] = []
    retried_sids: list[str] = []
    remaining: list[Failure] = []
    for f in failures:
        if (best := retryable_best(f)) is not None:
            bests.append(best)
            retried_sids.append(f.sid)
        else:
            remaining.append(f)
    if not bests:
        return list(successes), remaining
    logger.info(
        "%d structure(s) did not converge — retrying from their best geometry: %s",
        len(bests),
        ", ".join(retried_sids),
    )
    attempts.archive_previous(ctx_for_prepare.step_dir, (b.id for b in bests))
    succ, fail = submit_and_parse(engine, ctx_for_prepare, bests)
    return [*successes, *succ], [*remaining, *fail]


# ---------------------------------------------------------------------------
# Streaming — run a batch and its retries as one piece of work
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Pending:
    """A job in flight, and everything needed to interpret its result.

    One record rather than a side table per fact: a follow-up needs both *whose* verdict it
    contributes to and *what* it re-runs, and two dicts keyed by the same path would be two
    things to keep in step by hand.
    """

    scope: _JobScope
    """The set of jobs this one belongs to — whose ledger takes its result, whose directory
    its re-run archives under, and whose one-retry budget it draws on."""

    origin_sid: str
    """The structure this job's result belongs to within its scope — its own id for an
    original, and the id of the job it re-runs for a follow-up (which may differ: a fan-out
    job's best geometry is a *child*, so a retry of ``5`` is submitted as ``5-3``)."""

    structure: Structure | None
    """What this job re-runs, or ``None`` for an original. See :meth:`_JobScope.ctx_for`."""


class _ResultLedger:
    """A step's results as they arrive, emitted in the order its manifest lists them.

    Two things it exists to get right, both of which are silent when wrong:

    **Order.** :meth:`chemrefine.cache.StepKey.of` keys the next step over its parents
    *in iteration order*, so results in completion order — which is nondeterministic — would
    give the next step a different fingerprint on every run and invalidate its cache for
    nothing. Manifest order is stable. It is *not* the order a non-streaming run produced
    (that appended retries at the end), so a step that retries re-runs its downstream steps
    once; see the CHANGELOG.

    **Accumulation.** A fan-out job contributes both successes and a failure — a GOAT run with
    47 usable conformers and one bad one — and its retry must *add* to those 47, not replace
    them. So a record appends structures and replaces only the failure.
    """

    def __init__(self, manifest: StepInputs, *, records: bool = True) -> None:
        """``manifest`` fixes the order results come back in; it need not be what is parsed.

        The two differ where a job is re-run: :func:`resubmit_unusable` absorbs the whole
        manifest and then only the jobs it re-ran, and both land under the same origin in the
        same emitted position.

        ``records`` is set here rather than per call because owning the tree is a property of
        the *operation*: a ledger that recorded one pass and not the next would leave a step
        half-documented on disk, which no caller could want. :meth:`read_only` is the one
        case that does not own it.
        """
        self._order = [sid for _inp, _out, sid in manifest.files]
        self._by_sid: dict[str, tuple[list[Structure], Failure | None]] = {}
        self._records = records

    @classmethod
    def read_only(cls, manifest: StepInputs) -> _ResultLedger:
        """A ledger for a tree that must not be modified — it writes no result records.

        :func:`chemrefine.nms.rebuild_nms` re-reads round-2 children only to re-derive which
        one won, and a rebuild that wrote would be rewriting the outputs it was asked to read.
        """
        return cls(manifest, records=False)

    def restart(self, origin_sid: str) -> None:
        """Forget what an origin produced, because its job is about to be run from scratch.

        The counterpart to :meth:`absorb`'s appending. A *retry* re-runs one bad frame and its
        result joins the good ones, so appending is right; a *resubmission* re-runs the whole
        job, so every frame it produced is superseded. Without this, a resubmitted fan-out
        would emit both attempts' children — two structures wearing the same id.
        """
        self._by_sid.pop(origin_sid, None)

    def absorb(
        self, engine: CalculationEngine, ctx: StepContext, job: JobTriple, origin_sid: str
    ) -> tuple[list[Structure], Failure | None]:
        """Parse one finished job into ``origin_sid``'s results; return what it produced.

        The one place a job becomes results, for every caller: a batch parsed after the fact
        and a job reported the moment it finishes go through here alike, so they cannot
        disagree about accumulation, classification, ordering or what gets written beside the
        output.

        Both halves come back because a caller decides two things from them — whether a
        *failure* earns a re-run, and what a *success* earns beyond itself
        (:class:`ChildRound`). Reading the successes back off the ledger instead would answer
        a different question: everything that origin has accumulated, not what this job
        produced.

        ``ctx`` is the job's own, which is not always the step's —
        :meth:`_JobScope.ctx_for` explains when and why.
        """
        structures, failure = _parse_job(engine, job, ctx)
        if self._records and structures:
            # Written for failures too: an unconverged result is still a parsed one.
            cache.save_result_records(structures, job[1].parent, ctx.step_cfg.step)
        kept, _previous = self._by_sid.get(origin_sid, ([], None))
        self._by_sid[origin_sid] = ([*kept, *structures], failure)
        return structures, failure

    def absorb_all(self, engine: CalculationEngine, ctx: StepContext, inputs: StepInputs) -> None:
        """Parse every job of ``inputs`` into this ledger, each under its own structure id.

        ``inputs`` rather than the manifest this ledger was built from, so a re-run can be
        absorbed into the same ledger as the batch it replaces.
        """
        for job in inputs.files:
            self.absorb(engine, ctx, job, job[2])

    def emit(self) -> tuple[list[Structure], list[Failure]]:
        """The step's successes and failures, in manifest order.

        A job's *bad* frames are dropped here rather than at absorption: a fan-out job's
        unconverged frame is still the geometry its retry restarts from, so it has to survive
        until the verdict is final.
        """
        successes: list[Structure] = []
        failures: list[Failure] = []
        for sid in self._order:
            if sid not in self._by_sid:
                # Unreachable: no scheduler returns while work is outstanding. Raising
                # because the alternative is dropping a structure — which shrinks the
                # survivor set and shifts every downstream fingerprint, with nothing to
                # show for it.
                raise ChemRefineError(f"structure {sid} was submitted but never reported")
            structures, failure = self._by_sid[sid]
            successes.extend(s for s in structures if succeeded(s))
            if failure is not None:
                failures.append(failure)
        return successes, failures


@dataclass(frozen=True)
class ChildRun:
    """The jobs one finished structure earns, and the directory they run in.

    Two facts rather than a bare list, because they are decided together and by the same
    party: :mod:`chemrefine.nms` picks a fresh ``attemptK/`` *and* the ± displacements that go
    in it, and a scheduler that chose the directory itself would be choosing where the winner
    is later promoted from.
    """

    directory: Path
    structures: tuple[Structure, ...]


class ChildRound(Protocol):
    """Work a finished structure earns, run in the same queue but judged on its own.

    The seam :mod:`chemrefine.nms` reaches the scheduler through. Declared *here* because
    this module sits below ``nms`` and must not import it; ``nms`` implements it and
    :func:`chemrefine.step._run_full_step` hands the implementation to both sides.

    Distinct from a *retry*, which is the same structure run again into the same verdict. A
    child round is **different structures, in a directory of their own, whose results are not
    the step's**: NMS round-2 children belong to the parent they were displaced from, and a
    step whose ledger held them would emit a survivor per child and fingerprint the next step
    on geometries it discarded.

    Not ``runtime_checkable`` — nothing tests it with ``isinstance``, for the same reason
    :class:`~chemrefine.engines.api.CompletionSink` is not.
    """

    def children_for(self, structure: Structure) -> ChildRun | None:
        """The round ``structure`` earns, or ``None`` when it earns none.

        Asked once per finished structure, the moment it lands. The *decision* must not depend
        on when it is asked: a queue calls this in completion order, which is nondeterministic,
        so an answer that varied with call order would make a run's results vary with its
        scheduling.
        """
        ...

    def settled(self, origin_sid: str, successes: list[Structure], failures: list[Failure]) -> None:
        """Take one round's verdict, once the queue that ran it has drained."""
        ...


@dataclass(frozen=True)
class _JobScope:
    """A set of jobs that share a directory, a verdict and a one-retry budget.

    A step is one. Each NMS parent's round-2 children are another: they run under
    ``stepN/<parent>/attemptK/``, their results are that parent's rather than the step's, and
    each child earns its own retry — landing in ``attemptK/<child>/attempt1/``, because
    ``ctx.step_dir`` is what :func:`chemrefine.attempts.archive_previous` and ``prepare``
    nest by.

    The three facts travel together because they are one decision. The budget is what makes
    this a class rather than three fields on :class:`_Pending`: it is per-*scope*, not
    per-job, so spread across jobs it would need a side table keyed by an identity only those
    fields imply — the second mapping over the same key that :class:`_Pending` exists to
    avoid, reintroduced one level up.
    """

    ctx: StepContext
    ledger: _ResultLedger
    attempted: set[str] = field(default_factory=set)

    def ctx_for(self, structure: Structure | None) -> StepContext:
        """The context a job must be parsed with — its own, not necessarily the scope's.

        :func:`chemrefine.engines._job.build_structures` resolves lineage by looking the job's
        id up in ``prev_state`` and, finding nothing, sets ``parent_id=None`` *silently*. A
        retry's id is the best geometry's, which for a fan-out job is a child id the scope's
        parents do not contain — so parsing it against the scope context would emit an orphan,
        permanently, into the cache. :func:`submit_and_parse` avoids the same trap by
        rebinding ``prev_state``; this is that, per job.
        """
        if structure is None:
            return self.ctx
        return replace(self.ctx, prev_state=PipelineState((structure,)))

    def claim_retry(self, origin_sid: str) -> bool:
        """Whether this origin may be re-run — true at most once per origin, per scope.

        Test and set in one call, so "at most once" is one statement rather than a check here
        and an add three lines later, with a return in between.
        """
        if origin_sid in self.attempted:
            return False
        self.attempted.add(origin_sid)
        return True


class _QueueSink:
    """Parse each job as it finishes, and ask for a re-run when one is warranted.

    The :class:`~chemrefine.engines.api.CompletionSink` a step runs under. It owns everything
    the scheduler must not know: which context a job is parsed with, where its result belongs,
    and the one-attempt budget.

    Named for the queue rather than the step because one queue can carry jobs answering to
    several verdicts at once; each job's :class:`_Pending` names the :class:`_JobScope` its
    result belongs to.
    """

    def __init__(
        self,
        engine: CalculationEngine,
        root: _JobScope,
        inputs: StepInputs,
        children: ChildRound | None = None,
    ) -> None:
        self._engine = engine
        self._root = root
        self._children = children
        self._pending = {inp: _Pending(root, sid, None) for inp, _out, sid in inputs.files}
        self._rounds: dict[str, _JobScope] = {}

    def on_complete(self, job: JobTriple) -> tuple[JobTriple, ...]:
        """Record one finished job; return its re-run and any round it earned.

        Both, not either: a fan-out job contributes good frames *and* a bad one, so the good
        frames' rounds start now while the bad frame goes round the queue again.
        """
        pending = self._pending[job[0]]
        scope = pending.scope
        structures, failure = scope.ledger.absorb(
            self._engine, scope.ctx_for(pending.structure), job, pending.origin_sid
        )
        return (*self._retry(pending, failure), *self._follow(pending, structures))

    def _retry(self, pending: _Pending, failure: Failure | None) -> tuple[JobTriple, ...]:
        """The job's own re-run, from the best geometry it reached — at most once."""
        if failure is None:
            return ()
        best = retryable_best(failure)
        if best is None or not pending.scope.claim_retry(pending.origin_sid):
            return ()
        logger.info("structure %s did not converge — retrying from its best geometry", failure.sid)
        triple, structure = _retry_input(self._engine, pending.scope.ctx, best)
        self._pending[triple[0]] = _Pending(pending.scope, pending.origin_sid, structure)
        return (triple,)

    def _follow(self, pending: _Pending, structures: list[Structure]) -> tuple[JobTriple, ...]:
        """The rounds this job's results earned, if any.

        Two rules, both load-bearing. **Only the root scope spawns**: a round is a second
        round by definition, and a child that spawned its own would recurse with nothing to
        stop it. **Only a structure that succeeded spawns**: an unconverged frame is about to
        be re-run, and building a round off a geometry the retry will replace spends a whole
        batch on a superseded structure — the retry's own frames spawn when they land.
        """
        if self._children is None or pending.scope is not self._root:
            return ()
        return self.spawn(self._children, structures)

    def spawn(self, children: ChildRound, parents: Iterable[Structure]) -> tuple[JobTriple, ...]:
        """Register a round for each of ``parents`` that earns one; return its jobs.

        Each round gets a scope of its own — its own ledger, its own directory, its own
        one-retry budget — so its children are judged as that parent's rather than the
        queue's. Inputs are written here, between one completion and the next admission, the
        same shape of synchronous I/O the convergence retry already does in
        :func:`_retry_input`.

        Takes the round rather than reading ``self._children`` so there is one place that
        decides whether there is a round at all: :meth:`_follow` for a queue that has one,
        :func:`run_child_rounds` for a caller that brought its own.
        """
        out: list[JobTriple] = []
        for parent in parents:
            if not succeeded(parent) or parent.id in self._rounds:
                continue
            run = children.children_for(parent)
            if run is None:
                continue
            child_ctx = replace(
                self._root.ctx,
                step_dir=run.directory,
                prev_state=PipelineState(run.structures),
            )
            prepared = self._engine.prepare(child_ctx)
            scope = _JobScope(child_ctx, _ResultLedger(prepared))
            self._rounds[parent.id] = scope
            for triple in prepared.files:
                self._pending[triple[0]] = _Pending(scope, triple[2], None)
            out.extend(prepared.files)
        return tuple(out)

    def settle(self) -> None:
        """Hand every round's drained verdict back to the :class:`ChildRound` that asked.

        Once, after the queue is empty and before any caller reads a result — a round settled
        early would report a parent resolved from children still in flight.
        """
        if self._children is None:
            return
        for sid, scope in self._rounds.items():
            self._children.settled(sid, *scope.ledger.emit())


def _retry_input(
    engine: CalculationEngine, ctx: StepContext, best: Structure
) -> tuple[JobTriple, Structure]:
    """Archive ``best``'s failed attempt and prepare its re-run as a single job.

    Prepares but does not submit: the caller owns the queue, which is the whole point — the
    re-run goes in behind whatever is already waiting rather than jumping the batch.
    """
    attempts.archive_previous(ctx.step_dir, (best.id,))
    files = engine.prepare(replace(ctx, prev_state=PipelineState((best,)))).files
    if len(files) != 1:
        # Every streaming engine is a `JobEngine`, whose `prepare` emits one job per
        # structure. Named rather than indexed blindly, so an engine that ever stops doing
        # that says so instead of raising IndexError three frames down.
        raise ChemRefineError(f"retry of {best.id}: expected one prepared job, got {len(files)}")
    return files[0], best


def run_with_retries(
    engine: CalculationEngine,
    ctx: StepContext,
    inputs: StepInputs,
    *,
    children: ChildRound | None = None,
) -> tuple[list[Structure], list[Failure]]:
    """Run ``inputs``, re-running each unconverged structure once from its best geometry.

    One sink and one ledger, driven by whichever scheduler the engine supports — so the two
    agree on results, lineage and the on-disk layout by construction rather than by two
    implementations being kept in step.

    A streaming engine parses each job the moment it finishes, so a re-run is submitted into
    the slot its own failed job just freed. Everything else submits the batch, sweeps it, and
    repeats for whatever the sweep asks for — the same work, without the overlap.

    ``children`` lets a finished structure earn a round of its own — NMS round-2
    displacements — which joins the same queue rather than waiting for this batch to drain.
    Those results are **not** in the returned pair: they belong to the parent that earned
    them and go back through :meth:`ChildRound.settled`.
    """
    root = _JobScope(ctx, _ResultLedger(inputs))
    sink = _QueueSink(engine, root, inputs, children)
    _schedule(engine, ctx, inputs, sink)
    sink.settle()
    return root.ledger.emit()


def run_child_rounds(
    engine: CalculationEngine,
    ctx: StepContext,
    parents: Sequence[Structure],
    children: ChildRound,
) -> None:
    """Run every parent's child round in one queue, retrying each child once.

    For the caller whose parents finished elsewhere — :func:`chemrefine.nms.reattempt_nms`
    reuses a round 1 that is already on disk, so it has no batch of its own for the children
    to stream behind. One queue all the same: every parent's children go out together under
    one budget, rather than one parent's batch at a time.

    Returns nothing; verdicts go back through :meth:`ChildRound.settled` exactly as on the
    streamed path, so the two cannot disagree about what a round produced.
    """
    root = _JobScope(ctx, _ResultLedger(StepInputs(files=())))
    sink = _QueueSink(engine, root, StepInputs(files=()), children)
    if follow := sink.spawn(children, parents):
        _schedule(engine, ctx, StepInputs(files=follow), sink)
    sink.settle()


def _schedule(
    engine: CalculationEngine, ctx: StepContext, inputs: StepInputs, sink: _QueueSink
) -> None:
    """Run ``inputs`` and everything the sink asks for, under the scheduler the engine has.

    A streaming engine hears about each job the moment it finishes; anything else submits the
    batch, sweeps it, and repeats for the follow-ups — the same work, without the overlap.
    """
    if isinstance(engine, StreamingSubmit):
        engine.submit_streaming(inputs, ctx, sink)
    else:
        _drain(engine, ctx, inputs, sink)


def _drain(
    engine: CalculationEngine, ctx: StepContext, inputs: StepInputs, sink: _QueueSink
) -> None:
    """Submit a batch, block, hand every job to ``sink``; repeat for the follow-ups.

    ``submit`` is called before the emptiness test rather than gated on it. ``mlip-train``
    prepares no files at all and does its entire job inside ``submit`` — a loop that skipped
    an empty batch would skip the training, with no jobs and no error to show for it. Nothing
    else changes for it: the sweep is empty, and the step yields no structures exactly as it
    does today.

    Follow-up batches are submitted with the step context. Submission does not read
    ``prev_state`` — the context that matters for a re-run is the one it is *parsed* with,
    which the sink holds per job.
    """
    batch = inputs
    while True:
        engine.submit(batch, ctx)
        follow = sweep(batch, sink)
        if not follow:
            return
        batch = StepInputs(follow)


def apply_failure_policy(
    successes: list[Structure],
    failures: list[Failure],
    ctx: StepContext,
    step_cfg: StepConfig,
) -> StepResults:
    """Record failures to the ledger and resolve them per ``step_cfg.on_failure``.

    The ``failed_jobs.json`` ledger is **always** written for any failures (so
    they're visible regardless of policy) and cleared on a clean step. ``skip``
    drops the failures and keeps the successes; ``best`` keeps every
    structure, backfilling a failure with the best geometry obtained for it
    (else its submitted input); ``stop`` (the default) keeps the successes too but the run is
    halted by :func:`chemrefine.step.halt_if_pending` (from the pipeline)
    *after* the cache is written (so ``resume`` / ``rerun-errors`` re-attempt
    only those failed jobs).
    """
    if not failures:
        cache.clear_failed_jobs(ctx.step_dir)
        return StepResults(structures=tuple(successes))

    cache.save_failure_records(ctx.step_dir, [FailureRecord.of(f) for f in failures])
    for f in failures:
        logger.debug("step %d: structure %s failed — %s", step_cfg.step, f.sid, f.reason)
    logger.warning(
        "step %d: %d/%d structure(s) failed [on_failure=%s]",
        step_cfg.step,
        len(failures),
        len(successes) + len(failures),
        step_cfg.on_failure,
    )

    # Matched on the closed literal rather than tested with `==`, and with no catch-all —
    # the same shape, for the same reason, as `chemrefine.filtering._dispatch` over its
    # discriminated union. A chain of `==` needs a fall-through, and a fall-through cannot be
    # told apart from a policy nobody wrote a branch for: a fourth value would silently behave
    # like `skip`, dropping the structures it was meant to keep, with no error anywhere.
    # Unmatched here, the implicit `None` contradicts the return type and mypy says so.
    match step_cfg.on_failure:
        case "best":
            prev_by_id = ctx.prev_state.by_id
            # Build a new list rather than appending into the caller's: every other value
            # crossing this module is frozen, and a policy function quietly rewriting its
            # argument is the one aliasing bug this file would not survive.
            backfilled = [
                fallback
                for f in failures
                if (fallback := (f.best if f.best is not None else prev_by_id.get(f.sid)))
                is not None
            ]
            return StepResults(structures=(*successes, *backfilled))
        case "skip" | "stop":
            # `stop` keeps the successes too; the run is halted afterwards by
            # `chemrefine.step.halt_if_pending`, once this cache is written.
            return StepResults(structures=tuple(successes))


def finalize(
    engine: CalculationEngine,
    ctx: StepContext,
    step_cfg: StepConfig,
    key: cache.StepKey,
    successes: list[Structure],
    failures: list[Failure],
) -> StepResults:
    """Resolve a step's failures and persist the result — the one way a step ends.

    Every path that finishes a step does the same two things in the same order: apply
    ``on_failure`` (:func:`apply_failure_policy`), then write the cache
    (:func:`chemrefine.cache.save`). Four callers need it — the full run, ``rebuild-cache``,
    the failed-job resubmit, and the NMS re-attempt — and how they *reach* this point differs
    (some retry unconverged structures first, some run NMS, some filter afterwards and some
    return the raw results), which is why only the tail is shared and only the tail is
    extracted.

    One function rather than a convention, because a site that applies the policy and skips
    the write — or writes under a key of its own derivation — produces no crash, just a
    silent re-run or a silent reuse much later. The key is a value
    (:class:`chemrefine.cache.StepKey`), so this takes the one its caller already built.
    """
    results = apply_failure_policy(successes, failures, ctx, step_cfg)
    cache.save(
        step_cfg=step_cfg,
        key=key,
        results=results,
        step_dir=ctx.step_dir,
        chemrefine_version=__version__,
    )
    return results

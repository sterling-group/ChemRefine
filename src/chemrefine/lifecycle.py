"""The body of a step: run a set of structures, classify, apply the policy, persist.

Given an engine, a :class:`~chemrefine.state.StepContext` and some structures, this produces
:class:`~chemrefine.state.StepResults` honouring the step's ``on_failure`` setting. It is the
part :mod:`chemrefine.step` and :mod:`chemrefine.nms` both need — ``step`` wraps it in caching
and filtering, ``nms`` runs it for each round of displaced children — so it sits below both.

The phases:

* :func:`submit_and_parse` — prepare, submit, parse a set of structures in one directory.
* :func:`parse_with_failures` — parse outputs that already exist, classifying each
  (:func:`succeeded`, :func:`failure_kind`).
* :func:`rerun_from_best` / :func:`retry_unconverged` — give a convergence failure one more
  try from the best geometry it reached.
* :func:`apply_failure_policy` — ``stop | skip | best``, and the ledger that records it.
* :func:`finalize` — the two of those a step always ends with, in the right order.

It does **not** own the order of a whole step. :mod:`chemrefine.step` interleaves phases with
work of its own — the manifest is written between ``prepare`` and ``submit``, so an interrupted
run leaves proof of what its outputs were computed for — and that composition belongs there.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from chemrefine import __version__, attempts, cache
from chemrefine.config import StepConfig
from chemrefine.engines.api import CalculationEngine
from chemrefine.errors import OutputParseError, OutputTerminationError
from chemrefine.state import (
    Failure,
    FailureKind,
    FailureRecord,
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
    engine: CalculationEngine, triple: tuple[Path, Path, str], ctx: StepContext
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


def _parse_each(
    engine: CalculationEngine, inputs: StepInputs, ctx: StepContext
) -> list[tuple[Path, list[Structure], Failure | None]]:
    """Every job's ``(directory, structures, failure)``, parsed independently.

    Per job rather than per batch so one bad output never crashes the step — its failure is
    captured and the rest still parse.
    """
    return [(triple[1].parent, *_parse_job(engine, triple, ctx)) for triple in inputs.files]


def _split(
    jobs: list[tuple[Path, list[Structure], Failure | None]],
) -> tuple[list[Structure], list[Failure]]:
    """Flatten parsed jobs into the successes and failures a step reasons about."""
    successes = [s for _dir, parsed, _f in jobs for s in parsed if succeeded(s)]
    failures = [f for _dir, _parsed, f in jobs if f is not None]
    return successes, failures


def parse_with_failures(
    engine: CalculationEngine, inputs: StepInputs, ctx: StepContext
) -> tuple[list[Structure], list[Failure]]:
    """Parse a batch's outputs and classify them. Writes nothing.

    Use this to read a tree you must not modify — :func:`chemrefine.nms.rebuild_nms` re-reads
    round-2 children only to re-derive which one won. Everything that *owns* the results it
    parses wants :func:`parse_and_record`.
    """
    return _split(_parse_each(engine, inputs, ctx))


def parse_and_record(
    engine: CalculationEngine, inputs: StepInputs, ctx: StepContext
) -> tuple[list[Structure], list[Failure]]:
    """:func:`parse_with_failures`, and drop each job's canonical result record beside it.

    The record is the engine-independent JSON every calculation leaves next to its native
    output — written for failures too, since an unconverged result is still a parsed one.
    """
    jobs = _parse_each(engine, inputs, ctx)
    for job_dir, parsed, _failure in jobs:
        if parsed:
            cache.save_result_records(parsed, job_dir, ctx.step_cfg.step)
    return _split(jobs)


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


def rerun_from_best(
    engine: CalculationEngine, ctx_for_prepare: StepContext, best: Structure
) -> tuple[list[Structure], list[Failure]]:
    """Archive a structure's failed attempt and re-run it once from ``best``.

    ``best`` is the last good geometry obtained; its id locates the per-structure dir
    under ``ctx_for_prepare.step_dir``. That dir's files are archived into a numbered
    ``attemptK/``, a fresh input is prepared from ``best`` at the canonical place,
    resubmitted, and re-parsed. ``ctx_for_prepare.step_dir`` is what nests the
    structure, so the SAME helper serves a top-level structure (round-1, ``step_dir``
    = the step dir) and an NMS round-2 child (``step_dir`` = the parent's dir).

    **``best`` is resubmitted whole, not rebuilt from its parts.** Reconstructing it as
    ``Structure(id=..., atoms=...)`` would drop every field not named — ``parent_id`` above
    all, which :func:`chemrefine.engines._job.build_structures` reads back off
    ``prev_state``, so a structure that merely needed a second attempt would leave the step
    an orphan, permanently, into the cache. That is not cosmetic downstream:
    :func:`chemrefine.filtering._filter_by_parent` groups on ``parent_id or id``, so an
    orphan is indistinguishable from a seed, forms its own singleton group, and survives a
    filter that should have discarded it.

    The structure is frozen and nothing here mutates it, so there is nothing to copy — its
    stale result fields are overwritten by the re-parse. Passing it whole is also what
    carries a field added to :class:`~chemrefine.state.Structure` later through the retry
    without anyone remembering to.
    """
    attempts.archive(ctx_for_prepare.step_dir / best.id)
    return submit_and_parse(engine, ctx_for_prepare, [best])


def retry_unconverged(
    engine: CalculationEngine,
    ctx_for_prepare: StepContext,
    successes: list[Structure],
    failures: list[Failure],
) -> tuple[list[Structure], list[Failure]]:
    """Retry each *unconverged* failure once, from its best geometry.

    Convergence-only — a crashed / missing-output failure (or one with no best
    geometry) is left untouched for the ``on_failure`` policy. A single inline pass
    (no recursion): each structure is retried at most once per run; a later ``resume``
    archives into the next ``attemptK/``. Serves both round-1 and NMS round-2 children
    (the caller passes the matching ``ctx_for_prepare``).
    """
    kept = list(successes)
    remaining: list[Failure] = []
    for f in failures:
        if f.kind is not FailureKind.NOT_CONVERGED or f.best is None:
            remaining.append(f)
            continue
        logger.info("structure %s did not converge — retrying from its best geometry", f.sid)
        succ, fail = rerun_from_best(engine, ctx_for_prepare, f.best)
        kept.extend(succ)
        remaining.extend(fail)
    return kept, remaining


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

    if step_cfg.on_failure == "best":
        prev_by_id = {s.id: s for s in ctx.prev_state.structures}
        # Build a new list rather than appending into the caller's: every other value
        # crossing this module is frozen, and a policy function quietly rewriting its
        # argument is the one aliasing bug this file would not survive.
        backfilled = [
            fallback
            for f in failures
            if (fallback := (f.best if f.best is not None else prev_by_id.get(f.sid))) is not None
        ]
        return StepResults(structures=(*successes, *backfilled))
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
    (:func:`chemrefine.cache.save`). Four call sites spelled that out
    identically — the full run, ``rebuild-cache``, the failed-job resubmit, and the NMS
    re-attempt — and how they *reach* this point differs (some retry unconverged structures
    first, some run NMS, some filter afterwards and some return the raw results), which is
    why only the tail is shared and only the tail is extracted.

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

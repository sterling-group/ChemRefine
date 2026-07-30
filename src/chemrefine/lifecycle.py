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

from chemrefine import __version__, attempts, cache
from chemrefine.config import StepConfig
from chemrefine.engines.api import CalculationEngine
from chemrefine.errors import OutputParseError
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
    otherwise :attr:`FailureKind.FAILED` (a flag the engine set we don't name).
    """
    if s.terminated_normally is False:
        return FailureKind.NOT_TERMINATED
    if s.converged is False:
        return FailureKind.NOT_CONVERGED
    return FailureKind.FAILED


def parse_with_failures(
    engine: CalculationEngine, inputs: StepInputs, ctx: StepContext
) -> tuple[list[Structure], list[Failure]]:
    """Parse each output independently; classify into successes and failures.

    A job is a *failure* when its output is missing, unparseable, or parses to a
    structure the engine marks unconverged / not-terminated. Parsing per input
    (rather than the whole batch at once) means one bad job never crashes the
    step — its failure is captured and the rest still parse. Engine success
    flags are set in the single parse pass (see ``orca.output``).
    """
    successes: list[Structure] = []
    failures: list[Failure] = []
    for triple in inputs.files:
        _inp, out, sid = triple
        if not out.is_file():
            failures.append(Failure(sid, FailureKind.MISSING_OUTPUT, None))
            continue
        try:
            parsed = list(engine.parse(StepInputs(files=(triple,)), ctx).structures)
        except OutputParseError as e:
            failures.append(Failure(sid, FailureKind.UNPARSEABLE, None, detail=str(e)))
            continue
        # Drop the canonical parsed-result record next to the native output —
        # the engine-independent JSON every calculation leaves behind, for
        # failures too (an unconverged result is still a parsed result).
        cache.save_result_records(parsed, out.parent, ctx.step_cfg.step)
        successes.extend(s for s in parsed if succeeded(s))
        bad = [s for s in parsed if not succeeded(s)]
        if bad:
            best = min(
                bad,
                key=lambda s: (s.energy_hartree is None, s.energy_hartree or 0.0),
            )
            # The reason must describe the geometry we carry forward, not some other
            # frame of the same job. For a fan-out — a GOAT ensemble, a PES scan —
            # frame 0 crashing while frame 3 merely failed to converge would otherwise
            # ledger "did not terminate normally" against frame 3's geometry, and
            # `retry_unconverged` keys on that exact string, so the mismatch routes the
            # structure to the wrong recovery.
            failures.append(Failure(sid, failure_kind(best), best))
    return successes, failures


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
    return parse_with_failures(engine, inputs, run_ctx)


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
    """
    attempts.archive(ctx_for_prepare.step_dir / best.id)
    return submit_and_parse(engine, ctx_for_prepare, [Structure(id=best.id, atoms=best.atoms)])


def retry_unconverged(
    engine: CalculationEngine,
    ctx_for_prepare: StepContext,
    successes: list[Structure],
    failures: list[Failure],
) -> tuple[list[Structure], list[Failure]]:
    """Retry each *unconverged* failure once, from its best geometry (B10).

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
    parent_ids: tuple[str, ...],
    successes: list[Structure],
    failures: list[Failure],
) -> StepResults:
    """Resolve a step's failures and persist the result — the one way a step ends.

    Every path that finishes a step does the same two things in the same order: apply
    ``on_failure`` (:func:`apply_failure_policy`), then write the cache
    (:func:`chemrefine.cache.save_step_results`). Four call sites spelled that out
    identically — the full run, ``rebuild-cache``, the failed-job resubmit, and the NMS
    re-attempt — and how they *reach* this point differs (some retry unconverged structures
    first, some run NMS, some filter afterwards and some return the raw results), which is
    why only the tail is shared and only the tail is extracted.

    Worth having as one function because the shape has already cost something: this pair is
    what :func:`chemrefine.cache.save_step_results` was itself extracted for, after a site
    drifted and wrote a cache without its reuse fingerprint. A cache written with an
    inconsistent key is not a crash — it is a silent re-run, or a silent reuse, much later.

    """
    results = apply_failure_policy(successes, failures, ctx, step_cfg)
    cache.save_step_results(
        step_cfg=step_cfg,
        parent_ids=parent_ids,
        results=results,
        ctx=ctx,
        template_digest=engine.input_digest(ctx),
        chemrefine_version=__version__,
    )
    return results

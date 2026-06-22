"""Failure classification and the per-step ``on_failure`` policy.

A *failure* is a structure whose job produced no output, an unparseable
output, or an output the engine flagged unconverged / not-terminated.
This module owns that vocabulary — the :class:`Failure` record, the
success test (:func:`succeeded`), per-output classification
(:func:`parse_with_failures`), the ``stop | skip | best`` policy
(:func:`apply_failure_policy`), and the shared "attempt"/retry primitive
(:func:`archive_failed_attempt`, :func:`retry_from_best`,
:func:`retry_unconverged`) — used by both the generic step lifecycle
(:mod:`chemrefine.step`) and the two-round NMS coordinator
(:mod:`chemrefine.nms`). The ``failed_jobs.json`` ledger itself is
persisted via :mod:`chemrefine.cache`.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

from chemrefine import cache, ids
from chemrefine.config import StepConfig
from chemrefine.engines.base import CalculationEngine
from chemrefine.errors import OutputParseError
from chemrefine.state import PipelineState, StepContext, StepInputs, StepResults, Structure

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Failure:
    """One failed structure: its id, why, and the best geometry obtained (if any)."""

    sid: str
    reason: str
    best: Structure | None


@dataclass(frozen=True)
class NmsResolution:
    """Outcome of NMS resolution: the resolved survivors plus unresolved failures.

    The generic NMS coordinator (:mod:`chemrefine.nms`) returns this; the step
    lifecycle then applies the ``on_failure`` policy to ``failures`` exactly as for a
    plain step, so NMS reuses the same failure handling.
    """

    survivors: tuple[Structure, ...]
    failures: tuple[Failure, ...]


def succeeded(s: Structure) -> bool:
    """A parsed structure failed only when an engine success flag is explicitly False.

    ``None`` (engine doesn't report it) is treated as 'not a failure signal', so
    backends that don't set termination/convergence flags are never gated.
    """
    return s.terminated is not False and s.converged is not False


def failure_reason(s: Structure) -> str:
    """Human-readable reason a parsed structure counts as a failure.

    ``terminated is False`` → the engine crashed / didn't finish cleanly;
    ``converged is False`` → it finished but the SCF/geometry didn't converge;
    otherwise a generic ``"failed"`` (a flag the engine set we don't name).
    """
    if s.terminated is False:
        return "did not terminate normally"
    if s.converged is False:
        return "did not converge"
    return "failed"


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
            failures.append(Failure(sid, "output missing", None))
            continue
        try:
            parsed = list(engine.parse(StepInputs(files=(triple,)), ctx).structures)
        except OutputParseError as e:
            failures.append(Failure(sid, f"unparseable: {e}", None))
            continue
        successes.extend(s for s in parsed if succeeded(s))
        bad = [s for s in parsed if not succeeded(s)]
        if bad:
            best = min(
                bad,
                key=lambda s: (s.energy_hartree is None, s.energy_hartree or 0.0),
            )
            failures.append(Failure(sid, failure_reason(bad[0]), best))
    return successes, failures


# ---------------------------------------------------------------------------
# Attempt / retry — the shared "resolve a structure in subdirs" primitive
# ---------------------------------------------------------------------------


def archive_failed_attempt(structure_dir: Path) -> Path:
    """Move a structure dir's loose files into the next free ``attemptK/``; return it.

    The shared "attempt" primitive (path via :func:`chemrefine.ids.next_attempt_dir`):
    only loose **files** move — existing ``attempt*/`` (and any nested) sub-directories
    stay put — so a re-run never clobbers an earlier attempt.
    """
    dest = ids.next_attempt_dir(structure_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for item in structure_dir.iterdir():
        if item.is_dir():
            continue  # leave attempt*/ (and any nested) sub-directories in place
        shutil.move(str(item), str(dest / item.name))
    return dest


def retry_from_best(
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
    archive_failed_attempt(ctx_for_prepare.step_dir / best.id)
    seed = Structure(id=best.id, atoms=best.atoms)
    retry_ctx = replace(ctx_for_prepare, prev_state=PipelineState(structures=(seed,)))
    inputs = engine.prepare(retry_ctx)
    engine.submit(inputs, retry_ctx)
    return parse_with_failures(engine, inputs, retry_ctx)


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
        if f.reason != "did not converge" or f.best is None:
            remaining.append(f)
            continue
        logger.info("structure %s did not converge — retrying from its best geometry", f.sid)
        succ, fail = retry_from_best(engine, ctx_for_prepare, f.best)
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
    (default) drops the failures and keeps the successes; ``best`` keeps every
    structure, backfilling a failure with the best geometry obtained for it
    (else its submitted input); ``stop`` keeps the successes too but the run is
    halted by :func:`chemrefine.step.halt_if_pending` (from the pipeline)
    *after* the cache is written (so ``resume`` / ``rerun-errors`` re-attempt
    only those failed jobs).
    """
    if not failures:
        cache.clear_failed_jobs(ctx.step_dir)
        return StepResults(structures=tuple(successes))

    cache.save_failed_jobs(
        ctx.step_dir,
        [{"structure_id": f.sid, "reason": f.reason} for f in failures],
    )
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
        for f in failures:
            fallback = f.best if f.best is not None else prev_by_id.get(f.sid)
            if fallback is not None:
                successes.append(fallback)
    return StepResults(structures=tuple(successes))

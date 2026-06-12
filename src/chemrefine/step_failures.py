"""Failure classification and the per-step ``on_failure`` policy.

A *failure* is a structure whose job produced no output, an unparseable
output, or an output the engine flagged unconverged / not-terminated.
This module owns that vocabulary — the :class:`Failure` record, the
success test (:func:`succeeded`), per-output classification
(:func:`parse_with_failures`), and the ``stop | skip | best`` policy
resolution (:func:`apply_failure_policy`) — shared by the generic step
lifecycle (:mod:`chemrefine.step`) and the two-round NMS resolution
(:mod:`chemrefine.step_nms`). The ``failed_jobs.json`` ledger itself is
persisted via :mod:`chemrefine.cache`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from chemrefine import cache
from chemrefine.config import StepConfig
from chemrefine.engines.base import CalculationEngine
from chemrefine.errors import OutputParseError
from chemrefine.state import StepContext, StepInputs, StepResults, Structure

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Failure:
    """One failed structure: its id, why, and the best geometry obtained (if any)."""

    sid: str
    reason: str
    best: Structure | None


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

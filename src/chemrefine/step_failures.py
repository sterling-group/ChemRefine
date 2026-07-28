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
from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

from chemrefine import cache, ids
from chemrefine.config import StepConfig
from chemrefine.engines.api import CalculationEngine
from chemrefine.errors import OutputParseError
from chemrefine.state import PipelineState, StepContext, StepInputs, StepResults, Structure

logger = logging.getLogger(__name__)


class FailureKind(StrEnum):
    """Why a structure failed — the classification recovery branches on.

    A closed vocabulary rather than free text. ``retry_unconverged``,
    ``_resubmit_failed`` and ``reattempt_nms`` all route on *which* kind of failure
    this is, and they used to do that by comparing against the exact wording of a
    human-readable message. Rewording one — the sort of thing that looks like a docs
    change — silently disabled a recovery path.

    The values are the wording, so the ledger on disk stays readable and messages
    stay unchanged; what moved is that the comparison is now against a name.
    """

    MISSING_OUTPUT = "output missing"
    """The job produced no output file at all — crashed, killed, never started."""

    UNPARSEABLE = "unparseable"
    """An output exists but the engine could not read it (truncated, corrupt)."""

    NOT_TERMINATED = "did not terminate normally"
    """The program ran but did not exit cleanly."""

    NOT_CONVERGED = "did not converge"
    """It finished, but the SCF or the geometry did not converge. The only kind
    that is retried from its best geometry — resubmitting the identical input for
    any of the others would just fail the same way."""

    UNRESOLVED_NMS = "NMS: target stationary point not reached"
    """Normal-mode sampling could not reach the requested stationary point."""

    FAILED = "failed"
    """The engine set a failure flag we don't have a more specific name for."""


@dataclass(frozen=True)
class Failure:
    """One failed structure: its id, why, and the best geometry obtained (if any)."""

    sid: str
    kind: FailureKind
    best: Structure | None
    detail: str = ""
    """Extra context for kinds that have some (the parser message for
    ``UNPARSEABLE``); empty otherwise."""

    @property
    def reason(self) -> str:
        """The human-readable reason, as written to the ledger and the logs."""
        return f"{self.kind.value}: {self.detail}" if self.detail else self.kind.value


@dataclass(frozen=True)
class FailureRecord:
    """A ledger entry — one failed structure, as persisted to ``failed_jobs.json``.

    The recovery paths read this back to decide what to re-attempt, so it is a typed
    record rather than a bare dict indexed with string literals at four call sites.
    """

    structure_id: str
    kind: FailureKind
    reason: str

    @classmethod
    def of(cls, failure: Failure) -> FailureRecord:
        """The ledger entry for an in-flight :class:`Failure`."""
        return cls(structure_id=failure.sid, kind=failure.kind, reason=failure.reason)

    def to_json(self) -> dict[str, str]:
        """Serialize for ``failed_jobs.json``."""
        return {"structure_id": self.structure_id, "kind": self.kind.value, "reason": self.reason}

    @classmethod
    def from_json(cls, data: dict[str, str]) -> FailureRecord:
        """Rebuild from a ``failed_jobs.json`` entry."""
        return cls(
            structure_id=data["structure_id"],
            kind=FailureKind(data["kind"]),
            reason=data.get("reason", ""),
        )


def load_failure_records(step_dir: Path) -> list[FailureRecord]:
    """Read a step's failure ledger back into typed records.

    The JSON↔domain half of the ledger; :func:`chemrefine.cache.load_failed_jobs` is
    the bytes↔JSON half. Split that way because this module already depends on
    ``cache``, so the typing has to live on this side of the boundary.
    """
    return [FailureRecord.from_json(rec) for rec in cache.load_failed_jobs(step_dir)]


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


def archive_previous_attempts(step_dir: Path, structure_ids: Iterable[str]) -> list[Path]:
    """Archive any prior artifacts of ``structure_ids`` before they are re-executed (B1).

    The invariant this protects: **a parsed output must have been produced by this run's
    submission of that job.** :func:`parse_with_failures` decides success by
    ``out.is_file()``, which cannot tell this run's output from a leftover — so without
    this, a re-executed step whose job dies before writing anything silently re-reads the
    *previous* run's result and reports it as current. The exposed paths are the ones that
    do not truncate their output in place: the script engines (whose JSON is written in
    ``$WORK_DIR`` and only copied back on success) and every ORCA ensemble operation
    (which reads a ``*.finalensemble.xyz``-style sidecar, not the ``.out``).

    Only structures with loose files are touched, so a first run is a no-op. Reuses
    :func:`archive_failed_attempt`, so an ``attemptK/`` from an earlier run is preserved
    rather than clobbered and the whole history stays recoverable.
    """
    archived: list[Path] = []
    for sid in structure_ids:
        struct_dir = step_dir / sid
        if struct_dir.is_dir() and any(p.is_file() for p in struct_dir.iterdir()):
            archived.append(archive_failed_attempt(struct_dir))
    return archived


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
        if f.kind is not FailureKind.NOT_CONVERGED or f.best is None:
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

    cache.save_failed_jobs(
        ctx.step_dir,
        [FailureRecord.of(f).to_json() for f in failures],
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

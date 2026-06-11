"""Step-level NMS orchestration: resolution, re-attempt, and reuse fingerprint.

The generic per-step lifecycle lives in :mod:`chemrefine.step`; this module
holds the parts that are specific to two-round normal-mode sampling — grouping
round-2 outputs back to their round-1 parent, re-attempting only the unresolved
parents, and the search-param-stable reuse fingerprint. It is distinct from
:mod:`chemrefine.engines.orca.nms` (the ORCA displacement math + ``NmsOptions``).

It reuses the generic failure helpers from :mod:`chemrefine.step` (imported at
module scope); :mod:`chemrefine.step` imports *this* module at function scope so
there is no import cycle.
"""

from __future__ import annotations

import logging

from chemrefine import cache
from chemrefine.config import StepConfig
from chemrefine.engines.base import CalculationEngine
from chemrefine.errors import CacheError
from chemrefine.state import StepContext, StepInputs, StepResults, Structure
from chemrefine.step import (
    _apply_failure_policy,
    _Failure,
    _parse_with_failures,
    _succeeded,
)

logger = logging.getLogger(__name__)


# NMS *search* params: tuning these doesn't change what counts as resolved, so
# changing them reuses round-1 (the reuse fingerprint ignores them). The
# resolution *criterion* (target / ts_mode_index) stays in, so changing it
# forces a full re-run.
_NMS_SEARCH_KEYS = frozenset({"displacement_value", "num_random_displacements", "seed"})


def _nms_reuse_fingerprint(
    step_cfg: StepConfig, parent_ids: tuple[str, ...], *, parents_digest: str = ""
) -> str:
    """Fingerprint that's stable across NMS search-param tuning.

    Same as :func:`chemrefine.cache.fingerprint` (including the
    ``parents_digest`` content key) but with the NMS search parameters
    stripped from ``options`` — so bumping ``displacement_value`` leaves it
    unchanged (reuse round-1 + resolved, re-attempt only the unresolved),
    while changing the criterion / template / parents changes it.
    Returns ``""`` for non-NMS steps (the reuse path is NMS-only).
    """
    if not step_cfg.nms:
        return ""
    trimmed = {k: v for k, v in (step_cfg.options or {}).items() if k not in _NMS_SEARCH_KEYS}
    return cache.fingerprint(
        step_cfg.model_copy(update={"options": trimmed}),
        parent_ids,
        parents_digest=parents_digest,
    )


def _reattempt_nms(
    engine: CalculationEngine,
    ctx: StepContext,
    step_cfg: StepConfig,
    cached: cache.StepCache,
    parent_ids: tuple[str, ...],
    version: str,
) -> StepResults:
    """Re-attempt only the ledgered-unresolved NMS parents, reusing round-1.

    Round-1 freq is reused from disk (re-parsed, not resubmitted) for parents
    whose output exists; a parent whose round-1 output is genuinely *missing*
    has it resubmitted first. NMS round-2 is re-run for those parents under the
    current ``NmsOptions``; the still-valid resolved structures from the old
    cache are kept, and the merged result is re-cached.
    """
    manifest = cache.load_manifest(ctx.step_dir)
    if manifest is None:
        raise CacheError(f"step {step_cfg.step}: cannot re-attempt NMS — no manifest on disk")
    failed = cache.load_failed_jobs(ctx.step_dir)
    failed_ids = {f["structure_id"] for f in failed}
    missing_ids = {f["structure_id"] for f in failed if f.get("reason") == "output missing"}

    failed_manifest = StepInputs(files=tuple(f for f in manifest.files if f[2] in failed_ids))
    missing_inputs = StepInputs(
        files=tuple(f for f in failed_manifest.files if f[2] in missing_ids)
    )
    if missing_inputs.files:
        logger.info(
            "step %d: NMS re-attempt resubmitting %d missing round-1 job(s)",
            step_cfg.step,
            len(missing_inputs.files),
        )
        engine.wait(engine.submit(missing_inputs, ctx))

    # Re-parse the failed parents' round-1 outputs (reuse on disk; caches freqs),
    # then re-run NMS round-2 for them under the current options. Round-1 jobs
    # that still produced no output stay failures, carried into the resolution.
    r1_succ, r1_fail = _parse_with_failures(engine, failed_manifest, ctx)
    round1_failed = StepResults(structures=tuple(r1_succ))
    reattempt = _resolve_nms(
        engine.normal_mode_sample(round1_failed, ctx),
        round1_failed,
        ctx,
        step_cfg,
        round1_failures=r1_fail,
    )
    kept = tuple(
        s
        for s in cached.results.structures
        if s.id not in failed_ids and (s.parent_id not in failed_ids)
    )
    merged = StepResults(structures=kept + reattempt.structures)
    digest = cache.parents_digest(ctx.prev_state.structures)
    cache.save(
        step_cfg=step_cfg,
        parent_ids=parent_ids,
        results=merged,
        step_dir=ctx.step_dir,
        chemrefine_version=version,
        reuse_fingerprint=_nms_reuse_fingerprint(step_cfg, parent_ids, parents_digest=digest),
        parents_digest=digest,
    )
    return merged


def _resolve_nms(
    nms_results: StepResults,
    round1: StepResults,
    ctx: StepContext,
    step_cfg: StepConfig,
    round1_failures: list[_Failure] | tuple[_Failure, ...] = (),
) -> StepResults:
    """Group two-round NMS outputs by the round-1 structure each resolves.

    A round-1 structure is *resolved* if it passed through already at the
    target (``converged=True``) or any of its displaced ± children resolved.
    Resolved children are kept (both ±, no dedup); a round-1 structure with no
    resolved outcome is a failure handed to the step's ``on_failure`` policy.
    ``round1_failures`` (jobs that produced no/unparseable output) are carried
    in unchanged, so the ledger records *both* failure kinds in one write
    (recorded to ``failed_jobs.json`` so ``resume`` can re-attempt them).
    """
    round1_ids = {s.id for s in round1.structures}
    successes: list[Structure] = []
    resolved_parents: set[str] = set()
    attempts: dict[str, list[Structure]] = {}
    for o in nms_results.structures:
        # An already-resolved pass-through carries its own round-1 id; a
        # displaced child carries its round-1 parent in ``parent_id``.
        key = o.id if o.id in round1_ids else (o.parent_id or o.id)
        attempts.setdefault(key, []).append(o)
        if _succeeded(o):
            successes.append(o)
            resolved_parents.add(key)

    failures: list[_Failure] = list(round1_failures)
    for s in round1.structures:
        if s.id in resolved_parents:
            continue
        group = attempts.get(s.id, [])
        best = (
            min(group, key=lambda a: (a.energy_hartree is None, a.energy_hartree or 0.0))
            if group
            else s
        )
        failures.append(_Failure(s.id, "NMS: target stationary point not reached", best))
    return _apply_failure_policy(successes, failures, ctx, step_cfg)

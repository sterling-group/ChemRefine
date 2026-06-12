"""Per-step lifecycle: prepare → submit → wait → parse → (nms) → filter → cache.

This is the only place that knows the order in which an engine's
lifecycle methods are called. :func:`run_step` is intentionally short
(~30 LOC) — every concern it touches lives in its own module:

* Caching + manifest:  :mod:`chemrefine.cache`
* Filtering:           :mod:`chemrefine.filtering`
* Engine lookup:  :mod:`chemrefine.engines.base`
* Failure vocabulary + ``on_failure`` policy:  :mod:`chemrefine.step_failures`
* NMS resolution / recovery:  :mod:`chemrefine.step_nms`
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from chemrefine import __version__, cache, filtering, step_failures, step_nms
from chemrefine.config import Config, StepConfig
from chemrefine.engines.base import CalculationEngine, get_engine
from chemrefine.errors import CacheError, ChemRefineError
from chemrefine.state import PipelineState, StepContext, StepInputs, StepResults

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context construction
# ---------------------------------------------------------------------------


def step_dir_for(config: Config, step_cfg: StepConfig) -> Path:
    """The on-disk directory for a step's artifacts (cache, ledger, outputs)."""
    return (config.output_dir / step_cfg.dir_name()).resolve()


def build_context(config: Config, step_cfg: StepConfig, prev_state: PipelineState) -> StepContext:
    """Bundle the per-step inputs into a :class:`StepContext`."""
    return StepContext(
        step_cfg=step_cfg,
        step_dir=step_dir_for(config, step_cfg),
        template_dir=config.template_dir.resolve(),
        scratch_dir=config.scratch_dir.resolve() if config.scratch_dir is not None else None,
        prev_state=prev_state,
        charge=step_cfg.charge if step_cfg.charge is not None else config.charge,
        multiplicity=(
            step_cfg.multiplicity if step_cfg.multiplicity is not None else config.multiplicity
        ),
        max_cores=config.max_cores,
        slurm_template=config.slurm_template,
        executables=config.executables,
        max_gpus=config.max_gpus,
    )


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepOutcome:
    """What :func:`run_step` returns — the survivors plus whether the cache was hit."""

    state: PipelineState
    cache_hit: bool


def run_step(
    config: Config,
    step_cfg: StepConfig,
    prev_state: PipelineState,
    *,
    engine: CalculationEngine | None = None,
    use_cache: bool = True,
    resubmit_step: int | None = None,
) -> StepOutcome:
    """Execute one step end-to-end and return its surviving state.

    * If ``use_cache`` is true and the on-disk cache fingerprint matches the
      current step config + parent structures (IDs and content), the cached
      :class:`~chemrefine.state.StepResults` are reused and only filtering runs
      again — **unless** the step is ``on_failure: stop`` and has a failed-jobs
      ledger, in which case ``resume`` re-attempts only the still-failed
      structures (see :func:`_resubmit_failed`). ``skip`` / ``best`` steps keep
      their ledger for visibility but are never re-attempted. ``resubmit_step``
      (set by ``rerun-errors``) scopes the re-attempt to one step; ``None``
      re-attempts whichever ``stop`` step is pending. ``rerun`` (redo a whole
      step) is the caller invalidating the cache first.
    * Otherwise the engine's full lifecycle runs and the result is cached. An
      ``on_failure: stop`` step that ends with failures caches its successes and
      ledgers the failures; the pipeline then halts the run once via
      :func:`halt_if_pending` (this function never raises).
    """
    ctx = build_context(config, step_cfg, prev_state)
    parent_ids = tuple(s.id for s in prev_state.structures)
    ctx.step_dir.mkdir(parents=True, exist_ok=True)
    engine = engine if engine is not None else get_engine(step_cfg.engine)
    is_nms = step_cfg.nms and engine.supports_nms

    if use_cache:
        cached = _cached_outcome(
            ctx,
            step_cfg,
            parent_ids,
            engine,
            is_nms=is_nms,
            resubmit_step=resubmit_step,
        )
        if cached is not None:
            return cached
        # Full fingerprint invalid. For an NMS step whose *search* params changed
        # but whose round-1 + criterion are unchanged (reuse fingerprint matches),
        # reuse the round-1 freq + already-resolved children and re-attempt only
        # the ledgered-unresolved parents — instead of re-running the whole step.
        if is_nms:
            reused = _nms_reuse_outcome(ctx, step_cfg, parent_ids, engine)
            if reused is not None:
                return reused

    return _run_full_step(ctx, step_cfg, parent_ids, engine, is_nms=is_nms)


def _cached_outcome(
    ctx: StepContext,
    step_cfg: StepConfig,
    parent_ids: tuple[str, ...],
    engine: CalculationEngine,
    *,
    is_nms: bool,
    resubmit_step: int | None,
) -> StepOutcome | None:
    """Outcome from a valid on-disk cache, or ``None`` if the cache is invalid.

    A pending ``on_failure: stop`` ledger (scoped by ``resubmit_step``) re-attempts
    only the still-failed structures; otherwise it's a plain cache hit (refilter).
    """
    if not cache.is_valid(
        step_cfg=step_cfg,
        parent_ids=parent_ids,
        step_dir=ctx.step_dir,
        parents_digest=cache.parents_digest(ctx.prev_state.structures),
    ):
        return None
    cached = cache.load(ctx.step_dir)
    if cached is None:
        raise CacheError("is_valid returned True but load returned None")
    failed = cache.load_failed_jobs(ctx.step_dir)
    if (
        failed
        and step_cfg.on_failure == "stop"
        and (resubmit_step is None or resubmit_step == step_cfg.step)
    ):
        results = (
            step_nms.reattempt_nms(engine, ctx, step_cfg, cached, parent_ids)
            if is_nms
            else _resubmit_failed(engine, ctx, step_cfg, failed, parent_ids)
        )
        return StepOutcome(state=filtering.apply(results, step_cfg.sample), cache_hit=False)
    logger.info(
        "step %d: cache hit, reusing %d structures",
        step_cfg.step,
        len(cached.results.structures),
    )
    return StepOutcome(state=filtering.apply(cached.results, step_cfg.sample), cache_hit=True)


def _nms_reuse_outcome(
    ctx: StepContext,
    step_cfg: StepConfig,
    parent_ids: tuple[str, ...],
    engine: CalculationEngine,
) -> StepOutcome | None:
    """Reuse a cached NMS round-1 when only the *search* params changed.

    Returns ``None`` (re-run the whole step) unless a cache exists whose reuse
    fingerprint matches; then it re-attempts the ledgered-unresolved parents, or
    re-stamps the cache when everything was already resolved.
    """
    try:
        cached = cache.load(ctx.step_dir)
    except CacheError:
        cached = None
    digest = cache.parents_digest(ctx.prev_state.structures)
    fingerprint = step_nms.nms_reuse_fingerprint(step_cfg, parent_ids, parents_digest=digest)
    if cached is None or getattr(cached, "reuse_fingerprint", "") != fingerprint:
        return None
    if cache.load_failed_jobs(ctx.step_dir):
        results = step_nms.reattempt_nms(engine, ctx, step_cfg, cached, parent_ids)
    else:
        logger.info(
            "step %d: NMS search params changed, all resolved — reusing cache", step_cfg.step
        )
        cache.save(
            step_cfg=step_cfg,
            parent_ids=parent_ids,
            results=cached.results,
            step_dir=ctx.step_dir,
            chemrefine_version=__version__,
            reuse_fingerprint=fingerprint,
            parents_digest=digest,
        )
        results = cached.results
    return StepOutcome(state=filtering.apply(results, step_cfg.sample), cache_hit=False)


def _run_full_step(
    ctx: StepContext,
    step_cfg: StepConfig,
    parent_ids: tuple[str, ...],
    engine: CalculationEngine,
    *,
    is_nms: bool,
) -> StepOutcome:
    """Run the engine's full lifecycle (prepare → submit → parse → nms/policy → cache)."""
    logger.info("step %d (%s): preparing inputs", step_cfg.step, step_cfg.engine)
    inputs = engine.prepare(ctx)
    cache.save_manifest(inputs, ctx.step_dir, operation=step_cfg.operation, engine=step_cfg.engine)

    logger.info("step %d: submitting %d jobs", step_cfg.step, len(inputs.files))
    batch = engine.submit(inputs, ctx)
    engine.wait(batch)

    logger.info("step %d: parsing outputs", step_cfg.step)
    successes, failures = step_failures.parse_with_failures(engine, inputs, ctx)

    if is_nms:
        # Run NMS on the round-1 survivors, then apply the failure policy once
        # over BOTH round-1 job failures and NMS-unresolved parents (a single
        # ledger write — otherwise the NMS resolution would clobber the round-1
        # failures).
        logger.info("step %d: running normal-mode sampling", step_cfg.step)
        round1 = StepResults(structures=tuple(successes))
        results = step_nms.resolve_nms(
            engine.normal_mode_sample(round1, ctx),
            round1,
            ctx,
            step_cfg,
            round1_failures=failures,
        )
    else:
        results = step_failures.apply_failure_policy(successes, failures, ctx, step_cfg)

    digest = cache.parents_digest(ctx.prev_state.structures)
    cache.save(
        step_cfg=step_cfg,
        parent_ids=parent_ids,
        results=results,
        step_dir=ctx.step_dir,
        chemrefine_version=__version__,
        reuse_fingerprint=step_nms.nms_reuse_fingerprint(
            step_cfg, parent_ids, parents_digest=digest
        ),
        parents_digest=digest,
    )
    return StepOutcome(state=filtering.apply(results, step_cfg.sample), cache_hit=False)


def halt_if_pending(config: Config, step_cfg: StepConfig, resubmit_step: int | None) -> None:
    """Halt the run when an ``on_failure: stop`` step still has failed jobs.

    Called **once** from the pipeline after a step executes (never after a
    ``rebuild_cache_step``), so the step's successes are already cached. Only
    ``stop`` turns its ledgered failures into a hard stop; ``skip`` / ``best``
    keep their ledger for visibility but never halt. The ``resubmit_step`` gate
    lets ``rerun-errors N`` cache-hit past a *different* step's pending failures
    (it re-attempts only step N), matching the per-step scoping of ``run_step``.
    """
    if step_cfg.on_failure != "stop":
        return
    if resubmit_step is not None and resubmit_step != step_cfg.step:
        return
    if cache.load_failed_jobs(step_dir_for(config, step_cfg)):
        raise ChemRefineError(
            f"step {step_cfg.step} halted (on_failure=stop); fix the failed "
            f"job(s) and run `chemrefine resume` (or `rerun-errors {step_cfg.step}`)"
        )


def rebuild_cache_step(
    config: Config, step_cfg: StepConfig, prev_state: PipelineState
) -> StepOutcome:
    """Rebuild one step's cache from outputs already on disk — **no submission**.

    Re-parses the step's existing outputs (via its manifest), re-applies the
    ``on_failure`` policy and — for NMS steps — re-resolves from the existing
    ``nms/`` round-2 outputs, then rewrites the ``StepCache`` with the same
    fingerprint a normal run would produce. Backs ``chemrefine rebuild-cache``.
    """
    ctx = build_context(config, step_cfg, prev_state)
    parent_ids = tuple(s.id for s in prev_state.structures)
    engine = get_engine(step_cfg.engine)
    manifest = cache.load_manifest(ctx.step_dir)
    if manifest is None:
        raise CacheError(f"step {step_cfg.step}: cannot rebuild-cache — no manifest on disk")
    logger.info("step %d: rebuilding cache from existing outputs", step_cfg.step)
    successes, failures = step_failures.parse_with_failures(engine, manifest, ctx)
    if step_cfg.nms and engine.supports_nms:
        round1 = StepResults(structures=tuple(successes))
        results = step_nms.resolve_nms(
            # Not on the CalculationEngine Protocol: only NMS-capable engines
            # (supports_nms=True, the gate above) provide this rebuild hook.
            engine.resolve_nms_from_existing(round1, ctx),  # type: ignore[attr-defined]
            round1,
            ctx,
            step_cfg,
            round1_failures=failures,
        )
    else:
        results = step_failures.apply_failure_policy(successes, failures, ctx, step_cfg)
    digest = cache.parents_digest(ctx.prev_state.structures)
    cache.save(
        step_cfg=step_cfg,
        parent_ids=parent_ids,
        results=results,
        step_dir=ctx.step_dir,
        chemrefine_version=__version__,
        reuse_fingerprint=step_nms.nms_reuse_fingerprint(
            step_cfg, parent_ids, parents_digest=digest
        ),
        parents_digest=digest,
    )
    return StepOutcome(state=filtering.apply(results, step_cfg.sample), cache_hit=False)


def _resubmit_failed(
    engine: CalculationEngine,
    ctx: StepContext,
    step_cfg: StepConfig,
    failed: list[dict],
    parent_ids: tuple[str, ...],
) -> StepResults:
    """Resubmit only the failed structures, then re-parse + re-cache the full step.

    Rehydrates the per-structure inputs from the manifest, resubmits the failed
    subset (their input files already exist on disk from the original prepare),
    then re-parses the *whole* step so fan-out lineage stays consistent and the
    ledger is refreshed (cleared if all now succeeded). NMS steps use
    :func:`chemrefine.step_nms.reattempt_nms` instead (they reuse round-1
    rather than resubmit it).
    """
    manifest = cache.load_manifest(ctx.step_dir)
    if manifest is None:
        raise CacheError(f"step {step_cfg.step}: cannot rerun — no manifest to rehydrate inputs")
    failed_ids = {f["structure_id"] for f in failed}
    failed_inputs = StepInputs(files=tuple(f for f in manifest.files if f[2] in failed_ids))
    logger.info(
        "step %d: rerun — resubmitting %d failed job(s)",
        step_cfg.step,
        len(failed_inputs.files),
    )
    engine.wait(engine.submit(failed_inputs, ctx))

    successes, failures = step_failures.parse_with_failures(engine, manifest, ctx)
    results = step_failures.apply_failure_policy(successes, failures, ctx, step_cfg)
    cache.save(
        step_cfg=step_cfg,
        parent_ids=parent_ids,
        results=results,
        step_dir=ctx.step_dir,
        chemrefine_version=__version__,
        parents_digest=cache.parents_digest(ctx.prev_state.structures),
    )
    return results

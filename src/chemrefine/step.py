"""Per-step lifecycle: prepare → submit → wait → parse → (nms) → filter → cache.

This is the only place that knows the order in which an engine's
lifecycle methods are called. :func:`run_step` is intentionally short
(~30 LOC) — every concern it touches lives in its own module:

* Caching + manifest:  :mod:`chemrefine.cache`
* Filtering:           :mod:`chemrefine.filtering`
* Engine lookup:  :mod:`chemrefine.engines.base`
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from chemrefine import cache, filtering
from chemrefine.config import Config, StepConfig
from chemrefine.engines.base import CalculationEngine, get_engine
from chemrefine.errors import CacheError, OutputParseError
from chemrefine.state import PipelineState, StepContext, StepInputs, StepResults

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context construction
# ---------------------------------------------------------------------------


def build_context(
    config: Config, step_cfg: StepConfig, prev_state: PipelineState
) -> StepContext:
    """Bundle the per-step inputs into a :class:`StepContext`."""
    return StepContext(
        step_cfg=step_cfg,
        step_dir=(config.output_dir / step_cfg.dir_name()).resolve(),
        template_dir=config.template_dir.resolve(),
        scratch_dir=config.scratch_dir.resolve() if config.scratch_dir is not None else None,
        prev_state=prev_state,
        charge=step_cfg.charge if step_cfg.charge is not None else config.charge,
        multiplicity=(
            step_cfg.multiplicity
            if step_cfg.multiplicity is not None
            else config.multiplicity
        ),
        max_cores=config.max_cores,
        slurm_template=config.slurm_template,
        executables=config.executables,
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
    rerun: bool = False,
) -> StepOutcome:
    """Execute one step end-to-end and return its surviving state.

    * If ``use_cache`` is true and the on-disk cache fingerprint matches
      the current step config + parent IDs, the cached
      :class:`~chemrefine.state.StepResults` are reused and only
      filtering runs again — unless ``rerun`` is set and a failed-jobs
      ledger exists, in which case only the failed structures are
      resubmitted (see :func:`_resubmit_failed`).
    * Otherwise the engine's full lifecycle runs and the resulting
      :class:`~chemrefine.state.StepResults` is cached for next time.
    """
    # Deferred import: chemrefine.__init__ imports chemrefine.engines, which
    # imports this module (step.py) to register engines — a circular chain.
    # Deferring to function scope breaks the cycle cleanly.
    from chemrefine import __version__

    ctx = build_context(config, step_cfg, prev_state)
    parent_ids = tuple(s.id for s in prev_state.structures)
    ctx.step_dir.mkdir(parents=True, exist_ok=True)
    engine = engine if engine is not None else get_engine(step_cfg.engine)

    if use_cache and cache.is_valid(
        step_cfg=step_cfg, parent_ids=parent_ids, step_dir=ctx.step_dir
    ):
        cached = cache.load(ctx.step_dir)
        if cached is None:  # pragma: no cover
            raise CacheError("is_valid returned True but load returned None")
        failed = cache.load_failed_jobs(ctx.step_dir)
        if rerun and failed:
            results = _resubmit_failed(
                engine, ctx, step_cfg, failed, parent_ids, __version__
            )
            return StepOutcome(
                state=filtering.apply(results, step_cfg.sample), cache_hit=False
            )
        logger.info(
            "step %d: cache hit, reusing %d structures",
            step_cfg.step,
            len(cached.results.structures),
        )
        return StepOutcome(
            state=filtering.apply(cached.results, step_cfg.sample),
            cache_hit=True,
        )

    logger.info("step %d (%s): preparing inputs", step_cfg.step, step_cfg.engine)
    inputs = engine.prepare(ctx)
    cache.save_manifest(
        inputs, ctx.step_dir, operation=step_cfg.operation, engine=step_cfg.engine
    )

    logger.info("step %d: submitting %d jobs", step_cfg.step, len(inputs.files))
    batch = engine.submit(inputs, ctx)
    engine.wait(batch)

    logger.info("step %d: parsing outputs", step_cfg.step)
    results = _parse_capturing_failures(engine, inputs, ctx, step_cfg)

    if step_cfg.nms and engine.supports_nms:
        logger.info("step %d: running normal-mode sampling", step_cfg.step)
        results = engine.normal_mode_sample(results, ctx)

    cache.save(
        step_cfg=step_cfg,
        parent_ids=parent_ids,
        results=results,
        step_dir=ctx.step_dir,
        chemrefine_version=__version__,
    )

    return StepOutcome(
        state=filtering.apply(results, step_cfg.sample),
        cache_hit=False,
    )


def _parse_capturing_failures(
    engine: CalculationEngine,
    inputs: StepInputs,
    ctx: StepContext,
    step_cfg: StepConfig,
) -> StepResults:
    """Parse the structures whose output exists; ledger the ones that produced none.

    A job that crashed / timed out leaves no output file. Rather than fail the
    whole step, record those structure IDs to ``_cache/failed_jobs.json`` (so
    ``chemrefine rerun`` can resubmit just them) and parse the rest. Raises only
    if *nothing* produced output. A clean run clears any stale ledger.
    """
    present = tuple(f for f in inputs.files if f[1].is_file())
    missing = [sid for _inp, out, sid in inputs.files if not out.is_file()]
    if missing:
        cache.save_failed_jobs(
            ctx.step_dir,
            [{"structure_id": sid, "reason": "output missing"} for sid in missing],
        )
        logger.warning(
            "step %d: %d job(s) produced no output; recorded to failed_jobs.json "
            "(resubmit with `chemrefine rerun`)",
            step_cfg.step,
            len(missing),
        )
        if not present:
            raise OutputParseError(
                f"step {step_cfg.step}: no job produced output (all {len(missing)} failed)"
            )
    else:
        cache.clear_failed_jobs(ctx.step_dir)
    return engine.parse(StepInputs(files=present), ctx)


def _resubmit_failed(
    engine: CalculationEngine,
    ctx: StepContext,
    step_cfg: StepConfig,
    failed: list[dict],
    parent_ids: tuple[str, ...],
    version: str,
) -> StepResults:
    """Resubmit only the failed structures, then re-parse + re-cache the full step.

    Rehydrates the per-structure inputs from the manifest, resubmits the failed
    subset (their input files already exist on disk from the original prepare),
    then re-parses the *whole* step so fan-out lineage stays consistent and the
    ledger is refreshed (cleared if all now succeeded).
    """
    manifest = cache.load_manifest(ctx.step_dir)
    if manifest is None:
        raise CacheError(
            f"step {step_cfg.step}: cannot rerun — no manifest to rehydrate inputs"
        )
    failed_ids = {f["structure_id"] for f in failed}
    failed_inputs = StepInputs(
        files=tuple(f for f in manifest.files if f[2] in failed_ids)
    )
    logger.info(
        "step %d: rerun — resubmitting %d failed job(s)",
        step_cfg.step,
        len(failed_inputs.files),
    )
    engine.wait(engine.submit(failed_inputs, ctx))

    results = _parse_capturing_failures(engine, manifest, ctx, step_cfg)
    if step_cfg.nms and engine.supports_nms:
        results = engine.normal_mode_sample(results, ctx)
    cache.save(
        step_cfg=step_cfg,
        parent_ids=parent_ids,
        results=results,
        step_dir=ctx.step_dir,
        chemrefine_version=version,
    )
    return results

"""Per-step lifecycle: prepare → submit → wait → parse → (nms) → filter → cache.

This is the only place that knows the order in which an engine's
lifecycle methods are called. :func:`run_step` is intentionally short
(~30 LOC) — every concern it touches lives in its own module:

* Caching:        :mod:`chemrefine.cache`
* File manifest:  :mod:`chemrefine.manifest`
* Filtering:      :mod:`chemrefine.filtering`
* Engine lookup:  :mod:`chemrefine.engines.base`
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from chemrefine import cache, filtering, manifest
from chemrefine.config import Config, StepConfig
from chemrefine.engines.base import CalculationEngine, get_engine
from chemrefine.state import PipelineState, StepContext

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
        orca_executable=config.orca_executable,
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
) -> StepOutcome:
    """Execute one step end-to-end and return its surviving state.

    * If ``use_cache`` is true and the on-disk cache fingerprint matches
      the current step config + parent IDs, the cached
      :class:`~chemrefine.state.StepResults` are reused and only
      filtering runs again.
    * Otherwise the engine's full lifecycle runs and the resulting
      :class:`~chemrefine.state.StepResults` is cached for next time.
    """
    from chemrefine import __version__

    ctx = build_context(config, step_cfg, prev_state)
    parent_ids = tuple(s.id for s in prev_state.structures)
    ctx.step_dir.mkdir(parents=True, exist_ok=True)

    if use_cache and cache.is_valid(
        step_cfg=step_cfg, parent_ids=parent_ids, step_dir=ctx.step_dir
    ):
        cached = cache.load(ctx.step_dir)
        assert cached is not None  # is_valid guarantees this
        logger.info(
            "step %d: cache hit, reusing %d structures",
            step_cfg.step,
            len(cached.results.structures),
        )
        return StepOutcome(
            state=filtering.apply(cached.results, step_cfg.sample),
            cache_hit=True,
        )

    engine = engine if engine is not None else get_engine(step_cfg.engine)

    logger.info("step %d (%s): preparing inputs", step_cfg.step, step_cfg.engine)
    inputs = engine.prepare(ctx)
    manifest.save(
        inputs, ctx.step_dir, operation=step_cfg.operation, engine=step_cfg.engine
    )

    logger.info("step %d: submitting %d jobs", step_cfg.step, len(inputs.files))
    batch = engine.submit(inputs, ctx)
    engine.wait(batch)

    logger.info("step %d: parsing outputs", step_cfg.step)
    results = engine.parse(inputs, ctx)

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

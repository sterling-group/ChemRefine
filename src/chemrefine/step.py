"""Per-step lifecycle: prepare → submit → wait → parse → (nms) → filter → cache.

This is the only place that knows the order in which an engine's
lifecycle methods are called. :func:`run_step` is intentionally short
(~30 LOC) — every concern it touches lives in its own module:

* Caching + manifest:  :mod:`chemrefine.cache`
* Filtering:           :mod:`chemrefine.filtering`
* Engine lookup:  :mod:`chemrefine.engines.base`
* NMS resolution / recovery:  :mod:`chemrefine.step_nms`
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from chemrefine import cache, filtering
from chemrefine.config import Config, StepConfig
from chemrefine.engines.base import CalculationEngine, get_engine
from chemrefine.errors import CacheError, ChemRefineError, OutputParseError
from chemrefine.state import (
    PipelineState,
    StepContext,
    StepInputs,
    StepResults,
    Structure,
)

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
) -> StepOutcome:
    """Execute one step end-to-end and return its surviving state.

    * If ``use_cache`` is true and the on-disk cache fingerprint matches
      the current step config + parent IDs, the cached
      :class:`~chemrefine.state.StepResults` are reused and only filtering
      runs again — **unless** a failed-jobs ledger exists, in which case
      ``resume`` is incremental: only the still-failed structures are
      resubmitted (see :func:`_resubmit_failed`). ``rerun`` (redo a whole
      step) is expressed by the caller invalidating the cache first, so the
      step falls through to a full re-execution.
    * Otherwise the engine's full lifecycle runs and the resulting
      :class:`~chemrefine.state.StepResults` is cached for next time.
    """
    # Deferred imports: chemrefine.__init__ imports chemrefine.engines, which
    # imports this module (step.py) to register engines — a circular chain.
    # ``step_nms`` imports the failure helpers from *this* module at its module
    # scope; deferring both to function scope breaks the cycle cleanly.
    from chemrefine import __version__, step_nms

    ctx = build_context(config, step_cfg, prev_state)
    parent_ids = tuple(s.id for s in prev_state.structures)
    ctx.step_dir.mkdir(parents=True, exist_ok=True)
    engine = engine if engine is not None else get_engine(step_cfg.engine)

    is_nms = step_cfg.nms and engine.supports_nms

    if use_cache and cache.is_valid(
        step_cfg=step_cfg, parent_ids=parent_ids, step_dir=ctx.step_dir
    ):
        cached = cache.load(ctx.step_dir)
        if cached is None:  # pragma: no cover
            raise CacheError("is_valid returned True but load returned None")
        failed = cache.load_failed_jobs(ctx.step_dir)
        if failed:
            results = (
                step_nms._reattempt_nms(engine, ctx, step_cfg, cached, parent_ids, __version__)
                if is_nms
                else _resubmit_failed(engine, ctx, step_cfg, failed, parent_ids, __version__)
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

    # Full fingerprint invalid. For an NMS step whose *search* params changed but
    # whose round-1 + criterion are unchanged (reuse fingerprint matches), reuse
    # the round-1 freq + already-resolved children and re-attempt only the
    # ledgered-unresolved parents — instead of re-running the whole step.
    if use_cache and is_nms:
        try:
            cached = cache.load(ctx.step_dir)
        except CacheError:
            cached = None
        if cached is not None and getattr(cached, "reuse_fingerprint", "") == (
            step_nms._nms_reuse_fingerprint(step_cfg, parent_ids)
        ):
            if cache.load_failed_jobs(ctx.step_dir):
                results = step_nms._reattempt_nms(
                    engine, ctx, step_cfg, cached, parent_ids, __version__
                )
            else:
                # All were resolved already; just re-stamp the new fingerprints.
                logger.info(
                    "step %d: NMS search params changed, all resolved — reusing cache",
                    step_cfg.step,
                )
                cache.save(
                    step_cfg=step_cfg, parent_ids=parent_ids, results=cached.results,
                    step_dir=ctx.step_dir, chemrefine_version=__version__,
                    reuse_fingerprint=step_nms._nms_reuse_fingerprint(step_cfg, parent_ids),
                )
                results = cached.results
            return StepOutcome(
                state=filtering.apply(results, step_cfg.sample), cache_hit=False
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
    successes, failures = _parse_with_failures(engine, inputs, ctx)

    if is_nms:
        # Run NMS on the round-1 survivors, then apply the failure policy once
        # over BOTH round-1 job failures and NMS-unresolved parents (a single
        # ledger write — otherwise the NMS resolution would clobber the round-1
        # failures).
        logger.info("step %d: running normal-mode sampling", step_cfg.step)
        round1 = StepResults(structures=tuple(successes))
        results = step_nms._resolve_nms(
            engine.normal_mode_sample(round1, ctx), round1, ctx, step_cfg,
            round1_failures=failures,
        )
    else:
        results = _apply_failure_policy(successes, failures, ctx, step_cfg)

    cache.save(
        step_cfg=step_cfg,
        parent_ids=parent_ids,
        results=results,
        step_dir=ctx.step_dir,
        chemrefine_version=__version__,
        reuse_fingerprint=step_nms._nms_reuse_fingerprint(step_cfg, parent_ids),
    )

    return StepOutcome(
        state=filtering.apply(results, step_cfg.sample),
        cache_hit=False,
    )


@dataclass(frozen=True)
class _Failure:
    """One failed structure: its id, why, and the best geometry obtained (if any)."""

    sid: str
    reason: str
    best: Structure | None


def _succeeded(s: Structure) -> bool:
    """A parsed structure failed only when an engine success flag is explicitly False.

    ``None`` (engine doesn't report it) is treated as 'not a failure signal', so
    backends that don't set termination/convergence flags are never gated.
    """
    return s.terminated is not False and s.converged is not False


def _failure_reason(s: Structure) -> str:
    """Human reason for a parsed-but-unsuccessful structure."""
    if s.terminated is False:
        return "did not terminate normally"
    if s.converged is False:
        return "did not converge"
    return "failed"


def _parse_with_failures(
    engine: CalculationEngine, inputs: StepInputs, ctx: StepContext
) -> tuple[list[Structure], list[_Failure]]:
    """Parse each output independently; classify into successes and failures.

    A job is a *failure* when its output is missing, unparseable, or parses to a
    structure the engine marks unconverged / not-terminated. Parsing per input
    (rather than the whole batch at once) means one bad job never crashes the
    step — its failure is captured and the rest still parse. Engine success
    flags are set in the single parse pass (see ``orca.output``).
    """
    successes: list[Structure] = []
    failures: list[_Failure] = []
    for triple in inputs.files:
        _inp, out, sid = triple
        if not out.is_file():
            failures.append(_Failure(sid, "output missing", None))
            continue
        try:
            parsed = list(engine.parse(StepInputs(files=(triple,)), ctx).structures)
        except OutputParseError as e:
            failures.append(_Failure(sid, f"unparseable: {e}", None))
            continue
        successes.extend(s for s in parsed if _succeeded(s))
        bad = [s for s in parsed if not _succeeded(s)]
        if bad:
            best = min(
                bad,
                key=lambda s: (s.energy_hartree is None, s.energy_hartree or 0.0),
            )
            failures.append(_Failure(sid, _failure_reason(bad[0]), best))
    return successes, failures


def _apply_failure_policy(
    successes: list[Structure],
    failures: list[_Failure],
    ctx: StepContext,
    step_cfg: StepConfig,
) -> StepResults:
    """Record failures to the ledger and resolve them per ``step_cfg.on_failure``.

    ``stop`` raises (halts the pipeline); ``skip`` (default) drops the failures
    and keeps the successes; ``best`` keeps every structure, backfilling a
    failure with the best geometry obtained for it (else its submitted input).
    The ``failed_jobs.json`` ledger is always written (so ``rerun`` / ``resume``
    can re-attempt) and cleared on a clean step.
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

    if step_cfg.on_failure == "stop":
        raise ChemRefineError(
            f"step {step_cfg.step}: {len(failures)} structure(s) failed and "
            "on_failure='stop'"
        )
    if step_cfg.on_failure == "best":
        prev_by_id = {s.id: s for s in ctx.prev_state.structures}
        for f in failures:
            fallback = f.best if f.best is not None else prev_by_id.get(f.sid)
            if fallback is not None:
                successes.append(fallback)
    return StepResults(structures=tuple(successes))


def rebuild_cache_step(
    config: Config, step_cfg: StepConfig, prev_state: PipelineState
) -> StepOutcome:
    """Rebuild one step's cache from outputs already on disk — **no submission**.

    Re-parses the step's existing outputs (via its manifest), re-applies the
    ``on_failure`` policy and — for NMS steps — re-resolves from the existing
    ``nms/`` round-2 outputs, then rewrites the ``StepCache`` with the same
    fingerprint a normal run would produce. Backs ``chemrefine rebuild-cache``.
    """
    from chemrefine import __version__, step_nms

    ctx = build_context(config, step_cfg, prev_state)
    parent_ids = tuple(s.id for s in prev_state.structures)
    engine = get_engine(step_cfg.engine)
    manifest = cache.load_manifest(ctx.step_dir)
    if manifest is None:
        raise CacheError(
            f"step {step_cfg.step}: cannot rebuild-cache — no manifest on disk"
        )
    logger.info("step %d: rebuilding cache from existing outputs", step_cfg.step)
    successes, failures = _parse_with_failures(engine, manifest, ctx)
    if step_cfg.nms and engine.supports_nms:
        round1 = StepResults(structures=tuple(successes))
        results = step_nms._resolve_nms(
            engine.resolve_nms_from_existing(round1, ctx), round1, ctx, step_cfg,
            round1_failures=failures,
        )
    else:
        results = _apply_failure_policy(successes, failures, ctx, step_cfg)
    cache.save(
        step_cfg=step_cfg,
        parent_ids=parent_ids,
        results=results,
        step_dir=ctx.step_dir,
        chemrefine_version=__version__,
        reuse_fingerprint=step_nms._nms_reuse_fingerprint(step_cfg, parent_ids),
    )
    return StepOutcome(state=filtering.apply(results, step_cfg.sample), cache_hit=False)


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
    ledger is refreshed (cleared if all now succeeded). NMS steps use
    :func:`_reattempt_nms` instead (they reuse round-1 rather than resubmit it).
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

    successes, failures = _parse_with_failures(engine, manifest, ctx)
    results = _apply_failure_policy(successes, failures, ctx, step_cfg)
    cache.save(
        step_cfg=step_cfg,
        parent_ids=parent_ids,
        results=results,
        step_dir=ctx.step_dir,
        chemrefine_version=version,
    )
    return results

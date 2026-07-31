"""Composing one step: which lifecycle it runs, and what happens around it.

:func:`run_step` decides between the cache-recovery modes and a full run, inserts NMS
resolution when the step asks for it and the engine can do it, then filters and caches what
comes back. The work itself belongs to other modules:

* Running, classifying, the ``on_failure`` policy: :mod:`chemrefine.lifecycle`
* NMS resolution (engine-independent): :mod:`chemrefine.nms`
* Caching + manifest: :mod:`chemrefine.cache`
* Attempt directories: :mod:`chemrefine.attempts`
* Filtering: :mod:`chemrefine.filtering`
* Engine lookup: :mod:`chemrefine.engines.api`

The one lifecycle sequence spelled out here rather than delegated is a full run's
``prepare → save_manifest → submit``, because the manifest has to be stamped *between* the
first two: an interrupted run then leaves proof of what its outputs were computed for. Every
other run of a structure set goes through :func:`chemrefine.lifecycle.submit_and_parse`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path

from chemrefine import __version__, attempts, cache, filtering, lifecycle, nms
from chemrefine.config import Config, StepConfig
from chemrefine.engines.api import CalculationEngine, NmsCapableEngine, get_engine
from chemrefine.errors import CacheError, ChemRefineError, ConfigError
from chemrefine.state import (
    FailureKind,
    FailureRecord,
    PipelineState,
    StepContext,
    StepResults,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context construction
# ---------------------------------------------------------------------------


def step_dir_for(config: Config, step_cfg: StepConfig) -> Path:
    """The on-disk directory for a step's artifacts (cache, ledger, outputs).

    Resolves :meth:`Config.step_dir` (the single source of the
    ``output_dir / dir_name`` construction) to an absolute path.
    """
    return config.step_dir(step_cfg).resolve()


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
        slurm_array=config.slurm_array,
        dispatch=config.dispatch,
        job_timeout_seconds=config.job_timeout_seconds,
    )


class StepMode(StrEnum):
    """How one step is to be handled on this run.

    Resolved **once**, here, from the requested :class:`Action` — rather than
    re-derived at each layer from nullable step numbers, which is how the same
    question ended up being asked at three different depths.
    """

    EXECUTE = "execute"
    """Ignore any cache and run the engine's full lifecycle."""

    RESUME = "resume"
    """Honour a valid cache, but re-attempt an ``on_failure: stop`` step's pending
    failures rather than serving them from it."""

    CACHE_ONLY = "cache-only"
    """Honour a valid cache and leave pending failures alone — the mode every step
    that isn't the target of a scoped action runs in. Submits nothing: see
    :meth:`may_submit`."""

    REBUILD = "rebuild"
    """Re-parse the outputs already on disk and rewrite the cache. Never submits."""

    def may_submit(self) -> bool:
        """Whether a step in this mode is allowed to send work to the engine.

        Asked once, at each point in :func:`run_step` that can reach a submission, so the
        modes that promise not to run anything cannot do so by any route. ``CACHE_ONLY``
        is the one that says no: it belongs to the steps a scoped action is *not*
        targeting, and both ``rebuild-cache`` and ``rerun-errors`` are documented to leave
        those alone. (``REBUILD`` never reaches here — see :meth:`runs_through_run_step` —
        but it answers honestly for the same reason.)
        """
        return self not in (StepMode.CACHE_ONLY, StepMode.REBUILD)

    def runs_through_run_step(self) -> bool:
        """Whether :func:`chemrefine.pipeline.run` drives this mode through :func:`run_step`.

        ``REBUILD`` is the one that does not: it re-parses outputs already on disk, which is
        :func:`rebuild_cache_step`, a different function with a different precondition (a
        manifest whose fingerprint it can check) rather than a branch inside the recovery
        chain.

        A predicate on the enum rather than a member test in the pipeline, for the same
        reason :meth:`may_submit` is one: three separate questions are asked about
        :class:`StepMode` — may it submit, does it go through ``run_step``, can it halt —
        and each answered in a different module means adding a fourth mode is a search
        rather than a compiler error.
        """
        return self is not StepMode.REBUILD

    def can_halt(self) -> bool:
        """Whether an ``on_failure: stop`` step in this mode may stop the run.

        ``CACHE_ONLY`` may not, and that is load-bearing rather than an optimisation: it is
        the mode every *non-target* step runs in under a scoped action, so
        ``rerun-errors N`` has to be able to reach step N past an earlier step's pending
        failures. Every other mode halts — including ``REBUILD``, since re-parsing from disk
        cannot make a failed structure succeed, and continuing would run the next step
        against the partial survivor set the user asked to stop on.
        """
        return self is not StepMode.CACHE_ONLY


@dataclass(frozen=True)
class RunPlan:
    """Which :class:`StepMode` each step runs in.

    ``default`` covers the whole pipeline; ``overrides`` scopes a different mode to
    individual steps, which is what the step-targeted actions (``rerun-errors``,
    ``rebuild-cache``) need and all that distinguishes them from a plain resume.
    """

    default: StepMode = StepMode.RESUME
    """``RESUME`` so a bare ``pipeline.run(config)`` behaves the way "run the
    pipeline" reads: honour the caches, repair whatever is pending, halt on a
    ``stop`` step. ``CACHE_ONLY`` is opted into by the scoped actions for the steps
    they are *not* targeting."""

    overrides: Mapping[int, StepMode] = field(default_factory=dict)

    def for_step(self, step: int) -> StepMode:
        """The mode this step runs in."""
        return self.overrides.get(step, self.default)


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
    mode: StepMode = StepMode.RESUME,
) -> StepOutcome:
    """Execute one step end-to-end and return its surviving state.

    ``mode`` decides how the cache is treated — see :class:`StepMode`:

    * ``EXECUTE`` ignores any cache outright.
    * ``CACHE_ONLY`` and ``RESUME`` both reuse a cache whose fingerprint matches the
      current step config + parent structures (IDs and content), re-running only the
      filter. They differ on a pending ledger: ``RESUME`` re-attempts an
      ``on_failure: stop`` step's still-failed structures (see
      :func:`_resubmit_failed`), ``CACHE_ONLY`` leaves them alone. ``skip`` / ``best``
      steps keep their ledger for visibility but are never re-attempted under either.
    * ``rerun`` needs no mode of its own: the caller invalidates the target's cache,
      so it misses and executes.

    On a miss the engine's full lifecycle runs and the result is cached. An
    ``on_failure: stop`` step that ends with failures caches its successes and ledgers
    the failures; the pipeline then halts the run once via :func:`halt_if_pending`
    (this function never raises).
    """
    ctx = build_context(config, step_cfg, prev_state)
    parent_ids = tuple(s.id for s in prev_state.structures)
    ctx.step_dir.mkdir(parents=True, exist_ok=True)
    engine = engine if engine is not None else get_engine(step_cfg.engine)
    # Narrow once, here, instead of asserting the capability again with a cast at
    # each of the four places that need it. `nms_engine is not None` then carries
    # both facts — the step asked for NMS, and this engine can do it — and mypy
    # checks the calls rather than being told to trust them.
    nms_engine: NmsCapableEngine | None = None
    if step_cfg.nms and isinstance(engine, NmsCapableEngine):
        nms_engine = engine

    # Every recovery route below either serves what is already on disk or sends work to
    # the engine, and which of those a step is allowed to do is the whole of what the mode
    # decides here. Derived once and passed down, so no route re-asks it in its own words.
    may_submit = mode.may_submit()

    if mode is not StepMode.EXECUTE:
        cached = _cached_outcome(
            ctx, step_cfg, parent_ids, engine, nms_engine=nms_engine, may_submit=may_submit
        )
        if cached is not None:
            return cached
        # Full fingerprint invalid. For an NMS step whose *search* params changed
        # but whose round-1 + criterion are unchanged (reuse fingerprint matches),
        # reuse the round-1 freq + already-resolved children and re-attempt only
        # the ledgered-unresolved parents — instead of re-running the whole step.
        if nms_engine is not None:
            reused = _nms_reuse_outcome(
                ctx, step_cfg, parent_ids, nms_engine, may_submit=may_submit
            )
            if reused is not None:
                return reused
        # No cache at all, but possibly a step this configuration already half-ran and
        # was interrupted before it could write one.
        partial = _partial_step_outcome(
            ctx, step_cfg, parent_ids, engine, nms_engine=nms_engine, may_submit=may_submit
        )
        if partial is not None:
            return partial

    if not may_submit:
        raise ChemRefineError(
            f"step {step_cfg.step} has no cache this configuration can use, and "
            f"`{mode.value}` does not submit work for a step it is not targeting. "
            f"Run `chemrefine resume` to bring it up to date, or "
            f"`chemrefine rerun {step_cfg.step}` to redo it."
        )
    return _run_full_step(ctx, step_cfg, parent_ids, engine, nms_engine=nms_engine)


def _cached_outcome(
    ctx: StepContext,
    step_cfg: StepConfig,
    parent_ids: tuple[str, ...],
    engine: CalculationEngine,
    *,
    nms_engine: NmsCapableEngine | None,
    may_submit: bool,
) -> StepOutcome | None:
    """Outcome from a valid on-disk cache, or ``None`` if the cache is invalid.

    A pending ``on_failure: stop`` ledger re-attempts only the still-failed structures
    when this step may submit; otherwise it is a plain cache hit (refilter), which is
    what a step a scoped action is not targeting wants.
    """
    cached = cache.load_if_valid(
        step_cfg=step_cfg,
        parent_ids=parent_ids,
        step_dir=ctx.step_dir,
        parents_digest=cache.parents_digest(ctx.prev_state.structures),
        template_digest=engine.input_digest(ctx),
    )
    if cached is None:
        return None
    failed = cache.load_failure_records(ctx.step_dir)
    if failed and step_cfg.on_failure == "stop" and may_submit:
        results = (
            nms.reattempt_nms(nms_engine, ctx, step_cfg, cached, parent_ids)
            if nms_engine is not None
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
    engine: NmsCapableEngine,
    *,
    may_submit: bool,
) -> StepOutcome | None:
    """Reuse a cached NMS round-1 when only the *search* params changed.

    Returns ``None`` (re-run the whole step) unless a cache exists whose reuse
    fingerprint matches; then it re-attempts the ledgered-unresolved parents, or
    re-stamps the cache when everything was already resolved.

    Re-attempting runs round-2 jobs, so it needs ``may_submit``. This branch is reached
    whenever the *reuse* fingerprint matches — the ordinary state after tuning a search
    parameter, not an error condition — so it is the likeliest way for a step nobody
    targeted to start computing.
    """
    try:
        cached = cache.load(ctx.step_dir)
    except CacheError:
        cached = None
    digest = cache.parents_digest(ctx.prev_state.structures)
    template_digest = engine.input_digest(ctx)
    fingerprint = cache.reuse_fingerprint(
        step_cfg, parent_ids, parents_digest=digest, template_digest=template_digest
    )
    if cached is None or getattr(cached, "reuse_fingerprint", "") != fingerprint:
        return None
    if cache.load_failure_records(ctx.step_dir):
        if not may_submit:
            return None
        results = nms.reattempt_nms(engine, ctx, step_cfg, cached, parent_ids)
    else:
        logger.info(
            "step %d: NMS search params changed, all resolved — reusing cache", step_cfg.step
        )
        # The one place a step is persisted *without* `lifecycle.finalize`, and
        # deliberately so: there is nothing to resolve. Every structure was already resolved
        # under the previous search params, so re-applying `on_failure` to an empty failure
        # list would only re-stamp a ledger that is already correct. This re-stamps the
        # cache under the new fingerprint and nothing else.
        cache.save_step_results(
            step_cfg=step_cfg,
            parent_ids=parent_ids,
            results=cached.results,
            ctx=ctx,
            template_digest=template_digest,
            chemrefine_version=__version__,
        )
        results = cached.results
    return StepOutcome(state=filtering.apply(results, step_cfg.sample), cache_hit=False)


def _current_fingerprint(
    ctx: StepContext,
    step_cfg: StepConfig,
    parent_ids: tuple[str, ...],
    engine: CalculationEngine,
) -> str:
    """The cache key for this step as configured right now."""
    return cache.fingerprint(
        step_cfg,
        parent_ids,
        parents_digest=cache.parents_digest(ctx.prev_state.structures),
        template_digest=engine.input_digest(ctx),
    )


def _partial_step_outcome(
    ctx: StepContext,
    step_cfg: StepConfig,
    parent_ids: tuple[str, ...],
    engine: CalculationEngine,
    *,
    nms_engine: NmsCapableEngine | None,
    may_submit: bool,
) -> StepOutcome | None:
    """Continue a step the driver died in the middle of, instead of redoing it.

    ``step.json`` is written **once**, at the end of a step, so a driver killed partway
    through — the batch job hits its walltime, a node fails, Ctrl-C — leaves no cache at all.
    The alternative is :func:`_run_full_step`, whose first act is to archive every finished
    ``.out`` into ``attemptK/`` and resubmit the lot: on HPC, days of completed compute left
    on disk and never read back, because
    :func:`~chemrefine.lifecycle.parse_with_failures` decides success by ``out.is_file()``
    at the canonical path, which archiving has just emptied.

    The manifest is what makes continuing *safe* rather than merely cheap. Written before
    submission and carrying the step's fingerprint, a match proves these outputs were
    produced for this step config and these parents — the distinction ``out.is_file()``
    cannot make alone, and the reason archiving is otherwise unconditional. Anything still
    missing is handed to :func:`_resubmit_failed`, the
    same per-structure archive-and-resubmit path ``rerun-errors`` uses; it re-parses the
    whole manifest afterwards, so the finished structures come back from disk.

    Returns ``None`` — meaning "run the whole step" — unless every condition holds:

    * The step may submit. Continuing means resubmitting whatever is still missing, so a
      mode that promises to run nothing has no use for this route.
    * A manifest exists and its fingerprint matches the current one.
    * The step is not NMS. An interrupted NMS step needs its round-2 children re-resolved,
      not just its round-1 outputs re-parsed, so it falls back to the full re-run rather
      than being silently half-recovered.
    """
    if not may_submit or nms_engine is not None:
        return None
    manifest = cache.load_manifest(ctx.step_dir)
    if manifest is None:
        return None
    if cache.load_manifest_fingerprint(ctx.step_dir) != _current_fingerprint(
        ctx, step_cfg, parent_ids, engine
    ):
        return None
    missing = [
        FailureRecord(
            structure_id=sid,
            kind=FailureKind.MISSING_OUTPUT,
            reason=FailureKind.MISSING_OUTPUT.value,
        )
        for _inp, out, sid in manifest.files
        if not out.is_file()
    ]
    logger.info(
        "step %d: resuming an interrupted step — %d of %d structure(s) still to run",
        step_cfg.step,
        len(missing),
        len(manifest.files),
    )
    results = _resubmit_failed(engine, ctx, step_cfg, missing, parent_ids)
    return StepOutcome(state=filtering.apply(results, step_cfg.sample), cache_hit=False)


def _run_full_step(
    ctx: StepContext,
    step_cfg: StepConfig,
    parent_ids: tuple[str, ...],
    engine: CalculationEngine,
    *,
    nms_engine: NmsCapableEngine | None,
) -> StepOutcome:
    """Run the engine's full lifecycle (prepare → submit → parse → retry → nms/policy → cache)."""
    if nms_engine is not None:
        _check_nms_freq_gate(nms_engine, ctx, step_cfg)
    logger.info("step %d (%s): preparing inputs", step_cfg.step, step_cfg.engine)
    # Anything already in these structure dirs is a previous run's work. Move it aside
    # before writing new inputs, so a job that dies without producing output is seen as
    # a failure rather than re-reading the old result. Done here rather than inside
    # ``prepare``, because the retry and NMS paths call ``prepare`` too and already
    # manage their own attempt dirs.
    attempts.archive_previous(ctx.step_dir, (s.id for s in ctx.prev_state.structures))
    inputs = engine.prepare(ctx)
    cache.save_manifest(
        inputs,
        ctx.step_dir,
        operation=step_cfg.operation,
        engine=step_cfg.engine,
        # Stamped before submission, so an interrupted run leaves proof of *what* these
        # outputs were computed for — see :func:`_partial_step_outcome`.
        fingerprint=_current_fingerprint(ctx, step_cfg, parent_ids, engine),
    )

    logger.info("step %d: submitting %d jobs", step_cfg.step, len(inputs.files))
    engine.submit(inputs, ctx)

    logger.info("step %d: parsing outputs", step_cfg.step)
    successes, failures = lifecycle.parse_and_record(engine, inputs, ctx)
    # Round-1 convergence failures are retried from best before NMS / the policy.
    successes, failures = lifecycle.retry_unconverged(engine, ctx, successes, failures)

    if nms_engine is not None:
        logger.info("step %d: running normal-mode sampling", step_cfg.step)
        resolution = nms.run_nms(
            nms_engine,
            StepResults(structures=tuple(successes)),
            failures,
            ctx,
        )
        successes, failures = list(resolution.survivors), list(resolution.failures)
    results = lifecycle.finalize(engine, ctx, step_cfg, parent_ids, successes, failures)
    return StepOutcome(state=filtering.apply(results, step_cfg.sample), cache_hit=False)


def _check_nms_freq_gate(engine: NmsCapableEngine, ctx: StepContext, step_cfg: StepConfig) -> None:
    """B9: reject an ``nms: true`` step whose input computes no frequencies.

    NMS acts on imaginary modes, so a step that won't produce frequencies can only
    ever leave every structure unresolved — caught here before any job is submitted.
    ``operation`` never changes the generated input (it only picks the parser), so an
    explicit one cannot rescue a frequency-less input and does not bypass this check.
    """
    if not engine.nms_input_info(ctx).computes_frequencies:
        raise ConfigError(
            f"step {step_cfg.step}: `nms: true` needs a frequency calculation, but the "
            f"input computes none. Request frequencies (e.g. ORCA `! Opt Freq`), or drop "
            f"`nms: true` if you don't want normal-mode sampling."
        )


def halt_if_pending(config: Config, step_cfg: StepConfig, mode: StepMode) -> None:
    """Halt the run when an ``on_failure: stop`` step still has failed jobs.

    Called **once** from the pipeline after a step executes — including after a
    ``rebuild_cache_step`` — so the step's successes are already cached. Only ``stop``
    turns its ledgered failures into a hard stop; ``skip`` / ``best`` keep their ledger for
    visibility but never halt. Which *modes* may halt is :meth:`StepMode.can_halt`, which
    is where the ``CACHE_ONLY`` exemption and its reason live.
    """
    if step_cfg.on_failure != "stop":
        return
    if not mode.can_halt():
        return
    if cache.load_failure_records(step_dir_for(config, step_cfg)):
        raise ChemRefineError(
            f"step {step_cfg.step} halted (on_failure=stop); fix the failed "
            f"job(s) and run `chemrefine resume` (or `rerun-errors {step_cfg.step}`)"
        )


def rebuild_cache_step(
    config: Config, step_cfg: StepConfig, prev_state: PipelineState
) -> StepOutcome:
    """Rebuild one step's cache from outputs already on disk — **no submission**.

    Re-parses the step's existing outputs (via its manifest), re-applies the
    ``on_failure`` policy and — for NMS steps — re-resolves from the displaced
    children already on disk (under each structure's latest ``attemptK/``), then
    rewrites the ``StepCache`` with the same fingerprint a normal run would produce.
    Backs ``chemrefine rebuild-cache``.

    Refuses when the manifest's stamped fingerprint *disagrees* with the current one.
    Re-parsing outputs produced for a different template, options or upstream survivors
    would write a cache that is internally valid and describes a run that never happened,
    and the next ``resume`` would serve it rather than compute what was asked for.

    A manifest carrying **no** fingerprint is not evidence of a mismatch, and this
    proceeds: an output tree predating that stamp is precisely what the command exists to
    re-parse, and the caller has named the step. The distinction is between *proving* the
    outputs are wrong and merely being unable to prove they are right. It is drawn
    differently in :func:`_partial_step_outcome`, which treats an unproven match as a
    reason to re-run — it can afford to, being an optimisation over doing the work anyway.
    """
    ctx = build_context(config, step_cfg, prev_state)
    parent_ids = tuple(s.id for s in prev_state.structures)
    engine = get_engine(step_cfg.engine)
    manifest = cache.load_manifest(ctx.step_dir)
    if manifest is None:
        raise CacheError(f"step {step_cfg.step}: cannot rebuild-cache — no manifest on disk")
    stamped = cache.load_manifest_fingerprint(ctx.step_dir)
    if stamped and stamped != _current_fingerprint(ctx, step_cfg, parent_ids, engine):
        raise CacheError(
            f"step {step_cfg.step}: the outputs on disk were produced for a different "
            f"configuration — its template, options or upstream results have changed "
            f"since — so re-parsing them would cache results this configuration never "
            f"produced. Run `chemrefine rerun {step_cfg.step}` to recompute it."
        )
    logger.info("step %d: rebuilding cache from existing outputs", step_cfg.step)
    successes, failures = lifecycle.parse_and_record(engine, manifest, ctx)
    if step_cfg.nms and isinstance(engine, NmsCapableEngine):
        resolution = nms.rebuild_nms(
            engine,
            StepResults(structures=tuple(successes)),
            failures,
            ctx,
        )
        successes, failures = list(resolution.survivors), list(resolution.failures)
    results = lifecycle.finalize(engine, ctx, step_cfg, parent_ids, successes, failures)
    return StepOutcome(state=filtering.apply(results, step_cfg.sample), cache_hit=False)


def _resubmit_failed(
    engine: CalculationEngine,
    ctx: StepContext,
    step_cfg: StepConfig,
    failed: list[FailureRecord],
    parent_ids: tuple[str, ...],
) -> StepResults:
    """Re-prepare and resubmit only the failed structures, then re-parse + re-cache the step.

    The failed structures' prior artifacts are archived into ``attemptK/`` and their
    inputs **regenerated** before resubmission, rather than reusing the input files the
    original prepare left on disk. Two reasons: archiving is what stops a job that dies
    without producing output from re-reading the old result, and regenerating means
    a ``rerun-errors`` after a template edit actually runs the edited template — the
    on-disk input would otherwise contradict the fingerprint the cache is keyed on.

    The *whole* step is then re-parsed from the manifest so fan-out lineage stays
    consistent and the ledger is refreshed (cleared if all now succeeded). NMS steps use
    :func:`chemrefine.nms.reattempt_nms` instead (they reuse round-1 rather than
    resubmit it).
    """
    manifest = cache.load_manifest(ctx.step_dir)
    if manifest is None:
        raise CacheError(f"step {step_cfg.step}: cannot rerun — no manifest to rehydrate inputs")
    # Convergence failures are re-attempted from their best geometry by the retry
    # pass below (resubmitting the identical input would just fail again); the
    # plain resubmit handles crashed / missing-output jobs.
    failed_ids = {f.structure_id for f in failed if f.kind is not FailureKind.NOT_CONVERGED}
    failed_seeds = tuple(s for s in ctx.prev_state.structures if s.id in failed_ids)
    if failed_seeds:
        logger.info(
            "step %d: rerun — resubmitting %d failed job(s)",
            step_cfg.step,
            len(failed_seeds),
        )
        attempts.archive_previous(ctx.step_dir, (s.id for s in failed_seeds))
        retry_ctx = replace(ctx, prev_state=PipelineState(structures=failed_seeds))
        engine.submit(engine.prepare(retry_ctx), retry_ctx)

    successes, failures = lifecycle.parse_and_record(engine, manifest, ctx)
    successes, failures = lifecycle.retry_unconverged(engine, ctx, successes, failures)
    return lifecycle.finalize(engine, ctx, step_cfg, parent_ids, successes, failures)

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
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import assert_never

from chemrefine import __version__, attempts, cache, filtering, ids, lifecycle, nms
from chemrefine.config import Config, StepConfig
from chemrefine.engines.api import (
    ArtifactEngine,
    CalculationEngine,
    NmsCapableEngine,
    TemplateDriven,
    get_engine,
)
from chemrefine.errors import CacheError, ChemRefineError, ConfigError, JobFailureError
from chemrefine.state import (
    PipelineState,
    StepContext,
    StepInputs,
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


def _template_for(config: Config, step_cfg: StepConfig, engine: CalculationEngine) -> Path | None:
    """Where this step's input template lives, or ``None`` if the engine reads none.

    Existence is not checked: a step whose template is missing must still be able to compute
    its cache key (the digest is then ``""``, which changes the fingerprint and forces a
    re-run), and the actionable error belongs at the moment of rendering —
    :func:`chemrefine.ids.require_template`.
    """
    if not isinstance(engine, TemplateDriven):
        return None
    return ids.step_template_path(
        config.template_dir.resolve(),
        step_cfg.step,
        template=step_cfg.template,
        suffix=engine.template_suffix,
    )


def build_context(
    config: Config, step_cfg: StepConfig, prev_state: PipelineState, engine: CalculationEngine
) -> StepContext:
    """Bundle the per-step inputs into a :class:`StepContext`.

    Takes the engine because the step's template is part of its specification and only the
    engine knows the extension to look for. Resolving it **here, once** is the point: every
    later reader — ``prepare``, ORCA's ``pal`` and run-type detection, the NMS input probe,
    the cache — takes it off the context instead of re-deriving and re-reading the file.
    """
    return StepContext(
        step_cfg=step_cfg,
        step_dir=step_dir_for(config, step_cfg),
        template_dir=config.template_dir.resolve(),
        template=_template_for(config, step_cfg, engine),
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

    Resolved **once**, here, from the requested :class:`Action`. Re-deriving it per layer
    from nullable step numbers is how one question comes to be asked at three depths, in
    three wordings.
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
        match self:
            case StepMode.EXECUTE | StepMode.RESUME:
                return True
            case StepMode.CACHE_ONLY | StepMode.REBUILD:
                return False
            case _:
                assert_never(self)

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
        rather than a compiler error. The exhaustive ``match`` — every member named, the
        wildcard arm holding only :func:`typing.assert_never` — is what makes it the
        compiler error, in all three predicates: a fifth member stops narrowing to
        ``Never`` there and strict mypy rejects the call, where a negative membership test
        would hand the new mode the *permissive* answer silently.
        """
        match self:
            case StepMode.REBUILD:
                return False
            case StepMode.EXECUTE | StepMode.RESUME | StepMode.CACHE_ONLY:
                return True
            case _:
                assert_never(self)

    def can_halt(self) -> bool:
        """Whether an ``on_failure: stop`` step in this mode may stop the run.

        ``CACHE_ONLY`` may not, and that is load-bearing rather than an optimisation: it is
        the mode every *non-target* step runs in under a scoped action, so
        ``rerun-errors N`` has to be able to reach step N past an earlier step's pending
        failures. Every other mode halts — including ``REBUILD``, since re-parsing from disk
        cannot make a failed structure succeed, and continuing would run the next step
        against the partial survivor set the user asked to stop on.
        """
        match self:
            case StepMode.CACHE_ONLY:
                return False
            case StepMode.EXECUTE | StepMode.RESUME | StepMode.REBUILD:
                return True
            case _:
                assert_never(self)


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

    stop_after: int | None = None
    """Last step this plan covers; ``None`` runs to the end of the pipeline.

    ``rebuild-cache N`` is about steps 1..N and nothing else: it re-parses outputs already on
    disk and promises to submit nothing, so the steps after its target have no part in it.
    They cannot be left ``CACHE_ONLY`` either — a step the run never reached has no cache, and
    asking for one raises. Resuming them is not the alternative it is for ``rerun-errors``:
    that would submit, which is the one thing this command says it will not do, and would put
    the backend requirement back on a command whose purpose is to run where the backend is
    not installed (see :func:`chemrefine.pipeline.run`).
    """

    def for_step(self, step: int) -> StepMode:
        """The mode this step runs in."""
        return self.overrides.get(step, self.default)

    def covers(self, step: int) -> bool:
        """Whether the pipeline should go on to the step *after* ``step``."""
        return self.stop_after is None or step < self.stop_after


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
    engine = engine if engine is not None else get_engine(step_cfg.engine)
    ctx = build_context(config, step_cfg, prev_state, engine)
    # Derived once, here, and passed down as a value. Every route below needs the same
    # answer to "would this configuration have written the cache on disk", and each used
    # to re-derive it from whatever context it held. See `cache.StepKey`.
    key = cache.StepKey.of(step_cfg, prev_state.structures, ctx.template)
    ctx.step_dir.mkdir(parents=True, exist_ok=True)
    # Narrow once, here, instead of asserting the capability again with a cast at
    # each of the four places that need it. `nms_engine is not None` then carries
    # both facts — the step asked for NMS, and this engine can do it — and mypy
    # checks the calls rather than being told to trust them.
    nms_engine: NmsCapableEngine | None = None
    if step_cfg.nms and isinstance(engine, NmsCapableEngine):
        nms_engine = engine
    # Narrowed the same way and in the same place, so the two step kinds are decided together
    # rather than one here and one three functions down. An artifact step submits one job over
    # the whole ensemble and produces a file, so every route below that reasons per structure —
    # the partial-step resume, the failed-job resubmit — has to know before it starts.
    artifact_engine: ArtifactEngine | None = engine if isinstance(engine, ArtifactEngine) else None

    # Every recovery route below either serves what is already on disk or sends work to
    # the engine, and which of those a step is allowed to do is the whole of what the mode
    # decides here. Derived once and passed down, so no route re-asks it in its own words.
    may_submit = mode.may_submit()

    if mode is not StepMode.EXECUTE:
        cached = _cached_outcome(
            ctx, step_cfg, key, engine, nms_engine=nms_engine, may_submit=may_submit
        )
        if cached is not None:
            return cached
        # Full fingerprint invalid. For an NMS step whose *search* params changed
        # but whose round-1 + criterion are unchanged (reuse fingerprint matches),
        # reuse the round-1 freq + already-resolved children and re-attempt only
        # the ledgered-unresolved parents — instead of re-running the whole step.
        if nms_engine is not None:
            reused = _nms_reuse_outcome(ctx, step_cfg, key, nms_engine, may_submit=may_submit)
            if reused is not None:
                return reused
        # No cache at all, but possibly a step this configuration already half-ran and
        # was interrupted before it could write one.
        partial = _partial_step_outcome(
            ctx,
            step_cfg,
            key,
            engine,
            nms_engine=nms_engine,
            artifact_engine=artifact_engine,
            may_submit=may_submit,
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
    if artifact_engine is not None:
        return _run_artifact_step(ctx, step_cfg, key, artifact_engine)
    return _run_full_step(ctx, step_cfg, key, engine, nms_engine=nms_engine)


def _cached_outcome(
    ctx: StepContext,
    step_cfg: StepConfig,
    key: cache.StepKey,
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
    cached = cache.load_if_valid(key=key, step_dir=ctx.step_dir)
    if cached is None:
        return None
    failed = cache.load_failure_records(ctx.step_dir)
    if failed and step_cfg.leaves_failures_pending and may_submit:
        results = (
            nms.reattempt_nms(nms_engine, ctx, step_cfg, cached, key)
            if nms_engine is not None
            else _resubmit_failed(engine, ctx, step_cfg, key)
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
    key: cache.StepKey,
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
    if cached is None or cached.reuse_fingerprint != key.reuse_fingerprint:
        return None
    if cache.load_failure_records(ctx.step_dir):
        if not may_submit:
            return None
        results = nms.reattempt_nms(engine, ctx, step_cfg, cached, key)
    else:
        logger.info(
            "step %d: NMS search params changed, all resolved — reusing cache", step_cfg.step
        )
        # The one place a step is persisted *without* `lifecycle.finalize`, and
        # deliberately so: there is nothing to resolve. Every structure was already resolved
        # under the previous search params, so re-applying `on_failure` to an empty failure
        # list would only re-stamp a ledger that is already correct. This re-stamps the
        # cache under the new fingerprint and nothing else.
        cache.save(
            step_cfg=step_cfg,
            key=key,
            results=cached.results,
            step_dir=ctx.step_dir,
            chemrefine_version=__version__,
        )
        results = cached.results
    return StepOutcome(state=filtering.apply(results, step_cfg.sample), cache_hit=False)


def _partial_step_outcome(
    ctx: StepContext,
    step_cfg: StepConfig,
    key: cache.StepKey,
    engine: CalculationEngine,
    *,
    nms_engine: NmsCapableEngine | None,
    artifact_engine: ArtifactEngine | None,
    may_submit: bool,
) -> StepOutcome | None:
    """Continue a step the driver died in the middle of, instead of redoing it.

    ``step.json`` is written **once**, at the end of a step, so a driver killed partway
    through — the batch job hits its walltime, a node fails, Ctrl-C — leaves no cache at all.
    The alternative is :func:`_run_full_step`, whose first act is to archive every finished
    ``.out`` into ``attemptK/`` and resubmit the lot: on HPC, days of completed compute left
    on disk and never read back, because a structure is only ever parsed from the canonical
    path, which archiving has just emptied.

    The manifest is what makes continuing *safe* rather than merely cheap. Written before
    submission and carrying the step's fingerprint, a match proves these outputs were
    produced for this step config and these parents — the distinction ``out.is_file()``
    cannot make alone, and the reason archiving is otherwise unconditional. The tree is then
    handed to :func:`_resubmit_failed`, the same archive-and-resubmit path ``rerun-errors``
    uses: it re-runs whatever has no usable result and reads the rest back from disk.

    Returns ``None`` — meaning "run the whole step" — unless every condition holds:

    * The step may submit. Continuing means resubmitting whatever is still missing, so a
      mode that promises to run nothing has no use for this route.
    * A manifest exists and its fingerprint matches the current one.
    * The step is not NMS. An interrupted NMS step needs its round-2 children re-resolved,
      not just its round-1 outputs re-parsed, so it falls back to the full re-run rather
      than being silently half-recovered.
    * The step is not an artifact step. There is nothing per-structure to continue —
      :func:`_run_artifact_step` re-runs the one job, and adopting a *finished* product is
      ``rebuild-cache``'s job, not a resume's.
    * **The manifest names at least one job.** An empty manifest is a legitimate value, not a
      missing one: a step that prepares no per-structure inputs writes ``"files": []`` and
      ``load_manifest`` returns ``StepInputs(files=())``, which is not ``None``. Testing only
      for ``None`` sent such a step down the resubmit path, where a zero-job re-parse produced
      zero successes and zero failures — so an interrupted training cached an empty result and
      was never re-run. Nothing can be continued from a manifest with no jobs in it.
    """
    if not may_submit or nms_engine is not None or artifact_engine is not None:
        return None
    manifest = cache.load_manifest(ctx.step_dir)
    if manifest is None or not manifest.files:
        return None
    if cache.load_manifest_fingerprint(ctx.step_dir) != key.fingerprint:
        return None
    logger.info(
        "step %d: resuming an interrupted step — %d structure(s) on disk",
        step_cfg.step,
        len(manifest.files),
    )
    results = _resubmit_failed(engine, ctx, step_cfg, key)
    return StepOutcome(state=filtering.apply(results, step_cfg.sample), cache_hit=False)


def _run_full_step(
    ctx: StepContext,
    step_cfg: StepConfig,
    key: cache.StepKey,
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
        fingerprint=key.fingerprint,
    )

    # Built before submission, so a `target` that cannot be resolved says so before the step
    # spends a batch — and so round 1 and round 2 share one object, and one queue.
    round2 = nms.child_round(nms_engine, ctx) if nms_engine is not None else None

    logger.info("step %d: submitting %d round-1 job(s)", step_cfg.step, len(inputs.files))
    # Submission, parsing, the convergence retries and the NMS fan-out are one call rather
    # than four phases: a structure that fails to converge is re-run — and one that needs
    # displacing gets its children — as soon as its own job frees a slot, instead of after
    # the whole batch has drained. (No "parsing outputs" log line: parsing is interleaved
    # with submission, so a phase banner would be a lie. The count above is round 1's, for
    # the same reason — it does not bound what the step submits.)
    successes, failures = lifecycle.run_with_retries(engine, ctx, inputs, children=round2)

    if nms_engine is not None and round2 is not None:
        # Both come from the same condition; the second test is what narrows `round2`.
        logger.info("step %d: resolving normal-mode sampling", step_cfg.step)
        resolution = nms.run_nms(
            nms_engine,
            StepResults(structures=tuple(successes)),
            failures,
            ctx,
            round2=round2,
        )
        successes, failures = list(resolution.survivors), list(resolution.failures)
    results = lifecycle.finalize(engine, ctx, step_cfg, key, successes, failures)
    return StepOutcome(state=filtering.apply(results, step_cfg.sample), cache_hit=False)


def _run_artifact_step(
    ctx: StepContext, step_cfg: StepConfig, key: cache.StepKey, engine: ArtifactEngine
) -> StepOutcome:
    """Run a step that submits one job and produces one file.

    The same three lifecycle calls every other step makes, in the same order and with the
    manifest stamped between the first two — but without the per-structure ledger, which has
    nothing to say here: there is one job, it is not a structure, and the structures this step
    reports are the ones it was given.

    ``prepare`` writes the job's inputs and ``submit`` runs it through the ordinary scheduler,
    so an artifact step is throttled, headed, scratch-run and copied back like anything else.

    The archive is what makes :func:`_finish_artifact_step`'s success test mean anything. That
    test is ``artifact.exists()``, which cannot tell this run's product from the last one's —
    so a re-run (a changed template, a changed dataset) whose job dies having written nothing
    would find the *previous* model, digest those bytes into the sidecar and cache them under
    the **new** fingerprint. The consuming step then hashes the same stale file and cache-hits
    too, leaving a run that is internally consistent and describes a training that never
    happened. Moving the run directory aside first turns that into the failure it is.
    """
    attempts.archive_previous(ctx.step_dir, (ids.TRAINING_ID,))
    inputs = engine.prepare(ctx)
    cache.save_manifest(
        inputs,
        ctx.step_dir,
        operation=step_cfg.operation,
        engine=step_cfg.engine,
        fingerprint=key.fingerprint,
    )
    engine.submit(inputs, ctx)
    return _finish_artifact_step(ctx, step_cfg, key, engine)


def _finish_artifact_step(
    ctx: StepContext, step_cfg: StepConfig, key: cache.StepKey, engine: ArtifactEngine
) -> StepOutcome:
    """Decide an artifact step's outcome from its product, and cache it if there is one.

    **The product is the success test, and a missing one raises rather than ledgers.** A job
    that leaves the scheduler's queue having written nothing looks exactly like one that
    worked, so without this the step caches a model it never produced and every later
    ``resume`` serves that cache. Raising leaves no cache at all, which is what makes the
    recovery correct: ``resume`` finds nothing to reuse and runs the step again.

    A ledgered failure would not do — it writes a cache with zero structures, which is the
    same wrong state by a longer route. There is nothing per-structure to ledger anyway.

    Shared with :func:`rebuild_cache_step` so a product whose job finished before the driver
    died can be adopted without recomputing it — for a training that ran for days, that is the
    difference between a rebuild and a week.
    """
    artifact = engine.artifact(ctx)
    if not artifact.exists():
        raise JobFailureError(
            f"step {step_cfg.step} ({step_cfg.engine}) produced no {artifact.name}: "
            f"the job finished without writing {artifact}. Its log is under {ctx.step_dir}; "
            f"fix the cause and re-run — nothing was cached, so `chemrefine resume` will "
            f"redo this step."
        )
    logger.info("step %d: produced %s", step_cfg.step, artifact)
    results = engine.parse(StepInputs(files=()), ctx)
    results = lifecycle.finalize(engine, ctx, step_cfg, key, list(results.structures), [])
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
    if not step_cfg.halts_on_failure:
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
    Backs ``chemrefine rebuild-cache`` and ``chemrefine rebuild-nms``, which differ only in
    which step they aim at.

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
    engine = get_engine(step_cfg.engine)
    ctx = build_context(config, step_cfg, prev_state, engine)
    key = cache.StepKey.of(step_cfg, prev_state.structures, ctx.template)
    manifest = cache.load_manifest(ctx.step_dir)
    if manifest is None:
        # Named for what is missing rather than for the command that asked: `rebuild-cache`
        # and `rebuild-nms` both arrive here.
        raise CacheError(
            f"step {step_cfg.step}: nothing to rebuild from — no manifest on disk, so there "
            f"is no record of which output belongs to which structure"
        )
    stamped = cache.load_manifest_fingerprint(ctx.step_dir)
    if stamped and stamped != key.fingerprint:
        raise CacheError(
            f"step {step_cfg.step}: the outputs on disk were produced for a different "
            f"configuration — its template, options or upstream results have changed "
            f"since — so re-parsing them would cache results this configuration never "
            f"produced. Run `chemrefine rerun {step_cfg.step}` to recompute it."
        )
    if isinstance(engine, ArtifactEngine):
        # Past the same fingerprint guard as any other rebuild — the product on disk has to
        # belong to *this* configuration — but there are no per-structure outputs to re-parse
        # beyond it. This is the most valuable recovery the command offers for such a step: a
        # training whose job finished before the driver died is re-cached rather than re-run.
        logger.info("step %d: rebuilding cache from the product on disk", step_cfg.step)
        return _finish_artifact_step(ctx, step_cfg, key, engine)
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
    results = lifecycle.finalize(engine, ctx, step_cfg, key, successes, failures)
    return StepOutcome(state=filtering.apply(results, step_cfg.sample), cache_hit=False)


def _resubmit_failed(
    engine: CalculationEngine,
    ctx: StepContext,
    step_cfg: StepConfig,
    key: cache.StepKey,
) -> StepResults:
    """Re-run whatever the manifest has no usable result for, then re-cache the step.

    Which structures those are is not passed in but read off the outputs, by
    :func:`~chemrefine.lifecycle.resubmit_unusable` — one parse that both selects the work and
    supplies the results for everything it did not select. A ledger of what failed *last* time
    would be a second opinion about the same tree, and the two disagree exactly when it
    matters: a run killed while a re-run was being prepared leaves an output that exists and
    is worthless, which no ledger records and ``out.is_file()`` cannot see.

    The whole step is re-parsed, so fan-out lineage stays consistent and the ledger is
    refreshed (cleared if all now succeeded). Convergence failures are handled after, by
    :func:`~chemrefine.lifecycle.retry_unconverged`, from the geometry they reached. NMS steps
    use :func:`chemrefine.nms.reattempt_nms` instead (they reuse round-1 rather than
    resubmit it).
    """
    manifest = cache.load_manifest(ctx.step_dir)
    if manifest is None:
        raise CacheError(f"step {step_cfg.step}: cannot rerun — no manifest to rehydrate inputs")
    successes, failures = lifecycle.resubmit_unusable(engine, ctx, manifest)
    successes, failures = lifecycle.retry_unconverged(engine, ctx, successes, failures)
    return lifecycle.finalize(engine, ctx, step_cfg, key, successes, failures)

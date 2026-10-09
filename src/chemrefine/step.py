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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import assert_never

from chemrefine import attempts, cache, filtering, ids, io, job_log, lifecycle, nms, slurm
from chemrefine.config import (
    STEP_OPTION_PATHS,
    Config,
    StepConfig,
    is_failure_policy,
    policy_stores_backfills,
)
from chemrefine.engines import BatchPlan
from chemrefine.engines.api import (
    ArtifactEngine,
    AuxFileConsuming,
    CalculationEngine,
    JobExecutable,
    NmsCapableEngine,
    OptionsDeclaring,
    TemplateChecking,
    TemplateDriven,
    WhitespacePathIntolerant,
    get_engine,
    operation_notice,
    require_whitespace_free,
    resolve_operation,
)
from chemrefine.errors import (
    CacheError,
    ChemRefineError,
    JobFailureError,
    NoUsableCacheError,
)
from chemrefine.state import (
    JobTriple,
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
    later reader — ``prepare``, an engine's core count, the NMS input probe, the cache —
    takes it off the context instead of re-deriving it.

    The operation the step's outputs are read as is resolved here too, once
    (:func:`~chemrefine.engines.api.resolve_operation`): the cache key and every parse take
    it off the context, so no output re-reads the template to learn it. A template that
    cannot be read infers nothing, so the context still builds; the template is refused by
    name where its text is needed.
    """
    template = _template_for(config, step_cfg, engine)
    return StepContext(
        step_cfg=step_cfg,
        step_dir=step_dir_for(config, step_cfg),
        template_dir=config.template_dir.resolve(),
        template=template,
        operation=resolve_operation(engine, step_cfg, template),
        scratch_dir=config.scratch_dir.resolve() if config.scratch_dir is not None else None,
        prev_state=prev_state,
        charge=step_cfg.effective_charge(config.charge),
        multiplicity=step_cfg.effective_multiplicity(config.multiplicity),
        max_cores=config.max_cores,
        slurm_template=config.slurm_template,
        executables=config.executables,
        max_gpus=config.max_gpus,
        slurm_array=config.slurm_array,
        dispatch=config.dispatch,
        job_timeout_seconds=config.job_timeout_seconds,
    )


def check_step_dir(engine: CalculationEngine, ctx: StepContext) -> None:
    """Refuse a step whose engine cannot express the path its directory resolves to.

    The rule of :class:`~chemrefine.engines.api.WhitespacePathIntolerant`
    (:func:`~chemrefine.engines.api.require_whitespace_free`), asked of the *resolved*
    directory every path such an engine writes into its input descends from — so a step
    late in a run is refused before the steps ahead of it have run and been paid for, not
    when it comes to write its first input. Any other engine takes any directory.
    """
    if isinstance(engine, WhitespacePathIntolerant):
        require_whitespace_free(
            ctx.step_dir,
            what=f"step {ctx.step_cfg.step}'s directory",
            reason=engine.whitespace_path_reason,
        )


def preflight_batches(config: Config, steps: Iterable[StepConfig], *, local: bool) -> None:
    """Everything that would stop each step's jobs before they run, checked before any step.

    For every step in ``steps``, on the context the step will run with, less the structures
    neither check reads: the directory its inputs are written under, held to what its engine
    can express (:func:`check_step_dir`); its input template, read as its ``prepare`` reads it
    (:func:`chemrefine.ids.require_template`) and held to the engine's own template rules
    (:class:`~chemrefine.engines.api.TemplateChecking`), with what the run should say about
    the operation the step's outputs are read as said once, here
    (:func:`~chemrefine.engines.api.operation_notice`); an ``nms: true`` step's knobs read
    (:func:`chemrefine.nms.read_options`) and its input held to computing frequencies
    (:func:`chemrefine.nms.require_frequencies`); then, for a step the scheduler runs, its
    whole batch plan — SLURM header, job shape, GPU budget
    (:meth:`chemrefine.engines.BatchPlan.of`) — under this run's ``max_cores``, dispatch and
    resolved GPU budget.
    """
    budget = slurm.resolve_gpu_budget(config.max_gpus, local=local)
    for step_cfg in steps:
        engine = get_engine(step_cfg.engine)
        ctx = build_context(config, step_cfg, PipelineState(), engine)
        check_step_dir(engine, ctx)
        if isinstance(engine, TemplateDriven):
            template = ids.require_template(ctx.template, label=engine.label)
            if isinstance(engine, TemplateChecking):
                engine.check_template(ctx, template)
            if (notice := operation_notice(engine, step_cfg, template)) is not None:
                logger.warning("%s", notice)
        if step_cfg.nms and isinstance(engine, NmsCapableEngine):
            nms.read_options(step_cfg)
            nms.require_frequencies(engine, ctx)
        if isinstance(engine, JobExecutable):
            BatchPlan.of(engine, ctx, local=local, gpu_budget=budget)


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
        reason :meth:`may_submit` is one: separate questions are asked about
        :class:`StepMode` — may it submit, does it go through ``run_step``, can it halt —
        and each answered in a different module means adding a mode is a search
        rather than a compiler error. The exhaustive ``match`` — every member named, the
        wildcard arm holding only :func:`typing.assert_never` — is what makes it the
        compiler error, in every predicate: a new member stops narrowing to
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
    """Last step this plan *targets*; ``None`` runs the whole pipeline normally.

    ``rebuild-cache N`` re-parses step N from outputs already on disk and promises to
    submit nothing — the steps after N are not its business to *compute*. But the
    cumulative report (``steps.csv``) is rewritten from step 1 on every run, so ending the
    run at N would silently drop the later steps' rows even when their caches are still
    valid. The steps past ``stop_after`` therefore run **best-effort**
    (:meth:`best_effort`): still ``CACHE_ONLY``, still submitting nothing and needing no
    backend, but a step whose cache the current configuration cannot serve ends the run
    quietly (:class:`~chemrefine.errors.NoUsableCacheError`) instead of failing it. The
    report then covers exactly what the current configuration can vouch for — complete
    when the rebuild changed nothing, honestly cut where validity ends when it did.
    """

    def for_step(self, step: int) -> StepMode:
        """The mode this step runs in."""
        return self.overrides.get(step, self.default)

    def best_effort(self, step: int) -> bool:
        """Whether ``step`` is past the plan's target and runs as best-effort reporting.

        Past the target, a cache the configuration cannot serve is an ordinary place for
        the report to end — the pipeline stops there without error. At or before the
        target, and everywhere on an unscoped plan, the same miss is a real failure and
        must raise: ``rerun-errors`` over a broken earlier step has to say so, not
        silently report a shorter pipeline.
        """
        return self.stop_after is not None and step > self.stop_after


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepOutcome:
    """What :func:`run_step` returns — the survivors plus whether the cache was hit."""

    state: PipelineState
    cache_hit: bool


def _step_outcome(
    ctx: StepContext, step_cfg: StepConfig, results: StepResults, *, cache_hit: bool
) -> StepOutcome:
    """Filter ``results``, write the step's two ensemble XYZ files, return the outcome.

    **The one way a step's results become a :class:`StepOutcome`.** Every route out of
    :func:`run_step` — the fresh run, the cache hit, the policy re-attempt, the incremental
    the partial-step resume, the artifact step and ``rebuild-cache`` — ends here, which is
    what guarantees the ensemble files exist on a resumed, relocated or rebuilt tree and
    not only on the run that computed the results. Their content is a pure function of the
    results (stable energy sort over the cache's manifest order), so a cache-hit rewrite is
    byte-identical rather than churn.

    Two files, because a step has two survivor sets worth seeing as geometry: the full
    parsed results (``stepN_ensemble.xyz`` — what the cache holds, pre-filter) and what the
    ``sample:`` filter kept (``stepN_survivors.xyz`` — what feeds the next step). Both are
    ordered and captioned by the step's own ranking energy
    (:func:`chemrefine.filtering.ranking_energy`), the same energy ``steps.csv`` reports.

    An artifact step passes its structures through unchanged, so its
    ensemble duplicates the previous step's geometries under this step's number —
    deliberate uniformity: every step directory answers "what did this step end with" the
    same way.
    """
    state = filtering.apply(results, step_cfg.sample)
    ranking = filtering.ranking_energy(step_cfg.sample)
    io.write_ensemble_xyz(
        results.structures,
        ids.step_ensemble_path(ctx.step_dir, step_cfg.step),
        step=step_cfg.step,
        energy_attr=ranking.attr,
        energy_label=ranking.label,
    )
    io.write_ensemble_xyz(
        state.structures,
        ids.step_survivors_path(ctx.step_dir, step_cfg.step),
        step=step_cfg.step,
        energy_attr=ranking.attr,
        energy_label=ranking.label,
    )
    return StepOutcome(state=state, cache_hit=cache_hit)


def derive_step_key(
    ctx: StepContext, step_cfg: StepConfig, engine: CalculationEngine
) -> cache.StepKey:
    """Assemble the readings :meth:`chemrefine.cache.StepKey.of` keys a step by.

    The values are the run's own, not re-interpretations: charge and multiplicity are
    the **effective** ones off the context (the values jobs render — hashing the
    per-step override let a workflow-level edit change every job while every
    fingerprint stood still); the engine options are the engine's declared model's
    resolved reading (``{}`` for a non-declaring engine — an undeclared key can reach
    no job); the aux files are the engine's own enumeration of what its template
    references (empty for an engine whose templates name none); the resolution spec is
    :class:`~chemrefine.nms.NmsOptions`' validated reading, split criterion/search,
    present exactly when this step resolves. Living here rather than in
    :mod:`chemrefine.cache` keeps that module free of engine and NMS knowledge — it
    keys values it does not interpret.

    The operation and the NMS target are keyed **as the step resolves them**, not as the
    config spells them: the operation off the context
    (:attr:`~chemrefine.state.StepContext.operation`) — the value every parse of the step
    reads — and the target :func:`chemrefine.nms.resolved_options` resolves by. That is what
    selects the reader and decides a child resolved, so spelling out what the template
    implies moves no key, and a change in what it infers does. A template that is not there
    infers nothing: its digest is ``""`` (:func:`chemrefine.cache.template_digest`), so
    the step misses its cache anyway, and the actionable error arrives at rendering.

    A path-valued option (:data:`chemrefine.config.STEP_OPTION_PATHS`) is keyed by its
    **basename**. The loader resolves it to an absolute path so the job can find the file;
    keyed by that path, the same config over the same model bytes would derive a different
    row key at every directory it runs from, and a tree copied off a cluster would miss its
    cache on ``resume`` and be refused by ``rebuild-cache`` as a different configuration. The
    file's *bytes* are pinned by :func:`chemrefine.cache.option_file_digests` under the
    same option name, so the basename is all the key needs from the string — enough to
    tell two same-byte files with different names apart, and nothing that names a machine.
    A field that chooses where or how the job runs rather than what it computes — marked
    :data:`~chemrefine.engines._options.UNKEYED` where its model declares it — is not keyed
    at all.
    """
    engine_options: Mapping[str, object] = {}
    if isinstance(engine, OptionsDeclaring):
        model = engine.options_cls.from_raw_lenient(step_cfg.options)
        dumped = model.model_dump(mode="json", exclude=model.unkeyed())
        engine_options = {
            key: (
                Path(value).name if key in STEP_OPTION_PATHS and isinstance(value, str) else value
            )
            for key, value in dumped.items()
        }
    aux_files: Mapping[str, Path] = {}
    if isinstance(engine, AuxFileConsuming) and ctx.template is not None:
        aux_files = engine.template_aux_files(ctx.template)
    resolution = None
    if step_cfg.nms and isinstance(engine, NmsCapableEngine):
        readable = ctx.template is None or io.file_there(ctx.template)
        options = nms.resolved_options(engine, ctx) if readable else nms.read_options(step_cfg)
        resolution = options.resolution_spec()
    return cache.StepKey.of(
        step_cfg,
        ctx.prev_state.structures,
        ctx.template,
        operation=ctx.operation,
        charge=ctx.charge,
        multiplicity=ctx.multiplicity,
        engine_options=engine_options,
        resolution=resolution,
        aux_files=aux_files,
    )


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
    # Derived once, here, and passed down as a value: every route below needs the same
    # answer to "would this configuration have written the cache on disk". See
    # `cache.StepKey`.
    key = derive_step_key(ctx, step_cfg, engine)
    with io.tree_write(ctx.step_dir):
        ctx.step_dir.mkdir(parents=True, exist_ok=True)
    # Narrow once, here, instead of asserting the capability again with a cast at
    # each place that needs it. `nms_engine is not None` then carries
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
    if may_submit and nms_engine is not None:
        # Asked by the walk before the run started, and again here, before any route below
        # can submit — the cached re-attempt included: the template is read again at the
        # step's turn, so an edit made since meets the same rule.
        nms.require_frequencies(nms_engine, ctx)

    if mode is not StepMode.EXECUTE:
        cached = _cached_outcome(
            ctx, step_cfg, key, engine, nms_engine=nms_engine, may_submit=may_submit
        )
        if cached is not None:
            return cached
        # No cache this key can serve — but possibly a tree that can vouch for itself: an
        # interrupted run, parents that only partially changed, an NMS search retune, the
        # nms flag flipped over a finished non-NMS run, or an artifact step whose job
        # finished with no driver left to cache its product.
        incremental = _incremental_step_outcome(
            ctx,
            step_cfg,
            key,
            engine,
            nms_engine=nms_engine,
            artifact_engine=artifact_engine,
            may_submit=may_submit,
        )
        if incremental is not None:
            return incremental

    if not may_submit:
        # The two ways out cost very different things, and which one applies is decidable
        # here: the manifest's stamped fingerprint against the key in hand — the same
        # provenance test `rebuild_cache_step` and `_incremental_step_outcome` use. A match
        # means the outputs on disk were produced for exactly this configuration and only
        # the cache document cannot serve it, so `rebuild-cache` re-adopts them without
        # recomputing — which for an NMS step is what `resume` cannot promise.
        # A mismatch means `rebuild-cache` would refuse by that same guard, so only the
        # commands that recompute are honest advice. The attempts are asked the same way:
        # a resume killed before its resolution pass leaves a current fingerprint over a
        # carried-forward search stamp, and `rebuild-cache` would refuse those attempts.
        provenance = cache.load_manifest_provenance(ctx.step_dir)
        if provenance.fingerprint == key.fingerprint and not _attempts_foreign(
            step_cfg, engine, provenance, key
        ):
            raise NoUsableCacheError(
                f"step {step_cfg.step} has no cache this configuration can use, and "
                f"`{mode.value}` does not submit work for a step it is not targeting. "
                f"Its outputs on disk still match this configuration, so "
                f"`chemrefine rebuild-cache {step_cfg.step}` re-adopts them without "
                f"recomputing anything."
            )
        raise NoUsableCacheError(
            f"step {step_cfg.step} has no cache this configuration can use, and "
            f"`{mode.value}` does not submit work for a step it is not targeting. "
            f"Run `chemrefine resume` to bring it up to date, or "
            f"`chemrefine rerun {step_cfg.step}` to redo it."
        )
    if artifact_engine is not None:
        return _run_artifact_step(ctx, step_cfg, key, artifact_engine)
    return _run_full_step(ctx, step_cfg, key, engine, nms_engine=nms_engine)


def _policy_conflict(stored: str, current: StepConfig) -> bool:
    """Whether results finalized under ``stored`` cannot be served to ``current``.

    The fingerprint deliberately excludes ``on_failure`` — like ``sample:``, it shapes the
    *output* rather than the calculations — but unlike the filter, the policy is applied
    **before** :func:`chemrefine.cache.save`, so what is on disk already wears one policy's
    shape. Which policies persist the backfilled failures is
    :func:`chemrefine.config.policy_stores_backfills`'s answer, held exhaustive there; a
    change across that line would hand the user the previous policy's survivor set —
    silently, since the fingerprint still matches. A stored policy this version does not
    know has a shape it cannot judge, so it conflicts with every current policy.

    Only :func:`_cached_outcome` asks, and only when the failure ledger is non-empty: with
    no failures, every policy produces identical results and any edit is a free hit.
    """
    if not is_failure_policy(stored):
        return True
    return policy_stores_backfills(stored) != current.stores_backfills


def _attempts_foreign(
    step_cfg: StepConfig,
    engine: CalculationEngine,
    provenance: cache.ManifestProvenance,
    key: cache.StepKey,
) -> bool:
    """Whether the ``attemptK/`` children on disk were displaced under another search.

    The resolution's own stamp, held to the artifact rule: row keys exclude the NMS
    resolution by design, so no row check can see a retune. The *search* half is the one
    an attempt cannot survive — a changed displacement_value/seed/num_random_displacements
    changes the displaced geometries while the child ids stay the same, so re-parsing
    such an attempt would cache an exploration this configuration never ran. A
    *criterion* retune stays adoptable on purpose — it never moves a child's geometry,
    and re-reading the attempt under a new target is the very thing ``rebuild-nms``
    offers. An empty stored key is unprovable, never proven wrong — the row doctrine.

    One predicate for the two places that answer it: :func:`rebuild_cache_step`'s
    refusal, and :func:`run_step`'s advice on which command to run next, which must
    never name a rebuild that this refusal would then turn away.
    """
    return (
        step_cfg.nms
        and isinstance(engine, NmsCapableEngine)
        and bool(provenance.search_key)
        and provenance.search_key != key.search_key
    )


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

    A ledgered step whose ``on_failure`` moved across the storage line
    (:func:`_policy_conflict`) takes the same re-attempt route: the stored results wear
    the previous policy's shape, and re-attempting the failures then re-finalizing under
    the *current* config is what makes every direction of the edit honest — ``best``
    backfills what still fails, ``skip`` drops it, ``stop`` re-ledgers it — without ever
    recomputing a success. A mode that may not submit cannot make that repair, so it
    falls through to :func:`run_step`'s "no cache this configuration can use" error
    rather than serving the wrong survivor set.
    """
    cached = cache.load_if_valid(key=key, step_dir=ctx.step_dir)
    if cached is None:
        return None
    failed = cache.load_failure_records(ctx.step_dir)
    stale_policy = _policy_conflict(cached.on_failure, step_cfg)
    if failed and stale_policy and not may_submit:
        return None
    if failed and (step_cfg.leaves_failures_pending or stale_policy) and may_submit:
        results = (
            nms.reattempt_nms(nms_engine, ctx, step_cfg, cached, key)
            if nms_engine is not None
            else _resubmit_failed(engine, ctx, step_cfg, key)
        )
        return _step_outcome(ctx, step_cfg, results, cache_hit=False)
    logger.info(
        "step %d: cache hit, reusing %d structures",
        step_cfg.step,
        len(cached.results.structures),
    )
    return _step_outcome(ctx, step_cfg, cached.results, cache_hit=True)


def _prepare_stamped(
    engine: CalculationEngine,
    ctx: StepContext,
    step_cfg: StepConfig,
    key: cache.StepKey,
    *,
    archive: Iterable[str],
    adopted: Iterable[JobTriple] = (),
    stamp: cache.ManifestStamp | None = None,
) -> StepInputs:
    """Archive ``archive``, prepare what is not ``adopted``, stamp the manifest — the head
    every submitting route shares.

    In that order and no other. The previous artifacts in the named directories leave
    *before* the manifest is stamped with the keys that condemn them: stamped first, a
    driver killed during the submission pass left new row provenance vouching for old
    outputs still at canonical, and the next resume adopted a stale output (parse-usable,
    answering the previous parent's geometry) as the current parent's result — the
    internally-consistent wrong state the provenance exists to prevent. Archived away, the
    same crash re-reads as MISSING_OUTPUT and the row is resubmitted. The inputs are
    prepared before the manifest is written, so the manifest describes files that exist for
    every row it prepares; the stamp is on disk before any job goes out, so an interrupted
    run leaves proof of what its outputs were computed for — the per-row provenance is that
    proof at structure grain, for the incremental resume, and the step stamp is that proof
    for an artifact step's one product (:func:`_adopt_product`).

    ``adopted`` are the jobs a route takes from the manifest on disk instead of preparing
    them: the incremental resume's matching rows. They enter the manifest as they stand,
    because the input beside an adopted output belongs to the job that wrote it, and that
    need not be this context's rendering of the parent: a convergence retry starts from the
    geometry the first attempt reached, a promoted NMS winner is a displaced child's whole
    calculation, and a tree that moved, or a core budget that changed, renders other bytes
    under the same row key. Prepared again, the directory would pair an output with an
    input that never produced it. Every parent ``adopted`` does not name is prepared, and
    the manifest lists both in the parents' order (a job that is no parent's row, an
    artifact step's one job, follows them).

    Archiving happens here rather than inside ``prepare`` because the retry and NMS paths
    call ``prepare`` too and manage their own attempt directories. ``stamp`` is the full
    current stamp unless a route must carry a stored half forward: the incremental resume
    keeps the stored search key until the attempts it describes exist.
    """
    attempts.archive_previous(ctx.step_dir, archive)
    kept = {job[2]: job for job in adopted}
    parents = ctx.prev_state.structures
    rest = PipelineState(structures=tuple(s for s in parents if s.id not in kept))
    prepared = engine.prepare(replace(ctx, prev_state=rest))
    position = {s.id: i for i, s in enumerate(parents)}
    inputs = StepInputs(
        files=tuple(
            sorted(
                (*kept.values(), *prepared.files),
                key=lambda job: position.get(job[2], len(position)),
            )
        )
    )
    cache.save_manifest(
        inputs,
        ctx.step_dir,
        operation=key.operation,
        engine=step_cfg.engine,
        stamp=stamp if stamp is not None else key.manifest_stamp(),
    )
    return inputs


def _restamp(
    inputs: StepInputs, ctx: StepContext, step_cfg: StepConfig, key: cache.StepKey
) -> None:
    """Write the full current stamp over ``inputs``' manifest, after ``finalize``.

    Cache first (``finalize``), then the manifest's provenance: a crash between them leaves
    a current cache beside an unprovenanced manifest, which reads as "rebuild again" —
    cheap; the other order would leave provenance vouching for a cache that was never
    written.
    """
    cache.save_manifest(
        inputs,
        ctx.step_dir,
        operation=key.operation,
        engine=step_cfg.engine,
        stamp=key.manifest_stamp(),
    )


def _incremental_step_outcome(
    ctx: StepContext,
    step_cfg: StepConfig,
    key: cache.StepKey,
    engine: CalculationEngine,
    *,
    nms_engine: NmsCapableEngine | None,
    artifact_engine: ArtifactEngine | None,
    may_submit: bool,
) -> StepOutcome | None:
    """Continue a step from its provenanced manifest, computing only the rows that need it.

    The one rule: a row is **adopted** — its output re-parsed from disk — iff its stored
    row key equals the key this configuration derives for the same parent; every other
    current parent's row is computed. That serves the interrupted step (all rows match,
    only the unusable resubmit), the partially changed parent set (exactly the changed
    rows run), and any config edit that provably reaches no job (nothing runs at all) —
    where the alternative, :func:`_run_full_step`, archives every finished ``.out`` and
    resubmits the lot.

    The row provenance is what makes continuing *safe* rather than merely cheap: a
    stored key that matches proves the output on disk is the one this configuration
    would compute for this parent — the distinction an output's presence cannot make, and
    the reason archiving is otherwise unconditional. Stale rows (changed or new
    parents) are handed to :func:`~chemrefine.lifecycle.resubmit_unusable` as condemned
    sight-unseen: their outputs may parse perfectly and still answer a different
    geometry.

    An artifact step has one job and no rows, so it is continued at step grain: the
    manifest's step stamp is its provenance, and :func:`_adopt_product` adopts its product
    whole or answers ``None``.

    A manifest **without** provenance at its grain (no row keys; for an artifact step, no
    stamp) is a tree from before the current rules — unprovable, never wrong — and resume
    refuses it loudly, naming the adoption command, rather than silently archiving
    finished work (or, worse, trusting it): ``chemrefine rebuild-cache`` adopts it under
    the current rules, submits nothing, and records the provenance this route needs.

    Returns ``None`` — meaning "run the whole step" — when the route does not apply:

    * The step may not submit: a mode that submits nothing is served from a cache or not
      at all, and adopting from disk is a resume's or ``rebuild-cache``'s to do.
    * No manifest, or one naming no jobs — a step that never prepared anything has
      nothing to continue (an empty manifest is a legitimate value, not a missing one).
    * An artifact step whose product is another configuration's, or is not a finished
      job's (:func:`_adopt_product`).
    """
    if not may_submit:
        return None
    manifest = cache.load_manifest(ctx.step_dir)
    if manifest is None or not manifest.files:
        return None
    provenance = cache.load_manifest_provenance(ctx.step_dir)
    if not (provenance.fingerprint if artifact_engine is not None else provenance.rows):
        raise CacheError(
            f"step {step_cfg.step}: its manifest carries no provenance — a tree from before "
            f"the current cache rules, whose outputs resume can neither prove right nor "
            f"wrong. Run `chemrefine rebuild-cache {step_cfg.step}` to adopt it under the "
            f"current rules (it submits nothing), then resume — or `chemrefine rerun "
            f"{step_cfg.step}` to compute it again."
        )
    if artifact_engine is not None:
        return _adopt_product(ctx, step_cfg, key, artifact_engine, manifest, provenance)
    current = key.manifest_rows()
    changed = sorted(
        sid
        for sid in current
        if sid not in provenance.rows or provenance.rows[sid][0] != current[sid][0]
    )
    logger.info(
        "step %d: continuing from disk — %d row(s) adopted, %d to compute (changed or new)",
        step_cfg.step,
        len(current) - len(changed),
        len(changed),
    )
    # The search half of the stamp is carried forward from the stored one: it describes
    # the `attemptK/` children on disk, and those are still the previous submission's until
    # the resolution pass below replaces them. Stamped current here, a driver killed
    # anywhere in the hours between this write and that pass would leave old-search attempts
    # under a current-search stamp — adoptable by `rebuild-cache`, whose search guard reads
    # exactly this stamp. The current one is written only once the attempts it describes
    # exist (after `finalize`, cache first, then provenance — `_restamp`). Rows and
    # fingerprint stamp early as every route's do: they describe round-1 outputs, which the
    # archive of the condemned rows makes honest.
    stamp = key.manifest_stamp()
    stamp["search_key"] = provenance.search_key
    # Only the condemned rows are archived and prepared, each into the directory its stale
    # artifacts just left (the resubmission knows they arrive pre-archived). An adopted row
    # is read back as it stands: it keeps the job the manifest records for it and the
    # inputs that job ran from.
    adopted = current.keys() - changed
    inputs = _prepare_stamped(
        engine,
        ctx,
        step_cfg,
        key,
        archive=changed,
        adopted=(job for job in manifest.files if job[2] in adopted),
        stamp=stamp,
    )
    successes, failures = lifecycle.resubmit_unusable(engine, ctx, inputs, stale=changed)
    successes, failures = lifecycle.retry_unconverged(engine, ctx, successes, failures)
    if nms_engine is not None:
        # Round 1 is assembled; resolution runs exactly as a full step's would, with its
        # fan-out started here. A parent already at the target wears the child its latest
        # attempt names only under the criterion that attempt's record was resolved under,
        # which the record says for itself; one not at the target fans out fresh children
        # — which is what makes flipping `nms: true` over a finished run cost exactly the
        # displacement children and nothing else.
        resolution = nms.run_nms(
            nms_engine, StepResults(structures=tuple(successes)), failures, ctx
        )
        successes, failures = list(resolution.survivors), list(resolution.failures)
    results = lifecycle.finalize(ctx, step_cfg, key, successes, failures)
    _restamp(inputs, ctx, step_cfg, key)
    return _step_outcome(ctx, step_cfg, results, cache_hit=False)


def _adopt_product(
    ctx: StepContext,
    step_cfg: StepConfig,
    key: cache.StepKey,
    engine: ArtifactEngine,
    manifest: StepInputs,
    provenance: cache.ManifestProvenance,
) -> StepOutcome | None:
    """Continue an artifact step by adopting its product; ``None`` runs its job instead.

    The row rule at the one grain an artifact step has, under the two proofs a product
    needs. It is **this configuration's**: :func:`_prepare_stamped` archives the run
    directory, then stamps, then the job goes out, so a product beside a stamp equal to
    this key was written by a job this configuration submitted, or adopted for it by
    ``rebuild-cache``. And its job **finished** (:func:`_unfinished`, the success test
    every route shares). With both, the product is cached with nothing submitted: a driver
    that stops while its job runs on, or a write the output tree refuses once the product
    exists, costs a resume and not the job again.

    With either missing the job runs, as a step with no manifest runs it. A stamp for
    another key is another configuration's product — a changed template, option or
    upstream ensemble. A product its job never finished is a file a later step must not
    load. :func:`_run_artifact_step` moves both aside, into the next ``attemptK/``.
    """
    if provenance.fingerprint != key.fingerprint:
        return None
    if (unfinished := _unfinished(engine, ctx, manifest)) is not None:
        logger.warning("step %d: %s — running the job again", step_cfg.step, unfinished.why)
        return None
    logger.info("step %d: adopting the product its finished job left", step_cfg.step)
    return _finish_artifact_step(ctx, step_cfg, key, engine, manifest)


def _run_full_step(
    ctx: StepContext,
    step_cfg: StepConfig,
    key: cache.StepKey,
    engine: CalculationEngine,
    *,
    nms_engine: NmsCapableEngine | None,
) -> StepOutcome:
    """Run the engine's full lifecycle (prepare → submit → parse → retry → nms/policy → cache)."""
    logger.info("step %d (%s): preparing inputs", step_cfg.step, step_cfg.engine)
    # Anything already in these structure dirs is a previous run's work: every parent's
    # directory is archived, so a job that dies without producing output is seen as a
    # failure rather than re-reading the old result.
    inputs = _prepare_stamped(
        engine, ctx, step_cfg, key, archive=(s.id for s in ctx.prev_state.structures)
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
    results = lifecycle.finalize(ctx, step_cfg, key, successes, failures)
    return _step_outcome(ctx, step_cfg, results, cache_hit=False)


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

    The archive is what makes the success test (:func:`_unfinished`) mean anything. A
    product and a runlog footer at the canonical path cannot say which submission left them
    — so a re-run (a changed template, a changed dataset) whose job dies having written
    nothing would find the *previous* run's model and its footer, digest those bytes into the
    sidecar and cache them under the **new** fingerprint. The consuming step then hashes the
    same stale file and cache-hits too, leaving a run that is internally consistent and
    describes a training that never happened. Moving the run directory aside first turns that
    into the failure it is: what is at the canonical path is this submission's, or nothing.

    The cached results go before the archive, for the product is no part of them: a cache
    says only that the model at the product's path is the one its key trained. Left behind,
    another configuration's cache would still match its key once the configuration is
    edited back, over a path this run has since moved that model away from.
    """
    cache.invalidate(ctx.step_dir)
    inputs = _prepare_stamped(engine, ctx, step_cfg, key, archive=(engine.run_dir(ctx).name,))
    engine.submit(inputs, ctx)
    return _finish_artifact_step(ctx, step_cfg, key, engine, inputs)


@dataclass(frozen=True)
class _Unfinished:
    """Why an artifact step's product is not a finished job's (:func:`_unfinished`)."""

    why: str
    """What is missing and where, as a refusal or a log line says it."""

    adoptable: bool
    """Whether a caller's word can stand in for what is missing. ``True`` when the product
    is there and its job did not say its run failed: it said nothing, or said only that
    its results did not all come home. ``False`` when there is no product, or the run
    itself failed — proven wrong, which no word mends."""


def _unfinished(engine: ArtifactEngine, ctx: StepContext, inputs: StepInputs) -> _Unfinished | None:
    """Why this step's product is not a finished job's; ``None`` when it is one.

    **The one success test of an artifact step**, asked by the run that just submitted the
    job (:func:`_run_artifact_step`), by a resume deciding whether to adopt
    (:func:`_adopt_product`) and by ``rebuild-cache`` (:func:`rebuild_cache_step`). Two
    things, both needed: the product is on disk, and the job that wrote it says it
    finished — in the footer of its own runlog (:func:`chemrefine.job_log.read_footer`),
    a run block that exited 0 and results that all came home.

    The product alone proves neither. A job that leaves the queue having written nothing
    looks, to the scheduler, like one that worked; and a job that was stopped, or whose
    program died, can leave a file at the product's path all the same — a best-so-far
    checkpoint written as the run went, or a copy the output tree cut short. The footer is
    the job's own account of which it was, and it is the one record of that which is still
    there when no driver was.

    ``inputs`` names the job, as :func:`_prepare_stamped` prepared it or the manifest
    recorded it: its runlog sits beside its output, named after its input
    (:func:`chemrefine.job_log.runlog_path`).
    """
    artifact = engine.artifact(ctx)
    if io.absent(artifact):
        return _Unfinished(f"the job finished without writing {artifact}", adoptable=False)
    silent = f"nothing says the job that wrote {artifact} finished"
    unvouched: _Unfinished | None = None
    if not inputs.files:
        unvouched = _Unfinished(f"{silent}: its manifest names no job", adoptable=True)
    for config, output, _id in inputs.files:
        runlog = job_log.runlog_path(output.parent, config.stem)
        footer = job_log.read_footer(runlog)
        if footer is None:
            unvouched = unvouched or _Unfinished(
                f"{silent}: {runlog} is missing or holds no footer of its last run",
                adoptable=True,
            )
        elif footer.exit_code != 0:
            return _Unfinished(
                f"the job that wrote {artifact} did not finish: {runlog} ends with "
                f"exit_code={footer.exit_code}",
                adoptable=False,
            )
        elif footer.scratch_kept:
            unvouched = unvouched or _Unfinished(
                f"the results of the job that wrote {artifact} did not all come home: "
                f"{runlog} ends with scratch_kept=true, and the scratch directory its "
                f"header names was left holding them",
                adoptable=True,
            )
    return unvouched


def _finish_artifact_step(
    ctx: StepContext,
    step_cfg: StepConfig,
    key: cache.StepKey,
    engine: ArtifactEngine,
    inputs: StepInputs,
    *,
    on_the_callers_word: bool = False,
) -> StepOutcome:
    """Decide an artifact step's outcome from its product, and cache a finished job's.

    **The success test is :func:`_unfinished`, and failing it raises rather than ledgers.**
    Without it the step caches a model its job never produced, or never finished, and every
    later ``resume`` serves that cache. Raising leaves no cache at all, which is what makes
    the recovery correct: ``resume`` finds nothing to reuse, and runs the job again unless
    what is on disk has by then been proven a finished job's (:func:`_adopt_product`).

    A ledgered failure would not do — it writes a cache with zero structures, which is the
    same wrong state by a longer route. There is nothing per-structure to ledger anyway.

    ``on_the_callers_word`` is ``rebuild-cache``'s: a product nothing vouches for whole —
    unprovable, never proven wrong — is adopted with a warning, as that command adopts a
    manifest without provenance. A run that failed is refused there as everywhere.

    The engine's ``parse`` reads the product to record it. A read the filesystem refuses
    there says nothing against the job, so it is :class:`~chemrefine.errors.CacheError`
    naming the file, with nothing cached: once the file can be read, a resume adopts the
    product (:func:`_adopt_product`) and the job does not run again.
    """
    unfinished = _unfinished(engine, ctx, inputs)
    if unfinished is not None and not (on_the_callers_word and unfinished.adoptable):
        raise JobFailureError(
            f"step {step_cfg.step} ({step_cfg.engine}): {unfinished.why}. Nothing was "
            f"cached; its log is under {ctx.step_dir}. `chemrefine resume` runs the job "
            f"again"
            + (
                f"; `chemrefine rebuild-cache {step_cfg.step}` adopts the product instead, "
                f"on your word that it is whole."
                if unfinished.adoptable
                else ", so fix the cause first."
            )
        )
    if unfinished is not None:
        logger.warning(
            "step %d: %s — adopting the product on this command's word",
            step_cfg.step,
            unfinished.why,
        )
    artifact = engine.artifact(ctx)
    logger.info("step %d: produced %s", step_cfg.step, artifact)
    try:
        results = engine.parse(StepInputs(files=()), ctx)
    except OSError as e:
        if e.errno is None:
            # No refusal of the filesystem's: an error of the engine's own, and its defect.
            raise
        raise CacheError(
            f"step {step_cfg.step} ({step_cfg.engine}): cannot read {e.filename or artifact}: "
            f"{e.strerror or e} — so the step's product was not cached. Restore read access, "
            f"then `chemrefine resume` (or the `rebuild-cache` that stopped here) adopts it "
            f"without running the job again."
        ) from e
    results = lifecycle.finalize(ctx, step_cfg, key, list(results.structures), [])
    return _step_outcome(ctx, step_cfg, results, cache_hit=False)


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

    Refuses when the manifest's **row provenance** disagrees with the current key —
    a row keyed to a different template, engine options, effective charge or parent
    content, or a parent set the rows do not cover exactly. Re-parsing such outputs
    would write a cache that is internally valid and describes a run that never
    happened, and the next ``resume`` would serve it rather than compute what was
    asked for.

    A manifest carrying **no** provenance at its grain — no row keys, or for an artifact
    step no stamp — is not evidence of a mismatch, and this proceeds: it is a hand-written
    v1 adoption manifest, *unprovable* rather than wrong, and the caller has named the
    step. The distinction is between proving the outputs wrong and merely being unable to
    prove them right. :func:`_incremental_step_outcome` draws the same line and stops at
    it: a resume refuses such a tree by name and names this command. A successful rebuild
    then writes the manifest back **with** its provenance — the row keys, and the step
    stamp an artifact step's product is proven this configuration's by — so the adoption
    is recorded.

    An artifact step's product is held to the success test every route shares
    (:func:`_unfinished`), drawn along that same line: a run that failed, or left no
    product, is proven wrong and refused; a product nothing vouches for whole — its job
    said nothing, or said its results did not all come home — is unprovable, and adopted
    on the caller's word with a warning.

    A template-driven step whose template is not there is refused first, by name, as
    the run's t=0 walk refuses it. Its key would derive over an empty template digest,
    and a rebuild would stamp every adopted row with that key — the next ``resume``,
    the template restored, would then condemn every one of them. Refused ahead of the
    provenance check, which would blame the configuration instead, and of the artifact
    branch, which would re-cache a product under the same key. An ``nms: true`` step whose
    input computes no frequencies is refused there too, as the walk refuses it
    (:func:`chemrefine.nms.require_frequencies`): its outputs carry no table to resolve
    from, and re-parsing them would ledger every structure unresolved.
    """
    engine = get_engine(step_cfg.engine)
    ctx = build_context(config, step_cfg, prev_state, engine)
    if isinstance(engine, TemplateDriven):
        ids.require_template(ctx.template, label=engine.label)
    if step_cfg.nms and isinstance(engine, NmsCapableEngine):
        nms.require_frequencies(engine, ctx)
    key = derive_step_key(ctx, step_cfg, engine)
    manifest = cache.load_manifest(ctx.step_dir)
    if manifest is None:
        # Named for what is missing rather than for the command that asked: `rebuild-cache`
        # and `rebuild-nms` both arrive here.
        raise CacheError(
            f"step {step_cfg.step}: nothing to rebuild from — no manifest on disk, so there "
            f"is no record of which output belongs to which structure"
        )
    provenance = cache.load_manifest_provenance(ctx.step_dir)
    if provenance.rows:
        current = key.manifest_rows()
        foreign = sorted(
            sid
            for sid, (stored_row, _digest) in provenance.rows.items()
            if sid not in current or current[sid][0] != stored_row
        )
        unproven = sorted(sid for sid in current if sid not in provenance.rows)
        if foreign or unproven:
            raise CacheError(
                f"step {step_cfg.step}: the outputs on disk were produced for a different "
                f"configuration — {len(foreign)} row(s) disagree with the current key and "
                f"{len(unproven)} current parent(s) have no row — so re-parsing them would "
                f"cache results this configuration never produced. "
                f"Run `chemrefine rerun {step_cfg.step}` to recompute it."
            )
    if _attempts_foreign(step_cfg, engine, provenance, key):
        # Resume already refuses to reuse such an attempt ("displaced from a round this
        # run never produced"); the explicit command must not adopt what resume refuses.
        raise CacheError(
            f"step {step_cfg.step}: the NMS children on disk were displaced under "
            f"different search settings (displacement_value / "
            f"num_random_displacements / seed changed), so re-parsing them would cache "
            f"an exploration this configuration never ran. `chemrefine resume` re-runs "
            f"just the displaced children under the current settings; "
            f"`chemrefine rerun {step_cfg.step}` redoes the whole step."
        )
    if isinstance(engine, ArtifactEngine):
        # An artifact step has one whole-set product and no per-structure rows, so the
        # step stamp is the right grain for its provenance — the row doctrine above can
        # never see it. A stamp that disagrees is proven wrong exactly as a foreign row
        # is; an absent one is unprovable and proceeds under the explicit command.
        if provenance.fingerprint and provenance.fingerprint != key.fingerprint:
            raise CacheError(
                f"step {step_cfg.step}: the product on disk was produced for a different "
                f"configuration, so re-caching it would describe a training that never "
                f"happened. Run `chemrefine rerun {step_cfg.step}` to recompute it."
            )
        # Stamped once the cache is written, as the row branch below is: a later resume
        # that finds no cache can then prove the product this configuration's. It still
        # asks the job's own word, so a product adopted here with none is trained again.
        logger.info("step %d: rebuilding cache from the product on disk", step_cfg.step)
        outcome = _finish_artifact_step(
            ctx, step_cfg, key, engine, manifest, on_the_callers_word=True
        )
        _restamp(manifest, ctx, step_cfg, key)
        return outcome
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
    results = lifecycle.finalize(ctx, step_cfg, key, successes, failures)
    _restamp(manifest, ctx, step_cfg, key)
    return _step_outcome(ctx, step_cfg, results, cache_hit=False)


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
    is worthless, which no ledger records and its presence cannot show.

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
    return lifecycle.finalize(ctx, step_cfg, key, successes, failures)

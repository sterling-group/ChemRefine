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

from pydantic import ValidationError

from chemrefine import attempts, cache, filtering, ids, io, lifecycle, nms
from chemrefine.config import Config, StepConfig
from chemrefine.engines.api import (
    ArtifactEngine,
    AuxFileConsuming,
    CalculationEngine,
    NmsCapableEngine,
    OptionsDeclaring,
    TemplateDriven,
    get_engine,
)
from chemrefine.errors import (
    CacheError,
    ChemRefineError,
    ConfigError,
    JobFailureError,
    NoUsableCacheError,
)
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

    An artifact step (``mlip-train``) passes its structures through unchanged, so its
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
    """
    engine_options: Mapping[str, object] = {}
    if isinstance(engine, OptionsDeclaring):
        engine_options = engine.options_cls.from_raw_lenient(step_cfg.options).model_dump(
            mode="json"
        )
    aux_files: Mapping[str, Path] = {}
    if isinstance(engine, AuxFileConsuming) and ctx.template is not None:
        aux_files = engine.template_aux_files(ctx.template)
    resolution = None
    if step_cfg.nms and isinstance(engine, NmsCapableEngine):
        try:
            dump = nms.NmsOptions.from_raw(step_cfg.options).model_dump(mode="json")
        except ValidationError as e:
            raise ConfigError(f"step {step_cfg.step}: invalid NMS options:\n{e}") from e
        resolution = cache.ResolutionSpec(
            criterion={k: dump[k] for k in ("target", "ts_mode_index")},
            search={k: dump[k] for k in ("displacement_value", "num_random_displacements", "seed")},
        )
    return cache.StepKey.of(
        step_cfg,
        ctx.prev_state.structures,
        ctx.template,
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
    # Derived once, here, and passed down as a value. Every route below needs the same
    # answer to "would this configuration have written the cache on disk", and each used
    # to re-derive it from whatever context it held. See `cache.StepKey`.
    key = derive_step_key(ctx, step_cfg, engine)
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

    if mode is not StepMode.EXECUTE:
        cached = _cached_outcome(
            ctx, step_cfg, key, engine, nms_engine=nms_engine, may_submit=may_submit
        )
        if cached is not None:
            return cached
        # No cache this key can serve — but possibly a tree whose rows can vouch for
        # themselves: an interrupted run, parents that only partially changed, an NMS
        # search retune, or the nms flag flipped over a finished non-NMS run.
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
        # recomputing — which for an NMS or artifact step is what `resume` cannot promise.
        # A mismatch means `rebuild-cache` would refuse by that same guard, so only the
        # commands that recompute are honest advice.
        if cache.load_manifest_fingerprint(ctx.step_dir) == key.fingerprint:
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


def _policy_conflict(stored: str, current: str) -> bool:
    """Whether results finalized under ``stored`` cannot be served as ``current``.

    The fingerprint deliberately excludes ``on_failure`` — like ``sample:``, it shapes the
    *output* rather than the calculations — but unlike the filter, the policy is applied
    **before** :func:`chemrefine.cache.save`, so what is on disk already wears one policy's
    shape. ``stop`` and ``skip`` both persist the successes alone, so they serve each
    other; ``best`` persists the backfilled failures too, so a change across that line
    hands the user the previous policy's survivor set — silently, since the fingerprint
    still matches. ``""`` is a cache written before the policy was recorded and is treated
    as serving any policy, so older caches are not stranded.

    Only :func:`_cached_outcome` asks, and only when the failure ledger is non-empty: with
    no failures, every policy produces identical results and any edit is a free hit.
    """
    if not stored:
        return False
    return (stored == "best") != (current == "best")


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
    stale_policy = _policy_conflict(cached.on_failure, step_cfg.on_failure)
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
    would compute for this parent — the distinction ``out.is_file()`` cannot make, and
    the reason archiving is otherwise unconditional. Stale rows (changed or new
    parents) are handed to :func:`~chemrefine.lifecycle.resubmit_unusable` as condemned
    sight-unseen: their outputs may parse perfectly and still answer a different
    geometry.

    A manifest **without** row provenance is a tree from before the current rules —
    unprovable, never wrong — and resume refuses it loudly, naming the adoption command,
    rather than silently archiving finished work (or, worse, trusting it):
    ``chemrefine rebuild-cache`` re-parses under the current rules, submits nothing, and
    records the provenance this route needs.

    Returns ``None`` — meaning "run the whole step" — when the route does not apply:

    * The step may not submit: continuing means resubmitting whatever is missing.
    * The step is an artifact step. There is nothing per-structure to continue —
      :func:`_run_artifact_step` re-runs the one job, and adopting a *finished* product
      is ``rebuild-cache``'s job, not a resume's.
    * No manifest, or one naming no jobs — a step that never prepared anything has
      nothing to continue (an empty manifest is a legitimate value, not a missing one).
    """
    if not may_submit or artifact_engine is not None:
        return None
    manifest = cache.load_manifest(ctx.step_dir)
    if manifest is None or not manifest.files:
        return None
    if nms_engine is not None:
        # Before any adoption: a template that computes no frequencies must fail with
        # the actionable error, not an all-unresolved ledger read off perfectly good
        # outputs.
        _check_nms_freq_gate(nms_engine, ctx, step_cfg)
    provenance = cache.load_manifest_provenance(ctx.step_dir)
    if not provenance.rows:
        raise CacheError(
            f"step {step_cfg.step}: outputs are on disk but carry no per-row provenance — "
            f"a tree from before the current cache rules, which resume can neither prove "
            f"right nor wrong. Run `chemrefine rebuild-cache {step_cfg.step}` to adopt it "
            f"under the current rules (it re-parses and submits nothing), then resume."
        )
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
    # The condemned rows' canonical artifacts leave BEFORE the manifest is stamped with
    # the keys that condemn them. Stamped first, a driver killed during the resubmission
    # pass left the new row provenance vouching for the old outputs still at canonical —
    # and the next resume computed `changed = {}`, adopting a stale output (parse-usable,
    # answering the previous parent's geometry) as the changed parent's result: the
    # internally-consistent wrong state this provenance exists to prevent. Archived away,
    # the same crash re-reads as MISSING_OUTPUT and the row is resubmitted.
    attempts.archive_previous(ctx.step_dir, changed)
    # Fresh inputs for the whole current set: an adopted row re-renders byte-identically,
    # a condemned row renders into the directory its stale artifacts just left (the
    # resubmission knows they arrive pre-archived). Prepared before the manifest write so
    # the manifest describes files that exist.
    inputs = engine.prepare(ctx)
    cache.save_manifest(
        inputs,
        ctx.step_dir,
        operation=step_cfg.operation,
        engine=step_cfg.engine,
        fingerprint=key.fingerprint,
        criterion_key=key.criterion_key,
        search_key=key.search_key,
        rows=current,
    )
    successes, failures = lifecycle.resubmit_unusable(engine, ctx, inputs, stale=changed)
    successes, failures = lifecycle.retry_unconverged(engine, ctx, successes, failures)
    if nms_engine is not None:
        # Round 1 is assembled; resolution runs exactly as a full step's would, with one
        # verdict from the provenance: `resolved_from` labels on disk are trusted only
        # under the criterion they were written for. Parents not at the target fan out
        # fresh children either way — which is what makes flipping `nms: true` over a
        # finished run cost exactly the displacement children and nothing else.
        trust = bool(key.criterion_key) and provenance.criterion_key == key.criterion_key
        resolution = nms.resume_nms(
            nms_engine,
            StepResults(structures=tuple(successes)),
            failures,
            ctx,
            trust_resolutions=trust,
        )
        successes, failures = list(resolution.survivors), list(resolution.failures)
    results = lifecycle.finalize(engine, ctx, step_cfg, key, successes, failures)
    return _step_outcome(ctx, step_cfg, results, cache_hit=False)


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
        # outputs were computed for — see :func:`_incremental_step_outcome`. The per-row
        # provenance is the same proof at structure grain, for the incremental resume.
        fingerprint=key.fingerprint,
        criterion_key=key.criterion_key,
        search_key=key.search_key,
        rows=key.manifest_rows(),
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

    The archive is what makes :func:`_finish_artifact_step`'s success test mean anything. That
    test is ``artifact.exists()``, which cannot tell this run's product from the last one's —
    so a re-run (a changed template, a changed dataset) whose job dies having written nothing
    would find the *previous* model, digest those bytes into the sidecar and cache them under
    the **new** fingerprint. The consuming step then hashes the same stale file and cache-hits
    too, leaving a run that is internally consistent and describes a training that never
    happened. Moving the run directory aside first turns that into the failure it is.
    """
    attempts.archive_previous(ctx.step_dir, (engine.run_dir(ctx).name,))
    inputs = engine.prepare(ctx)
    cache.save_manifest(
        inputs,
        ctx.step_dir,
        operation=step_cfg.operation,
        engine=step_cfg.engine,
        fingerprint=key.fingerprint,
        criterion_key=key.criterion_key,
        search_key=key.search_key,
        rows=key.manifest_rows(),
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
    return _step_outcome(ctx, step_cfg, results, cache_hit=False)


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

    Refuses when the manifest's **row provenance** disagrees with the current key —
    a row keyed to a different template, engine options, effective charge or parent
    content, or a parent set the rows do not cover exactly. Re-parsing such outputs
    would write a cache that is internally valid and describes a run that never
    happened, and the next ``resume`` would serve it rather than compute what was
    asked for.

    A manifest carrying **no** row provenance is not evidence of a mismatch, and this
    proceeds: it is a tree from before the current rules (or a hand-written v1
    adoption manifest), *unprovable* rather than wrong, and the caller has named the
    step. The distinction is between proving the outputs wrong and merely being unable
    to prove them right. It is drawn differently in :func:`_incremental_step_outcome`,
    which treats an unproven match as a reason to re-run — it can afford to, being an
    optimisation over doing the work anyway. A successful rebuild then writes the
    manifest back **with** row provenance, so the adoption is recorded and every later
    question is answered per row.
    """
    engine = get_engine(step_cfg.engine)
    ctx = build_context(config, step_cfg, prev_state, engine)
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
    if (
        step_cfg.nms
        and isinstance(engine, NmsCapableEngine)
        and provenance.search_key
        and provenance.search_key != key.search_key
    ):
        # The resolution's own stamp, held to the artifact rule below: row keys exclude
        # the NMS resolution by design, so the row check above cannot see a retune. The
        # *search* half is the one an attempt cannot survive — a changed
        # displacement_value/seed/num_random_displacements changes the displaced
        # geometries while the child ids stay the same, so a rebuild would parse outputs
        # answering displacements this configuration never asked for and cache them
        # under the new resolution's fingerprint. Resume already refuses to reuse such
        # an attempt ("displaced from a round this run never produced"); the explicit
        # command must not adopt what resume refuses. A *criterion* retune stays
        # adoptable on purpose — it never moves a child's geometry, and re-reading the
        # attempt under a new target is the very thing `rebuild-nms` offers. An empty
        # stored key stays adoptable too — unprovable, never proven wrong, the row
        # doctrine.
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
        # This is the most valuable recovery the command offers for such a step: a
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
    # Cache first (finalize), then the manifest's provenance: a crash between the two
    # leaves a current cache beside an unprovenanced manifest, which reads as "rebuild
    # again" — cheap; the other order would leave provenance vouching for a cache that
    # was never written.
    cache.save_manifest(
        manifest,
        ctx.step_dir,
        operation=step_cfg.operation,
        engine=step_cfg.engine,
        fingerprint=key.fingerprint,
        criterion_key=key.criterion_key,
        search_key=key.search_key,
        rows=key.manifest_rows(),
    )
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

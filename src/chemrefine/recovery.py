"""Action dispatcher — run / resume / rerun-errors / rerun / rebuild-cache / rebuild-nms.

The CLI maps each subcommand to an :class:`Action` and calls :func:`execute`.
Failure handling is per step via ``on_failure: stop | skip | best``: ``stop``
(the default) halts the run after caching the step's successes, ``skip`` drops
failures and continues, ``best`` keeps all (backfilling the best geometry).
Only a ``stop`` step leaves failures pending — and these actions recover them:

* ``run`` — wipe every step's cache and re-execute from scratch.
* ``resume`` — honor the on-disk cache and re-attempt the pending failed jobs
  of any ``on_failure: stop`` step (only those: ``skip`` / ``best`` failures are
  already resolved), then continue. The ``failed_jobs.json`` ledger records the
  failures of *every* policy, so they're always visible.
* ``rerun-errors [step]`` — re-attempt one step's pending failed jobs (latest if no step
  given), then continue. Scoped backwards only: the steps before it cache-hit and submit
  nothing, the steps after it resume, because the halt that left the failures pending is
  what stopped them running in the first place.
* ``rerun [step]`` — redo one whole step from scratch; every other step resumes, so one
  whose fingerprint no longer holds re-executes too.
* ``rebuild-cache [step]`` — rebuild one step's cache from outputs already on
  disk (parse only, no submission). The run ends at that step.
* ``rebuild-nms [step]`` — a named alias of ``rerun`` for the NMS-tuning workflow. It
  targets the step named (the last if none is), not "the NMS step", and discards its cache,
  so round 1 is recomputed; tuning only the search parameters is what ``resume`` reuses
  round 1 for.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from enum import StrEnum

from chemrefine import cache, pipeline
from chemrefine.config import Config, StepConfig
from chemrefine.errors import ChemRefineError
from chemrefine.step import RunPlan, StepMode

logger = logging.getLogger(__name__)


class Action(StrEnum):
    """Lifecycle action requested by the CLI."""

    RUN = "run"
    RESUME = "resume"
    REBUILD_CACHE = "rebuild-cache"
    REBUILD_NMS = "rebuild-nms"
    RERUN = "rerun"
    RERUN_ERRORS = "rerun-errors"


def resolve_target(config: Config, key: str | int) -> StepConfig:
    """Look up a step by number or name; raise if missing."""
    step = config.find_step(key)
    if step is None:
        raise ChemRefineError(
            f"no step matches {key!r}; available: {[s.dir_name() for s in config.steps]}"
        )
    return step


def _resolve_target_or_last(config: Config, target: str | int | None) -> StepConfig:
    """Resolve ``target`` to a step, defaulting to the last step when it's None."""
    return config.steps[-1] if target is None else resolve_target(config, target)


def invalidate_step(config: Config, step_cfg: StepConfig) -> None:
    """Discard one step's state so the next run genuinely re-executes it.

    Results *and* manifest (:func:`chemrefine.cache.discard_step`): keeping the manifest
    would leave the step looking merely interrupted rather than deliberately discarded, and
    ``resume`` would then re-parse the outputs from disk instead of resubmitting them —
    which is the opposite of what ``run`` and ``rerun`` ask for.
    """
    step_dir = config.step_dir(step_cfg).resolve()
    cache.discard_step(step_dir)
    logger.info("discarded cached state for %s", step_cfg.dir_name())


def _action_run(config: Config, _target: str | int | None) -> None:
    """Invalidate every step's cache, then run from scratch."""
    for step_cfg in config.steps:
        invalidate_step(config, step_cfg)
    pipeline.run(config, RunPlan(default=StepMode.EXECUTE))


def _action_resume(config: Config, _target: str | int | None) -> None:
    """Run the pipeline honoring whatever caches are on disk.

    Every step is in ``RESUME`` mode, so whichever ``on_failure: stop`` step is
    pending gets its failures re-attempted.
    """
    pipeline.run(config, RunPlan(default=StepMode.RESUME))


def _action_rerun(config: Config, target: str | int | None) -> None:
    """Redo one whole step from scratch (latest if ``target`` is None), then resume.

    Invalidates the target step's cache — results *and* manifest, so an NMS step really
    re-displaces rather than reusing round 1 — and runs the pipeline in ``RESUME``. Prior
    steps therefore behave as they would under ``resume``: served from cache when it is
    valid, brought up to date when it is not. To repair only failed jobs without redoing
    successful ones, use ``resume`` (incremental).
    """
    target_step = _resolve_target_or_last(config, target)
    invalidate_step(config, target_step)
    pipeline.run(config, RunPlan(default=StepMode.RESUME))


def _action_rerun_errors(config: Config, target: str | int | None) -> None:
    """Re-attempt one step's pending failed jobs (latest if ``target`` is None), then continue.

    Scoped *backwards* only: steps before the target cache-hit and submit nothing, the target
    step's still-failed structures are resubmitted (it must be ``on_failure: stop`` to have
    pending failures), and everything after it resumes.

    Later steps resume rather than cache-hit because the run halted at the target — that is
    what left the failures pending — so they have no cache to hit and never ran. Left
    ``CACHE_ONLY`` they raise from :func:`chemrefine.step.run_step` for a cache that could not
    exist, which failed the command after it had already repaired what it was pointed at, and
    told the user to run ``resume`` when ``resume`` is what they had just run.
    """
    target_step = _resolve_target_or_last(config, target)
    step_dir = config.step_dir(target_step).resolve()
    n_failed = len(cache.load_failure_records(step_dir))
    if not n_failed:
        logger.info(
            "rerun-errors: %s has no recorded failures to rerun",
            target_step.dir_name(),
        )
    elif target_step.leaves_failures_pending:
        logger.info(
            "rerun-errors: re-attempting %d failed job(s) in %s",
            n_failed,
            target_step.dir_name(),
        )
    else:
        # The ledger is visibility-only for skip/best — run_step never
        # re-attempts those, so don't claim a re-attempt is happening.
        logger.info(
            "rerun-errors: %s recorded %d failure(s) but on_failure=%s resolved them; "
            "nothing is pending (use `rerun %s` to redo the whole step)",
            target_step.dir_name(),
            n_failed,
            target_step.on_failure,
            target_step.step,
        )
    pipeline.run(
        config,
        RunPlan(
            default=StepMode.CACHE_ONLY,
            overrides={
                step_cfg.step: StepMode.RESUME
                for step_cfg in config.steps
                if step_cfg.step >= target_step.step
            },
        ),
    )


def _action_rebuild_cache(config: Config, target: str | int | None) -> None:
    """Rebuild one step's cache from outputs already on disk (no submission).

    Prior steps cache-hit to supply the upstream state; the target step is
    re-parsed from its existing outputs (and re-resolved, for NMS) and its
    ``StepCache`` rewritten. Use after a parser/cache change to avoid re-running
    finished jobs.

    The run ends at the target (``stop_after``): rebuilding step N says nothing about the
    steps after it, and neither mode available to them is right — ``CACHE_ONLY`` raises for a
    cache a step that never ran cannot have, and resuming would submit, which is the one
    thing this command promises not to do.
    """
    target_step = _resolve_target_or_last(config, target)
    pipeline.run(
        config,
        RunPlan(
            default=StepMode.CACHE_ONLY,
            overrides={target_step.step: StepMode.REBUILD},
            stop_after=target_step.step,
        ),
    )


_HANDLERS: dict[Action, Callable[[Config, str | int | None], None]] = {
    Action.RUN: _action_run,
    Action.RESUME: _action_resume,
    Action.RERUN_ERRORS: _action_rerun_errors,
    Action.REBUILD_CACHE: _action_rebuild_cache,
    Action.RERUN: _action_rerun,
    # rebuild-nms re-runs the NMS step with the current NmsOptions (re-displace +
    # re-optimise); it's a named alias of rerun for the NMS-tuning workflow.
    Action.REBUILD_NMS: _action_rerun,
}


def execute(
    config: Config,
    action: Action,
    target: str | int | None = None,
) -> int:
    """Dispatch ``action`` for ``config`` and return a process exit code."""
    handler = _HANDLERS.get(action)
    if handler is None:
        raise ChemRefineError(f"unknown action: {action!r}")
    handler(config, target)
    return 0

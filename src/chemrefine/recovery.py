"""Action dispatcher — run / resume / rerun / rebuild-cache / rebuild-nms.

The CLI maps each subcommand to an :class:`Action` and calls :func:`execute`:

* ``run`` — wipe every step's cache and re-execute from scratch.
* ``resume`` — honor the on-disk cache; **incremental**, so a step with a
  ``failed_jobs.json`` ledger resubmits only its still-failed structures.
* ``rerun [step]`` — redo one whole step from scratch (others cache-hit).
* ``rebuild-cache [step]`` — rebuild one step's cache from outputs already on
  disk (parse only, no submission).
* ``rebuild-nms [step]`` — re-run the NMS step with the current options
  (a named alias of ``rerun`` for the NMS-tuning workflow).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from enum import StrEnum

from chemrefine import cache, pipeline
from chemrefine.config import Config, StepConfig
from chemrefine.errors import ChemRefineError

logger = logging.getLogger(__name__)


class Action(StrEnum):
    """Lifecycle action requested by the CLI."""

    RUN = "run"
    RESUME = "resume"
    REBUILD_CACHE = "rebuild-cache"
    REBUILD_NMS = "rebuild-nms"
    RERUN = "rerun"


def resolve_target(config: Config, key: str | int) -> StepConfig:
    """Look up a step by number or name; raise if missing."""
    step = config.find_step(key)
    if step is None:
        raise ChemRefineError(
            f"no step matches {key!r}; available: "
            f"{[s.dir_name() for s in config.steps]}"
        )
    return step


def invalidate_step(config: Config, step_cfg: StepConfig) -> None:
    """Drop the cache for one step so the next run re-executes it."""
    step_dir = (config.output_dir / step_cfg.dir_name()).resolve()
    cache.invalidate(step_dir)
    logger.info("invalidated cache for %s", step_cfg.dir_name())


def _action_run(config: Config, _target: str | int | None) -> None:
    """Invalidate every step's cache, then run from scratch."""
    for step_cfg in config.steps:
        invalidate_step(config, step_cfg)
    pipeline.run(config, use_cache=False)


def _action_resume(config: Config, _target: str | int | None) -> None:
    """Run the pipeline honoring whatever caches are on disk."""
    pipeline.run(config, use_cache=True)


def _action_rerun(config: Config, target: str | int | None) -> None:
    """Redo one whole step from scratch (latest if ``target`` is None), then resume.

    Invalidates the target step's cache so it re-executes end-to-end (both NMS
    rounds; with current options) while prior steps cache-hit. To repair only
    failed jobs without redoing successful ones, use ``resume`` (incremental).
    """
    target_step = (
        config.steps[-1] if target is None else resolve_target(config, target)
    )
    invalidate_step(config, target_step)
    pipeline.run(config, use_cache=True)


def _action_rebuild_cache(config: Config, target: str | int | None) -> None:
    """Rebuild one step's cache from outputs already on disk (no submission).

    Prior steps cache-hit to supply the upstream state; the target step is
    re-parsed from its existing outputs (and re-resolved, for NMS) and its
    ``StepCache`` rewritten. Use after a parser/cache change to avoid re-running
    finished jobs.
    """
    target_step = (
        config.steps[-1] if target is None else resolve_target(config, target)
    )
    pipeline.run(config, use_cache=True, rebuild_step=target_step.step)


_HANDLERS: dict[Action, Callable[[Config, str | int | None], None]] = {
    Action.RUN: _action_run,
    Action.RESUME: _action_resume,
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

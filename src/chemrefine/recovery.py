"""Action dispatcher — run / resume / rebuild-cache / rebuild-nms / rerun.

The CLI maps each subcommand to an :class:`Action` and calls
:func:`execute`. ``run`` wipes every step's cache and re-executes from
scratch; ``resume`` honors whatever cache is on disk; the three
targeted actions invalidate one step's cache (the latest if no target
is given) and then resume.
"""

from __future__ import annotations

import logging
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


def execute(
    config: Config,
    action: Action,
    target: str | int | None = None,
) -> int:
    """Dispatch ``action`` for ``config`` and return a process exit code."""
    if action is Action.RUN:
        for step_cfg in config.steps:
            invalidate_step(config, step_cfg)
        pipeline.run(config, use_cache=False)
        return 0

    if action is Action.RESUME:
        pipeline.run(config, use_cache=True)
        return 0

    if action in (Action.REBUILD_CACHE, Action.RERUN, Action.REBUILD_NMS):
        target_step = (
            config.steps[-1] if target is None else resolve_target(config, target)
        )
        invalidate_step(config, target_step)
        pipeline.run(config, use_cache=True)
        return 0

    raise ChemRefineError(f"unknown action: {action!r}")

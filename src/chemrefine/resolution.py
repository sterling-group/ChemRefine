"""Options a step's own environment decides, applied to the step and recorded in its tree.

:func:`chemrefine.engines._provision.resolve_step_options` asks each step that can submit
what its environment decides (``solvation_model: auto``); :func:`resolved_step` puts the
answer in the step's options and records it, and gives a step that cannot submit the answer
its last run recorded.

The one module that compares a step's raw options against a record, key by key: what the
config *asked* is the record's subject, before and apart from any engine's options model,
so the read is of the raw mapping on purpose (``test_engines_invariants`` allow-lists it).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping

from chemrefine import ids, io
from chemrefine.config import Config, StepConfig
from chemrefine.step import step_dir_for

logger = logging.getLogger(__name__)


def recorded_choice(config: Config, step_cfg: StepConfig) -> dict[str, object] | None:
    """What ``step_cfg``'s last run chose, while its options still ask what they asked then.

    ``None`` when there is no record, it cannot be read, or the options have moved on since:
    a choice made for other options says nothing about these.
    """
    path = ids.resolved_options_path(step_dir_for(config, step_cfg))
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        logger.warning("step %d: %s is unreadable, ignored: %s", step_cfg.step, path, exc)
        return None
    options = dict(step_cfg.options or {})
    if any(options.get(k) != v for k, v in record["asked"].items()):
        return None
    return dict(record["chosen"])


def resolved_step(
    config: Config, step_cfg: StepConfig, chosen: Mapping[str, object] | None
) -> StepConfig:
    """``step_cfg`` with the options its environment chose in place of the ones that asked.

    ``chosen`` comes from :func:`chemrefine.engines._provision.resolve_step_options` for a
    step that can submit. It is recorded in the step's ``_cache/`` beside what asked for it,
    and the step runs with it: its key, its jobs and its reports all see the chosen value.

    A step that cannot submit was not asked: a cache-only step, or ``rebuild-cache`` on a
    login node without the step's environment. It reuses :func:`recorded_choice`. A step
    whose options moved on is left as it is, and its key then misses the cache as any
    edited step's does.
    """
    options = dict(step_cfg.options or {})
    if chosen:
        record = {"asked": {k: options.get(k) for k in chosen}, "chosen": dict(chosen)}
        io.write_tree_file(
            ids.resolved_options_path(step_dir_for(config, step_cfg)),
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            atomic=True,
        )
    else:
        chosen = recorded_choice(config, step_cfg)
        if chosen is None:
            return step_cfg
    for key, value in chosen.items():
        logger.info("step %d: %s: %s → %s", step_cfg.step, key, options.get(key), value)
    return step_cfg.model_copy(update={"options": {**options, **chosen}})

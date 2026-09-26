"""Backend-side training for API-only libraries — one shared ``-m`` entry, per-library hook.

``python -m chemrefine.engines.mlip.train.driver <task_name> <config.yaml>`` — the
training counterpart of :mod:`chemrefine.engines._backend_server.server`, and shaped by
the same rule: the *process shell* is shared infrastructure, and everything
library-specific lives on a class the registry resolves. A library whose training is a
pure Python API (CHGNet has no CLI and no config format) implements one extra method on
the same trainer class its module already registers — ``run_training(config)``, where the
heavy imports live — so adding such a library is still **one dropped-in module**, the
registry docstring's promise. A library with a real CLI (MACE, SevenNet) never comes
here; its ``command`` names its own binary.

This runs under the *backend environment's* interpreter, where the library is importable
— and so is chemrefine, because a managed env is a ``pip install "chemrefine[<extra>]"``:
the same fact that lets the ExtOpt server run as a chemrefine module from one.

The config is the step template rendered by :func:`~chemrefine.engines.mlip.train.base.
render_config` — each trainer's ``run_training`` documents and validates its own keys,
because the schema is the trainer's (there is no native one to defer to). The **plan
facts** — device, seed, and the foundation weights — do not travel in it: they arrive on
this driver's own argv (written by :meth:`~chemrefine.engines.mlip.train.base.
ApiTrainerBase.command`) and are overlaid onto the parsed config as authoritative. Routed
through the template as placeholders, each was silently lost whenever a template did not
reference it — a ``device: cuda`` step trained on CPU with the GPU booked, a declared
foundation model trained from scratch while the sidecar recorded it. A template value
for any of them that *disagrees* with the step's own is refused by name, and so is one
the step does not set at all; silent precedence in either direction is how this started.
Which facts, and under which flags, is
:data:`~chemrefine.engines.mlip.train.base.PLAN_FACTS` — read here, not restated.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from chemrefine.engines.mlip.train.base import PLAN_FACTS


def _load_config(path: Path) -> dict[str, Any]:
    """The rendered YAML as a mapping, or the usage-shaped refusal."""
    import yaml

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise SystemExit(f"train_driver: config {path} is not a YAML mapping")
    return config


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """The driver's own CLI — the generic shape plus one flag per plan fact.

    A fact every run has (``device``, ``seed``) is required, because the engine always
    writes it; the two foundation flags are optional because "from scratch" is a
    legitimate run, and they are two flags rather than one string because a checkpoint
    path and a release name are different facts the hooks dispatch on.
    """
    parser = argparse.ArgumentParser(prog="chemrefine-train-driver")
    parser.add_argument("task_name", help="the registry task whose trainer runs")
    parser.add_argument("config", help="the rendered trainer config (backend knobs)")
    for fact in PLAN_FACTS:
        parser.add_argument(
            fact.flag,
            dest=fact.key,
            required=fact.always,
            type=fact.kind,
            default=None,
            help=f"from the step's `{fact.option}`",
        )
    return parser.parse_args(argv)


def _absent(value: Any) -> bool:
    """Whether a rendered or parsed value states nothing — ``None`` or empty, never ``0``.

    ``$FOUNDATION_MODEL`` renders empty for a from-scratch step, which YAML reads back as
    ``None``; a ``seed: 0`` is a value, which a truthiness test would drop.
    """
    return value is None or value == ""


def _overlay_plan_facts(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Overlay the argv plan facts onto the rendered config, refusing a disagreement.

    Every fact in :data:`~chemrefine.engines.mlip.train.base.PLAN_FACTS`, and a value the
    step does not set counts too: a template ``weights_path`` for a step with no
    ``model_path`` trained from that checkpoint while the runlog and the sidecar said
    "scratch". Equality is judged as strings because the template channel only carries
    strings: a template that rendered ``seed: $SEED`` holds the same digits argv carries,
    and must keep working — harmless duplication is not a conflict. ``start_from`` is the
    legacy spelling a pre-channel template may still render; the hooks no longer read it,
    so a hard-coded value that names something *else* must refuse rather than be silently
    outranked (the inversion of the silent loss this driver exists to end).
    """
    facts = {fact.key: v for fact in PLAN_FACTS if not _absent(v := getattr(args, fact.key))}
    for fact in PLAN_FACTS:
        rendered = config.get(fact.key)
        if _absent(rendered) or str(rendered) == str(facts.get(fact.key, "")):
            continue
        said = repr(facts[fact.key]) if fact.key in facts else "nothing"
        raise SystemExit(
            f"train_driver: the template renders {fact.key}: {rendered!r} but the step's "
            f"`{fact.option}` says {said} — the options are authoritative for a plan fact. "
            f"Remove `{fact.key}:` from the template; the driver hands the trainer the "
            f"step's value itself."
        )
    display = str(facts.get("weights_path") or facts.get("foundation") or "")
    if str(config.get("start_from") or "") not in ("", display):
        raise SystemExit(
            f"train_driver: the template renders start_from: {config['start_from']!r} but "
            f"the step's options name {display or 'nothing'} — the options are "
            f"authoritative. Set model_path/model_name on the step (or reference "
            f"$FOUNDATION_MODEL, which renders the same value)."
        )
    return {**config, **facts}


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve ``task_name``'s trainer and hand it the config with the plan facts overlaid."""
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))

    from chemrefine.engines.mlip.registry import trainer_for
    from chemrefine.engines.mlip.train.base import ApiTrainerBase

    trainer = trainer_for(args.task_name)()
    if not isinstance(trainer, ApiTrainerBase):
        # Reachable only by hand: a CLI-driven trainer's `command` never names this
        # driver — but a person running the module against the wrong task deserves the
        # real answer. Nominal, not structural: being drivable is a fact of inheritance.
        raise SystemExit(
            f"train_driver: {args.task_name!r} trains through its library's own CLI, "
            f"not through this driver"
        )
    return trainer.run_training(_overlay_plan_facts(_load_config(Path(args.config)), args))


if __name__ == "__main__":
    raise SystemExit(main())

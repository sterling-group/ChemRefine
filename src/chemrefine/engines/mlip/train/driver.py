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
because the schema is the trainer's (there is no native one to defer to).
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _load_config(path: Path) -> dict[str, Any]:
    """The rendered YAML as a mapping, or the usage-shaped refusal."""
    import yaml

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise SystemExit(f"train_driver: config {path} is not a YAML mapping")
    return config


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve ``task_name``'s trainer and hand it the rendered config."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        raise SystemExit(
            "usage: python -m chemrefine.engines.mlip.train.driver <task_name> <config>"
        )
    task_name, config_path = args

    from chemrefine.engines.mlip.registry import trainer_for
    from chemrefine.engines.mlip.train.base import ApiTrainerBase

    trainer = trainer_for(task_name)()
    if not isinstance(trainer, ApiTrainerBase):
        # Reachable only by hand: a CLI-driven trainer's `command` never names this
        # driver — but a person running the module against the wrong task deserves the
        # real answer. Nominal, not structural: being drivable is a fact of inheritance.
        raise SystemExit(
            f"train_driver: {task_name!r} trains through its library's own CLI, "
            f"not through this driver"
        )
    return trainer.run_training(_load_config(Path(config_path)))


if __name__ == "__main__":
    raise SystemExit(main())

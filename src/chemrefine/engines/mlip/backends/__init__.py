"""Built-in MLIP libraries — importing this package registers them all.

**One module per library**, holding everything chemrefine knows about it: the environment that
provides it, the ASE calculators it can build, and the trainer that fine-tunes it, if it has
one. They are declared together so the environment is declared *once* — a library cannot name
one env for running and another for training, because there is only one place to name it
(:mod:`chemrefine.engines.mlip.registry`).

Modules are **auto-discovered**: every bare-named module here is imported, which runs its
``@<LIBRARY>.calculator`` / ``@<LIBRARY>.trainer`` decorators. Making an MLIP available, or
making an available one trainable, is **fully self-contained** — drop a module here, or add a
decorator to the one that exists, with no import line to maintain and no central table.

Every heavy third-party import stays inside the builder or inside the job the trainer
generates, so this package imports cheaply on a machine with no MLIP library installed at all.
That matters more than it looks: the registry is built at import time, so a top-level
``import mace`` in one module would make the whole package unimportable wherever MACE is not
the library that happens to be installed.
"""

import importlib
import pkgutil


def _load_backends() -> None:
    """Import every bare-named module — each self-registers its library's capabilities.

    Underscored modules are skipped (helpers, not libraries). Idempotent; separate from the
    call below so tests can point it at a temporary path to prove drop-in additions.
    """
    for mod in pkgutil.iter_modules(__path__):
        if not mod.name.startswith("_"):
            importlib.import_module(f"{__name__}.{mod.name}")


_load_backends()

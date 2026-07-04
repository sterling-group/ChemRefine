"""Built-in MLIP backends — importing this package registers them all.

Backends are **auto-discovered**: every bare-named module in this package is imported here,
which runs its ``@register_backend`` decorator(s) (from
:mod:`chemrefine.engines.mlip.calculator`) and registers its builder + packaging metadata.
Adding a new MLIP is **fully self-contained**: drop a module here and it is discovered —
no import line to maintain. The heavy third-party import stays inside each builder, so this
package imports cheaply even when an optional dependency is missing.
"""

import importlib
import pkgutil


def _load_backends() -> None:
    """Import every bare-named module — each self-registers its backend builder(s).

    Underscored modules are skipped (helpers, not backends). Idempotent; separate from the
    call below so tests can point it at a temporary path to prove drop-in additions.
    """
    for mod in pkgutil.iter_modules(__path__):
        if not mod.name.startswith("_"):
            importlib.import_module(f"{__name__}.{mod.name}")


_load_backends()

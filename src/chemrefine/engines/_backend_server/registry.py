"""ExtOpt backend registry — resolves a backend name to its calculator class.

The ExtOpt engines declare the mapping themselves (their ``backend`` and
``calculator_cls`` ClassVars); this module just scans the engine registry.
Adding a third backend is: write ``engines/<name>/extopt_calc.py`` with a
``ComputeBackend``-conforming class and register the engine plugin — there
is no list here to update.
"""

from __future__ import annotations

from chemrefine.engines._backend_server.base import ComputeBackend, ExtOptServed


def _calculators() -> dict[str, type[ComputeBackend]]:
    """Scan the engine registry for ExtOpt engines and their calculators.

    Detected via :class:`ExtOptServed` — a capability Protocol, like every other
    ``isinstance`` test in ``engines/`` — so the two ClassVars the ExtOpt engines declare
    are read as their declared types rather than through an untyped ``getattr`` probe.
    """
    from chemrefine.engines.api import ENGINES, get_engine

    found: dict[str, type[ComputeBackend]] = {}
    for name in ENGINES:
        engine = get_engine(name)
        if isinstance(engine, ExtOptServed):
            found[engine.backend] = engine.calculator_cls
    return found


def known_backends() -> list[str]:
    """Every backend name an ExtOpt engine declares (drives the server CLI)."""
    return sorted(_calculators())


def load_calculator(name: str) -> type[ComputeBackend]:
    """Return the ``ComputeBackend`` class the ExtOpt engine registered for ``name``.

    Raises :class:`KeyError` (listing the known names) when no registered
    engine declares ``name``.
    """
    calculators = _calculators()
    if name not in calculators:
        raise KeyError(f"unknown ExtOpt backend {name!r} (known: {sorted(calculators)})")
    return calculators[name]

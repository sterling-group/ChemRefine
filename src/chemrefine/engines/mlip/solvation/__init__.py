"""Implicit-solvation corrections for MLIP steps — importing this package registers them all.

**One module per library**, auto-discovered like :mod:`chemrefine.engines.mlip.backends`:
every bare-named module here is imported, which runs its ``@<LIBRARY>.correction``
decorator. The registry itself is :mod:`._registry`; see it for the shape of a library.
:mod:`._choose` is run as a program in a step's environment, so it is not imported here.

Every third-party import stays inside the correction builder, so this package imports
cheaply wherever no solvation library is installed.
"""

import importlib
import pkgutil

from chemrefine.engines.mlip.solvation._registry import (
    AUTO,
    Correction,
    CorrectionBuilder,
    ExecutableRequirement,
    Locator,
    PythonRequirement,
    SolvationLibrary,
    SolvationRequirement,
    SolvationSpec,
    WeightsMissing,
    build_correction,
    registered_solvation,
    solvation_library,
    solvation_problem,
)

__all__ = [
    "AUTO",
    "Correction",
    "CorrectionBuilder",
    "ExecutableRequirement",
    "Locator",
    "PythonRequirement",
    "SolvationLibrary",
    "SolvationRequirement",
    "SolvationSpec",
    "WeightsMissing",
    "build_correction",
    "registered_solvation",
    "solvation_library",
    "solvation_problem",
]


def _load_libraries() -> None:
    """Import every bare-named module — each registers its library's correction."""
    for mod in pkgutil.iter_modules(__path__):
        if not mod.name.startswith("_"):
            importlib.import_module(f"{__name__}.{mod.name}")


_load_libraries()

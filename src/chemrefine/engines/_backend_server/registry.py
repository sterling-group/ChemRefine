"""ExtOpt backend registry — maps a backend name to a calculator class.

Looked up by :mod:`.server` (to choose which backend to instantiate)
and consulted by :mod:`.client` (for the ``--backend`` CLI choice).

Adding a third backend is: write ``engines/<name>/extopt_calc.py`` with
a ``ComputeBackend``-conforming class, then add one line here.
"""

from __future__ import annotations

import importlib

from chemrefine.engines._backend_server.base import ComputeBackend

CALCULATORS: dict[str, str] = {
    "mlip": "chemrefine.engines.mlip.extopt_calc:MlipExtOptCalculator",
    "pyscf": "chemrefine.engines.pyscf.extopt_calc:PyscfExtOptCalculator",
}


def load_calculator(name: str) -> type[ComputeBackend]:
    """Return the ``ComputeBackend`` class registered under ``name``.

    Raises :class:`KeyError` if ``name`` isn't in :data:`CALCULATORS`,
    :class:`ImportError` if the target module doesn't exist, or
    :class:`AttributeError` if the class isn't found in the module.
    """
    target = CALCULATORS[name]
    module_path, class_name = target.split(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

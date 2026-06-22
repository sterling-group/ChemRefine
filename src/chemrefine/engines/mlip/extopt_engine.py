"""ORCA-driven MLIP engine (ORCA optimises, MLIP provides gradients).

A pure declaration over :class:`~chemrefine.engines.orca.extopt.engine.ExtOptOrcaEngine`:
ORCA runs the optimisation and calls a wrapper that relays each geometry to a long-running
MLIP gradient server (the shared :mod:`chemrefine.engines._backend_server.server` with
``--backend mlip``). All shared machinery — the ``%method ProgExt`` block, the server-launch
``run_block``, the ``_server_cmd`` template — lives on the base; this engine only names the
backend, its wrapper file, and the option/calculator classes that describe it.
"""

from __future__ import annotations

from typing import ClassVar

from chemrefine.engines._backend_server.base import ComputeBackend
from chemrefine.engines._options import EngineOptions
from chemrefine.engines.api import register
from chemrefine.engines.mlip.extopt_calc import MlipExtOptCalculator
from chemrefine.engines.mlip.options import MlipOptions
from chemrefine.engines.orca.extopt.engine import ExtOptOrcaEngine


@register("mlip-extopt")
class MlipExtOptEngine(ExtOptOrcaEngine):
    """ORCA optimisation backed by an MLIP gradient server."""

    name: ClassVar[str] = "mlip-extopt"
    backend: ClassVar[str] = "mlip"
    wrapper_filename: ClassVar[str] = "mlip_extopt.sh"
    options_cls: ClassVar[type[EngineOptions]] = MlipOptions
    calculator_cls: ClassVar[type[ComputeBackend]] = MlipExtOptCalculator

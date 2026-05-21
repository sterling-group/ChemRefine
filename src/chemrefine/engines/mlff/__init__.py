"""MLFF engine — importing this module registers ``mlff`` and ``mlff-direct``.

Two engines ship in this package:

* ``MlffEngine`` (registered as ``"mlff"``) — drives ORCA's external
  optimizer protocol with an MLFF-based gradient server. The HTTP /
  ExtOpt plumbing lives in :mod:`chemrefine.engines._extopt`; only
  the per-call ASE-Atoms → energy/gradient adapter lives here as
  :mod:`.extopt_calc`.
* ``MlffDirectEngine`` (registered as ``"mlff-direct"``) — runs MLFF
  inference in-process, no ORCA, for fast pre-screening.
"""

from chemrefine.engines.mlff.direct import MlffDirectEngine
from chemrefine.engines.mlff.engine import MlffEngine
from chemrefine.engines.mlff.extopt_calc import MlffExtOptCalculator

__all__ = ["MlffDirectEngine", "MlffEngine", "MlffExtOptCalculator"]

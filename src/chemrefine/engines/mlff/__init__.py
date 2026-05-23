"""MLFF engine — importing this module registers ``mlff`` and ``mlff-extopt``.

Two engines ship in this package:

* ``MlffEngine`` (registered as ``"mlff"``) — runs MLFF inference
  in-process, no ORCA, for fast pre-screening. The GPU model loads
  once per pipeline run.
* ``MlffExtOptEngine`` (registered as ``"mlff-extopt"``) — drives ORCA's
  external optimizer protocol with an MLFF-based gradient server. The
  HTTP / ExtOpt plumbing lives in :mod:`chemrefine.engines._extopt`;
  only the per-call ASE-Atoms → energy/gradient adapter lives here as
  :mod:`.extopt_calc`.
"""

from chemrefine.engines.mlff.engine import MlffEngine
from chemrefine.engines.mlff.extopt_calc import MlffExtOptCalculator
from chemrefine.engines.mlff.extopt_engine import MlffExtOptEngine

__all__ = ["MlffEngine", "MlffExtOptCalculator", "MlffExtOptEngine"]

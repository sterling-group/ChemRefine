"""MLFF engine — importing this module registers ``mlff``, ``mlff-extopt``, ``mlff-train``.

Engines shipped in this package:

* ``MlffEngine`` (``"mlff"``) — in-process MLFF inference, no ORCA, for fast
  pre-screening. The model loads once per pipeline run.
* ``MlffExtOptEngine`` (``"mlff-extopt"``) — drives ORCA's external optimizer
  protocol with an MLFF gradient server. The HTTP / backend-server plumbing
  lives in :mod:`chemrefine.engines._backend_server`; only the per-call
  ASE-Atoms → energy/gradient adapter lives here as :mod:`.extopt_calc`.
* ``MlffTrainEngine`` (``"mlff-train"``) — trains an MLFF on the previous step's
  structures (energies + forces) via :mod:`.trainer`.
"""

from chemrefine.engines.mlff.engine import MlffEngine
from chemrefine.engines.mlff.extopt_calc import MlffExtOptCalculator
from chemrefine.engines.mlff.extopt_engine import MlffExtOptEngine
from chemrefine.engines.mlff.train_engine import MlffTrainEngine

__all__ = [
    "MlffEngine",
    "MlffExtOptCalculator",
    "MlffExtOptEngine",
    "MlffTrainEngine",
]

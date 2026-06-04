"""MLIP engine — importing this module registers ``mlip``, ``mlip-extopt``, ``mlip-train``.

Legacy ``mlff`` / ``mlff-extopt`` / ``mlff-train`` engine keys (and other v1.3.1
spellings) are rewritten to these canonical names by the config normalizer
(:func:`chemrefine.config._normalize_legacy`) — the single place that knows the
old vocabulary — so existing YAML keeps working.

Engines shipped in this package:

* ``MlipEngine`` (``"mlip"``) — in-process MLIP inference, no ORCA, for fast
  pre-screening. The model loads once per pipeline run.
* ``MlipExtOptEngine`` (``"mlip-extopt"``) — drives ORCA's external optimizer
  protocol with an MLIP gradient server. The HTTP / backend-server plumbing
  lives in :mod:`chemrefine.engines._backend_server`; only the per-call
  ASE-Atoms → energy/gradient adapter lives here as :mod:`.extopt_calc`.
* ``MlipTrainEngine`` (``"mlip-train"``) — trains an MLIP on the previous step's
  structures (energies + forces) via :mod:`.trainer`.
"""

from chemrefine.engines.mlip import backends
from chemrefine.engines.mlip.engine import MlipEngine
from chemrefine.engines.mlip.extopt_calc import MlipExtOptCalculator
from chemrefine.engines.mlip.extopt_engine import MlipExtOptEngine
from chemrefine.engines.mlip.train_engine import MlipTrainEngine

__all__ = [
    "MlipEngine",
    "MlipExtOptCalculator",
    "MlipExtOptEngine",
    "MlipTrainEngine",
    "backends",
]

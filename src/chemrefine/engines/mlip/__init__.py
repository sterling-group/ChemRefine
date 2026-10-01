"""MLIP engine — importing this module registers ``mlip``, ``mlip-extopt``, ``mlip-train``.

v1.3.1's spellings of these engines are rewritten to their canonical names by the
config normalizer (:func:`chemrefine.config_legacy.normalize`), the single place that
knows the old vocabulary.

Engines shipped in this package:

* ``MlipEngine`` (``"mlip"``) — in-process MLIP inference, no ORCA, for fast
  pre-screening. Each structure is its own job: one process that loads the model.
* ``MlipExtOptEngine`` (``"mlip-extopt"``) — drives ORCA's external optimizer
  protocol with an MLIP gradient server. The HTTP / backend-server plumbing
  lives in :mod:`chemrefine.engines._backend_server`; only the per-call
  ASE-Atoms → energy/gradient adapter lives here as :mod:`.extopt_calc`.
* ``MlipTrainEngine`` (``"mlip-train"``) — trains or fine-tunes an MLIP on the
  previous step's structures (energies + forces) and passes them through.

:mod:`.backends` is imported here for its registration side effects. It holds **one module per
library**, each declaring the environment it needs once and hanging its capabilities off it —
an ASE calculator, a trainer, or both (:mod:`.registry`). The modules auto-discover, so making
an MLIP available, or making it trainable, changes no file but its own.
"""

from chemrefine.engines.mlip import backends
from chemrefine.engines.mlip.engine import MlipEngine
from chemrefine.engines.mlip.extopt_calc import MlipExtOptCalculator
from chemrefine.engines.mlip.extopt_engine import MlipExtOptEngine
from chemrefine.engines.mlip.train.engine import MlipTrainEngine

__all__ = [
    "MlipEngine",
    "MlipExtOptCalculator",
    "MlipExtOptEngine",
    "MlipTrainEngine",
    "backends",
]

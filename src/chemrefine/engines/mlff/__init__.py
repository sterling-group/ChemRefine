"""MLFF engine — registers ``MlffEngine`` on import.

Two engines ship in this package:

* ``MlffEngine`` (registered as ``"mlff"``) — drives ORCA's external
  optimizer protocol with an MLFF-based gradient server.
* ``MlffDirectEngine`` (registered as ``"mlff-direct"``) — runs MLFF
  inference in-process, no ORCA, for fast pre-screening.
"""

from chemrefine.engines.mlff.direct import MlffDirectEngine  # noqa: F401
from chemrefine.engines.mlff.engine import MlffEngine  # noqa: F401

"""MLIP training — one package for the concern, three sharply named parts.

The same shape :mod:`chemrefine.engines._backend_server` gives the ExtOpt server, for the
same reason: a concern with a contract, an orchestrator half and a backend-side process
shell reads as one directory, not as three siblings competing for a prefix at the
package's top level.

* :mod:`~chemrefine.engines.mlip.train.base` — the contract and the backend-agnostic
  machinery: :class:`Trainer`, :class:`TrainingPlan`, the deterministic split, the
  template render, the shared dataset writers.
* :mod:`~chemrefine.engines.mlip.train.engine` — :class:`MlipTrainEngine`, the
  scheduler-facing half (``engine: mlip-train``).
* :mod:`~chemrefine.engines.mlip.train.driver` — the ``python -m`` shell an API-only
  library's trainer is driven through in the backend environment.

The names below are the package's public face; which library trains stays
:mod:`chemrefine.engines.mlip.registry`'s business, shared with the calculators — one
registry per library is the doctrine this package must not fork.
"""

from __future__ import annotations

from chemrefine.engines.mlip.train.base import (
    DatasetFiles,
    DatasetSplit,
    Trainer,
    TrainingPlan,
    base_placeholders,
    digest_of,
    labelled_atoms,
    placeholders_for,
    render_config,
    split_structures,
    write_labelled_extxyz,
)
from chemrefine.engines.mlip.train.engine import MlipTrainEngine

__all__ = [
    "DatasetFiles",
    "DatasetSplit",
    "MlipTrainEngine",
    "Trainer",
    "TrainingPlan",
    "base_placeholders",
    "digest_of",
    "labelled_atoms",
    "placeholders_for",
    "render_config",
    "split_structures",
    "write_labelled_extxyz",
]

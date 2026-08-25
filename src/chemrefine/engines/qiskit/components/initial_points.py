"""Built-in variational initial-point factories."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from chemrefine.engines.qiskit.context import AnsatzArtifacts
from chemrefine.engines.qiskit.registry import INITIAL_POINTS, NoComponentOptions
from chemrefine.errors import ConfigError


class RandomInitialPointOptions(BaseModel):
    """Options for reproducible uniform random parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int | None = None
    scale: float = Field(0.1, gt=0)


def _parameter_count(ansatz: AnsatzArtifacts) -> int:
    """Return the fixed circuit's parameter count or fail with component context."""
    if ansatz.circuit is None:
        raise ConfigError("the selected initial point requires a fixed ansatz circuit")
    return int(ansatz.circuit.num_parameters)


@INITIAL_POINTS.register("zeros", NoComponentOptions)
def build_zero_initial_point(*, options: BaseModel, ansatz: AnsatzArtifacts) -> np.ndarray:
    """Initialize every fixed-circuit parameter to zero."""
    del options
    return np.zeros(_parameter_count(ansatz), dtype=float)


@INITIAL_POINTS.register("random", RandomInitialPointOptions)
def build_random_initial_point(
    *, options: RandomInitialPointOptions, ansatz: AnsatzArtifacts
) -> np.ndarray:
    """Draw reproducible parameters uniformly from ``[-scale, scale]``."""
    generator = np.random.default_rng(options.seed)
    return generator.uniform(-options.scale, options.scale, _parameter_count(ansatz))

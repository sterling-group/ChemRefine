"""Built-in classical optimizer factories."""

from __future__ import annotations

from copy import deepcopy
from threading import RLock
from typing import Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from chemrefine.engines.qiskit.registry import OPTIMIZERS

_SPSA_RANDOM_LOCK = RLock()


class SLSQPOptions(BaseModel):
    """Options for sequential least-squares programming."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    maxiter: int = Field(100, ge=1)
    ftol: float = Field(1e-6, gt=0)
    disp: bool = False


class COBYLAOptions(BaseModel):
    """Options for the derivative-free COBYLA optimizer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    maxiter: int = Field(1000, ge=1)
    rhobeg: float = Field(1.0, gt=0)
    tol: float | None = Field(None, gt=0)
    disp: bool = False


class SPSAOptions(BaseModel):
    """Common scalar options for the noise-tolerant SPSA optimizer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    maxiter: int = Field(100, ge=1)
    blocking: bool = False
    trust_region: bool = False
    learning_rate: float | None = Field(None, gt=0)
    perturbation: float | None = Field(None, gt=0)
    second_order: bool = False
    seed: int | None = None

    @model_validator(mode="after")
    def _paired_schedule_values(self) -> Self:
        """Require scalar learning rate and perturbation together, as SPSA does at run time."""
        if (self.learning_rate is None) != (self.perturbation is None):
            raise ValueError("learning_rate and perturbation must be set together")
        return self


@OPTIMIZERS.register("slsqp", SLSQPOptions)
def build_slsqp(*, options: SLSQPOptions) -> object:
    """Build the deterministic SLSQP optimizer."""
    from qiskit_algorithms.optimizers import SLSQP

    return SLSQP(maxiter=options.maxiter, ftol=options.ftol, disp=options.disp)


@OPTIMIZERS.register("cobyla", COBYLAOptions)
def build_cobyla(*, options: COBYLAOptions) -> object:
    """Build the derivative-free COBYLA optimizer."""
    from qiskit_algorithms.optimizers import COBYLA

    return COBYLA(
        maxiter=options.maxiter,
        rhobeg=options.rhobeg,
        tol=options.tol,
        disp=options.disp,
    )


@OPTIMIZERS.register("spsa", SPSAOptions)
def build_spsa(*, options: SPSAOptions) -> object:
    """Build SPSA with a private random stream and the existing scalar schedules.

    Algorithms 0.4 reads a process-global generator during optimization. Swap
    only its public bit-generator state for the duration of each minimize call,
    restoring the caller's stream even on failure. The lock coordinates all
    ChemRefine SPSA instances; unrelated code using Qiskit's global generator
    concurrently does not participate in that lock.
    """
    from qiskit_algorithms.optimizers import SPSA
    from qiskit_algorithms.utils import algorithm_globals

    random = np.random.default_rng(options.seed)

    class IsolatedSPSA(SPSA):  # type: ignore[misc]
        """Delegate optimization while preserving each instance's random stream."""

        def minimize(self, fun: Any, x0: Any, jac: Any = None, bounds: Any = None) -> Any:
            """Advance this optimizer's stream without consuming ambient random state."""
            with _SPSA_RANDOM_LOCK:
                upstream = algorithm_globals.random
                previous_state = deepcopy(upstream.bit_generator.state)
                upstream.bit_generator.state = random.bit_generator.state
                try:
                    return super().minimize(fun, x0, jac=jac, bounds=bounds)
                finally:
                    random.bit_generator.state = upstream.bit_generator.state
                    upstream.bit_generator.state = previous_state

    return IsolatedSPSA(
        maxiter=options.maxiter,
        blocking=options.blocking,
        trust_region=options.trust_region,
        learning_rate=options.learning_rate,
        perturbation=options.perturbation,
        second_order=options.second_order,
    )

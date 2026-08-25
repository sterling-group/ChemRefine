"""Built-in classical optimizer factories."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from chemrefine.engines.qiskit.registry import OPTIMIZERS


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
    """Build SPSA using scalar learning-rate and perturbation schedules."""
    from qiskit_algorithms.optimizers import SPSA
    from qiskit_algorithms.utils import algorithm_globals

    if options.seed is not None:
        algorithm_globals.random_seed = options.seed

    return SPSA(
        maxiter=options.maxiter,
        blocking=options.blocking,
        trust_region=options.trust_region,
        learning_rate=options.learning_rate,
        perturbation=options.perturbation,
        second_order=options.second_order,
    )

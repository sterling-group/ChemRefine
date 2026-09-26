"""Additional deterministic optimizers for smooth or derivative-free VQE studies."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from chemrefine.engines.qiskit.registry import OPTIMIZERS


class DeterministicOptimizerOptions(BaseModel):
    """Finite iteration budget shared by the optional deterministic optimizers."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    maxiter: int = Field(1000, ge=1, strict=True)


class LBFGSBOptions(DeterministicOptimizerOptions):
    """Controls for bounded limited-memory BFGS and numerical differentiation."""

    maxfun: int = Field(15000, ge=1, strict=True)
    ftol: float = Field(1e-9, gt=0)
    gtol: float = Field(1e-5, gt=0)
    eps: float = Field(1e-8, gt=0)
    maxls: int = Field(20, ge=1, strict=True)


class PowellOptions(DeterministicOptimizerOptions):
    """Function budget and parameter/objective tolerances for Powell's method."""

    maxfev: int = Field(10000, ge=1, strict=True)
    xtol: float = Field(1e-4, gt=0)
    ftol: float = Field(1e-4, gt=0)


class NelderMeadOptions(DeterministicOptimizerOptions):
    """Simplex budgets, convergence tolerances and dimension-aware coefficients."""

    maxfev: int = Field(10000, ge=1, strict=True)
    xatol: float = Field(1e-4, gt=0)
    fatol: float = Field(1e-4, gt=0)
    adaptive: bool = False


class GradientOptimizerOptions(DeterministicOptimizerOptions):
    """Gradient convergence and finite-difference controls for CG or BFGS."""

    gtol: float = Field(1e-5, gt=0)
    eps: float = Field(1.4901161193847656e-8, gt=0)


def _build_scipy(method: str, options: BaseModel) -> object:
    """Use Algorithms' primitive-compatible SciPy wrapper with validated options."""
    from qiskit_algorithms.optimizers import SciPyOptimizer

    return SciPyOptimizer(method=method, options=options.model_dump())


@OPTIMIZERS.register("l_bfgs_b", LBFGSBOptions)
def build_l_bfgs_b(*, options: LBFGSBOptions) -> object:
    """Build bounded limited-memory BFGS for deterministic energy objectives."""
    return _build_scipy("L-BFGS-B", options)


@OPTIMIZERS.register("powell", PowellOptions)
def build_powell(*, options: PowellOptions) -> object:
    """Build Powell's derivative-free direction-set optimizer."""
    return _build_scipy("Powell", options)


@OPTIMIZERS.register("nelder_mead", NelderMeadOptions)
def build_nelder_mead(*, options: NelderMeadOptions) -> object:
    """Build the derivative-free Nelder-Mead simplex optimizer."""
    return _build_scipy("Nelder-Mead", options)


@OPTIMIZERS.register("cg", GradientOptimizerOptions)
def build_cg(*, options: GradientOptimizerOptions) -> object:
    """Build nonlinear conjugate gradients for smooth energy objectives."""
    return _build_scipy("CG", options)


@OPTIMIZERS.register("bfgs", GradientOptimizerOptions)
def build_bfgs(*, options: GradientOptimizerOptions) -> object:
    """Build full-memory BFGS without claiming support for parameter bounds."""
    return _build_scipy("BFGS", options)

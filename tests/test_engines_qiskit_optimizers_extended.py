"""Extended optimizer options and real numerical minimization contracts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from chemrefine.engines.qiskit.components import optimizers_extended as extended
from chemrefine.engines.qiskit.options import ComponentSelection
from chemrefine.engines.qiskit.registry import OPTIMIZERS

_OPTIONS = (
    extended.LBFGSBOptions,
    extended.PowellOptions,
    extended.NelderMeadOptions,
    extended.GradientOptimizerOptions,
)


@pytest.mark.parametrize("model", _OPTIONS)
@pytest.mark.parametrize("invalid", [{"maxiter": 0}, {"maxiter": True}, {"unknown": 1}])
def test_optimizer_options_reject_invalid_budgets_and_unknown_fields(model, invalid):
    """Bad controls fail at validation rather than being discarded by SciPy."""
    with pytest.raises(ValidationError):
        model(**invalid)


@pytest.mark.parametrize(
    ("model", "field"),
    [
        (extended.LBFGSBOptions, "ftol"),
        (extended.PowellOptions, "xtol"),
        (extended.NelderMeadOptions, "fatol"),
        (extended.GradientOptimizerOptions, "gtol"),
    ],
)
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_optimizer_tolerances_are_positive_and_finite(model, field, value):
    """Nonfinite convergence controls must not produce misleading successful runs."""
    with pytest.raises(ValidationError):
        model(**{field: value})


@pytest.mark.parametrize("name", ["l_bfgs_b", "powell", "nelder_mead", "cg", "bfgs"])
def test_each_optimizer_minimizes_a_coupled_quadratic(name):
    """Every registered optimizer performs real work against the supported Algorithms API."""
    pytest.importorskip("qiskit_algorithms")
    optimizer: Any = OPTIMIZERS.build(ComponentSelection(name=name, options={"maxiter": 500}))
    target = np.array([0.7, -0.3])
    matrix = np.array([[3.0, 0.8], [0.8, 1.5]])

    def objective(point):
        displacement = point - target
        return float(displacement @ matrix @ displacement)

    result = optimizer.minimize(objective, x0=np.array([-1.0, 1.2]))
    assert result.fun < 1e-7
    np.testing.assert_allclose(result.x, target, atol=3e-4)
    assert result.nfev > 1


def test_l_bfgs_b_honors_parameter_bounds():
    """A bounded optimum differs from the free optimum and must stay inside the box."""
    pytest.importorskip("qiskit_algorithms")
    optimizer: Any = extended.build_l_bfgs_b(options=extended.LBFGSBOptions())
    result = optimizer.minimize(lambda x: float((x[0] - 2) ** 2), [0.2], bounds=[(0, 1)])
    assert result.x[0] == pytest.approx(1.0, abs=1e-8)
    assert result.fun == pytest.approx(1.0, abs=1e-8)

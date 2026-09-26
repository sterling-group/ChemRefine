"""Sampler-backed quantum natural SPSA for an explicitly fixed circuit ansatz."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from chemrefine.engines.qiskit.components.optimizers import _SPSA_RANDOM_LOCK
from chemrefine.engines.qiskit.context import SolverComponents
from chemrefine.engines.qiskit.registry import OPTIMIZERS
from chemrefine.errors import ConfigError


class QNSPSAOptions(BaseModel):
    """Finite-shot metric estimates and bounded fixed-ansatz optimization."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    maxiter: int = Field(100, ge=1)
    blocking: bool = True
    allowed_increase: float | None = Field(None, ge=0)
    learning_rate: float | None = Field(None, gt=0)
    perturbation: float | None = Field(None, gt=0)
    resamplings: int = Field(1, ge=1)
    regularization: float = Field(0.01, gt=0)
    hessian_delay: int = Field(0, ge=0)
    fidelity_shots: int = Field(4096, ge=1)
    seed: int | None = Field(None, ge=0)

    @model_validator(mode="after")
    def _paired_schedules(self) -> Self:
        """Require scalar rate and perturbation together, matching Qiskit's contract."""
        if (self.learning_rate is None) != (self.perturbation is None):
            raise ValueError("learning_rate and perturbation must be set together")
        return self


@OPTIMIZERS.register("qnspsa", QNSPSAOptions, requires=frozenset({"sampler", "circuit"}))
def build_qnspsa(*, options: QNSPSAOptions, components: SolverComponents, **context: Any) -> Any:
    """Bind the metric to the same circuit and sampler owned by solver assembly."""
    del context
    circuit = components.ansatz.circuit
    if circuit is None or circuit.num_parameters < 1 or components.sampler is None:
        raise ConfigError("QNSPSA requires a parameterized fixed circuit and a sampler")
    from qiskit_algorithms.optimizers import QNSPSA
    from qiskit_algorithms.state_fidelities import ComputeUncompute
    from qiskit_algorithms.utils import algorithm_globals

    metric = ComputeUncompute(
        components.sampler,
        shots=options.fidelity_shots,
        transpiler=components.sampler_transpiler,
        transpiler_options=components.sampler_transpiler_options,
    )
    parameter_count = circuit.num_parameters

    def fidelity(left: Any, right: Any) -> np.ndarray:
        """Evaluate batched overlaps without substituting a local statevector primitive."""
        xs = np.reshape(left, (-1, parameter_count)).tolist()
        ys = np.reshape(right, (-1, parameter_count)).tolist()
        values = np.asarray(
            metric.run([circuit] * len(xs), [circuit] * len(ys), xs, ys).result().fidelities
        )
        if not np.isfinite(values).all():
            raise ConfigError("QNSPSA fidelity returned nonfinite values")
        return values

    random = np.random.default_rng(options.seed)

    class IsolatedQNSPSA(QNSPSA):  # type: ignore[misc]
        """Use a private perturbation stream without consuming ambient Qiskit randomness."""

        def minimize(self, fun: Any, x0: Any, jac: Any = None, bounds: Any = None) -> Any:
            """Restore the process-global stream even when a provider or callback fails."""
            with _SPSA_RANDOM_LOCK:
                upstream = algorithm_globals.random
                previous = deepcopy(upstream.bit_generator.state)
                upstream.bit_generator.state = random.bit_generator.state
                try:
                    return super().minimize(fun, x0, jac=jac, bounds=bounds)
                finally:
                    random.bit_generator.state = upstream.bit_generator.state
                    upstream.bit_generator.state = previous

    return IsolatedQNSPSA(
        fidelity=fidelity,
        **options.model_dump(exclude={"fidelity_shots", "seed"}),
    )

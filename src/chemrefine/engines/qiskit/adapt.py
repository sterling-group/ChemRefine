"""Per-run diagnostics around Qiskit's ADAPT implementation, without replacing its loop."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AdaptDiagnostics:
    """ADAPT candidates, gradient checks, and the final retained operator sequence.

    A terminal gradient check need not add an operator. Energy convergence can
    also roll back the final candidate, so the retained sequence is separate
    from the complete gradient history.
    """

    gradient_history: list[dict[str, Any]] = field(default_factory=list)
    selected_operator_indices: tuple[int, ...] = ()
    selected_operators: tuple[dict[str, Any], ...] = ()


def tracked_adapt_vqe(
    solver: Any,
    *,
    diagnostics: AdaptDiagnostics,
    pool_metadata: tuple[dict[str, Any], ...] = (),
    **options: Any,
) -> Any:
    """Construct ADAPT with instance-local observation of its gradient evaluations.

    Algorithms 0.4 has no public ADAPT iteration callback or selected-index
    result field. The single private hook used here delegates the gradient
    calculation unchanged. After the upstream loop returns, its retained
    excitation count accounts for cycle/gradient stops and energy rollback.
    This compatibility seam is covered by real-stack termination tests.
    """
    from qiskit_algorithms import AdaptVQE

    class TrackedAdaptVQE(AdaptVQE):  # type: ignore[misc]
        """Observe one ADAPT run while leaving selection and convergence upstream."""

        def retained_logical_circuit(self) -> Any:
            """Rebuild the retained ansatz before routing, including upstream rollback."""
            return self._build_ansatz()

        def _compute_gradients(self, theta: list[float], operator: Any) -> Any:
            """Record the same maximum candidate that the upstream loop will choose."""
            gradients = super()._compute_gradients(theta, operator)
            pool_index, maximum = max(enumerate(gradients), key=lambda item: np.abs(item[1][0]))
            gradient = float(np.real_if_close(maximum[0]))
            iteration = len(diagnostics.gradient_history) + 1
            diagnostics.gradient_history.append(
                {
                    "iteration": iteration,
                    "pool_index": pool_index,
                    "gradient": gradient,
                    "max_gradient": abs(gradient),
                    "retained": False,
                }
            )
            logger.info(
                "Qiskit ADAPT iteration %d: maximum gradient %.8g at pool index %d",
                iteration,
                abs(gradient),
                pool_index,
            )
            return gradients

        def compute_minimum_eigenvalue(self, operator: Any, aux_operators: Any = None) -> Any:
            """Reset run diagnostics and record the final sequence after upstream rollback."""
            diagnostics.gradient_history.clear()
            diagnostics.selected_operator_indices = ()
            diagnostics.selected_operators = ()
            result = super().compute_minimum_eigenvalue(operator, aux_operators)
            selected_count = len(self._excitation_list)
            retained = diagnostics.gradient_history[:selected_count]
            diagnostics.selected_operator_indices = tuple(entry["pool_index"] for entry in retained)
            for entry in retained:
                entry["retained"] = True
            diagnostics.selected_operators = tuple(
                dict(pool_metadata[index])
                if pool_metadata
                else {"pool_index": index, "label": f"operator_{index}"}
                for index in diagnostics.selected_operator_indices
            )
            return result

    return TrackedAdaptVQE(solver, **options)

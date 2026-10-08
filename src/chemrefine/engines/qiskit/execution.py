"""Bounded execution adapters for algorithm-generated logical publications."""

from __future__ import annotations

from threading import Lock
from typing import Any

import numpy as np

from chemrefine.engines.qiskit.context import EstimatorResource
from chemrefine.errors import ConfigError


def logical_estimator(resource: EstimatorResource, *, max_publications: int) -> Any:
    """Compile every generated publication, including gradient ancillas, once.

    The adapter preserves parameter binding order and broadcasts of observables.
    It does not own the provider lifecycle; callers keep the resource open.
    """
    from qiskit.primitives import BaseEstimatorV2
    from qiskit.primitives.containers.estimator_pub import EstimatorPub
    from qiskit.quantum_info import SparsePauliOp

    class LogicalEstimator(BaseEstimatorV2):  # type: ignore[misc]
        """A provider-transparent publication budget and logical-to-physical boundary."""

        def __init__(self) -> None:
            """Start a fresh workload counter for one scientific experiment."""
            self.publications = 0
            self.default_precision = getattr(resource.estimator, "default_precision", None)
            self._budget_lock = Lock()

        def run(self, pubs: Any, *, precision: float | None = None) -> Any:
            """Map each observable with the layout of its own compiled circuit."""
            prepared = [EstimatorPub.coerce(pub, precision) for pub in pubs]
            count = sum(pub.size for pub in prepared)
            with self._budget_lock:
                if self.publications + count > max_publications:
                    raise ConfigError("quantum execution exceeds max_publications")
                self.publications += count
            if resource.transpiler is not None:
                compiled = []
                for pub in prepared:
                    circuit = resource.transpiler.run(
                        pub.circuit, **(resource.transpiler_options or {})
                    )
                    try:
                        parameters = pub.parameter_values.as_array(circuit.parameters)
                    except ValueError as exc:
                        raise ConfigError(
                            "transpilation changed the logical parameter set"
                        ) from exc
                    observables = np.empty(pub.observables.shape, dtype=object)
                    for index in np.ndindex(pub.observables.shape):
                        terms = pub.observables[index]
                        operator = SparsePauliOp.from_list(list(terms.items()))
                        observables[index] = operator.apply_layout(circuit.layout)
                    compiled.append((circuit, observables, parameters, pub.precision))
                prepared = compiled
            return resource.estimator.run(prepared)

    return LogicalEstimator()

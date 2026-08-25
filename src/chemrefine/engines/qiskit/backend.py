"""Managed-environment declaration shared by Qiskit engine variants."""

from __future__ import annotations

from typing import Any

from chemrefine.engines.api import BackendRequirement

_QISKIT = BackendRequirement(
    extra="qiskit",
    import_name="qiskit_nature",
)


class QiskitBackend:
    """Mixin declaring the managed environment required by Qiskit engines."""

    def backend_requirement(self, options: dict[str, Any] | None) -> BackendRequirement:
        """Validate the graph and probe Aer when the selected algorithm will use it."""
        from chemrefine.engines.qiskit.options import QiskitOptions
        from chemrefine.engines.qiskit.registry import ALGORITHMS, ESTIMATORS
        from chemrefine.engines.qiskit.workflow import validate_options

        resolved = QiskitOptions.from_raw(options)
        validate_options(resolved)
        requirements = ALGORITHMS.spec(resolved.algorithm.name).requires
        estimator_requirement = ESTIMATORS.spec(resolved.estimator.name).backend_requirement
        if "estimator" in requirements and estimator_requirement is not None:
            return estimator_requirement
        return _QISKIT

    def backend_extras(self) -> frozenset[str]:
        """Return core Qiskit plus provider extras declared by estimators."""
        from chemrefine.engines.qiskit.registry import ESTIMATORS

        extras = {_QISKIT.extra}
        extras.update(
            requirement.extra
            for name in ESTIMATORS.names()
            if (requirement := ESTIMATORS.spec(name).backend_requirement) is not None
        )
        return frozenset(extras)

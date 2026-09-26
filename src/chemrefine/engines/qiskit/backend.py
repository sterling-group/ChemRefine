"""Managed-environment declaration shared by Qiskit engine variants."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from chemrefine.engines.api import BackendRequirement
from chemrefine.errors import ConfigError

_QISKIT = BackendRequirement(
    extra="qiskit",
    import_name="qiskit_nature",
)


class QiskitBackend:
    """Mixin declaring the managed environment required by Qiskit engines."""

    def backend_requirement(self, options: Mapping[str, Any] | None) -> BackendRequirement:
        """Select one worker environment containing every consumed optional component."""
        from chemrefine.engines.qiskit.options import QiskitOptions
        from chemrefine.engines.qiskit.registry import ALGORITHMS, REGISTRIES
        from chemrefine.engines.qiskit.workflow import validate_options

        resolved = QiskitOptions.from_raw(options)
        validate_options(resolved)
        algorithm = ALGORITHMS.spec(resolved.algorithm.name)
        categories = {"algorithm"} | (algorithm.requires & REGISTRIES.keys())
        if algorithm.execution == "nature":
            categories.add("mapper")
        if algorithm.requires & {"circuit", "operator_pool"}:
            categories.add("ansatz")
        if resolved.algorithm.name == "sqd" and resolved.algorithm.options.get("counts") is None:
            categories.update({"mapper", "ansatz", "initial_state", "initial_point"})
        selections = resolved.component_selections()
        requirements = {
            requirement.extra: requirement
            for category in categories
            if (
                requirement := REGISTRIES[category]
                .spec(selections[category].name)
                .backend_requirement
            )
            is not None
        }
        if not requirements:
            return _QISKIT
        # Built-in extras form a documented dependency chain. Unknown providers
        # cannot silently replace a second independently required environment.
        included = {
            "qiskit": {"qiskit"},
            "qiskit-aer": {"qiskit", "qiskit-aer"},
            "qiskit-fermionic": {"qiskit", "qiskit-aer", "qiskit-fermionic"},
        }
        for extra, requirement in requirements.items():
            if requirements.keys() <= included.get(extra, {extra, "qiskit"}):
                return requirement
        raise ConfigError(
            "qiskit selected components require incompatible managed environments: "
            f"{sorted(requirements)}; register a shared provider environment"
        )

    def backend_extras(self) -> frozenset[str]:
        """Return core Qiskit plus provider extras declared by all component registries."""
        from chemrefine.engines.qiskit.registry import REGISTRIES

        extras = {_QISKIT.extra}
        extras.update(
            requirement.extra
            for registry in REGISTRIES.values()
            for name in registry.names()
            if (requirement := registry.spec(name).backend_requirement) is not None
        )
        return frozenset(extras)

"""Managed-environment declaration shared by Qiskit engine variants."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from chemrefine.engines.api import BackendRequirement
from chemrefine.engines.qiskit.profiles import BACKEND_PROFILES


class QiskitBackend:
    """Mixin declaring the managed environment required by Qiskit engines."""

    def backend_requirement(self, options: Mapping[str, Any] | None) -> BackendRequirement:
        """Select one worker environment containing every consumed optional component."""
        from chemrefine.engines.qiskit.options import QiskitOptions
        from chemrefine.engines.qiskit.registry import REGISTRIES, consumed_component_categories
        from chemrefine.engines.qiskit.workflow import validate_options

        resolved = QiskitOptions.from_raw(options)
        validate_options(resolved)
        categories = consumed_component_categories(resolved)
        selections = resolved.component_selections()
        requirements = [
            requirement
            for category in categories
            if (
                requirement := REGISTRIES[category]
                .spec(selections[category].name)
                .backend_requirement
            )
            is not None
        ]
        return BACKEND_PROFILES.resolve(requirements)

    def backend_extras(self) -> frozenset[str]:
        """Return core Qiskit plus provider extras declared by all component registries."""
        from chemrefine.engines.qiskit.registry import REGISTRIES

        extras = set(BACKEND_PROFILES.names())
        extras.update(
            requirement.extra
            for registry in REGISTRIES.values()
            for name in registry.names()
            if (requirement := registry.spec(name).backend_requirement) is not None
        )
        return frozenset(extras)

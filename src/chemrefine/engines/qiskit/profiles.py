"""Declarative compatible environments for a graph of optional quantum providers.

A profile names an ordinary ChemRefine extra. It does not install packages or merge
existing environments: the extra itself must contain every provider it declares.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from chemrefine.engines.api import BackendRequirement
from chemrefine.errors import ConfigError


@dataclass(frozen=True)
class BackendProfile:
    """One installable environment and the provider extras it includes."""

    requirement: BackendRequirement
    includes: frozenset[str] = frozenset()


class BackendProfileRegistry:
    """Resolve required providers to the smallest declared compatible environment."""

    def __init__(self) -> None:
        self._profiles: dict[str, BackendProfile] = {}

    def register(
        self, requirement: BackendRequirement, *, includes: frozenset[str] = frozenset()
    ) -> None:
        """Declare a profile; referenced profiles may register before or after it."""
        if requirement.extra in self._profiles:
            raise ValueError(f"quantum backend profile {requirement.extra!r} is already registered")
        self._profiles[requirement.extra] = BackendProfile(requirement, includes)

    def names(self) -> frozenset[str]:
        """Return the installable profile extras, including combined environments."""
        return frozenset(self._profiles)

    def provided_extras(self, extra: str) -> frozenset[str]:
        """Expand included profiles transitively, rejecting cyclic declarations."""

        def visit(name: str, ancestors: frozenset[str]) -> set[str]:
            """Collect one declaration, treating an unregistered extra as a leaf."""
            if name in ancestors:
                raise ConfigError(f"cyclic quantum backend profile includes {name!r}")
            profile = self._profiles.get(name)
            result = {name}
            if profile is not None:
                for included in sorted(profile.includes):
                    result.update(visit(included, ancestors | {name}))
            return result

        return frozenset(visit(extra, frozenset()))

    def resolve(self, requirements: Iterable[BackendRequirement]) -> BackendRequirement:
        """Choose one profile; an independent custom provider can supply its own base.

        Existing third-party components may declare only a ``BackendRequirement``.
        A sole such provider remains usable and is expected to include core Qiskit;
        combining it with another provider requires an explicit shared profile.
        """
        requested = {requirement.extra: requirement for requirement in requirements}
        required = set(requested) | {"qiskit"}
        candidates: list[tuple[int, str, BackendRequirement]] = []
        for extra in sorted(self.names() | requested.keys()):
            if extra in self._profiles:
                supplied = self.provided_extras(extra)
                requirement = self._profiles[extra].requirement
            else:
                supplied = frozenset({extra, "qiskit"})
                requirement = requested[extra]
            if required <= supplied:
                candidates.append((len(supplied), extra, requirement))
        if not candidates:
            raise ConfigError(
                "qiskit selected components require incompatible managed environments: "
                f"{sorted(required)}; register a shared provider environment"
            )
        return min(candidates, key=lambda candidate: candidate[:2])[2]


BACKEND_PROFILES = BackendProfileRegistry()
BACKEND_PROFILES.register(BackendRequirement(extra="qiskit", import_name="qiskit_nature"))
BACKEND_PROFILES.register(
    BackendRequirement(extra="qiskit-aer", import_name="qiskit_aer"),
    includes=frozenset({"qiskit"}),
)
BACKEND_PROFILES.register(
    BackendRequirement(extra="qiskit-fermionic", import_name="ffsim"),
    includes=frozenset({"qiskit-aer"}),
)
BACKEND_PROFILES.register(
    BackendRequirement(extra="qiskit-resources", import_name="openfermion"),
    includes=frozenset({"qiskit"}),
)
BACKEND_PROFILES.register(
    BackendRequirement(extra="qiskit-runtime", import_name="qiskit_ibm_runtime"),
    includes=frozenset({"qiskit-aer"}),
)
BACKEND_PROFILES.register(
    BackendRequirement(extra="qiskit-cutting", import_name="qiskit_addon_cutting"),
    includes=frozenset({"qiskit-aer"}),
)
BACKEND_PROFILES.register(
    BackendRequirement(extra="qiskit-rdm", import_name="cvxpy"),
    includes=frozenset({"qiskit"}),
)
BACKEND_PROFILES.register(
    BackendRequirement(extra="qiskit-toolkit", import_name="qiskit_ibm_runtime"),
    includes=frozenset({"qiskit-fermionic", "qiskit-runtime", "qiskit-cutting", "qiskit-rdm"}),
)

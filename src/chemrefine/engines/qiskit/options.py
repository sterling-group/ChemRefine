"""Typed configuration for the modular Qiskit Nature engine.

Qiskit objects are deliberately not imported here. The orchestrator imports this
module while loading every ChemRefine engine, including installations that do not
have the optional ``qiskit`` extra. Concrete objects are created lazily by the
component builders inside the backend interpreter.
"""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from chemrefine.engines._options import EngineOptions


class ComponentSelection(BaseModel):
    """One named registry component and the options passed to its builder."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_-]*$")
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", mode="before")
    @classmethod
    def _normalize_name(cls, value: Any) -> Any:
        """Normalize human-entered component names before applying the key pattern."""
        if isinstance(value, str):
            return value.strip().lower().replace("-", "_")
        return value

    @classmethod
    def named(cls, name: str) -> Self:
        """Construct a selection with no component-specific options."""
        return cls(name=name)


class ActiveSpaceOptions(BaseModel):
    """Active-space reduction applied before mapper and solver construction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    electrons: int | tuple[int, int] = Field(default=2)
    orbitals: int = Field(default=2, ge=1)

    @field_validator("electrons")
    @classmethod
    def _positive_electron_counts(cls, value: int | tuple[int, int]) -> int | tuple[int, int]:
        """Require a positive total, or non-negative alpha/beta populations."""
        if isinstance(value, int):
            if value < 1:
                raise ValueError("electrons must be at least 1")
            return value
        if len(value) != 2 or any(count < 0 for count in value) or sum(value) < 1:
            raise ValueError("electrons must be [n_alpha, n_beta] with a positive total")
        return value

    @model_validator(mode="after")
    def _electrons_fit_orbitals(self) -> Self:
        """Reject electron populations that exceed the active spin-orbital capacity."""
        if isinstance(self.electrons, int):
            if self.electrons > 2 * self.orbitals:
                raise ValueError(
                    f"{self.electrons} active electrons do not fit in "
                    f"{self.orbitals} spatial orbitals"
                )
        elif any(population > self.orbitals for population in self.electrons):
            raise ValueError(
                f"active alpha/beta populations {self.electrons} do not fit in "
                f"{self.orbitals} spatial orbitals"
            )
        return self


class QiskitOptions(EngineOptions):
    """Validated component graph for a Qiskit Nature ground-state calculation."""

    basis: str = Field("sto-3g", min_length=1)
    active_space: ActiveSpaceOptions | None = None

    mapper: ComponentSelection = Field(
        default_factory=lambda: ComponentSelection.named("jordan_wigner")
    )
    algorithm: ComponentSelection = Field(default_factory=lambda: ComponentSelection.named("exact"))
    ansatz: ComponentSelection = Field(default_factory=lambda: ComponentSelection.named("uccsd"))
    initial_state: ComponentSelection = Field(
        default_factory=lambda: ComponentSelection.named("hartree_fock")
    )
    estimator: ComponentSelection = Field(
        default_factory=lambda: ComponentSelection.named("statevector")
    )
    optimizer: ComponentSelection = Field(default_factory=lambda: ComponentSelection.named("slsqp"))
    initial_point: ComponentSelection = Field(
        default_factory=lambda: ComponentSelection.named("zeros")
    )

    @field_validator(
        "mapper",
        "algorithm",
        "ansatz",
        "initial_state",
        "estimator",
        "optimizer",
        "initial_point",
        mode="before",
    )
    @classmethod
    def _accept_short_component_name(cls, value: Any) -> Any:
        """Accept ``algorithm: vqe`` as shorthand for ``algorithm: {name: vqe}``."""
        if isinstance(value, str):
            return {"name": value}
        return value

    def component_selections(self) -> dict[str, ComponentSelection]:
        """Return every registry-backed selection, keyed by component category."""
        return {
            "mapper": self.mapper,
            "algorithm": self.algorithm,
            "ansatz": self.ansatz,
            "initial_state": self.initial_state,
            "estimator": self.estimator,
            "optimizer": self.optimizer,
            "initial_point": self.initial_point,
        }

    def as_job_spec(self) -> dict[str, Any]:
        """Return a JSON-compatible, fully defaulted specification for the runner."""
        return self.model_dump(mode="json")

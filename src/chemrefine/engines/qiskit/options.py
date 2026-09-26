"""Typed configuration for the modular Qiskit Nature engine.

Qiskit objects are deliberately not imported here. The orchestrator imports this
module while loading every ChemRefine engine, including installations that do not
have the optional ``qiskit`` extra. Concrete objects are created lazily by the
component builders inside the backend interpreter.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from chemrefine.engines._options import EngineOptions

logger = logging.getLogger(__name__)


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
    active_orbitals: list[StrictInt] | None = None

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
        if self.active_orbitals is not None and (
            len(self.active_orbitals) != self.orbitals
            or len(set(self.active_orbitals)) != self.orbitals
            or any(index < 0 for index in self.active_orbitals)
        ):
            raise ValueError(
                "active_orbitals must contain one distinct non-negative index per orbital"
            )
        return self


class IntegralSourceOptions(BaseModel):
    """A portable MO integral input explicitly associated with the current geometry."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    bundle_path: str = Field(
        min_length=1, json_schema_extra={"input_file": True, "file_format": "quantum_bundle"}
    )
    max_input_bytes: StrictInt = Field(536870912, ge=1)
    geometry_tolerance_angstrom: float = Field(1e-7, gt=0, le=1e-3)


class CircuitExportOptions(BaseModel):
    """Retain logical bound preparations for supported circuit-producing algorithms."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    max_bytes: StrictInt = Field(33554432, ge=1)


class QiskitOptions(EngineOptions):
    """Validated component graph for a Qiskit Nature ground-state calculation."""

    basis: str = Field("sto-3g", min_length=1)
    active_space: ActiveSpaceOptions | None = None
    freeze_core: bool = False
    integral_source: IntegralSourceOptions | None = None
    circuit_export: CircuitExportOptions | None = None

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
    sampler: ComponentSelection = Field(
        default_factory=lambda: ComponentSelection.named("statevector")
    )
    optimizer: ComponentSelection = Field(default_factory=lambda: ComponentSelection.named("slsqp"))
    initial_point: ComponentSelection = Field(
        default_factory=lambda: ComponentSelection.named("zeros")
    )

    @classmethod
    def accepted_names(cls) -> set[str]:
        """Include the pre-release active-space spellings in every options reader."""
        return super().accepted_names() | {"active_electrons", "active_orbitals"}

    @model_validator(mode="before")
    @classmethod
    def _normalize_active_space(cls, data: Any) -> Any:
        """Keep the pre-release flat active-space pair compatible inside this engine."""
        if not isinstance(data, Mapping):
            return data
        if not {"active_electrons", "active_orbitals"}.intersection(data):
            return data
        options = dict(data)
        electrons = options.pop("active_electrons", None)
        orbitals = options.pop("active_orbitals", None)
        if "active_space" in options:
            raise ValueError(
                "use either active_space or active_electrons/active_orbitals, not both"
            )
        if electrons is None or orbitals is None:
            raise ValueError("active_electrons and active_orbitals must be provided together")
        logger.warning(
            "Qiskit `active_electrons`/`active_orbitals` are deprecated; "
            "use `active_space: {electrons: ..., orbitals: ...}`"
        )
        options["active_space"] = {"electrons": electrons, "orbitals": orbitals}
        return options

    @field_validator(
        "mapper",
        "algorithm",
        "ansatz",
        "initial_state",
        "estimator",
        "sampler",
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
            "sampler": self.sampler,
            "optimizer": self.optimizer,
            "initial_point": self.initial_point,
        }

    def as_job_spec(self) -> dict[str, Any]:
        """Return a JSON-compatible, fully defaulted specification for the runner."""
        return self.model_dump(mode="json")

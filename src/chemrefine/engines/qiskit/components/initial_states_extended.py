"""Explicit determinant preparation in the transformed problem's orbital basis."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, StrictInt, field_validator

from chemrefine.engines.qiskit.components.initial_states import (
    REFERENCE_OCCUPATIONS_KEY as REFERENCE_OCCUPATIONS_KEY,
)
from chemrefine.engines.qiskit.components.initial_states import build_explicit_reference
from chemrefine.engines.qiskit.context import ElectronicStructureContext
from chemrefine.engines.qiskit.registry import INITIAL_STATES
from chemrefine.errors import ConfigError


class DeterminantOptions(BaseModel):
    """Occupied spatial-orbital indices separately for alpha and beta electrons.

    Indices refer to the prepared active-space ordering, not the original molecule.
    The populations must match the prepared problem. A determinant with these
    populations need not be an eigenstate of total spin squared.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    alpha: tuple[StrictInt, ...]
    beta: tuple[StrictInt, ...]

    @field_validator("alpha", "beta")
    @classmethod
    def _distinct_nonnegative_indices(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """Reject repeated or negative orbitals before importing optional packages."""
        if any(index < 0 for index in value) or len(set(value)) != len(value):
            raise ValueError("occupied orbital indices must be distinct and non-negative")
        return value


@INITIAL_STATES.register("determinant", DeterminantOptions)
def build_determinant(
    *, options: DeterminantOptions, context: ElectronicStructureContext
) -> object:
    """Prepare an explicit determinant and record occupations for excitation builders."""
    populations = (len(options.alpha), len(options.beta))
    if populations != context.num_particles:
        raise ConfigError(
            f"qiskit determinant populations {populations} must match "
            f"the prepared particle counts {context.num_particles}"
        )
    orbitals = context.num_spatial_orbitals
    if any(index >= orbitals for index in options.alpha + options.beta):
        raise ConfigError(f"qiskit determinant orbital indices must be below {orbitals}")
    alpha = tuple(index in options.alpha for index in range(orbitals))
    beta = tuple(index in options.beta for index in range(orbitals))
    circuit: Any = build_explicit_reference(context, alpha + beta)
    circuit.metadata = {
        **(circuit.metadata or {}),
        REFERENCE_OCCUPATIONS_KEY: {"alpha": list(alpha), "beta": list(beta)},
    }
    return circuit

"""Versioned, pickle-free electronic-integral bundles with owned data validation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, Self, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from chemrefine.engines.qiskit.bundles import DEFAULT_MAX_BYTES, read_bundle, write_bundle
from chemrefine.engines.qiskit.data import ElectronicStructureData, MolecularMetadata
from chemrefine.errors import ConfigError, OutputParseError

INTEGRAL_ARRAY_FIELDS = frozenset(
    {
        "one_body_integrals",
        "two_body_integrals",
        "one_body_integrals_beta",
        "two_body_integrals_beta_beta",
        "two_body_integrals_beta_alpha",
        "overlap_alpha_beta",
        "orbital_energies",
        "orbital_energies_beta",
        "orbital_occupations",
        "orbital_occupations_beta",
    }
)


class IntegralDescription(BaseModel):
    """Physical parameters, array references and tensor conventions for one input."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    version: Literal[1] = 1
    units: Literal["hartree"] = "hartree"
    orbital_order: Literal["alpha_then_beta"] = "alpha_then_beta"
    num_alpha: StrictInt = Field(ge=0)
    num_beta: StrictInt = Field(ge=0)
    num_spatial_orbitals: StrictInt = Field(ge=1)
    two_body_order: Literal["chemist", "physicist"]
    nuclear_repulsion_energy: float | None = Field(None, ge=0)
    multiplicity: StrictInt = Field(ge=1)
    molecular_metadata: dict[str, Any] | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    arrays: dict[str, str]

    @model_validator(mode="after")
    def _array_names(self) -> Self:
        """Require owned integral tensors and reject undeclared data fields or aliases."""
        if not {"one_body_integrals", "two_body_integrals"} <= self.arrays.keys():
            raise ValueError("integral bundle requires one-body and two-body tensors")
        if self.arrays.keys() - INTEGRAL_ARRAY_FIELDS or len(set(self.arrays.values())) != len(
            self.arrays
        ):
            raise ValueError("integral bundle has unknown or aliased array fields")
        return self


def save_integrals(
    path: Path, data: ElectronicStructureData, *, max_bytes: int = DEFAULT_MAX_BYTES
) -> Path:
    """Write owned complex or unrestricted integral data as a native output bundle."""
    arrays = {
        name: np.asarray(getattr(data, name))
        for name in INTEGRAL_ARRAY_FIELDS
        if getattr(data, name) is not None
    }
    description = IntegralDescription(
        num_alpha=data.num_alpha,
        num_beta=data.num_beta,
        num_spatial_orbitals=data.num_spatial_orbitals,
        two_body_order=data.two_body_order,
        nuclear_repulsion_energy=data.nuclear_repulsion_energy,
        multiplicity=cast("int", data.multiplicity),
        molecular_metadata=None
        if data.molecular_metadata is None
        else asdict(data.molecular_metadata),
        provenance=data.provenance,
        metadata=data.metadata,
        arrays={name: name for name in arrays},
    )
    return write_bundle(
        path,
        kind="electronic_structure",
        arrays=arrays,
        metadata=description.model_dump(mode="json"),
        max_bytes=max_bytes,
    )


def load_integrals(path: Path, *, max_bytes: int = DEFAULT_MAX_BYTES) -> ElectronicStructureData:
    """Read verified arrays and revalidate physical dimensions and tensor symmetries."""
    bundle = read_bundle(path, max_bytes=max_bytes)
    try:
        if bundle.description.kind != "electronic_structure":
            raise ValueError("unsupported electronic structure bundle")
        description = IntegralDescription.model_validate(bundle.metadata)
        if set(description.arrays.values()) != bundle.arrays.keys():
            raise ValueError("integral arrays differ from their declared fields")
        values = description.model_dump(exclude={"arrays", "version", "units", "orbital_order"})
        if description.molecular_metadata is not None:
            values["molecular_metadata"] = MolecularMetadata(**description.molecular_metadata)
        values.update({name: bundle.arrays[key] for name, key in description.arrays.items()})
        return ElectronicStructureData(**values)
    except (KeyError, TypeError, ValueError, ConfigError) as exc:
        raise OutputParseError(f"invalid electronic structure integrals in {path}: {exc}") from exc

"""Bounded fermionic lattice dynamics, separate from molecular energy workflows.

Qiskit Fermions 0.1 supplies Jordan-Wigner and generic mapping synthesis.
Bravyi-Kitaev and untapered parity use Qiskit Nature through that public synthesis
interface. These encodings have one qubit per mode and require no auxiliary
stabilizers. Flow-set and Verstraete-Cirac encodings are not supplied by that
released SDK and are deliberately not represented as supported choices here.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal, Self, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from chemrefine.errors import ConfigError


class LatticeEdge(BaseModel):
    """An undirected edge contributing hopping and density-density interaction.

    The hopping term is ``-hopping * (a†_i a_j + a†_j a_i)`` per spin species.
    The interaction is ``density_interaction * n_i * n_j``, where each site's
    density includes both spins for a spinful model. Every edge is listed once.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    source: StrictInt = Field(ge=0)
    target: StrictInt = Field(ge=0)
    hopping: float = 1.0
    density_interaction: float = 0.0


class FermionicLatticeModel(BaseModel):
    """Real hopping graph with Hubbard onsite and optional extended interactions.

    Spinful mode order is all alpha sites followed by all beta sites. Onsite
    interaction means ``U n_up n_down``; potentials multiply total site density.
    No chemical-potential, half-filling, or constant-energy shifts are implicit.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    num_sites: StrictInt = Field(ge=1)
    edges: tuple[LatticeEdge, ...] = ()
    spinful: bool = True
    onsite_interaction: float = 0.0
    site_potentials: tuple[float, ...] | None = None

    @model_validator(mode="after")
    def _validate_graph(self) -> Self:
        """Reject duplicated, missing, or physically ambiguous graph entries."""
        seen = set()
        for edge in self.edges:
            if edge.source == edge.target or max(edge.source, edge.target) >= self.num_sites:
                raise ValueError("lattice edges must join different sites below num_sites")
            key = tuple(sorted((edge.source, edge.target)))
            if key in seen:
                raise ValueError("each undirected lattice edge must be specified once")
            seen.add(key)
        if not self.spinful and self.onsite_interaction != 0:
            raise ValueError("onsite Hubbard interaction requires a spinful model")
        if self.site_potentials is not None and len(self.site_potentials) != self.num_sites:
            raise ValueError("site_potentials requires one value per site")
        return self

    @property
    def num_modes(self) -> int:
        """Return the number of spin orbitals, which equals the untapered qubit count."""
        return self.num_sites * (2 if self.spinful else 1)


def chain_lattice(
    num_sites: int,
    *,
    periodic: bool = False,
    spinful: bool = True,
    hopping: float = 1.0,
    onsite_interaction: float = 0.0,
    density_interaction: float = 0.0,
    site_potentials: Sequence[float] | None = None,
) -> FermionicLatticeModel:
    """Construct a simple open chain or ring with each undirected edge once."""
    if isinstance(num_sites, bool) or not isinstance(num_sites, Integral) or num_sites < 1:
        raise ConfigError("lattice num_sites must be a positive integer")
    pairs = [(index, index + 1) for index in range(num_sites - 1)]
    if periodic and num_sites > 2:
        pairs.append((0, num_sites - 1))
    return FermionicLatticeModel(
        num_sites=int(num_sites),
        edges=tuple(
            LatticeEdge(
                source=first,
                target=second,
                hopping=hopping,
                density_interaction=density_interaction,
            )
            for first, second in pairs
        ),
        spinful=spinful,
        onsite_interaction=onsite_interaction,
        site_potentials=None if site_potentials is None else tuple(site_potentials),
    )


def square_lattice(
    rows: int,
    columns: int,
    *,
    periodic: bool = False,
    spinful: bool = True,
    hopping: float = 1.0,
    onsite_interaction: float = 0.0,
    density_interaction: float = 0.0,
    site_potentials: Sequence[float] | None = None,
) -> FermionicLatticeModel:
    """Construct a row-major rectangular nearest-neighbor simple graph.

    A periodic dimension of length two still has one edge between its sites,
    rather than silently introducing parallel bonds with doubled hopping.
    """
    if any(
        isinstance(size, bool) or not isinstance(size, Integral) or size < 1
        for size in (rows, columns)
    ):
        raise ConfigError("lattice rows and columns must be positive integers")
    pairs = set()
    for row in range(rows):
        for column in range(columns):
            here = row * columns + column
            for neighbor_row, neighbor_column in ((row + 1, column), (row, column + 1)):
                if periodic:
                    neighbor_row %= rows
                    neighbor_column %= columns
                elif neighbor_row >= rows or neighbor_column >= columns:
                    continue
                there = neighbor_row * columns + neighbor_column
                if here != there:
                    pairs.add(tuple(sorted((here, there))))
    return FermionicLatticeModel(
        num_sites=int(rows * columns),
        edges=tuple(
            LatticeEdge(
                source=first,
                target=second,
                hopping=hopping,
                density_interaction=density_interaction,
            )
            for first, second in sorted(pairs)
        ),
        spinful=spinful,
        onsite_interaction=onsite_interaction,
        site_potentials=None if site_potentials is None else tuple(site_potentials),
    )


class LatticeDynamicsOptions(BaseModel):
    """Ideal simulation and product-formula controls with explicit resource limits."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    time: float = 1.0
    steps: StrictInt = Field(1, ge=1)
    order: Literal[1, 2, 4] = 2
    mapping: Literal["jordan_wigner", "bravyi_kitaev", "parity"] = "jordan_wigner"
    exact_reference: bool = False
    max_qubits: StrictInt = Field(16, ge=1)
    max_evolution_blocks: StrictInt = Field(10000, ge=1)
    max_statevector_bytes: StrictInt = Field(268435456, ge=1)
    max_exact_qubits: StrictInt = Field(12, ge=1)


@dataclass(frozen=True)
class LatticeCircuit:
    """Explicitly mapped circuit and physical observables, with no auxiliary qubits."""

    circuit: Any
    fermionic_circuit: Any
    qubit_hamiltonian: Any
    number_observables: tuple[Any, ...]
    initial_circuit: Any
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LatticeDynamicsResult:
    """Encoded ideal state and lattice observables; never a molecular single-point result."""

    statevector: NDArray[np.complex128]
    mode_occupations: tuple[float, ...]
    particle_number: float
    energy_expectation: float
    energy_drift: float
    return_probability: float
    exact_state_fidelity: float | None
    metadata: dict[str, Any]

    def as_dict(self, *, include_statevector: bool = False) -> dict[str, Any]:
        """Return JSON-compatible observables and optional encoded complex amplitudes."""
        result = {
            "mode_occupations": list(self.mode_occupations),
            "particle_number": self.particle_number,
            "energy_expectation": self.energy_expectation,
            "energy_drift": self.energy_drift,
            "return_probability": self.return_probability,
            "exact_state_fidelity": self.exact_state_fidelity,
            "metadata": self.metadata,
        }
        if include_statevector:
            result["statevector"] = {
                "real": self.statevector.real.tolist(),
                "imag": self.statevector.imag.tolist(),
            }
        return cast("dict[str, Any]", json.loads(json.dumps(result, allow_nan=False)))

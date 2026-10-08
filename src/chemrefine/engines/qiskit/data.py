"""ChemRefine-owned, driver-independent molecular-orbital integral inputs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from chemrefine.errors import ConfigError


def _array(
    value: ArrayLike, shape: tuple[int, ...], name: str, *, complex_allowed: bool = False
) -> NDArray[Any]:
    """Copy finite input into a read-only array with a prescribed shape and domain."""
    try:
        if np.iscomplexobj(value) and not complex_allowed:
            raise ValueError("complex-valued integrals are not supported")
        array = np.array(value, dtype=complex if np.iscomplexobj(value) else float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"qiskit {name} must contain valid numbers: {exc}") from exc
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ConfigError(f"qiskit {name} must be finite with shape {shape}")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class MolecularMetadata:
    """Optional molecular geometry; coordinates are in angstroms."""

    symbols: tuple[str, ...]
    coordinates: tuple[tuple[float, float, float], ...]
    charge: int = 0

    def __post_init__(self) -> None:
        """Validate geometry without importing a quantum-chemistry driver."""
        from ase.data import atomic_numbers

        if not self.symbols or any(atomic_numbers.get(symbol, 0) == 0 for symbol in self.symbols):
            raise ConfigError("qiskit molecular symbols must contain valid element symbols")
        if isinstance(self.charge, bool) or not isinstance(self.charge, int):
            raise ConfigError("qiskit molecular charge must be an integer")
        coordinates = _array(self.coordinates, (len(self.symbols), 3), "coordinates")
        object.__setattr__(self, "symbols", tuple(self.symbols))
        object.__setattr__(self, "coordinates", tuple(map(tuple, coordinates.tolist())))


@dataclass(frozen=True)
class ElectronicStructureData:
    """Real or complex integrals in orthonormal molecular orbitals, in atomic units.

    ``two_body_order`` explicitly declares chemist ``(pq|rs)`` or Qiskit
    physicist ordering; the engine never guesses from tensor symmetries.
    Omitted beta blocks mean restricted/shared spatial orbitals. Unrestricted
    inputs supply all three beta blocks and the alpha-beta orbital overlap
    required for the spin observable. Spin orbitals are alpha then beta.
    Default occupations fill the lowest orbitals independently for each spin.
    ``energy_offsets`` names electronic scalar constants excluded from the supplied
    integral tensors. Nuclear repulsion remains its own separate field. Preparation
    namespaces these constants as ``input:<name>`` to preserve transformation offsets.
    """

    num_alpha: int
    num_beta: int
    num_spatial_orbitals: int
    one_body_integrals: ArrayLike
    two_body_integrals: ArrayLike
    two_body_order: Literal["chemist", "physicist"] = "chemist"
    nuclear_repulsion_energy: float | None = None
    one_body_integrals_beta: ArrayLike | None = None
    two_body_integrals_beta_beta: ArrayLike | None = None
    two_body_integrals_beta_alpha: ArrayLike | None = None
    overlap_alpha_beta: ArrayLike | None = None
    orbital_energies: ArrayLike | None = None
    orbital_energies_beta: ArrayLike | None = None
    orbital_occupations: ArrayLike | None = None
    orbital_occupations_beta: ArrayLike | None = None
    multiplicity: int | None = None
    molecular_metadata: MolecularMetadata | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    energy_offsets: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject inconsistent sizes, populations, spin blocks, and integral tensors."""
        n = self.num_spatial_orbitals
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (n, self.num_alpha, self.num_beta)
            )
            or n < 1
        ):
            raise ConfigError("qiskit orbital and electron counts must be integers; orbitals >= 1")
        if any(count < 0 or count > n for count in self.num_particles):
            raise ConfigError("qiskit alpha/beta electrons must lie between zero and orbital count")
        if self.two_body_order not in {"chemist", "physicist"}:
            raise ConfigError("qiskit two_body_order must be 'chemist' or 'physicist'")
        multiplicity = self.multiplicity
        if multiplicity is None:
            multiplicity = abs(self.num_alpha - self.num_beta) + 1
        electron_count = sum(self.num_particles)
        max_twice_spin = min(electron_count, 2 * n - electron_count)
        if (
            isinstance(multiplicity, bool)
            or not isinstance(multiplicity, int)
            or multiplicity < 1
            or multiplicity - 1 < abs(self.num_alpha - self.num_beta)
            or multiplicity - 1 > max_twice_spin
            or (electron_count - multiplicity + 1) % 2
        ):
            raise ConfigError("qiskit multiplicity is incompatible with the electron populations")
        object.__setattr__(self, "multiplicity", multiplicity)
        for name in ("one_body_integrals", "two_body_integrals"):
            if getattr(self, name) is None:
                raise ConfigError(f"qiskit {name} is required")
        beta_names = (
            "one_body_integrals_beta",
            "two_body_integrals_beta_beta",
            "two_body_integrals_beta_alpha",
        )
        beta_present = [getattr(self, name) is not None for name in beta_names]
        if any(beta_present) and not all(beta_present):
            raise ConfigError("qiskit unrestricted input requires all three beta integral blocks")
        if all(beta_present) and self.overlap_alpha_beta is None:
            raise ConfigError("qiskit unrestricted input requires overlap_alpha_beta for spin")
        for name in ("one_body_integrals", "two_body_integrals", *beta_names):
            value = getattr(self, name)
            if value is None:
                continue
            rank = 2 if name.startswith("one_body") else 4
            array = _array(value, (n,) * rank, name, complex_allowed=True)
            if rank == 2:
                symmetric = np.allclose(array, array.T.conj(), atol=1e-10, rtol=1e-10)
            else:
                chemist = array if self.two_body_order == "chemist" else array.transpose(0, 3, 1, 2)
                symmetric = np.allclose(
                    chemist, chemist.transpose(1, 0, 3, 2).conj(), atol=1e-10, rtol=1e-10
                )
                if not np.iscomplexobj(array):
                    symmetric &= np.allclose(
                        chemist, chemist.swapaxes(0, 1), atol=1e-10, rtol=1e-10
                    )
                if name != "two_body_integrals_beta_alpha":
                    symmetric &= np.allclose(
                        chemist, chemist.transpose(2, 3, 0, 1), atol=1e-10, rtol=1e-10
                    )
            if not symmetric:
                raise ConfigError(f"qiskit {name} violates integral Hermitian symmetry")
            object.__setattr__(self, name, array)
        for name in ("orbital_energies", "orbital_energies_beta", "overlap_alpha_beta"):
            value = getattr(self, name)
            if value is not None:
                shape = (n, n) if name == "overlap_alpha_beta" else (n,)
                object.__setattr__(self, name, _array(value, shape, name))
        for name, count in zip(
            ("orbital_occupations", "orbital_occupations_beta"), self.num_particles, strict=True
        ):
            value = getattr(self, name)
            if value is None:
                value = [1.0] * count + [0.0] * (n - count)
            occupation = _array(value, (n,), name)
            if not np.all((occupation == 0) | (occupation == 1)) or sum(occupation) != count:
                raise ConfigError(f"qiskit {name} must contain 0/1 occupations summing to {count}")
            object.__setattr__(self, name, occupation)
        if self.nuclear_repulsion_energy is not None and (
            isinstance(self.nuclear_repulsion_energy, bool)
            or not isinstance(self.nuclear_repulsion_energy, Real)
            or not np.isfinite(self.nuclear_repulsion_energy)
            or self.nuclear_repulsion_energy < 0
        ):
            raise ConfigError("qiskit nuclear repulsion energy must be finite and non-negative")
        for name in ("provenance", "metadata"):
            try:
                value = json.loads(json.dumps(getattr(self, name), allow_nan=False))
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    f"qiskit {name} must contain finite JSON-compatible data"
                ) from exc
            if not isinstance(value, dict):
                raise ConfigError(f"qiskit {name} must be a mapping")
            object.__setattr__(self, name, value)
        if not isinstance(self.energy_offsets, Mapping) or any(
            not isinstance(name, str)
            or not name.strip()
            or name == "nuclear_repulsion_energy"
            or isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
            for name, value in self.energy_offsets.items()
        ):
            raise ConfigError("qiskit energy_offsets require named finite non-nuclear constants")
        object.__setattr__(
            self,
            "energy_offsets",
            {name: float(value) for name, value in self.energy_offsets.items()},
        )

    @property
    def num_particles(self) -> tuple[int, int]:
        """Return alpha and beta electron counts in Qiskit ordering."""
        return self.num_alpha, self.num_beta

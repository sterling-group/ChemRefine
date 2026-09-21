"""Construct quantum problems from owned integral data or the optional PySCF adapter."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from chemrefine.engines.qiskit.active_space import apply_active_space
from chemrefine.engines.qiskit.data import ElectronicStructureData
from chemrefine.engines.qiskit.options import ActiveSpaceOptions, QiskitOptions
from chemrefine.errors import ConfigError

logger = logging.getLogger(__name__)


def _atom_spec(xyz_path: Path) -> str:
    """Read one XYZ frame into the semicolon format expected by PySCFDriver."""
    try:
        lines = xyz_path.read_text(encoding="utf-8").splitlines()
        atom_count = int(lines[0])
    except (OSError, IndexError, ValueError) as exc:
        raise ConfigError(f"cannot read Qiskit XYZ input {xyz_path}: {exc}") from exc
    rows = lines[2 : 2 + atom_count]
    if len(rows) != atom_count:
        raise ConfigError(
            f"Qiskit XYZ input {xyz_path} declares {atom_count} atoms but contains {len(rows)}"
        )
    atoms: list[str] = []
    for index, line in enumerate(rows, start=1):
        fields = line.split()
        if len(fields) < 4:
            raise ConfigError(f"Qiskit XYZ input {xyz_path} has malformed atom row {index}")
        symbol, x, y, z = fields[:4]
        atoms.append(f"{symbol} {x} {y} {z}")
    return "; ".join(atoms)


@dataclass(frozen=True)
class PreparedProblem:
    """Prepared chemistry boundary, including Hamiltonian and original orbital indices.

    ``problem`` and ``fermionic_hamiltonian`` are Qiskit implementation artifacts;
    scientific provenance and result models remain owned by ChemRefine.
    """

    problem: Any
    fermionic_hamiltonian: Any
    original_num_spatial_orbitals: int
    active_orbitals: list[int]
    multiplicity: int
    provenance: dict[str, Any]
    metadata: dict[str, Any]

    @property
    def num_particles(self) -> tuple[int, int]:
        """Return the transformed alpha and beta electron populations."""
        alpha, beta = self.problem.num_particles
        return int(alpha), int(beta)

    @property
    def num_spatial_orbitals(self) -> int:
        """Return the transformed spatial-orbital count."""
        return int(self.problem.num_spatial_orbitals)

    @property
    def num_spin_orbitals(self) -> int:
        """Return the transformed spin-orbital count before mapping."""
        return 2 * self.num_spatial_orbitals

    @property
    def energy_offsets(self) -> dict[str, float]:
        """Return constants excluded from the fermionic and mapped Hamiltonians."""
        return {name: float(value) for name, value in self.problem.hamiltonian.constants.items()}


def _finish_preparation(
    problem: Any,
    *,
    multiplicity: int,
    active_space: ActiveSpaceOptions | None,
    freeze_core: bool,
    provenance: dict[str, Any],
    metadata: dict[str, Any],
) -> PreparedProblem:
    """Share transformations and provenance across independent upstream providers."""
    original_count = int(problem.num_spatial_orbitals)
    problem, indices, transformations = apply_active_space(
        problem, active_space=active_space, freeze_core=freeze_core
    )
    logger.info(
        "Qiskit electronic problem created: %s spatial orbitals", problem.num_spatial_orbitals
    )
    return PreparedProblem(
        problem=problem,
        fermionic_hamiltonian=problem.hamiltonian.second_q_op(),
        original_num_spatial_orbitals=original_count,
        active_orbitals=indices,
        multiplicity=multiplicity,
        provenance=dict(provenance),
        metadata={**metadata, "transformations": transformations},
    )


def prepare_problem(
    electronic_structure: ElectronicStructureData,
    *,
    active_space: ActiveSpaceOptions | None = None,
    freeze_core: bool = False,
) -> PreparedProblem:
    """Build an electronic Hamiltonian from MO integrals without importing PySCF."""
    from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
    from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
    from qiskit_nature.second_q.problems import ElectronicBasis, ElectronicStructureProblem
    from qiskit_nature.second_q.properties import AngularMomentum, Magnetization, ParticleNumber

    data = electronic_structure

    def physicist(value: Any) -> Any:
        """Convert the declared real two-body convention without symmetry inference."""
        if value is None:
            return None
        array = np.asarray(value)
        return array.transpose(0, 2, 3, 1) if data.two_body_order == "chemist" else array

    energy = ElectronicEnergy.from_raw_integrals(
        np.asarray(data.one_body_integrals),
        physicist(data.two_body_integrals),
        None if data.one_body_integrals_beta is None else np.asarray(data.one_body_integrals_beta),
        physicist(data.two_body_integrals_beta_beta),
        physicist(data.two_body_integrals_beta_alpha),
        auto_index_order=False,
    )
    if data.nuclear_repulsion_energy is not None:
        energy.nuclear_repulsion_energy = data.nuclear_repulsion_energy
    problem = ElectronicStructureProblem(energy)
    problem.basis = ElectronicBasis.MO
    problem.num_particles = data.num_particles
    problem.num_spatial_orbitals = data.num_spatial_orbitals
    problem.orbital_occupations = np.asarray(data.orbital_occupations)
    problem.orbital_occupations_b = np.asarray(data.orbital_occupations_beta)
    problem.orbital_energies = (
        None if data.orbital_energies is None else np.asarray(data.orbital_energies)
    )
    problem.orbital_energies_b = (
        None if data.orbital_energies_beta is None else np.asarray(data.orbital_energies_beta)
    )
    overlap = None if data.overlap_alpha_beta is None else np.asarray(data.overlap_alpha_beta)
    problem.properties.add(AngularMomentum(data.num_spatial_orbitals, overlap))
    problem.properties.add(Magnetization(data.num_spatial_orbitals))
    problem.properties.add(ParticleNumber(data.num_spatial_orbitals))
    multiplicity = int(data.multiplicity or 1)
    if data.molecular_metadata is not None:
        molecule = data.molecular_metadata
        problem.molecule = MoleculeInfo(
            symbols=molecule.symbols,
            coords=molecule.coordinates,
            charge=molecule.charge,
            multiplicity=multiplicity,
        )
    return _finish_preparation(
        problem,
        multiplicity=multiplicity,
        active_space=active_space,
        freeze_core=freeze_core,
        provenance={"source": "integrals", **data.provenance},
        metadata={"integral_order": data.two_body_order, **data.metadata},
    )


def prepare_pyscf_problem(
    xyz_path: str | Path,
    *,
    charge: int,
    multiplicity: int,
    options: QiskitOptions,
) -> PreparedProblem:
    """Adapt an XYZ/PySCF calculation to the same preparation boundary as integral inputs."""
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.units import DistanceUnit

    if multiplicity < 1:
        raise ConfigError("multiplicity must be at least 1")
    problem = PySCFDriver(
        atom=_atom_spec(Path(xyz_path)),
        unit=DistanceUnit.ANGSTROM,
        charge=charge,
        spin=multiplicity - 1,
        basis=options.basis,
    ).run()
    return _finish_preparation(
        problem,
        multiplicity=multiplicity,
        active_space=options.active_space,
        freeze_core=options.freeze_core,
        provenance={
            "source": "pyscf",
            "basis": options.basis,
            "charge": charge,
            "multiplicity": multiplicity,
        },
        metadata={},
    )

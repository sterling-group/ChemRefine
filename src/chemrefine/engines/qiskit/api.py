"""Public preparation and solving interfaces for callers supplying chemistry data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from chemrefine.engines.qiskit.data import ElectronicStructureData, MolecularMetadata
from chemrefine.engines.qiskit.determinants import (
    DeterminantState,
    FermionicHamiltonian,
    FermionTerm,
    ProjectedEigensystem,
    ReducedDensityMatrices,
    projected_eigensystem,
    projected_operator,
)
from chemrefine.engines.qiskit.lattice import (
    FermionicLatticeModel,
    LatticeCircuit,
    LatticeDynamicsOptions,
    LatticeDynamicsResult,
    LatticeEdge,
    build_lattice_dynamics,
    chain_lattice,
    simulate_lattice_dynamics,
    square_lattice,
)
from chemrefine.engines.qiskit.mapping import map_problem
from chemrefine.engines.qiskit.operators import Excitation, OperatorPool
from chemrefine.engines.qiskit.options import ActiveSpaceOptions, ComponentSelection, QiskitOptions
from chemrefine.engines.qiskit.orbitals import (
    OrbitalOptimizationOptions,
    OrbitalOptimizationResult,
    optimize_orbitals,
    rotate_hamiltonian,
)
from chemrefine.engines.qiskit.problem import (
    PreparedProblem,
    prepare_problem,
    prepare_pyscf_problem,
)
from chemrefine.engines.qiskit.result import CircuitMetrics, QiskitRunResult
from chemrefine.engines.qiskit.sampling import SampleBatch, sample_circuit
from chemrefine.engines.qiskit.workflow import EvaluationCallback, run_job, run_problem

__all__ = [
    "ActiveSpaceOptions",
    "CircuitMetrics",
    "DeterminantState",
    "ElectronicStructureData",
    "FermionTerm",
    "FermionicHamiltonian",
    "FermionicLatticeModel",
    "LatticeCircuit",
    "LatticeDynamicsOptions",
    "LatticeDynamicsResult",
    "LatticeEdge",
    "MolecularMetadata",
    "OperatorPool",
    "OrbitalOptimizationOptions",
    "OrbitalOptimizationResult",
    "PreparedProblem",
    "ProjectedEigensystem",
    "QiskitOptions",
    "QiskitRunResult",
    "ReducedDensityMatrices",
    "SampleBatch",
    "build_lattice_dynamics",
    "chain_lattice",
    "map_problem",
    "optimize_orbitals",
    "prepare_problem",
    "prepare_pyscf_problem",
    "projected_eigensystem",
    "projected_operator",
    "rotate_hamiltonian",
    "run_adapt_vqe",
    "run_job",
    "run_problem",
    "run_vqe",
    "sample_circuit",
    "simulate_lattice_dynamics",
    "solve_exact",
    "square_lattice",
]


def _select_algorithm(
    options: QiskitOptions | Mapping[str, Any] | None, name: str
) -> QiskitOptions:
    """Keep settings for the requested algorithm; discard another solver's options."""
    resolved = options if isinstance(options, QiskitOptions) else QiskitOptions.from_raw(options)
    if resolved.algorithm.name == name:
        return resolved
    return resolved.model_copy(update={"algorithm": ComponentSelection.named(name)})


def solve_exact(
    prepared: PreparedProblem, *, options: QiskitOptions | Mapping[str, Any] | None = None
) -> QiskitRunResult:
    """Solve a small prepared problem in its requested particle/spin sector."""
    return run_problem(prepared, options=_select_algorithm(options, "exact"))


def run_vqe(
    prepared: PreparedProblem,
    *,
    options: QiskitOptions | Mapping[str, Any] | None = None,
    excitations: Sequence[Excitation] | None = None,
    initial_point: Sequence[float] | None = None,
    callback: EvaluationCallback | None = None,
    reference_energy_hartree: float | None = None,
) -> QiskitRunResult:
    """Run fixed VQE, optionally using an explicitly supplied chemistry excitation list."""
    resolved = _select_algorithm(options, "vqe")
    if excitations is not None:
        ansatz_options = (
            dict(resolved.ansatz.options) if resolved.ansatz.name in {"ucc", "uccsd"} else {}
        )
        ansatz_options["excitations"] = list(excitations)
        resolved = resolved.model_copy(
            update={"ansatz": ComponentSelection(name="ucc", options=ansatz_options)}
        )
    return run_problem(
        prepared,
        options=resolved,
        initial_point=initial_point,
        callback=callback,
        reference_energy_hartree=reference_energy_hartree,
    )


def run_adapt_vqe(
    prepared: PreparedProblem,
    *,
    options: QiskitOptions | Mapping[str, Any] | None = None,
    operator_pool: OperatorPool | None = None,
    callback: EvaluationCallback | None = None,
    reference_energy_hartree: float | None = None,
) -> QiskitRunResult:
    """Run ADAPT with its chemistry pool or a caller's mapped operator pool."""
    return run_problem(
        prepared,
        options=_select_algorithm(options, "adapt_vqe"),
        operator_pool=operator_pool,
        callback=callback,
        reference_energy_hartree=reference_energy_hartree,
    )

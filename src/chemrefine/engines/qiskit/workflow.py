"""Composition root for modular Qiskit Nature ground-state calculations."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from chemrefine.engines.qiskit import components as _builtins  # noqa: F401
from chemrefine.engines.qiskit.context import (
    AnsatzArtifacts,
    ElectronicStructureContext,
    EstimatorResource,
    SolverComponents,
)
from chemrefine.engines.qiskit.options import QiskitOptions
from chemrefine.engines.qiskit.registry import (
    ALGORITHMS,
    ANSATZE,
    ESTIMATORS,
    INITIAL_POINTS,
    INITIAL_STATES,
    MAPPERS,
    OPTIMIZERS,
    REGISTRIES,
    validate_component_graph,
)
from chemrefine.errors import ConfigError


@dataclass(frozen=True)
class QiskitRunResult:
    """Energy and reproducibility diagnostics returned to the rendered job script."""

    energy_hartree: float
    metadata: dict[str, Any]


def validate_options(options: QiskitOptions) -> None:
    """Fail fast on unknown components, bad knobs, or incompatible artifacts."""
    validate_component_graph(options)


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


def _build_problem(
    xyz_path: Path,
    *,
    charge: int,
    multiplicity: int,
    options: QiskitOptions,
) -> Any:
    """Run the classical driver and optional active-space transformation."""
    if multiplicity < 1:
        raise ConfigError("multiplicity must be at least 1")
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
    from qiskit_nature.units import DistanceUnit

    problem: Any = PySCFDriver(
        atom=_atom_spec(xyz_path),
        unit=DistanceUnit.ANGSTROM,
        charge=charge,
        spin=multiplicity - 1,
        basis=options.basis,
    ).run()
    if options.active_space is not None:
        problem = ActiveSpaceTransformer(
            num_electrons=options.active_space.electrons,
            num_spatial_orbitals=options.active_space.orbitals,
        ).transform(problem)
    return problem


def _electronic_context(
    problem: Any, options: QiskitOptions, *, multiplicity: int
) -> ElectronicStructureContext:
    """Construct the mapper and immutable facts every later builder receives."""
    mapper = MAPPERS.build(options.mapper, problem=problem)
    hamiltonian = mapper.map(problem.hamiltonian.second_q_op())
    particles = tuple(int(value) for value in problem.num_particles)
    if len(particles) != 2:
        raise ConfigError(f"Qiskit electronic problem reported invalid particles {particles!r}")
    return ElectronicStructureContext(
        problem=problem,
        mapper=mapper,
        qubit_hamiltonian=hamiltonian,
        num_spatial_orbitals=int(problem.num_spatial_orbitals),
        num_particles=(particles[0], particles[1]),
        num_qubits=int(hamiltonian.num_qubits),
        multiplicity=multiplicity,
    )


def _jsonable(value: Any) -> Any:
    """Convert common Qiskit/numpy diagnostics into JSON-compatible values."""
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return str(value)


def _result_metadata(
    result: Any,
    options: QiskitOptions,
    context: ElectronicStructureContext,
    evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Capture resolved configuration and stable solver diagnostics for provenance."""
    raw_result = getattr(result, "raw_result", None)
    diagnostics: dict[str, Any] = {}
    for name in (
        "cost_function_evals",
        "num_iterations",
        "optimal_point",
        "optimal_value",
        "termination_criterion",
    ):
        value = getattr(raw_result, name, None)
        if value is not None:
            diagnostics[name] = _jsonable(value)
    return {
        "components": {
            category: {
                "name": selection.name,
                "options": REGISTRIES[category].options_for(selection).model_dump(mode="json"),
            }
            for category, selection in options.component_selections().items()
        },
        "basis": options.basis,
        "device": options.device,
        "cores": options.cores,
        "active_space": (
            options.active_space.model_dump(mode="json")
            if options.active_space is not None
            else None
        ),
        "num_spatial_orbitals": context.num_spatial_orbitals,
        "num_particles": list(context.num_particles),
        "num_qubits": context.num_qubits,
        "evaluations": evaluations,
        "solver": diagnostics,
    }


def run_job(
    xyz_path: str | Path,
    *,
    charge: int,
    multiplicity: int,
    options: QiskitOptions | Mapping[str, Any],
) -> QiskitRunResult:
    """Build the configured component graph, solve the molecule, and return its energy."""
    resolved = (
        options if isinstance(options, QiskitOptions) else QiskitOptions.from_raw(dict(options))
    )
    validate_options(resolved)
    problem = _build_problem(
        Path(xyz_path),
        charge=charge,
        multiplicity=multiplicity,
        options=resolved,
    )
    context = _electronic_context(problem, resolved, multiplicity=multiplicity)
    requirements = ALGORITHMS.spec(resolved.algorithm.name).requires

    initial_state: Any | None = None
    ansatz = AnsatzArtifacts()
    initial_point: Any | None = None
    optimizer: Any | None = None
    if requirements & {"initial_state", "circuit", "operator_pool"}:
        initial_state = INITIAL_STATES.build(resolved.initial_state, context=context)
    if requirements & {"circuit", "operator_pool"}:
        ansatz = ANSATZE.build(
            resolved.ansatz,
            context=context,
            initial_state=initial_state,
        )
    if "initial_point" in requirements:
        initial_point = INITIAL_POINTS.build(resolved.initial_point, ansatz=ansatz)
    if "optimizer" in requirements:
        optimizer = OPTIMIZERS.build(resolved.optimizer)

    evaluations: list[dict[str, Any]] = []
    previous_algorithm_evaluation: int | None = None
    inner_run = 0

    def callback(
        evaluation: int,
        _parameters: np.ndarray,
        mean: float,
        metadata: dict[str, Any],
    ) -> None:
        nonlocal inner_run, previous_algorithm_evaluation
        algorithm_evaluation = int(evaluation)
        if (
            previous_algorithm_evaluation is None
            or algorithm_evaluation <= previous_algorithm_evaluation
        ):
            inner_run += 1
        evaluations.append(
            {
                # The callback counter resets for each inner VQE in ADAPT. Keep a
                # workflow-global index while retaining the algorithm-provided value.
                "evaluation": len(evaluations) + 1,
                "inner_run": inner_run,
                "algorithm_evaluation": algorithm_evaluation,
                # VQE evaluates the mapped electronic Hamiltonian here; nuclear
                # repulsion and transformer constants are added to the final total.
                "objective_value_hartree": float(mean),
                "metadata": _jsonable(metadata),
            }
        )
        previous_algorithm_evaluation = algorithm_evaluation

    estimator_resource: EstimatorResource | nullcontext[None]
    if "estimator" in requirements:
        estimator_resource = ESTIMATORS.build(
            resolved.estimator,
            device=resolved.device,
            cores=resolved.cores,
        )
    else:
        estimator_resource = nullcontext(None)

    transpiler = (
        estimator_resource.transpiler if isinstance(estimator_resource, EstimatorResource) else None
    )
    transpiler_options = (
        estimator_resource.transpiler_options
        if isinstance(estimator_resource, EstimatorResource)
        else None
    )
    with estimator_resource as estimator:
        assembled = SolverComponents(
            estimator=estimator,
            optimizer=optimizer,
            initial_state=initial_state,
            ansatz=ansatz,
            initial_point=initial_point,
            callback=callback,
            transpiler=transpiler,
            transpiler_options=transpiler_options,
        )
        algorithm = ALGORITHMS.build(
            resolved.algorithm,
            context=context,
            components=assembled,
        )
        from qiskit_nature.second_q.algorithms import GroundStateEigensolver

        result = GroundStateEigensolver(context.mapper, algorithm.solver).solve(problem)

    if not getattr(result, "total_energies", None):
        raise ConfigError("Qiskit solver returned no total ground-state energy")
    energy = complex(result.total_energies[0])
    if abs(energy.imag) > 1e-10:
        raise ConfigError(f"Qiskit solver returned a complex total energy {energy!r}")
    energy_hartree = float(energy.real)
    if not np.isfinite(energy_hartree):
        raise ConfigError(f"Qiskit solver returned non-finite total energy {energy_hartree!r}")
    return QiskitRunResult(
        energy_hartree=energy_hartree,
        metadata=_result_metadata(result, resolved, context, evaluations),
    )

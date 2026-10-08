"""Scientific workloads for the quantum benchmark driver; providers import lazily."""

from __future__ import annotations

import copy
import json
import os
import runpy
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class Unsupported(Exception):
    """A declared configuration cannot execute on this provider or resource budget."""


def _aer_options(case: dict) -> dict:
    """Build identical seeded Aer settings for CPU and GPU comparison."""
    options = {
        "seed_simulator": case["seed"],
        "seed_transpiler": case["seed"],
        "simulation_precision": case["precision"],
        "optimization_level": 1,
    }
    if case["provider"] != "aer_statevector":
        options["method"] = case["method"]
        if case["noisy"]:
            from qiskit_aer.noise import NoiseModel, depolarizing_error

            noise = NoiseModel()
            noise.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ["ry", "rz"])
            options["noise_model"] = noise.to_dict(serializable=True)
    return options


def _check_backend(case: dict, device: str) -> None:
    """Refuse known unsupported combinations before allocating a simulator state."""
    if device == "cuda" and case["provider"] == "ibm_runtime":
        raise Unsupported("ChemRefine's Runtime fake-backend adapter does not support device: cuda")
    if device == "cuda" and not case["provider"].startswith("aer"):
        raise Unsupported("reference and BasicSimulator primitives are CPU-only")
    if case["provider"].startswith("aer"):
        from qiskit_aer import AerSimulator

        simulator = AerSimulator()
        requested = "GPU" if device == "cuda" else "CPU"
        if requested not in simulator.available_devices():
            raise Unsupported(f"{requested} unavailable in this Aer build")
        if case["method"] not in simulator.available_methods():
            raise Unsupported(f"Aer build does not provide {case['method']}")
        if device == "cuda" and case["method"] == "matrix_product_state":
            raise Unsupported("Aer matrix_product_state is CPU-only")
        if device == "cpu" and case["method"] == "tensor_network":
            raise Unsupported("Aer tensor_network is GPU-only")
        if case["method"] == "density_matrix" and 16 * 4 ** case["qubits"] > 256 * 1024**2:
            raise Unsupported("density matrix exceeds the 256 MiB benchmark state budget")
        if case["noisy"] and case["qubits"] > 12:
            raise Unsupported("noisy trajectory benchmarks are bounded to 12 qubits")


def primitive(case: dict, device: str):
    """Benchmark ChemRefine's actual estimator and sampler resource adapters."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp, Statevector

    from chemrefine.engines.qiskit.execution import logical_estimator
    from chemrefine.engines.qiskit.options import ComponentSelection
    from chemrefine.engines.qiskit.registry import ESTIMATORS
    from chemrefine.engines.qiskit.sampling import SamplingSession

    _check_backend(case, device)
    n = case["qubits"]
    rng = np.random.default_rng(case["seed"])
    circuit = QuantumCircuit(n)
    for layer in range(case["layers"]):
        for qubit in range(n):
            circuit.ry(float(rng.uniform(-1, 1)), qubit)
            circuit.rz(float(rng.uniform(-1, 1)), qubit)
        for qubit in range(layer % 2, n - 1, 2):
            circuit.cx(qubit, qubit + 1)
    observable = SparsePauliOp("I" * (n - 1) + "Z")
    exact = float(Statevector(circuit).expectation_value(observable).real)
    temporary = None
    if case["provider"].startswith("aer"):
        options = _aer_options(case)
    elif case["provider"] == "ibm_runtime":
        journal_dir = os.environ.get("CHEMREFINE_BENCHMARK_JOURNAL_DIR")
        if journal_dir is None:
            temporary = tempfile.TemporaryDirectory(prefix="chemrefine-runtime-bench-")
            journal_dir = temporary.name
        options = {
            "journal_dir": journal_dir,
            "fake_backend": case["fake_backend"],
            "implementation": case["implementation"],
            "mode": case["mode"],
            "seed_simulator": case["seed"],
            "seed_transpiler": case["seed"],
        }
        if case["family"] == "estimator":
            options.update(
                default_precision=1 / np.sqrt(case["shots"]),
                resilience_level=case["resilience_level"],
            )
    else:
        options = {"seed" if case["provider"] == "statevector" else "seed_simulator": case["seed"]}
    if case["family"] == "estimator" and case["provider"] in {"aer_shots", "basic_backend"}:
        options["default_precision"] = 1 / np.sqrt(case["shots"])
    selection = ComponentSelection(name=case["provider"], options=options)
    if case["provider"] == "ibm_runtime":
        from chemrefine.engines.qiskit.registry import SAMPLERS
        from chemrefine.errors import ConfigError

        try:
            (ESTIMATORS if case["family"] == "estimator" else SAMPLERS).options_for(selection)
        except ConfigError as exc:
            raise Unsupported(str(exc)) from exc
    setup = {
        "depth": circuit.depth(),
        "gates": circuit.size(),
        "reference_expectation": exact,
        "statevector_bytes": 16 * 2**n,
        "options": selection.model_dump(mode="json"),
    }

    def run():
        """Complete the provider job and check a physical expectation against its reference."""
        _ = temporary
        if case["family"] == "estimator":
            resource = ESTIMATORS.build(selection, device=device, cores=case["cores"])
            with resource:
                result = (
                    logical_estimator(resource, max_publications=1)
                    .run([(circuit, observable)])
                    .result()[0]
                )
                value = float(result.data.evs)
                metadata = result.metadata
            sampled = case["provider"] in {"aer_shots", "basic_backend", "ibm_runtime"}
        else:
            with SamplingSession(
                selection, device=device, cores=case["cores"], seed=case["seed"]
            ) as session:
                batch = session.sample(circuit, shots=case["shots"])
            if sum(batch.counts.values()) != case["shots"]:
                raise AssertionError("shot accounting mismatch")
            value = (
                sum((1 if bits[-1] == "0" else -1) * count for bits, count in batch.counts.items())
                / batch.shots
            )
            metadata = batch.metadata
            sampled = True
        error = abs(value - exact)
        tolerance = (
            6 / np.sqrt(case["shots"])
            if sampled
            else (1e-5 if case["precision"] == "single" else 1e-10)
        )
        if not np.isfinite(value) or abs(value) > 1 + 1e-5:
            raise AssertionError("expectation is outside physical bounds")
        if not case["noisy"] and error > tolerance:
            raise AssertionError(f"expectation error {error} exceeds {tolerance}")
        return {
            "expectation": value,
            "absolute_error": error,
            "tolerance": tolerance,
            "check": "physical_bounds" if case["noisy"] else "exact_reference",
            "provider_metadata": metadata,
        }

    return run, setup


def molecular(case: dict, device: str):
    """Prepare deterministic integral problems and exercise every molecular solver."""
    from chemrefine.engines.qiskit.data import ElectronicStructureData
    from chemrefine.engines.qiskit.mapping import map_problem
    from chemrefine.engines.qiskit.options import ComponentSelection, QiskitOptions
    from chemrefine.engines.qiskit.problem import prepare_problem
    from chemrefine.engines.qiskit.registry import (
        consumed_component_categories,
        validate_component_graph,
    )
    from chemrefine.engines.qiskit.workflow import run_problem
    from chemrefine.errors import ConfigError

    n = case["qubits"] // 2
    if n == 2:
        data = ElectronicStructureData(
            **json.loads((ROOT / "tests/data/engines/qiskit/h2_integrals.json").read_text())
        )
    else:
        h1 = np.diag(np.linspace(-1.5, 0.5, n))
        h1 += np.diag(np.full(n - 1, 0.1), 1) + np.diag(np.full(n - 1, 0.1), -1)
        h2 = np.zeros((n,) * 4)
        for i in range(n):
            h2[i, i, i, i] = 0.3
        data = ElectronicStructureData(1, 1, n, h1, h2, nuclear_repulsion_energy=0.0)
    prepared = prepare_problem(data)
    algorithm = case["algorithm"]
    options = {
        key: case[key]
        for key in ("algorithm", "mapper", "ansatz", "initial_state", "initial_point")
    }
    if "ansatz_options" in case:
        options["ansatz"] = {"name": case["ansatz"], "options": case["ansatz_options"]}
    optimizer: dict[str, Any] = {"maxiter": 30}
    if case["optimizer"] in {"spsa", "qnspsa"}:
        optimizer.update(seed=case["seed"], learning_rate=0.05, perturbation=0.05)
    options["optimizer"] = {"name": case["optimizer"], "options": optimizer}
    if case["initial_state"] == "determinant":
        options["initial_state"] = {"name": "determinant", "options": {"alpha": [0], "beta": [0]}}
    if case["initial_point"] == "random":
        options["initial_point"] = {"name": "random", "options": {"seed": case["seed"]}}
    if algorithm in {"ceo_adapt", "tetris_adapt"}:
        options["ansatz"] = "ceo" if algorithm == "ceo_adapt" else "qe"
    if algorithm in {"ceo_adapt", "tetris_adapt", "adapt_vqe"}:
        options["algorithm"] = {"name": algorithm, "options": {"max_iterations": 6}}
    if algorithm in {"sqd", "extended_sqd", "sqdrift", "skqd"}:
        options.pop("optimizer")
        controls = {
            "shots": 1024,
            "samples_per_batch": 32,
            "num_batches": 1,
            "max_iterations": 1,
            "seed": case["seed"],
        }
        if algorithm == "skqd":
            controls.update(num_steps=2, time_step=1.0)
        if algorithm == "sqdrift":
            controls.update(times=[0.1], randomizations=2)
        options["algorithm"] = {"name": algorithm, "options": controls}
    if algorithm == "vqd":
        options["algorithm"] = {"name": "vqd", "options": {"k": 2, "fidelity_shots": 4096}}
        options["initial_point"] = {"name": "random", "options": {"seed": 31}}
        options["optimizer"] = {"name": "cobyla", "options": {"maxiter": 150}}
    if "algorithm_options" in case:
        options["algorithm"]["options"].update(case["algorithm_options"])
    base = QiskitOptions.from_raw(options)
    consumed = consumed_component_categories(base)
    if "estimator" in consumed:
        options["estimator"] = {
            "name": "aer_statevector",
            "options": {"seed_simulator": case["seed"], "seed_transpiler": case["seed"]},
        }
    if "sampler" in consumed:
        options["sampler"] = {
            "name": "aer",
            "options": {"seed_simulator": case["seed"], "seed_transpiler": case["seed"]},
        }
    options["device"] = device
    try:
        resolved = QiskitOptions.from_raw(options)
        validate_component_graph(resolved)
    except ConfigError as exc:
        raise Unsupported(str(exc)) from exc
    context = map_problem(prepared, ComponentSelection.named("jordan_wigner"))
    lower_bound = float(np.linalg.eigvalsh(context.qubit_hamiltonian.to_matrix())[0]) + sum(
        prepared.energy_offsets.values()
    )
    setup = {
        "spatial_orbitals": n,
        "electrons": sum(prepared.num_particles),
        "pauli_terms": len(context.qubit_hamiltonian),
        "full_space_lower_bound": lower_bound,
        "resolved_options": resolved.model_dump(mode="json"),
    }

    def run():
        """Record approximation quality separately from numerical execution correctness."""
        result = run_problem(prepared, options=resolved)
        energy = result.energy_hartree
        if not np.isfinite(energy) or energy < lower_bound - 1e-5:
            raise AssertionError(f"energy {energy} violates the exact lower bound {lower_bound}")
        metrics = result.logical_circuit_metrics
        return {
            "energy_hartree": energy,
            "error_from_lower_bound_hartree": energy - lower_bound,
            "converged": result.converged,
            "termination_reason": result.termination_reason,
            "qubits": result.num_qubits,
            "parameters": result.parameter_count,
            "evaluations": result.energy_evaluation_count,
            "depth": metrics.depth if metrics else None,
            "gates": metrics.size if metrics else None,
            "check": "finite_variational_bound",
        }

    return run, setup


def tutorial(case: dict, device: str):
    """Time public artifact workflows, preserving all generated input/output bundles."""
    import yaml

    from chemrefine.engines.qiskit.bundles import read_bundle
    from chemrefine.engines.qiskit.experiment import (
        EXPERIMENTS,
        QiskitExperimentOptions,
        run_experiment,
        validate_experiment,
    )
    from chemrefine.errors import ConfigError

    temporary = tempfile.TemporaryDirectory(prefix="chemrefine-tutorial-bench-")
    root = Path(temporary.name)
    shutil.copytree(
        ROOT / "examples/tutorials",
        root / "examples/tutorials",
        ignore=shutil.ignore_patterns("outputs*", "__pycache__"),
    )
    path = root / case["path"]
    directory = path.parent
    generator = directory / "make_inputs.py"
    if generator.exists():
        try:
            runpy.run_path(str(generator))
        except ImportError as exc:
            raise Unsupported(str(exc)) from exc
    raw = yaml.safe_load(path.read_text())
    configured = []
    for step in raw["steps"]:
        options = copy.deepcopy(step["options"])
        options["device"] = device
        options["cores"] = 1
        experiment = options["experiment"]
        spec = EXPERIMENTS.spec(experiment["name"])
        nested = experiment["options"]
        for category in spec.requires & {"estimator", "sampler"}:
            selected = nested.get(category, "statevector")
            name = selected if isinstance(selected, str) else selected["name"]
            controls = {} if isinstance(selected, str) else selected.get("options", {})
            if category == "estimator":
                nested[category] = {
                    "name": "aer_statevector",
                    "options": {"seed_simulator": 19, "seed_transpiler": 19},
                }
            elif name != "aer":
                nested[category] = {
                    "name": "aer",
                    "options": {"seed_simulator": 19, "seed_transpiler": 19},
                }
            else:
                nested[category] = {
                    "name": "aer",
                    "options": {**controls, "seed_simulator": 19, "seed_transpiler": 19},
                }
        try:
            parsed = QiskitExperimentOptions.from_raw(options)
            validate_experiment(parsed)
        except ConfigError as exc:
            raise Unsupported(str(exc)) from exc
        configured.append((step["step"], options))
    setup = {
        "input_bytes": sum(p.stat().st_size for p in directory.iterdir() if p.is_file()),
        "resolved_options": [options for _, options in configured],
        "steps": len(configured),
    }

    def run():
        """Execute steps through the native bundle contract and validate retained arrays."""
        # Retain ownership until this closure is released by the worker.
        _ = temporary
        previous = Path.cwd()
        sizes = {}
        output_bytes = 0
        try:
            os.chdir(directory)
            for number, options in configured:
                output = (
                    directory / raw["output_dir"] / f"step{number}" / "experiment/artifact.json"
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                try:
                    run_experiment(options, output)
                except ImportError as exc:
                    raise Unsupported(str(exc)) from exc
                bundle = read_bundle(output)
                for name, array in bundle.arrays.items():
                    sizes[f"step{number}.{name}"] = list(array.shape)
                    output_bytes += array.nbytes
                    if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
                        raise AssertionError(f"nonfinite output array {name}")
        finally:
            os.chdir(previous)
        return {
            "array_shapes": sizes,
            "output_array_bytes": output_bytes,
            "check": "native_bundle_contract_and_finite_arrays",
        }

    return run, setup


def prepare(case: dict, device: str):
    """Dispatch only known workload families with no implicit remote execution."""
    if case["family"] in {"estimator", "sampler"}:
        return primitive(case, device)
    if case["family"] == "molecular":
        return molecular(case, device)
    if case["family"] == "tutorial":
        return tutorial(case, device)
    raise ValueError(f"unknown benchmark family {case['family']!r}")

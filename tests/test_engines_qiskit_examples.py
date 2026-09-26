"""Example coverage for the Qiskit engine's nested component options.

The shared example tests file top-level engine knobs through ``qiskit/knobs.json``.
These nested models and registry components belong to Qiskit, so their example-versus-
test-only coverage travels with the engine's own tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from chemrefine.engines.qiskit.options import ActiveSpaceOptions, ComponentSelection
from chemrefine.engines.qiskit.registry import REGISTRIES

REPO = Path(__file__).resolve().parent.parent

REQUIRED = {
    "active_space": {"electrons", "orbitals"},
    "selection": {"name", "options"},
    "mapper": set(),
    "algorithm": {
        "energy_increase_tolerance",
        "evolution",
        "gradient_norm",
        "max_pool_size",
        "selection_threshold",
        "tetris",
        "variant",
        "betas",
        "excitation_ranks",
        "fidelity_shots",
        "k",
        "max_excitations",
        "max_measurements",
        "overlap_tolerance",
        "num_roots",
        "num_steps",
        "projection",
        "spin_constraint",
        "target_root",
        "time_step",
    },
    "ansatz": {"reps", "preserve_spin", "max_pool_size"},
    "initial_state": set(),
    "estimator": {"default_precision", "seed"},
    "sampler": {"seed"},
    "optimizer": {"maxiter", "fidelity_shots"},
    "initial_point": set(),
}

TESTS_ONLY = {
    "active_space": {
        "active_orbitals",
    },
    "selection": set(),
    "mapper": {
        "base_mapper",
        "generators",
        "max_qubits",
        "max_statevector_bytes",
        "max_symmetries",
        "min_qubits",
        "sectors",
        "tolerance",
        "two_qubit_reduction",
    },
    "algorithm": {
        "conditioning_tolerance",
        "frequency_tolerance",
        "initial_points",
        "max_excitation_operators",
        "max_generated_determinants",
        "max_pauli_terms",
        "max_product_terms",
        "minimum_probability",
        "orbital_optimization",
        "product_formula",
        "repetitions",
        "residual_tolerance",
        "sector_tolerance",
        "suzuki_order",
        "target_s2",
        "ansatz",
        "configuration_recovery",
        "counts",
        "eigenvalue_threshold",
        "energy_tol",
        "gradient_threshold",
        "initial_parameters",
        "initial_scale",
        "initialization",
        "interaction_pairs",
        "max_circuits",
        "max_evaluations",
        "max_iterations",
        "max_memory_mb",
        "max_parameters",
        "max_statevector_dimension",
        "max_subspace_dimension",
        "max_total_diagonalizations",
        "max_total_shots",
        "n_reps",
        "num_batches",
        "num_groups",
        "occupancies_tol",
        "parameter_values",
        "randomizations",
        "reps",
        "samples_per_batch",
        "sci_max_cycle",
        "sci_max_space",
        "seed",
        "shots",
        "spin_tolerance",
        "spin_variant",
        "symmetrize_spin",
        "times",
        "with_final_orbital_rotation",
    },
    "ansatz": {
        "entanglement",
        "excitations",
        "flatten",
        "generalized",
        "include_imaginary",
        "mode",
        "ranks",
        "skip_final_rotation_layer",
        "su2_gates",
    },
    "initial_state": {
        "alpha",
        "beta",
    },
    "estimator": {
        "abelian_grouping",
        "backend_name",
        "method",
        "noise_model",
        "optimization_level",
        "seed_simulator",
        "seed_transpiler",
        "simulation_precision",
    },
    "sampler": {
        "method",
        "noise_model",
        "optimization_level",
        "seed_simulator",
        "seed_transpiler",
        "simulation_precision",
    },
    "optimizer": {
        "allowed_increase",
        "hessian_delay",
        "regularization",
        "resamplings",
        "adaptive",
        "blocking",
        "disp",
        "eps",
        "fatol",
        "ftol",
        "gtol",
        "learning_rate",
        "maxfev",
        "maxfun",
        "maxls",
        "perturbation",
        "rhobeg",
        "second_order",
        "seed",
        "tol",
        "trust_region",
        "xatol",
        "xtol",
    },
    "initial_point": {
        "scale",
        "seed",
    },
}

_UNIVERSE = {
    "active_space": set(ActiveSpaceOptions.model_fields),
    "selection": set(ComponentSelection.model_fields),
    **{
        category: {
            field
            for name in registry.names()
            for field in registry.spec(name).options_cls.model_fields
        }
        for category, registry in REGISTRIES.items()
    },
}


def _example_options() -> list[dict[str, Any]]:
    """The raw options of each shipped Qiskit example, before defaults are filled in."""
    return [
        step.get("options") or {}
        for path in sorted(
            set(REPO.glob("examples/**/input.yaml"))
            | set(REPO.glob("examples/tutorials/qiskit_sp/*.yaml"))
        )
        for step in yaml.safe_load(path.read_text(encoding="utf-8")).get("steps", [])
        if step.get("engine") == "qiskit"
    ]


def test_qiskit_nested_knob_universe_is_fully_filed() -> None:
    """Every nested field is assigned to examples or tests only, never both or neither."""
    assert set(REQUIRED) | set(TESTS_ONLY) == set(_UNIVERSE)
    for section, universe in _UNIVERSE.items():
        required = REQUIRED.get(section, set())
        tests_only = TESTS_ONLY.get(section, set())
        assert required.isdisjoint(tests_only), f"{section}: knob filed in both sets"
        assert required | tests_only == universe, (
            f"{section}: unfiled or stale knobs: {sorted(universe ^ (required | tests_only))}"
        )


def test_qiskit_examples_cover_required_nested_knobs() -> None:
    """Each example-required field appears at its own nesting level in a Qiskit step."""
    examples = _example_options()
    assert examples, "no shipped example uses the Qiskit engine"
    used: dict[str, set[str]] = {section: set() for section in _UNIVERSE}
    for options in examples:
        used["active_space"] |= set(options.get("active_space") or {})
        for category in REGISTRIES:
            selection = options.get(category)
            if not isinstance(selection, dict):
                continue
            used["selection"] |= set(selection)
            used[category] |= set(selection.get("options") or {})
    for section, required in REQUIRED.items():
        missing = required - used[section]
        assert not missing, f"{section}: no Qiskit example uses {sorted(missing)}"

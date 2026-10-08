"""SDK-free scientific product contracts for quantum experiment recovery.

Bundle integrity establishes that stored bytes agree with their descriptor. These
contracts additionally establish that the selected experiment produced its expected
product. They inspect only the output and resolved options, never mutable input
files, optional providers, or remote services.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from itertools import pairwise
from typing import Any

import numpy as np
from numpy.typing import NDArray

from chemrefine.engines.qiskit.bundles import QuantumBundle
from chemrefine.engines.qiskit.qpy_validation import validate_qpy_payload
from chemrefine.errors import ConfigError, OutputParseError

OutputValidator = Callable[[QuantumBundle, Mapping[str, Any]], None]
_CONTRACTS: dict[str, tuple[str, OutputValidator]] = {}


def register_experiment_output(name: str, kind: str, validator: OutputValidator) -> None:
    """Declare a custom experiment's product kind and local semantic validator.

    Register in both orchestrator and worker alongside the scientific component.
    The callback receives a validated numerical bundle and fully defaulted component
    options. Raise ``ValueError`` for malformed products; no provider calls belong
    in this contract. Existing declarations cannot be silently replaced.
    """
    name = name.strip().lower().replace("-", "_")
    if not name or not kind or name in _CONTRACTS:
        raise ValueError("experiment output contracts require unique names and nonempty kinds")
    _CONTRACTS[name] = (kind, validator)


def _contract(name: str) -> tuple[str, OutputValidator]:
    """Resolve the declared local contract without executing a provider or callback."""
    name = name.strip().lower().replace("-", "_")
    if name not in _CONTRACTS:
        raise ValueError(f"experiment {name!r} must declare register_experiment_output")
    return _CONTRACTS[name]


def require_experiment_output(name: str) -> None:
    """Refuse an undeclared extension contract during SDK-free configuration preflight."""
    try:
        _contract(name)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def validate_experiment_output(
    name: str, bundle: QuantumBundle, options: Mapping[str, Any]
) -> None:
    """Validate selected scientific output after generic integrity checks."""
    try:
        kind, validator = _contract(name)
        _require(bundle.description.kind == kind, f"expected bundle kind {kind!r}")
        validator(bundle, options)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise OutputParseError(f"invalid {name} experiment product: {exc}") from exc


def _require(condition: Any, message: str) -> None:
    """Raise one parseable contract failure for an unsuccessful scientific predicate."""
    if not condition:
        raise ValueError(message)


def _array(
    bundle: QuantumBundle, name: str, shape: tuple[int | None, ...], kinds: str = "fc"
) -> NDArray[Any]:
    """Require numerical rank, compatible dimensions and storage semantics."""
    value = bundle.arrays[name]
    _require(value.dtype.kind in kinds, f"{name} has an incompatible dtype")
    _require(value.ndim == len(shape), f"{name} has an incompatible rank")
    _require(
        all(
            expected is None or actual == expected
            for actual, expected in zip(value.shape, shape, strict=True)
        ),
        f"{name} has incompatible dimensions",
    )
    return value


def _fields(record: Mapping[str, Any], fields: Mapping[str, type]) -> None:
    """Require critical metadata types, keeping booleans distinct from integer counters."""
    for name, expected in fields.items():
        _require(type(record[name]) is expected, f"metadata {name} must be {expected.__name__}")


def _times(bundle: QuantumBundle, expected: Any) -> int:
    """Compare retained sampling times with the selected trajectory grid."""
    times = np.asarray(expected, dtype=float)
    values = _array(bundle, "times", times.shape, "f")
    _require(np.allclose(values, times, rtol=1e-12, atol=1e-14), "trajectory times disagree")
    return len(times)


def _packed(bundle: QuantumBundle, bits_name: str, counts_name: str, width: int) -> NDArray[Any]:
    """Check physical count storage, padding and positive outcome frequencies."""
    counts = _array(bundle, counts_name, (None,), "iu")
    bits = _array(bundle, bits_name, (len(counts), (width + 7) // 8), "u")
    _require(bits.dtype.itemsize == 1 and width > 0, "packed outcomes require byte storage")
    _require(np.all(counts > 0), "physical counts must be positive")
    padding = (-width) % 8
    _require(
        not padding or np.all((bits[:, -1] & ((1 << padding) - 1)) == 0), "nonzero bit padding"
    )
    return counts


def _qpy(
    bundle: QuantumBundle,
    name: str,
    circuits: int,
    *,
    num_qubits: int | None = None,
    num_clbits: int | None = None,
) -> None:
    """Apply shared local QPY structural checks without importing the SDK decoder."""
    value = _array(bundle, name, (None,), "u")
    validate_qpy_payload(value, circuits=circuits, num_qubits=num_qubits, num_clbits=num_clbits)


def _lattice(bundle: QuantumBundle, options: Mapping[str, Any]) -> None:
    """Validate trajectory dimensions in the selected fermionic encoding."""
    from chemrefine.engines.qiskit.lattice import (
        FermionicLatticeModel,
        LatticeIntegratorOptions,
        lattice_encoding_qubits,
    )

    model = FermionicLatticeModel.model_validate(options["model"])
    dynamics = LatticeIntegratorOptions.model_validate(options["dynamics"])
    size = _times(bundle, options["times"])
    _array(bundle, "statevectors", (size, 1 << lattice_encoding_qubits(model, dynamics)))
    _array(bundle, "occupations", (size, model.num_modes), "f")
    _array(bundle, "energies", (size,), "f")
    metadata = bundle.metadata
    _fields(metadata, {"model": dict, "dynamics": dict, "observations": list, "mode_order": str})
    _require(
        FermionicLatticeModel.model_validate(metadata["model"]) == model,
        "lattice model disagrees",
    )
    _require(
        LatticeIntegratorOptions.model_validate(metadata["dynamics"]) == dynamics,
        "lattice integrator disagrees",
    )
    _require(len(metadata["observations"]) == size, "missing lattice observations")
    _require(
        metadata["units"] == {"energy": "model_energy", "time": "hbar/model_energy", "hbar": 1},
        "invalid lattice units",
    )


def _variational(bundle: QuantumBundle, options: Mapping[str, Any]) -> None:
    """Validate parameter, observable and geometric diagnostic trajectories."""
    controls = options["dynamics"]
    size = _times(bundle, np.linspace(0, controls["time"], controls["steps"] + 1))
    parameters = len(options["initial_parameters"])
    _array(bundle, "parameters", (size, parameters), "f")
    _array(bundle, "expectations", (size, len(options["observables"]) + 1), "f")
    evaluations = controls["steps"] * {"euler": 1, "rk4": 4}[controls["integrator"]]
    _array(bundle, "metric_diagnostics", (evaluations, 5), "f")
    metadata = bundle.metadata
    _fields(metadata, {"parameter_order": list, "publications": int, "units": dict})
    _require(len(metadata["parameter_order"]) == parameters, "parameter ordering disagrees")
    _require(metadata["method"] == controls["method"], "variational method disagrees")
    _require(
        metadata["metric_columns"]
        == ["retained_rank", "min_eigenvalue", "max_eigenvalue", "residual_norm", "velocity_norm"],
        "invalid metric column convention",
    )


def _double_factorized(bundle: QuantumBundle, options: Mapping[str, Any]) -> None:
    """Check factors, orbitals and retained circuits without reopening source integrals."""
    size = _times(bundle, options["times"])
    one = _array(bundle, "factor_one_body", (None, None))
    modes = len(one)
    _require(modes > 0 and one.shape == (modes, modes), "invalid one-body factor")
    factors = _array(bundle, "diagonal_coulomb", (None, modes, modes))
    _array(bundle, "orbital_rotations", (len(factors), modes, modes))
    _array(bundle, "statevectors", (size, 1 << (2 * modes)))
    _array(bundle, "energies", (size,), "f")
    _array(bundle, "occupations", (size, 2 * modes), "f")
    _qpy(bundle, "circuits_qpy", size, num_qubits=2 * modes, num_clbits=0)
    metadata = bundle.metadata
    _fields(metadata, {"observations": list, "reference_occupations": dict, "circuits_format": str})
    _require(len(metadata["observations"]) == size, "missing factorized observations")
    for spin in ("alpha", "beta"):
        _require(
            len(metadata["reference_occupations"][spin]) == modes,
            "reference orbital dimension disagrees",
        )
    _require(metadata["orbital_order"] == "alpha_then_beta", "invalid orbital order")
    _require(
        metadata["units"] == {"energy": "hartree", "time": "hbar/hartree", "hbar": 1},
        "invalid evolution units",
    )


def _measurement(bundle: QuantumBundle, options: Mapping[str, Any]) -> None:
    """Certify measurement frames and reconstruct statistics from production counts."""
    from chemrefine.engines.qiskit.measurement_validation import validate_measurement

    validate_measurement(bundle, options)


def _shadows(bundle: QuantumBundle, options: Mapping[str, Any]) -> None:
    """Validate ensemble-dependent settings, RDM clusters and physical shot intervals."""
    controls, metadata = options["shadows"], bundle.metadata
    _fields(
        metadata,
        {
            "num_modes": int,
            "num_settings": int,
            "total_shots": int,
            "acceptance": list,
            "rdm_convention": str,
        },
    )
    modes, settings = metadata["num_modes"], controls["num_settings"]
    _require(modes > 0 and metadata["num_settings"] == settings, "shadow dimensions disagree")
    _require(metadata["ensemble"] == controls["ensemble"], "shadow ensemble disagrees")
    width = modes if controls["ensemble"] == "orbital_haar" else 2 * modes
    _array(bundle, "settings", (settings, width, width))
    for order, name in ((1, "one_body"), (2, "two_body")):
        if order > controls["max_order"]:
            continue
        shape = (modes,) * (2 * order)
        _array(bundle, name, shape)
        _array(bundle, "setting_" + name, (settings, *shape))
        if settings > 1:
            for part in ("real", "imag"):
                errors = _array(bundle, f"standard_error_{part}_{name}", shape, "f")
                _require(np.all(errors >= 0), "negative shadow uncertainty")
    counts = _packed(bundle, "bitstrings", "counts", modes)
    offsets = _array(bundle, "setting_offsets", (settings + 1,), "iu")
    _require(
        offsets[0] == 0 and offsets[-1] == len(counts) and np.all(offsets[1:] > offsets[:-1]),
        "invalid shadow setting intervals",
    )
    for begin, end in pairwise(offsets):
        _require(
            sum(map(int, counts[int(begin) : int(end)])) == controls["shots_per_setting"],
            "shadow setting shot total disagrees",
        )
    _require(
        metadata["total_shots"] == settings * controls["shots_per_setting"],
        "shadow total shots disagree",
    )
    _require(len(metadata["acceptance"]) == settings, "missing shadow acceptance statistics")


def _rdm(bundle: QuantumBundle, options: Mapping[str, Any]) -> None:
    """Check matching raw/fitted complex tensors and explicit solver verdicts."""
    one = _array(bundle, "one_body", (None, None))
    modes = len(one)
    _require(modes > 0 and one.shape == (modes, modes), "invalid RDM mode count")
    _array(bundle, "raw_one_body", (modes, modes))
    for name in ("two_body", "raw_two_body"):
        _array(bundle, name, (modes,) * 4)
    metadata, controls = bundle.metadata, options["reconstruction"]
    _fields(metadata, {"constraint_checks": dict, "rdm_convention": str, "units": dict})
    allowed = {"optimal", "optimal_inaccurate"} if controls["accept_inaccurate"] else {"optimal"}
    _require(
        metadata["status"] in allowed and metadata["solver"] == "SCS", "invalid RDM solver verdict"
    )
    for key in ("constraints", "energy_weight", "loss"):
        _require(metadata[key] == controls[key], f"RDM {key} disagrees")
    _require(
        metadata["necessary_not_sufficient"] is True and metadata["variational_bound"] is False,
        "invalid RDM guarantee declaration",
    )
    _require(controls["num_particles"] <= modes, "RDM particle count exceeds modes")


def _spacetime(bundle: QuantumBundle, options: Mapping[str, Any]) -> None:
    """Require joint/syndrome count accounting and the retained check circuit."""
    metadata, controls = bundle.metadata, options["spacetime"]
    _fields(
        metadata,
        {
            "num_data_qubits": int,
            "num_check_qubits": int,
            "raw_shots": int,
            "accepted_shots": int,
            "input_checks": list,
            "output_checks": list,
        },
    )
    modes, checks = metadata["num_data_qubits"], metadata["num_check_qubits"]
    _require(checks == len(controls["checks"]) and checks > 0, "check register disagrees")
    _require(
        metadata["input_checks"] == controls["checks"] and len(metadata["output_checks"]) == checks,
        "missing Pauli check identities",
    )
    _qpy(bundle, "checked_circuit_qpy", 1, num_qubits=modes + checks, num_clbits=0)
    totals = {}
    for name, width in (("raw", modes + checks), ("accepted", modes), ("rejected", modes + checks)):
        totals[name] = sum(map(int, _packed(bundle, name + "_bitstrings", name + "_counts", width)))
    _require(
        totals["raw"] == controls["shots"] == metadata["raw_shots"], "raw shot total disagrees"
    )
    _require(
        totals["accepted"] == metadata["accepted_shots"]
        and totals["raw"] == totals["accepted"] + totals["rejected"],
        "postselection shot totals disagree",
    )
    _require(
        np.isclose(metadata["acceptance_rate"], totals["accepted"] / totals["raw"]),
        "acceptance rate disagrees",
    )


def _cutting(bundle: QuantumBundle, options: Mapping[str, Any]) -> None:
    """Require reconstruction coefficients, QPY and every referenced physical register."""
    metadata = bundle.metadata
    _fields(
        metadata,
        {
            "observable_labels": list,
            "weight_types": list,
            "records": list,
            "subobservables": dict,
            "logical_qpy_partition_counts": dict,
        },
    )
    number = len(metadata["observable_labels"])
    _require(
        number > 0 and metadata["observable"] == options["observable"], "cut observable disagrees"
    )
    _array(bundle, "term_expectations", (number,), "f")
    coefficients = _array(bundle, "observable_coefficients", (number,), "f")
    _require(
        np.array_equal(
            coefficients, [options["observable"][key] for key in metadata["observable_labels"]]
        ),
        "observable coefficients disagree",
    )
    weights = _array(bundle, "qpd_coefficients", (len(metadata["weight_types"]),), "f")
    _require(
        weights.size > 0
        and metadata["signed_reconstruction"] is True
        and metadata["clipped"] is False,
        "invalid signed reconstruction contract",
    )
    _qpy(bundle, "logical_experiments_qpy", sum(metadata["logical_qpy_partition_counts"].values()))
    _require(len(metadata["records"]) > 0, "missing physical cutting records")
    for record in metadata["records"]:
        _fields(record, {"shots": int, "registers": dict})
        _require(
            record["shots"] == options["cutting"]["shots"] and record["registers"],
            "invalid cutting shot record",
        )
        for register in record["registers"].values():
            _fields(register, {"array": str, "num_bits": int})
            data = _array(
                bundle, register["array"], (record["shots"], (register["num_bits"] + 7) // 8), "u"
            )
            _require(
                data.dtype.itemsize == 1 and register["num_bits"] > 0,
                "invalid cutting register byte storage",
            )


def _resources(bundle: QuantumBundle, options: Mapping[str, Any]) -> None:
    """Validate typed report discriminators, units and mandatory quantitative costs."""
    metadata = bundle.metadata
    _require(
        not bundle.arrays and metadata["executable_circuit"] is False,
        "resource output is an analytical report",
    )
    _fields(metadata, {"estimate_kind": str, "exclusions": list})
    kind = metadata["estimate_kind"]
    if "hamiltonian" in options:
        _require(kind == "analytical_query_bound", "expected Pauli resource report")
        _fields(
            metadata,
            {
                "system_qubits": int,
                "controlled_walk_queries": int,
                "phase_bits": int,
                "budget": dict,
            },
        )
        _require(
            metadata["system_qubits"] == len(next(iter(options["hamiltonian"]))),
            "resource system width disagrees",
        )
        _require(metadata["budget"] == options["budget"], "QPE budget disagrees")
        _require(
            metadata["normalization_hartree"] >= 0 and metadata["controlled_walk_queries"] >= 0,
            "invalid query cost",
        )
    elif "method" in options:
        _require(
            kind == "provider_analytical_cost_model" and metadata["method"] == options["method"],
            "expected factorized resource report",
        )
        _fields(
            metadata,
            {
                "system_qubits": int,
                "provider": str,
                "version": str,
                "factorization_parameters": dict,
                "conservative_standard_qpe": dict,
                "provider_toffoli_total_single_run": int,
                "provider_logical_qubits_including_system_and_phase": int,
            },
        )
        _require(
            metadata["normalization_hartree"] > 0
            and metadata["provider_logical_qubits_including_system_and_phase"] > 0
            and metadata["provider_toffoli_total_single_run"] > 0,
            "invalid factorized cost",
        )
    else:
        _require(kind == "phenomenological_surface_code_model", "expected physical resource report")
        _fields(
            metadata,
            {
                "physical_qubits": int,
                "assumptions": dict,
                "meets_failure_budget_at_runtime_lower_bound": bool,
            },
        )
        _require(metadata["assumptions"] == dict(options), "physical resource assumptions disagree")
        _require(
            metadata["physical_qubits"] > 0
            and metadata["runtime_cycles_lower_bound"] > 0
            and metadata["runtime_seconds_lower_bound"] > 0
            and 0 <= metadata["failure_union_bound_at_runtime_lower_bound"] <= 1,
            "invalid physical resource cost",
        )


for _name, _kind, _validator in (
    ("lattice_dynamics", "lattice_trajectory", _lattice),
    ("variational_dynamics", "variational_trajectory", _variational),
    ("double_factorized_evolution", "double_factorized_trajectory", _double_factorized),
    ("pauli_measurement", "pauli_measurement", _measurement),
    ("fermionic_shadows", "fermionic_shadows", _shadows),
    ("rdm_reconstruction", "reconstructed_rdms", _rdm),
    ("spacetime_postselection", "spacetime_postselection", _spacetime),
    ("circuit_cutting", "circuit_cutting", _cutting),
    ("pauli_resources", "resource_estimate", _resources),
    ("factorized_resources", "resource_estimate", _resources),
    ("surface_code_resources", "resource_estimate", _resources),
):
    register_experiment_output(_name, _kind, _validator)

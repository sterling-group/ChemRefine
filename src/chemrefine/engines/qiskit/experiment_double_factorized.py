"""Native double-factorized trajectories from portable electronic-integral inputs."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from chemrefine.engines.api import BackendRequirement
from chemrefine.engines.qiskit.bundles import DEFAULT_MAX_BYTES
from chemrefine.engines.qiskit.double_factorized import (
    DoubleFactorizedIntegratorOptions,
    DoubleFactorizedOptions,
    simulate_double_factorized_evolution,
)
from chemrefine.engines.qiskit.experiment import EXPERIMENTS, ExperimentResult
from chemrefine.engines.qiskit.integral_io import load_integrals
from chemrefine.errors import ConfigError


class DoubleFactorizedExperimentOptions(BaseModel):
    """One integral bundle and independent times evolving the declared occupied orbitals."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    integral_bundle_path: str = Field(
        min_length=1, json_schema_extra={"input_file": True, "file_format": "quantum_bundle"}
    )
    max_input_bytes: StrictInt = Field(DEFAULT_MAX_BYTES, ge=1)
    times: tuple[float, ...] = Field((0, 1), min_length=1, max_length=512)
    evolution: DoubleFactorizedIntegratorOptions = Field(
        default_factory=DoubleFactorizedIntegratorOptions
    )


@EXPERIMENTS.register(
    "double_factorized_evolution",
    DoubleFactorizedExperimentOptions,
    status="experimental",
    supported_domains=(
        "complex shared-spin one-body with real shared-spin chemist two-body integrals",
        "open-shell or closed-shell declared occupied-orbital references",
        "bounded ideal Jordan-Wigner trajectories with explicit nuclear and supplied phases",
    ),
    backend_requirement=BackendRequirement(extra="qiskit-fermionic", import_name="ffsim"),
)
def double_factorized_experiment(
    *, options: DoubleFactorizedExperimentOptions, **context: Any
) -> ExperimentResult:
    """Retain executable QPY circuits and separate factorization and propagation diagnostics."""
    data = load_integrals(Path(options.integral_bundle_path), max_bytes=options.max_input_bytes)
    n, controls = data.num_spatial_orbitals, options.evolution
    factors = n * (n + 1) // 2 if controls.max_factors is None else controls.max_factors
    operations = (
        controls.steps * {1: 1, 2: 2, 4: 10}[controls.order] * (2 * factors + 2) * (20 * n * n + 1)
    )
    required = len(options.times) * (16 * (1 << (2 * n)) + 512 * operations)
    required += 16 * n * n * (1 + 2 * factors)
    if required > context["max_output_bytes"]:
        raise ConfigError("double-factorized trajectory exceeds max_output_bytes")
    results = [
        simulate_double_factorized_evolution(
            data,
            DoubleFactorizedOptions(**controls.model_dump(), time=time),
            cores=context["cores"],
        )
        for time in options.times
    ]
    from qiskit import qpy

    stream = io.BytesIO()
    qpy.dump([result.circuit for result in results], stream, version=13)
    return ExperimentResult(
        kind="double_factorized_trajectory",
        arrays={
            "times": np.asarray(options.times),
            "statevectors": np.asarray([result.statevector for result in results]),
            "energies": np.asarray([result.metadata["total_energy_hartree"] for result in results]),
            "occupations": np.asarray([result.metadata["mode_occupations"] for result in results]),
            "factor_one_body": results[0].one_body,
            "diagonal_coulomb": results[0].diagonal_coulomb,
            "orbital_rotations": results[0].orbital_rotations,
            "circuits_qpy": np.frombuffer(stream.getvalue(), dtype=np.uint8),
        },
        metadata={
            "experimental": True,
            "units": {"energy": "hartree", "time": "hbar/hartree", "hbar": 1},
            "orbital_order": "alpha_then_beta",
            "circuits_format": (
                "QPY version 13; one circuit per time; excludes reference preparation"
            ),
            "reference_occupations": {
                "alpha": np.asarray(data.orbital_occupations).tolist(),
                "beta": np.asarray(data.orbital_occupations_beta).tolist(),
            },
            "input_provenance": data.provenance,
            "observations": [result.metadata for result in results],
        },
    )

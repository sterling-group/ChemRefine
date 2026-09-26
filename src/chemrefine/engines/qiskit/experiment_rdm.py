"""Artifact reconstruction of measured complex RDMs with explicit convex constraints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from chemrefine.engines.api import BackendRequirement
from chemrefine.engines.qiskit.bundles import DEFAULT_MAX_BYTES, read_bundle
from chemrefine.engines.qiskit.determinants import (
    FermionicHamiltonian,
    FermionTerm,
    ReducedDensityMatrices,
)
from chemrefine.engines.qiskit.experiment import EXPERIMENTS, ExperimentResult
from chemrefine.engines.qiskit.integral_io import load_integrals
from chemrefine.engines.qiskit.rdm_reconstruction import RDMReconstructionOptions, reconstruct_rdms
from chemrefine.errors import ConfigError


class RDMExperimentOptions(BaseModel):
    """Measured tensors, optional objective Hamiltonian and explicit fitting weights."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    rdm_bundle_path: str = Field(
        min_length=1, json_schema_extra={"input_file": True, "file_format": "quantum_bundle"}
    )
    hamiltonian_bundle_path: str | None = Field(
        None, min_length=1, json_schema_extra={"input_file": True, "file_format": "quantum_bundle"}
    )
    max_input_bytes: StrictInt = Field(DEFAULT_MAX_BYTES, ge=1)
    one_body_weights_array: str | None = Field(None, min_length=1)
    two_body_weights_array: str | None = Field(None, min_length=1)
    reconstruction: RDMReconstructionOptions

    @model_validator(mode="after")
    def _objective_contract(self) -> Self:
        """Reject energy regularization without a declared Hamiltonian before provisioning."""
        if self.reconstruction.energy_weight and self.hamiltonian_bundle_path is None:
            raise ValueError("energy regularization requires hamiltonian_bundle_path")
        return self


@EXPERIMENTS.register(
    "rdm_reconstruction",
    RDMExperimentOptions,
    status="experimental",
    supported_domains=(
        "complex fixed-total-particle one/two-body RDMs",
        "D/DQ/DQG necessary representability constraints, not sufficient certificates",
        "explicit data-fit or energy-regularized objectives without variational guarantees",
    ),
    backend_requirement=BackendRequirement(extra="qiskit-rdm", import_name="cvxpy"),
)
def rdm_experiment(*, options: RDMExperimentOptions, **context: Any) -> ExperimentResult:
    """Retain raw and reconstructed tensors and solver/feasibility diagnostics separately."""
    source = read_bundle(Path(options.rdm_bundle_path), max_bytes=options.max_input_bytes)
    if source.description.kind not in {"fermionic_shadows", "measured_rdms"}:
        raise ConfigError("RDM reconstruction requires a measured_rdms or fermionic_shadows bundle")
    try:
        raw = ReducedDensityMatrices(source.arrays["one_body"], source.arrays["two_body"])
        weights = {
            key: None if name is None else source.arrays[name]
            for key, name in (
                ("one_body_weights", options.one_body_weights_array),
                ("two_body_weights", options.two_body_weights_array),
            )
        }
    except KeyError as exc:
        raise ConfigError(
            f"RDM input bundle is missing a required tensor or weight: {exc}"
        ) from exc
    # Both raw and fitted output copies use complex128 even for real input files.
    if 2 * 16 * (raw.one_body.size + np.asarray(raw.two_body).size) > context["max_output_bytes"]:
        raise ConfigError("RDM reconstruction output exceeds max_output_bytes")
    hamiltonian = None
    if options.hamiltonian_bundle_path is not None:
        from chemrefine.engines.qiskit.problem import prepare_problem

        data = load_integrals(
            Path(options.hamiltonian_bundle_path), max_bytes=options.max_input_bytes
        )
        prepared = prepare_problem(data)
        active = FermionicHamiltonian.from_operator(
            prepared.fermionic_hamiltonian, num_modes=prepared.num_spin_orbitals
        )
        hamiltonian = FermionicHamiltonian(
            active.num_modes,
            (*active.terms, FermionTerm((), (), sum(prepared.energy_offsets.values()))),
        )
    result = reconstruct_rdms(raw, options.reconstruction, hamiltonian=hamiltonian, **weights)
    return ExperimentResult(
        kind="reconstructed_rdms",
        arrays={
            "raw_one_body": result.raw.one_body,
            "raw_two_body": np.asarray(result.raw.two_body),
            "one_body": result.reconstructed.one_body,
            "two_body": np.asarray(result.reconstructed.two_body),
        },
        metadata={
            **result.diagnostics,
            "source_kind": source.description.kind,
            "source_ensemble": source.metadata.get("ensemble"),
            "source_property_origin": source.metadata.get(
                "property_source", "user-supplied measurements"
            ),
            "property_source": "constrained reconstruction of measured RDMs",
            "mode_order": source.metadata.get("mode_order", "input mode order preserved"),
            "rdm_convention": "gamma[p,q]=<a†p aq>; Gamma[p,q,r,s]=<a†p a†q a_s a_r>",
            "units": {"rdm": "dimensionless", "energy": None if hamiltonian is None else "hartree"},
            "uncertainty": "solver output does not propagate measurement uncertainty",
        },
    )

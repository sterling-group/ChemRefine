"""Credential-free, versioned Runtime controls, validated without importing the provider."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator


class RuntimeModel(BaseModel):
    """Strict immutable controls shared by Runtime execution and mitigation."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class DynamicalDecouplingOptions(RuntimeModel):
    """Runtime's scheduled idle-time pulse suppression options."""

    enable: bool = False
    sequence_type: Literal["XX", "XpXm", "XY4"] = "XX"
    extra_slack_distribution: Literal["middle", "edges"] = "middle"
    scheduling_method: Literal["alap", "asap"] = "alap"
    skip_reset_qubits: bool = False


class TwirlingOptions(RuntimeModel):
    """Finite Pauli-randomization controls with optional resilience-level inheritance."""

    enable_gates: bool | None = None
    enable_measure: bool | None = None
    num_randomizations: StrictInt | Literal["auto"] = "auto"
    shots_per_randomization: StrictInt | Literal["auto"] = "auto"
    strategy: Literal["active", "active-accum", "active-circuit", "all"] = "active-accum"
    group: Literal["pauli", "balanced_pauli", "local_c1", "local_pauli"] = "pauli"

    @model_validator(mode="after")
    def _positive_counts(self) -> Self:
        """Retain auto sentinels while rejecting zero and negative randomization counts."""
        if any(
            isinstance(value, int) and value < 1
            for value in (
                self.num_randomizations,
                self.shots_per_randomization,
            )
        ):
            raise ValueError("twirling counts must be positive or auto")
        return self


Extrapolator = Literal[
    "linear",
    "exponential",
    "double_exponential",
    "polynomial_degree_1",
    "polynomial_degree_2",
    "polynomial_degree_3",
    "polynomial_degree_4",
    "polynomial_degree_5",
    "polynomial_degree_6",
    "polynomial_degree_7",
    "fallback",
]


class ZNEOptions(RuntimeModel):
    """Noise scaling and extrapolators; all submitted factors and fit degrees are explicit."""

    enable: bool | None = None
    amplifier: Literal["gate_folding", "gate_folding_front", "gate_folding_back", "pea"] = (
        "gate_folding"
    )
    noise_factors: tuple[float, ...] = (1, 3, 5)
    extrapolator: tuple[Extrapolator, ...] = ("linear",)
    extrapolated_noise_factors: tuple[float, ...] = (0,)

    @model_validator(mode="after")
    def _fit_data(self) -> Self:
        """Reject duplicate scales and underdetermined polynomial/exponential fits."""
        factors = self.noise_factors
        if len(factors) < 2 or len(set(factors)) != len(factors) or any(x < 1 for x in factors):
            raise ValueError("ZNE requires distinct noise_factors >= 1")
        if not self.extrapolator or len(set(self.extrapolator)) != len(self.extrapolator):
            raise ValueError("ZNE requires distinct extrapolators")
        if not self.extrapolated_noise_factors or any(
            x < 0 for x in self.extrapolated_noise_factors
        ):
            raise ValueError("extrapolated_noise_factors must be non-negative")
        for name in self.extrapolator:
            parameters = (
                int(name.removeprefix("polynomial_degree_")) + 1
                if name.startswith("polynomial_degree_")
                else {"linear": 2, "exponential": 2, "double_exponential": 4, "fallback": 1}[name]
            )
            if len(factors) < parameters:
                raise ValueError(f"ZNE {name} needs at least {parameters} distinct noise factors")
        return self


class PECOptions(RuntimeModel):
    """Bound cancellation overhead, retaining the provider's optional partial-noise removal."""

    enable: bool = False
    max_overhead: float = Field(100, ge=1)
    noise_gain: float | Literal["auto"] = "auto"

    @model_validator(mode="after")
    def _gain(self) -> Self:
        """Reject nonphysical negative noise gain."""
        if isinstance(self.noise_gain, float) and self.noise_gain < 0:
            raise ValueError("PEC noise_gain must be non-negative or auto")
        return self


class NoiseLearningOptions(RuntimeModel):
    """Explicit executor PEC/PEA calibration, with finite layer and circuit controls."""

    enabled: bool = False
    max_layers: StrictInt = Field(128, ge=1)
    shots_per_randomization: StrictInt = Field(128, ge=1)
    num_randomizations: StrictInt = Field(32, ge=1)
    layer_pair_depths: tuple[StrictInt, ...] = (0, 1, 2, 4, 16, 32)

    @model_validator(mode="after")
    def _depths(self) -> Self:
        """Require informative, distinct non-negative calibration depths."""
        if (
            len(self.layer_pair_depths) < 2
            or len(set(self.layer_pair_depths)) != len(self.layer_pair_depths)
            or any(x < 0 for x in self.layer_pair_depths)
        ):
            raise ValueError("noise learning needs at least two distinct non-negative depths")
        return self


class RuntimeOptions(RuntimeModel):
    """Execution placement, compilation, recoverable jobs and bounded publication batches."""

    implementation: Literal["executor", "legacy_v2"] = "executor"
    backend_name: str | None = Field(None, pattern=r"^[A-Za-z0-9_.-]+$")
    fake_backend: str | None = Field(None, pattern=r"^Fake[A-Za-z0-9]+$")
    account_name: str | None = Field(None, min_length=1)
    channel: Literal["ibm_quantum_platform", "ibm_cloud"] | None = None
    instance: str | None = Field(None, min_length=1)
    mode: Literal["job", "session", "batch"] = "job"
    mode_id: str | None = Field(None, min_length=1)
    max_time: StrictInt = Field(3600, ge=1)
    max_execution_time: StrictInt = Field(300, ge=1)
    close_mode: bool = True
    initial_layout: tuple[StrictInt, ...] | None = None
    optimization_level: Literal[0, 1, 2, 3] = 1
    seed_transpiler: StrictInt | None = Field(None, ge=0)
    seed_simulator: StrictInt | None = Field(None, ge=0)
    journal_dir: str | None = None
    journal_history_dirs: tuple[str, ...] = Field(default=(), max_length=128)
    max_journal_records: StrictInt = Field(10000, ge=1)
    retrieve_job_ids: tuple[str, ...] = ()
    max_jobs: StrictInt = Field(1000, ge=1)
    max_pubs_per_job: StrictInt = Field(128, ge=1)
    max_parameter_sets: StrictInt = Field(10000, ge=1)
    max_nominal_shots: StrictInt = Field(10000000, ge=1)
    max_request_bytes: StrictInt = Field(67108864, ge=1)
    dynamical_decoupling: DynamicalDecouplingOptions = Field(
        default_factory=DynamicalDecouplingOptions
    )
    twirling: TwirlingOptions = Field(default_factory=TwirlingOptions)

    @model_validator(mode="after")
    def _execution_contract(self) -> Self:
        """Reject ignored placement choices, ambiguous providers and invalid recovery options."""
        if (self.backend_name is None) == (self.fake_backend is None):
            raise ValueError("select exactly one backend_name or fake_backend")
        if self.journal_dir is not None and not Path(self.journal_dir).expanduser().is_absolute():
            raise ValueError("journal_dir must be absolute and durable")
        if any(not Path(value).expanduser().is_absolute() for value in self.journal_history_dirs):
            raise ValueError("journal_history_dirs must be absolute")
        if self.mode == "job" and self.mode_id is not None:
            raise ValueError("mode_id requires session or batch mode")
        if self.mode_id and self.close_mode:
            raise ValueError("an existing mode is caller-owned; set close_mode false")
        if self.retrieve_job_ids and self.mode != "job":
            raise ValueError("explicit retrieval uses job mode without creating a session/batch")
        if self.initial_layout is not None and (
            not self.initial_layout
            or any(q < 0 for q in self.initial_layout)
            or len(set(self.initial_layout)) != len(self.initial_layout)
        ):
            raise ValueError("initial_layout requires distinct non-negative physical indices")
        if self.fake_backend and any(
            (self.account_name, self.channel, self.instance, self.mode_id, self.retrieve_job_ids)
        ):
            raise ValueError("fake backends cannot use remote credentials, mode IDs or retrieval")
        if not self.fake_backend and self.seed_simulator is not None:
            raise ValueError("seed_simulator requires a fake backend")
        if self.implementation == "legacy_v2" and self.seed_simulator == 0:
            raise ValueError("legacy local SDK ignores seed_simulator zero; choose a positive seed")
        if self.implementation == "legacy_v2" and self.twirling.group != "pauli":
            raise ValueError("non-Pauli twirling groups require executor implementation")
        if len(set(self.retrieve_job_ids)) != len(self.retrieve_job_ids) or any(
            not value for value in self.retrieve_job_ids
        ):
            raise ValueError("retrieve_job_ids must be distinct nonempty identifiers")
        return self


class RuntimeEstimatorOptions(RuntimeOptions):
    """Estimator precision and mitigation, with executor-specific noise learning requirements."""

    default_precision: float = Field(0.015625, gt=0)
    default_shots: StrictInt | None = Field(None, ge=1)
    resilience_level: Literal[0, 1, 2] = 0
    trex: bool | None = None
    zne: ZNEOptions = Field(default_factory=ZNEOptions)
    pec: PECOptions = Field(default_factory=PECOptions)
    noise_learning: NoiseLearningOptions = Field(default_factory=NoiseLearningOptions)

    @model_validator(mode="after")
    def _mitigation_contract(self) -> Self:
        """Resolve inherited toggles before checking mitigation and calibration compatibility."""
        zne = self.zne.enable if self.zne.enable is not None else self.resilience_level == 2
        trex = self.trex if self.trex is not None else self.resilience_level >= 1
        learned = self.pec.enable or (zne and self.zne.amplifier == "pea")
        if self.pec.enable and zne:
            raise ValueError("PEC and ZNE cannot be enabled together")
        if (trex or learned) and self.twirling.enable_measure is False:
            raise ValueError("TREX/PEC/PEA requires measurement twirling")
        if learned and self.twirling.enable_gates is False:
            raise ValueError("PEC/PEA requires gate twirling")
        if self.implementation == "executor" and learned and not self.noise_learning.enabled:
            raise ValueError("executor PEC/PEA requires explicit noise_learning.enabled")
        if self.noise_learning.enabled and (not learned or self.implementation != "executor"):
            raise ValueError("explicit noise_learning is used by executor PEC/PEA only")
        if self.fake_backend and self.noise_learning.enabled:
            raise ValueError("Runtime0.50 NoiseLearnerV3 does not support local fake backends")
        if (
            self.fake_backend
            and self.implementation == "legacy_v2"
            and (
                trex
                or zne
                or self.pec.enable
                or self.dynamical_decoupling.enable
                or self.twirling.enable_gates
                or self.twirling.enable_measure
            )
        ):
            raise ValueError(
                "legacy local mode ignores mitigation/DD/twirling; use executor or hardware"
            )
        return self


class RuntimeSamplerOptions(RuntimeOptions):
    """Classified finite-shot sampling; estimator-only mitigation is deliberately absent."""

    default_shots: StrictInt = Field(4096, ge=1)

    @model_validator(mode="after")
    def _local_legacy(self) -> Self:
        """Reject mitigation controls the legacy local sampler silently ignores."""
        if (
            self.fake_backend
            and self.implementation == "legacy_v2"
            and (
                self.dynamical_decoupling.enable
                or self.twirling.enable_gates
                or self.twirling.enable_measure
            )
        ):
            raise ValueError("legacy local mode ignores DD/twirling; use executor or hardware")
        return self

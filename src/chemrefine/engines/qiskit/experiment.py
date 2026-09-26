"""Artifact-engine adapter for configurable quantum experiments.

One scheduled worker writes a native bundle and passes the pipeline's structures
through unchanged. Experiment builders are reusable Python functions; neither the
builder nor the engine makes decisions about pipeline caching or recovery.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from chemrefine.config import StepConfig
from chemrefine.engines import _provision
from chemrefine.engines._input_files import typed_input_references
from chemrefine.engines._options import EngineOptions
from chemrefine.engines.api import BackendRequirement, ComponentCategory, RunBlock, register
from chemrefine.engines.qiskit.bundles import (
    DEFAULT_MAX_BYTES,
    bundle_dependencies,
    read_bundle,
    write_bundle,
)
from chemrefine.engines.qiskit.lattice import (
    FermionicLatticeModel,
    LatticeDynamicsOptions,
    LatticeIntegratorOptions,
    lattice_encoding_qubits,
)
from chemrefine.engines.qiskit.options import ComponentSelection
from chemrefine.engines.qiskit.profiles import BACKEND_PROFILES
from chemrefine.engines.qiskit.registry import REGISTRIES, ComponentRegistry
from chemrefine.errors import ConfigError
from chemrefine.state import JobBatch, StepContext, StepInputs, StepResults

EXPERIMENTS = ComponentRegistry("experiment")


@dataclass(frozen=True)
class ExperimentResult:
    """Provider-independent arrays and provenance ready for durable publication."""

    kind: str
    arrays: Mapping[str, NDArray[Any]]
    metadata: Mapping[str, Any]


class QiskitExperimentOptions(EngineOptions):
    """One named quantum workflow and its validated artifact allocation limit."""

    experiment: ComponentSelection = Field(
        default_factory=lambda: ComponentSelection.named("lattice_dynamics")
    )
    max_output_bytes: StrictInt = Field(DEFAULT_MAX_BYTES, ge=1)

    @field_validator("experiment", mode="before")
    @classmethod
    def _short_name(cls, value: Any) -> Any:
        """Accept the same named-component shorthand as molecular Qiskit steps."""
        return {"name": value} if isinstance(value, str) else value


class LatticeExperimentOptions(BaseModel):
    """Sample an ideal lattice trajectory at explicit physical times."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    model: FermionicLatticeModel = Field(
        default_factory=lambda: FermionicLatticeModel(num_sites=1, spinful=False)
    )
    occupied_modes: tuple[StrictInt, ...] = (0,)
    times: tuple[float, ...] = Field((0.0, 1.0), min_length=1)
    dynamics: LatticeIntegratorOptions = Field(default_factory=LatticeIntegratorOptions)

    @model_validator(mode="after")
    def _physical_inputs(self) -> Self:
        """Refuse decidable occupation and allocation errors before scheduling a worker."""
        if lattice_encoding_qubits(self.model, self.dynamics) > self.dynamics.max_qubits:
            raise ValueError("lattice mode count exceeds max_qubits")
        if len(set(self.occupied_modes)) != len(self.occupied_modes) or any(
            mode < 0 or mode >= self.model.num_modes for mode in self.occupied_modes
        ):
            raise ValueError("occupied_modes must be distinct and within the mode count")
        return self


@EXPERIMENTS.register(
    "lattice_dynamics",
    LatticeExperimentOptions,
    status="experimental",
    supported_domains=(
        "number-conserving complex lattice Hamiltonians",
        "bounded ideal statevector trajectories",
        "graph BKSF or open-square VC/DK local encodings",
    ),
    backend_requirement=BackendRequirement(extra="qiskit-fermionic", import_name="qiskit_fermions"),
)
def lattice_experiment(*, options: LatticeExperimentOptions, **context: Any) -> ExperimentResult:
    """Evaluate independent time points with the same reference and integrator."""
    from chemrefine.engines.qiskit.lattice import simulate_lattice_dynamics

    # Validate the aggregate allocation before constructing any statevector.
    required = (
        len(options.times) * (1 << lattice_encoding_qubits(options.model, options.dynamics)) * 16
    )
    if required > context["max_output_bytes"]:
        raise ConfigError("lattice trajectory exceeds max_output_bytes")
    outcomes = [
        simulate_lattice_dynamics(
            options.model,
            occupied_modes=options.occupied_modes,
            options=LatticeDynamicsOptions(**options.dynamics.model_dump(), time=time),
        )
        for time in options.times
    ]
    return ExperimentResult(
        kind="lattice_trajectory",
        arrays={
            "times": np.asarray(options.times),
            "statevectors": np.stack([outcome.statevector for outcome in outcomes]),
            "occupations": np.asarray([outcome.mode_occupations for outcome in outcomes]),
            "energies": np.asarray([outcome.energy_expectation for outcome in outcomes]),
        },
        metadata={
            "units": {"energy": "model_energy", "time": "hbar/model_energy", "hbar": 1},
            "mode_order": "alpha_then_beta" if options.model.spinful else "site",
            "model": options.model.model_dump(mode="json"),
            "dynamics": options.dynamics.model_dump(mode="json"),
            "observations": [outcome.as_dict() for outcome in outcomes],
        },
    )


def validate_experiment(options: QiskitExperimentOptions) -> None:
    """Resolve typed scientific options without importing optional SDKs."""
    component = EXPERIMENTS.options_for(options.experiment)
    for category in EXPERIMENTS.spec(options.experiment.name).requires & REGISTRIES.keys():
        selection = getattr(component, category)
        REGISTRIES[category].options_for(selection)
        if (
            options.device == "cuda"
            and "cuda" not in REGISTRIES[category].spec(selection.name).capabilities
        ):
            raise ConfigError(f"qiskit {category} {selection.name!r} requires device: cpu")
    if (
        options.device == "cuda"
        and "cuda" not in EXPERIMENTS.spec(options.experiment.name).capabilities
    ):
        raise ConfigError(f"qiskit experiment {options.experiment.name!r} requires device: cpu")


def run_experiment(options: Mapping[str, Any], output_path: Path) -> Path:
    """Execute a configured experiment and atomically publish its native bundle."""
    resolved = QiskitExperimentOptions.from_raw(options)
    validate_experiment(resolved)
    result: ExperimentResult = EXPERIMENTS.build(
        resolved.experiment,
        cores=resolved.cores,
        device=resolved.device,
        max_output_bytes=resolved.max_output_bytes,
    )
    return write_bundle(
        output_path,
        kind=result.kind,
        arrays=result.arrays,
        metadata={**result.metadata, "resolved_options": resolved.model_dump(mode="json")},
        max_bytes=resolved.max_output_bytes,
    )


@register("qiskit-experiment")
class QiskitExperimentEngine:
    """Run one artifact-producing quantum experiment through the shared scheduler."""

    name: ClassVar[str] = "qiskit-experiment"
    options_cls: ClassVar[type[QiskitExperimentOptions]] = QiskitExperimentOptions
    preflight_refuses: ClassVar[str] = "unknown experiments, invalid options or unsupported devices"
    output_globs: ClassVar[tuple[str, ...]] = ("*.json", "*.npz", "*.qpy")

    def component_catalog(self) -> dict[str, ComponentCategory]:
        """Publish registered experiment schemas using the validating model's default."""
        return {"experiment": EXPERIMENTS.describe(self.options_cls().experiment.name)}

    def _options(self, ctx: StepContext) -> QiskitExperimentOptions:
        """Read the same strict model used by preflight, schema and provisioning."""
        return self.options_cls.from_raw(ctx.step_cfg.engine_options())

    def backend_requirement(self, options: Mapping[str, Any] | None) -> BackendRequirement:
        """Resolve the selected experiment's declared worker environment."""
        resolved = self.options_cls.from_raw(options)
        validate_experiment(resolved)
        spec = EXPERIMENTS.spec(resolved.experiment.name)
        requirements = [] if spec.backend_requirement is None else [spec.backend_requirement]
        component = EXPERIMENTS.options_for(resolved.experiment)
        for category in spec.requires & REGISTRIES.keys():
            selected = REGISTRIES[category].spec(getattr(component, category).name)
            if selected.backend_requirement is not None:
                requirements.append(selected.backend_requirement)
        return BACKEND_PROFILES.resolve(requirements)

    def backend_extras(self) -> frozenset[str]:
        """Discover all registered experiment providers for backend installation."""
        return frozenset(
            set(BACKEND_PROFILES.names())
            | {
                requirement.extra
                for name in EXPERIMENTS.names()
                if (requirement := EXPERIMENTS.spec(name).backend_requirement) is not None
            }
        )

    def input_file_options(self, options: Mapping[str, Any]) -> tuple[tuple[str | int, ...], ...]:
        """Discover all nested input leaves of the selected typed experiment."""
        component = EXPERIMENTS.options_for(self.options_cls.from_raw(options).experiment)
        return tuple(
            reference.location
            for reference in typed_input_references(component, prefix=("experiment", "options"))
        )

    def input_file_dependencies(
        self, options: Mapping[str, Any], files: Mapping[str, Path]
    ) -> Mapping[str, Path]:
        """Interpret manifest payloads using only selected-component format declarations."""
        from chemrefine.input_files import option_pointer

        component = EXPERIMENTS.options_for(self.options_cls.from_raw(options).experiment)
        references = typed_input_references(component, prefix=("experiment", "options"))
        bundle_fields = {
            option_pointer(reference.location)
            for reference in references
            if reference.file_format == "quantum_bundle"
        }
        return {
            f"{pointer.lstrip('/')}/{key}": payload
            for pointer, path in files.items()
            if pointer in bundle_fields
            for key, payload in bundle_dependencies(path).items()
        }

    def check_step(self, step_cfg: StepConfig, *, charge: int, multiplicity: int) -> None:
        """Refuse invalid science before any upstream step submits work."""
        validate_experiment(self.options_cls.from_raw(step_cfg.engine_options()))
        if step_cfg.on_failure != "stop":
            raise ConfigError("qiskit-experiment requires on_failure: stop for its single artifact")

    def run_dir(self, ctx: StepContext) -> Path:
        """Locate the one scheduled job without relying on submission state."""
        return ctx.step_dir / "experiment"

    def artifact(self, ctx: StepContext) -> Path:
        """Locate the descriptor that commits this experiment's complete output."""
        return self.run_dir(ctx) / "artifact.json"

    def prepare(self, ctx: StepContext) -> StepInputs:
        """Render a thin worker with the scheduler's granted CPU budget."""
        self.check_step(ctx.step_cfg, charge=ctx.charge, multiplicity=ctx.multiplicity)
        options = self._options(ctx).model_copy(update={"cores": self.slurm_layout(ctx)[1]})
        directory = self.run_dir(ctx)
        directory.mkdir(parents=True, exist_ok=True)
        script = directory / "experiment.py"
        serialized = json.dumps(options.model_dump(mode="json"), allow_nan=False)
        script.write_text(
            "from pathlib import Path\nimport json\n"
            "from chemrefine.engines.qiskit.experiment import run_experiment\n"
            f"run_experiment(json.loads({serialized!r}), Path('artifact.json'))\n",
            encoding="utf-8",
        )
        return StepInputs(files=((script, self.artifact(ctx), "experiment"),))

    def submit(self, inputs: StepInputs, ctx: StepContext) -> JobBatch:
        """Use ChemRefine's common local/SLURM dispatch and resource accounting."""
        from chemrefine.engines import _execution

        return _execution.run_batch(self, inputs, ctx)

    def validate_outputs(self, inputs: StepInputs, ctx: StepContext) -> None:
        """Validate the complete output bundle without executing provider code."""
        read_bundle(self.artifact(ctx), max_bytes=self._options(ctx).max_output_bytes)

    def parse(self, inputs: StepInputs, ctx: StepContext) -> StepResults:
        """Validate the product and return the prior structures without new lineage."""
        self.validate_outputs(inputs, ctx)
        return StepResults(structures=ctx.prev_state.structures)

    def run_block(self, ctx: StepContext, inp_path: Path, out_path: Path) -> RunBlock:
        """Launch the worker with granted threads and let the scheduler own cleanup."""
        cores = self.slurm_layout(ctx)[1]
        interpreter = _provision.launcher_for(self, ctx.step_cfg.options)
        input_word = '"$INP_NAME"' if inp_path.name == "$INP_NAME" else shlex.quote(inp_path.name)
        return RunBlock(
            body=f"export OMP_NUM_THREADS={cores}\n"
            f"export MKL_NUM_THREADS={cores}\nexport OPENBLAS_NUM_THREADS={cores}\n"
            f"{shlex.quote(interpreter)} {input_word}"
        )

    def pal(self, ctx: StepContext) -> int:
        """Return the requested per-job CPU count."""
        return self._options(ctx).cores

    def slurm_layout(self, ctx: StepContext) -> tuple[int, int]:
        """Run one threaded process, clamped to the granted CPU budget."""
        return (1, min(self.pal(ctx), ctx.max_cores))

    def gpus(self, ctx: StepContext) -> int:
        """Read GPU demand from the authoritative options model."""
        return self._options(ctx).gpu_demand

    def single_node(self, ctx: StepContext) -> bool:
        """One task cannot be split across nodes, so no additional constraint is needed."""
        return False

    def memory_mb(self, ctx: StepContext) -> int | None:
        """Leave allocation to the header; artifact limits are not memory reservations."""
        return None

    def output_dirs(self, ctx: StepContext) -> tuple[str, ...]:
        """Preserve execution records and checkpoints through scratch cleanup."""
        return ("checkpoints", "provider_jobs")

    def extra_header_fields(self, ctx: StepContext) -> tuple[tuple[str, object], ...]:
        """Record the selected experiment in the shared run log."""
        return (("experiment", self._options(ctx).experiment.name),)


# Built-ins register in both the orchestrator and worker without importing SDKs.
from chemrefine.engines.qiskit import (  # noqa: E402, F401
    experiment_double_factorized,
    experiment_dynamics,
    experiment_measurement,
    experiment_shadows,
)

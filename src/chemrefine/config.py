"""Pydantic models + YAML loader for the v4 ChemRefine config.

The config is the **single source of truth**: every runtime knob comes
from here, never from `sys.argv` reads in the pipeline, never from
mutable module-level state. The loader raises :class:`ConfigError` on
any malformed file so the CLI can map it to a single non-zero exit code.

Schema shape (see ``Examples/`` for full examples):

.. code-block:: yaml

    template_dir: ./templates
    scratch_dir:  ./scratch
    output_dir:   ./outputs
    input:        ./input.xyz
    charge: 0
    multiplicity: 1
    max_cores: 64
    slurm_template: cpu.slurm.header
    orca_executable: orca

    steps:
      - step: 1
        name: screen        # optional human-readable label
        engine: mlff
        operation: opt_sp
        options: { model: mace_off23 }
        sample: { method: boltzmann, percent_cumulative: 99 }
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from chemrefine.constants import DEFAULT_TEMPERATURE_K
from chemrefine.errors import ConfigError

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


# ---------------------------------------------------------------------------
# Sample (filter) configuration — one model per method, discriminated union
# ---------------------------------------------------------------------------


class _SampleBase(BaseModel):
    """Common fields shared by every sample method."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    by_parent: bool = False
    """Apply the filter within each parent-ID group instead of globally."""

    temperature_k: float = Field(DEFAULT_TEMPERATURE_K, gt=0)
    """Temperature used by Boltzmann-style filters (K)."""


class BoltzmannSample(_SampleBase):
    """Keep structures whose cumulative Boltzmann weight reaches ``percent_cumulative``."""

    method: Literal["boltzmann"]
    percent_cumulative: float = Field(99.0, gt=0, le=100)


class EnergyWindowSample(_SampleBase):
    """Keep structures within ``window_kcal`` of the lowest-energy structure."""

    method: Literal["energy_window"]
    window_kcal: float = Field(..., gt=0)


class IntegerSample(_SampleBase):
    """Keep the ``count`` lowest-energy structures (0 = keep all)."""

    method: Literal["integer"]
    count: int = Field(..., ge=0)


class HighEnergySample(_SampleBase):
    """Keep the ``count`` highest-energy structures (e.g. for PES sampling)."""

    method: Literal["high_energy"]
    count: int = Field(..., ge=1)


SampleConfig: TypeAlias = Annotated[
    BoltzmannSample | EnergyWindowSample | IntegerSample | HighEnergySample,
    Field(discriminator="method"),
]


# ---------------------------------------------------------------------------
# Per-step configuration
# ---------------------------------------------------------------------------


class StepConfig(BaseModel):
    """One stage of the pipeline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int = Field(..., ge=1)
    """Canonical 1-based step number — drives directory naming and ordering."""

    name: str | None = None
    """Optional human label. Appears in logs/CSV and may target CLI subcommands."""

    engine: str
    """Engine key looked up in :data:`chemrefine.engines.base.ENGINES`."""

    operation: str
    """Engine-defined operation (``opt_sp``, ``goat``, ``pes``, ``solvator``, ``mlff_train``...)."""

    template: str | None = None
    """Engine input template (basename relative to ``template_dir`` if not absolute)."""

    charge: int | None = None
    """Per-step charge override; falls back to ``Config.charge`` when ``None``."""

    multiplicity: int | None = None
    """Per-step multiplicity override; falls back to ``Config.multiplicity`` when ``None``."""

    options: dict[str, Any] = Field(default_factory=dict)
    """Engine-specific knobs (model name, basis, device, etc.). Engine decides what to read."""

    sample: SampleConfig | None = None
    """How to filter survivors at the end of the step. ``None`` = keep all."""

    nms: bool = False
    """Opt-in normal-mode sampling (only honored if the engine ``supports_nms``)."""

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str | None) -> str | None:
        """Reject step names that aren't filesystem-safe."""
        if v is None:
            return v
        if not _NAME_RE.match(v):
            raise ValueError("step name must contain only letters, digits, underscores, and hyphens")
        return v

    def dir_name(self) -> str:
        """Return the directory name for this step (``stepN`` or ``stepN_name``)."""
        return f"step{self.step}_{self.name}" if self.name else f"step{self.step}"

    def matches(self, key: str | int) -> bool:
        """Return True if ``key`` (a CLI argument) targets this step."""
        if isinstance(key, int):
            return key == self.step
        if key.isdigit():
            return int(key) == self.step
        return key == self.name


# ---------------------------------------------------------------------------
# Top-level configuration
# ---------------------------------------------------------------------------


class Config(BaseModel):
    """Top-level YAML config."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    template_dir: Path = Path("./templates")
    scratch_dir: Path | None = None
    """Fast-storage base for per-calculation working directories.

    Leave unset (``None``) and ChemRefine will auto-derive a
    ``_work_<jobid>_<ts>_<rand>`` subdirectory under ``output_dir`` for
    each calculation — fine for laptop / local development. Set to a
    fast filesystem on the compute node (e.g. ``/scratch/$USER``) for
    HPC runs.
    """
    output_dir: Path = Path("./outputs")
    input: Path | None = None
    """Initial structure source: ``.xyz`` file, ``.csv`` of SMILES, or directory of xyz."""

    charge: int = 0
    multiplicity: int = Field(1, ge=1)
    max_cores: int = Field(32, ge=1)
    slurm_template: str = "cpu.slurm.header"
    orca_executable: str = "orca"
    steps: list[StepConfig]

    @model_validator(mode="after")
    def _reject_scratch_equal_output(self) -> Config:
        """``scratch_dir == output_dir`` is ambiguous; require ``None`` instead."""
        if self.scratch_dir is not None and self.scratch_dir == self.output_dir:
            raise ValueError(
                "scratch_dir must differ from output_dir; "
                "omit scratch_dir to get per-calculation work dirs under output_dir"
            )
        return self

    @field_validator("steps")
    @classmethod
    def _check_contiguous_steps(cls, v: list[StepConfig]) -> list[StepConfig]:
        """Ensure ``step:`` numbers form a contiguous 1..N sequence."""
        if not v:
            raise ValueError("'steps' must contain at least one step")
        numbers = [s.step for s in v]
        expected = list(range(1, len(numbers) + 1))
        if numbers != expected:
            raise ValueError(
                f"step numbers must be a contiguous 1..N sequence; got {numbers}"
            )
        return v

    @model_validator(mode="after")
    def _check_unique_names(self) -> Config:
        """Step names, if given, must be unique."""
        names = [s.name for s in self.steps if s.name is not None]
        if len(names) != len(set(names)):
            raise ValueError("step names must be unique when provided")
        return self

    def step_dir(self, step_cfg: StepConfig) -> Path:
        """Return the absolute output directory for ``step_cfg``."""
        return self.output_dir / step_cfg.dir_name()

    def find_step(self, key: str | int) -> StepConfig | None:
        """Return the step matching ``key`` (a number or a name), or ``None``."""
        for step_cfg in self.steps:
            if step_cfg.matches(key):
                return step_cfg
        return None


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> Config:
    """Load and validate a ChemRefine YAML config file.

    Raises :class:`ConfigError` for any malformed file, unknown top-level
    field, or per-step validation failure. The exception message is the
    raw Pydantic error text — already actionable for the user.
    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigError(f"could not read {p}: {e}") from e
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ConfigError(f"malformed YAML in {p}: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"config at {p} is not a YAML mapping")
    try:
        return Config(**raw)
    except ValidationError as e:
        raise ConfigError(f"invalid config {p}:\n{e}") from e

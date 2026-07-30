"""Pydantic models + YAML loader for the v4 ChemRefine config.

The config is the **single source of truth**: every runtime knob comes
from here, never from `sys.argv` reads in the pipeline, never from
mutable module-level state. The loader raises :class:`ConfigError` on
any malformed file so the CLI can map it to a single non-zero exit code.

Schema shape (see ``examples/`` for full examples):

.. code-block:: yaml

    template_dir: ./templates
    scratch_dir:  ./scratch
    output_dir:   ./outputs
    input:        ./input.xyz
    charge: 0
    multiplicity: 1
    max_cores: 64
    slurm_template: cpu.slurm.header
    executables: { orca: /opt/orca/orca }   # tool -> path, for external-binary engines

    steps:
      - step: 1
        name: screen        # optional human-readable label
        engine: mlip
        operation: opt_sp
        options: { model: mace_off23 }
        sample: { method: boltzmann, percent_cumulative: 99 }
"""

from __future__ import annotations

import logging
import os
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

from chemrefine import config_legacy
from chemrefine.errors import ConfigError
from chemrefine.quantities import DEFAULT_TEMPERATURE_K

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


# ---------------------------------------------------------------------------
# Sample (filter) configuration — one model per method, discriminated union
# ---------------------------------------------------------------------------


#: Short alias → canonical ``energy_type`` value (case-insensitive).
_ENERGY_TYPE_ALIASES = {
    "e": "electronic",
    "g": "gibbs",
    "h": "enthalpy",
    "e_zpe": "electronic_zero_point",
}


class _SampleBase(BaseModel):
    """Common fields shared by every sample method."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    by_parent: bool = False
    """Apply the filter within each parent-ID group instead of globally."""

    temperature_k: float = Field(DEFAULT_TEMPERATURE_K, gt=0)
    """Temperature used by Boltzmann-style filters (K)."""

    energy_type: Literal["electronic", "gibbs", "enthalpy", "electronic_zero_point"] = "electronic"
    """Which energy the filter sorts / selects on (default electronic).

    ``gibbs`` / ``enthalpy`` / ``electronic_zero_point`` require a frequency calc
    (thermochemistry) to have run; filtering raises if the chosen energy is
    missing. Short aliases are accepted: ``E`` / ``G`` / ``H`` / ``E_ZPE``."""

    @field_validator("energy_type", mode="before")
    @classmethod
    def _normalize_energy_type(cls, v: Any) -> Any:
        """Map short aliases (``G`` → ``gibbs``) and lowercase before validation."""
        if isinstance(v, str):
            key = v.strip().lower()
            return _ENERGY_TYPE_ALIASES.get(key, key)
        return v


class BoltzmannSample(_SampleBase):
    """Keep structures whose cumulative Boltzmann weight reaches ``percent_cumulative``."""

    method: Literal["boltzmann"]
    percent_cumulative: float = Field(99.0, gt=0, le=100)


class MinSample(_SampleBase):
    """Keep the lowest-energy structures.

    Set **exactly one** selector: ``count`` keeps the N lowest (``0`` = keep all);
    ``window_kcalmol`` keeps every structure within that window of the minimum.
    """

    method: Literal["min"]
    count: int | None = Field(None, ge=0)
    window_kcalmol: float | None = Field(None, gt=0)

    @model_validator(mode="after")
    def _exactly_one_selector(self) -> MinSample:
        """Require exactly one of ``count`` / ``window_kcalmol``."""
        if (self.count is None) == (self.window_kcalmol is None):
            raise ValueError("min: set exactly one of 'count' or 'window_kcalmol'")
        return self


class MaxSample(_SampleBase):
    """Keep the highest-energy structures (e.g. for PES sampling).

    Set **exactly one** selector: ``count`` keeps the N highest;
    ``window_kcalmol`` keeps every structure within that window of the maximum.
    """

    method: Literal["max"]
    count: int | None = Field(None, ge=1)
    window_kcalmol: float | None = Field(None, gt=0)

    @model_validator(mode="after")
    def _exactly_one_selector(self) -> MaxSample:
        """Require exactly one of ``count`` / ``window_kcalmol``."""
        if (self.count is None) == (self.window_kcalmol is None):
            raise ValueError("max: set exactly one of 'count' or 'window_kcalmol'")
        return self


SampleConfig: TypeAlias = Annotated[
    BoltzmannSample | MinSample | MaxSample,
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
    """Engine key looked up in :data:`chemrefine.engines.api.ENGINES`."""

    operation: str | None = None
    """Engine-defined operation (``opt_sp``, ``goat``, ``pes``, ``docker``,
    ``solvator``...). **Optional**: when omitted, an engine that can inspect its
    input (ORCA reads the template's keywords) auto-detects the run type; an
    explicit value here always wins. Provide it when inspection can't decide."""

    template: str | None = None
    """Engine input template (basename relative to ``template_dir`` if not absolute)."""

    slurm_template: str | None = None
    """Per-step SLURM header override (basename in ``template_dir``); falls back to
    ``Config.slurm_template`` when ``None``. A GPU-capable engine with
    ``options.device: cuda`` (or ``options.gpu``) auto-picks ``cuda.slurm.header``
    unless this is set explicitly."""

    charge: int | None = None
    """Per-step charge override; falls back to ``Config.charge`` when ``None``."""

    multiplicity: int | None = None
    """Per-step multiplicity override; falls back to ``Config.multiplicity`` when ``None``."""

    options: dict[str, Any] = Field(default_factory=dict)
    """Engine-specific knobs (model name, basis, device, etc.). Engine decides what to read."""

    sample: SampleConfig | None = None
    """How to filter survivors at the end of the step. ``None`` = keep all."""

    nms: bool = False
    """Opt-in normal-mode sampling (only honored for an NMS-capable engine: ORCA / ExtOpt)."""

    on_failure: Literal["stop", "skip", "best"] = "stop"
    """What to do when some structures fail this step (job error / no valid output,
    or NMS-unresolved): ``stop`` (default) halts the pipeline after caching the
    step's successes, ``skip`` drops the failures and keeps the successes, ``best``
    keeps every structure using the best geometry obtained for a failed one (else
    its submitted input). The default is ``stop`` so failures are never silently
    dropped — opt into ``skip``/``best`` per step when that is what you want."""

    @field_validator("operation")
    @classmethod
    def _reject_unsafe_operation(cls, v: str | None) -> str | None:
        """Hold ``operation`` to the same shell rule as the paths and ``executables``.

        This value is interpolated raw into the runlog header's ``cat <<EOF`` — the same
        unquoted heredoc :meth:`Config._reject_unsafe_executables` exists for, and unquotable
        for the same reason (``$(hostname)`` / ``${SLURM_JOB_ID:-$$}`` must still expand). So an
        ``operation: opt_sp$(id -un)`` is a command substitution the job would run, and a
        stray ``$`` corrupts the runlog without saying so.

        Its two neighbours in that header were already covered by construction — ``engine``
        must be a registry key, ``name`` is matched against :data:`_NAME_RE` — which is
        precisely why this one was missed: the rule had been attached to the fields that
        needed it first rather than to "every config value that reaches generated bash".

        Only the metacharacters are refused, not the vocabulary: ``OPT+SP`` (normalised to
        ``opt_sp`` before this runs) and any engine's own operation name stay legal.
        """
        if v is not None:
            reject_shell_unsafe(v, what="operation", fix="rename the operation")
        return v

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str | None) -> str | None:
        """Reject step names that aren't filesystem-safe or that shadow step numbers."""
        if v is None:
            return v
        if not _NAME_RE.match(v):
            raise ValueError(
                "step name must contain only letters, digits, underscores, and hyphens"
            )
        if v.isdigit():
            # `matches` resolves an all-digit CLI target as a step *number*,
            # so a digits-only name could never be addressed.
            raise ValueError("step name must not be all digits (ambiguous with a step number)")
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


Dispatch: TypeAlias = Literal["auto", "local", "slurm"]

#: Characters that cannot survive interpolation into the generated bash. ``"``, ``$`` and
#: a backtick end a quoted string or start a command substitution; a backslash is refused
#: because inside double quotes it escapes the very characters above, which is enough to
#: smuggle one past the check.
_SHELL_UNSAFE = ('"', "$", "`", "\\")


def reject_shell_unsafe(text: str, *, what: str, fix: str) -> None:
    """Raise unless ``text`` is safe to interpolate into the generated SLURM script.

    **One rule, one home, keyed to the property rather than to a list of fields.** Every
    config value that reaches generated bash goes through here — the table below is the whole
    set, and `tests/test_engines_invariants.py` fails if a new one appears without a decision
    about which column it belongs in. Remembered as "the path fields", the rule missed values
    that reached bash by other routes.

    ==========================  ===============================================================
    Value                       How it is protected
    ==========================  ===============================================================
    ``template_dir`` /          this rule — interpolated into ``export WORK_DIR=…``
    ``output_dir`` /
    ``scratch_dir``
    ``executables``             this rule — embedded raw in the runlog heredoc
    ``operation``               this rule — same heredoc
    ``tensor_folder``           this rule (via :class:`~chemrefine.engines.pyscf.options.\
PyscfOptions`) — reaches ``cp -r "…"``, and bash substitutes *inside* double quotes
    ``engine``                  safe by construction — must be a registry key
    ``step.name``               safe by construction — matched against :data:`_NAME_RE`
    ``step`` / ``cores``        safe by construction — integers
    ``structure_id``            safe by construction — minted by :mod:`chemrefine.ids`
    ``output_globs``            safe by construction — an engine ``ClassVar``
    ``job_name`` (trainer)      safe by construction — its own field ``pattern``
    ==========================  ===============================================================

    Public, not underscored, because the engine option models import it: a knob that reaches
    bash must be able to reach the rule, wherever it is declared.
    """
    bad = {c for c in _SHELL_UNSAFE if c in text}
    if "\n" in text or "\r" in text:
        bad.add("newline")
    if bad:
        raise ValueError(
            f"{what} {text!r} contains {sorted(bad)}, which cannot be safely embedded "
            f"in the generated SLURM script; {fix}"
        )


class Config(BaseModel):
    """Top-level YAML config."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_yaml(cls, data: Any) -> Any:
        """Rewrite legacy (v1.3.1 / ``mlff``-named) keys before validation."""
        return config_legacy.normalize(data) if isinstance(data, dict) else data

    template_dir: Path = Path("./templates")
    scratch_dir: Path | None = None
    """Fast-storage base for per-calculation working directories.

    Leave unset (``None``) and ChemRefine auto-derives a
    ``_work_<jobid>_<ts>_<rand>`` subdirectory under ``output_dir`` for each
    calculation — fine for a laptop / local development.

    On HPC, set it to the **compute node's fast local disk** (e.g.
    ``/scratch/$USER``, node-local NVMe/SSD) — distinct from the shared network
    filesystem (Lustre/NFS) that holds your home/project. The calculation's heavy
    I/O then runs on that fast local disk instead of hammering the shared FS, and
    the generated SLURM script's exit trap copies the result files back to
    ``output_dir`` when the job ends.
    """
    output_dir: Path = Path("./outputs")
    input: Path | None = None
    """Initial structure source: ``.xyz`` file, ``.csv`` of SMILES, or directory of xyz."""

    charge: int = 0
    multiplicity: int = Field(1, ge=1)
    max_cores: int = Field(4, ge=1)
    max_gpus: int | None = Field(None, ge=0)
    """GPU budget for concurrent local jobs. ``None`` (default) auto-resolves:
    unlimited under SLURM (the scheduler places GPUs via ``--gres``) and the
    detected device count locally (``nvidia-smi -L``), so a single-GPU desktop
    serialises CUDA jobs while CPU jobs still parallelise under ``max_cores``."""
    slurm_template: str = "cpu.slurm.header"
    slurm_array: bool = False
    """Submit each step as SLURM job array(s) (``sbatch --array``) instead of
    one job per structure. The scheduler then enforces the ``max_cores``
    budget natively via the array's ``%limit`` (``max_cores // PAL``), and a
    10⁴-structure step is one submission instead of 10⁴. Steps beyond the
    per-array task cap are split into chunks. Per-structure outputs, runlogs,
    and the failure ledger are identical to the per-job path. Ignored when
    running locally (no ``sbatch`` on PATH)."""
    job_timeout_seconds: float | None = Field(None, gt=0)
    """Wall-clock deadline for a step's jobs to finish, in seconds.

    ``None`` (default) waits indefinitely, which is the right thing under SLURM: the
    scheduler already enforces the partition's own time limit and will kill the job itself.
    Set it when nothing else will — a local ``dispatch: local`` run, or a cluster where a
    job can sit in ``PD`` forever — so a stuck batch fails with
    :class:`~chemrefine.errors.ThrottleTimeoutError` (exit code 8) instead of blocking the
    pipeline with no diagnostic.

    The deadline covers *waiting*, not compute: it is how long ChemRefine will wait for the
    scheduler to free up room or drain the batch, so set it well above the longest job you
    expect."""
    dispatch: Dispatch = "auto"
    """How jobs are executed. ``auto`` (default) submits via ``sbatch`` when it
    is on PATH and runs the generated scripts locally via ``bash`` otherwise.
    ``local`` forces the local runner even when an ``sbatch`` binary exists
    (e.g. a workstation with SLURM client tools but no reachable cluster);
    ``slurm`` requires ``sbatch`` and fails fast when it is missing instead of
    silently running locally."""
    executables: dict[str, str] = Field(default_factory=dict)
    """Global tool-name → binary-path map for external-binary engines (e.g.
    ``{"orca": "/opt/orca/orca"}``). Set once and shared by every step using
    that engine. Importable backends (mlip, pyscf, …) are installed as extras
    and need no entry here; conda/module activation belongs in the SLURM header."""
    steps: list[StepConfig]

    @field_validator("template_dir", "output_dir", "scratch_dir")
    @classmethod
    def _reject_shell_metacharacters(cls, v: Path | None) -> Path | None:
        """Reject directory paths carrying characters that break the generated bash.

        These three paths are interpolated into the generated SLURM script (see
        :func:`chemrefine.slurm._run_body_lines`), which exports them inside
        double quotes — so a metacharacter would end the quoted string or introduce a
        command substitution. Refused at config-load time rather than producing a
        corrupt — or actively dangerous — job script much later.
        """
        if v is None:
            return v
        reject_shell_unsafe(str(v), what="path", fix="rename the directory")
        return v

    @field_validator("executables")
    @classmethod
    def _reject_unsafe_executables(cls, v: dict[str, str]) -> dict[str, str]:
        """Hold ``executables`` to the same rule as the directory paths.

        These values reach generated bash by two routes with different protections.
        Where the binary is *run*, the engine shell-quotes it
        (:meth:`chemrefine.engines.orca.engine.OrcaEngine.orca_command`). But the runlog
        header embeds it raw inside ``cat <<EOF`` — a heredoc that must stay unquoted so
        ``$(hostname)``, ``$WORK_DIR`` and ``${SLURM_JOB_ID:-$$}`` expand — so
        ``executables: {orca: /opt/orca-$(id -un)/orca}`` is a command substitution the job
        would run, and an innocent ``$`` in a path corrupts the runlog without saying so.

        Quoting the header is not the fix: the heredoc's expansion is load-bearing for
        the fields around it. Refusing the character at the boundary is, and it is the
        rule this config already applies to every other value that reaches bash.

        Spaces stay legal — the run site quotes them correctly and ``/opt/my orca/orca``
        is a perfectly ordinary path.
        """
        for tool, value in v.items():
            reject_shell_unsafe(value, what=f"executable for {tool!r}", fix="move the binary")
        return v

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
            raise ValueError(f"step numbers must be a contiguous 1..N sequence; got {numbers}")
        return v

    @model_validator(mode="after")
    def _check_unique_names(self) -> Config:
        """Step names, if given, must be unique."""
        names = [s.name for s in self.steps if s.name is not None]
        if len(names) != len(set(names)):
            raise ValueError("step names must be unique when provided")
        return self

    @model_validator(mode="after")
    def _warn_missing_executable_paths(self) -> Config:
        """Warn (don't fail) when an ``executables`` entry is a path that's absent.

        Validation never hard-fails here: on HPC the binary is often provided by a
        ``module load`` *inside* the SLURM job, so the login node parsing the YAML
        legitimately can't see it. A bare command name (no path separator) is left
        alone — it is resolved on the executing host at submit time. Only an
        explicit path that doesn't exist on this host earns a warning, since that
        is almost always a typo.
        """
        for tool, value in self.executables.items():
            if os.sep in value and not Path(value).exists():
                logger.warning(
                    "executable %r for %r does not exist on this host (%s); "
                    "ignore if a module load provides it inside the job",
                    value,
                    tool,
                    value,
                )
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


def _resolve_relative_paths(cfg: Config, *, base: Path) -> Config:
    """Resolve a config's relative paths against ``base`` (the config file's dir).

    ChemRefine resolves ``template_dir`` / ``output_dir`` / ``scratch_dir`` /
    ``input`` relative to the **config file's** location, not the process working
    directory, so a config is portable: ``chemrefine run sub/proj/input.yaml``
    from anywhere finds ``sub/proj/templates`` and writes ``sub/proj/outputs``.
    Absolute paths pass through unchanged.
    """
    updates: dict[str, Path] = {}
    if not cfg.template_dir.is_absolute():
        updates["template_dir"] = base / cfg.template_dir
    if not cfg.output_dir.is_absolute():
        updates["output_dir"] = base / cfg.output_dir
    if cfg.scratch_dir is not None and not cfg.scratch_dir.is_absolute():
        updates["scratch_dir"] = base / cfg.scratch_dir
    if cfg.input is not None and not cfg.input.is_absolute():
        updates["input"] = base / cfg.input
    return cfg.model_copy(update=updates) if updates else cfg


def load_config(path: str | Path) -> Config:
    """Load and validate a ChemRefine YAML config file.

    Relative ``template_dir`` / ``output_dir`` / ``scratch_dir`` / ``input``
    paths are resolved against the config file's own directory (see
    :func:`_resolve_relative_paths`), so the file is portable regardless of the
    process working directory.

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
        cfg = Config(**raw)
    except ValidationError as e:
        raise ConfigError(f"invalid config {p}:\n{e}") from e
    return _resolve_relative_paths(cfg, base=p.parent.resolve())

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
    """Engine key looked up in :data:`chemrefine.engines.base.ENGINES`."""

    operation: str
    """Engine-defined operation (``opt_sp``, ``goat``, ``pes``, ``docker``, ``solvator``...)."""

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
    """Opt-in normal-mode sampling (only honored if the engine ``supports_nms``)."""

    on_failure: Literal["stop", "skip", "best"] = "stop"
    """What to do when some structures fail this step (job error / no valid output,
    or NMS-unresolved): ``stop`` (default) halts the pipeline after caching the
    step's successes, ``skip`` drops the failures and keeps the successes, ``best``
    keeps every structure using the best geometry obtained for a failed one (else
    its submitted input). The default is ``stop`` so failures are never silently
    dropped — opt into ``skip``/``best`` per step when that is what you want."""

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
# Legacy-YAML normalizer — the single place that knows the old (v1.3.1) vocabulary
# ---------------------------------------------------------------------------

#: Old engine name → canonical. Includes the ``mlff`` ↔ ``mlip`` rename aliases.
_ENGINE_RENAMES = {
    "mlff": "mlip",
    "mlff-extopt": "mlip-extopt",
    "mlff-train": "mlip-train",
    "dft": "orca",
}

#: Legacy step-level engine-config block → the canonical engine it implies.
#: v1.3.1 ran MLFF/PySCF through ORCA (the ``mlff:`` block configured the
#: gradient server), so they map to the ``*-extopt`` engines.
_LEGACY_BLOCKS = {
    "mlff": "mlip-extopt",
    "pyscf": "pyscf-extopt",
    "trainer": "mlip-train",
}

#: Option sub-keys that are auto-managed now and dropped from a moved block.
_OBSOLETE_OPTION_KEYS = frozenset({"bind"})

#: Old sample-method name → canonical v2 name. ``integer``/``high_energy`` became
#: ``min``/``max``; ``energy_window`` folded into ``min`` + a ``window_kcalmol`` knob.
_SAMPLE_METHOD_RENAMES = {
    "integer": "min",
    "high_energy": "max",
    "energy_window": "min",
}

#: Old sample parameter key → canonical v2 key (applies across methods).
_SAMPLE_KEY_RENAMES = {
    "num_structures": "count",
    "weight": "percent_cumulative",
    "energy": "window_kcalmol",
    "window_kcal": "window_kcalmol",
}


def _normalize_legacy(raw: dict) -> dict:
    """Rewrite legacy (v1.3.1 / ``mlff``-named) YAML keys to the current schema.

    The single place that knows the old vocabulary. Idempotent — new-style input
    passes through unchanged — and logs one warning per legacy feature rewritten.
    ``calculation_type`` is intentionally unsupported and raises
    :class:`~chemrefine.errors.ConfigError`.
    """
    out = dict(raw)
    if "orca_executable" in out:
        logger.warning("`orca_executable` is deprecated; use `executables: {orca: ...}`")
        execs = dict(out.get("executables") or {})
        execs.setdefault("orca", out.pop("orca_executable"))
        out["executables"] = execs
    if "initial_xyz" in out:
        if "input" not in out:
            logger.warning("`initial_xyz` is deprecated; use `input`")
            out["input"] = out["initial_xyz"]
        out.pop("initial_xyz")
    if isinstance(out.get("steps"), list):
        out["steps"] = [_normalize_step(s) for s in out["steps"]]
    return out


def _move_engine_block(s: dict) -> str | None:
    """Fold a legacy engine block (``mlff:``/``pyscf:``/``trainer:``) into ``options``.

    Mutates ``s`` (pops the block, merges its keys into ``options``) and returns the
    canonical engine that block implies, or ``None`` when no block is present.
    """
    options = dict(s.get("options") or {})
    block_engine: str | None = None
    for block, engine_name in _LEGACY_BLOCKS.items():
        if isinstance(s.get(block), dict):
            logger.warning("step-level `%s:` block is deprecated; use `options:`", block)
            for k, v in s.pop(block).items():
                if k not in _OBSOLETE_OPTION_KEYS:
                    options.setdefault(k, v)
            block_engine = engine_name
    if options or "options" in s:
        s["options"] = options
    return block_engine


def _normalize_nms_keys(s: dict) -> None:
    """Rewrite legacy ``normal_mode_sampling{,_parameters}`` into ``nms`` + ``options``.

    main's knobs are renamed: ``calc_type`` → ``target`` (``rm_imag`` → ``ts``, the
    default; ``random`` unchanged), ``displacement_vector`` → ``displacement_value``;
    other keys pass through. Mutates ``s`` in place.
    """
    if "normal_mode_sampling" not in s and "normal_mode_sampling_parameters" not in s:
        return
    logger.warning("`normal_mode_sampling*` is deprecated; use `nms` + `options`")
    nms_on = bool(s.pop("normal_mode_sampling", False))
    if nms_on:
        s["nms"] = True
    params = dict(s.pop("normal_mode_sampling_parameters", None) or {})
    opts = dict(s.get("options") or {})
    calc_type = str(params.pop("calc_type", "rm_imag")).lower()  # main default rm_imag → ts
    if nms_on:
        opts.setdefault("target", {"rm_imag": "ts"}.get(calc_type, calc_type))
    if "displacement_vector" in params:
        opts.setdefault("displacement_value", params.pop("displacement_vector"))
    for k, v in params.items():
        opts.setdefault(k, v)
    if opts:
        s["options"] = opts


def _normalize_step(step: Any) -> Any:
    """Rewrite one legacy step dict to the current schema (helper for :func:`_normalize_legacy`)."""
    if not isinstance(step, dict):
        return step
    s = dict(step)

    if "calculation_type" in s:
        raise ConfigError(
            "`calculation_type` is no longer supported; use `engine:` + `operation:` "
            "(see docs/migrating-from-main.md)"
        )

    # Engine name: a moved engine-config block decides it, else the rename map.
    block_engine = _move_engine_block(s)
    if block_engine is not None:
        s["engine"] = block_engine
    elif isinstance(s.get("engine"), str):
        eng = s["engine"].lower()
        s["engine"] = _ENGINE_RENAMES.get(eng, eng)

    # Operation: ``OPT+SP`` → ``opt_sp``, ``GOAT`` → ``goat`` (engines lower/replace too).
    if isinstance(s.get("operation"), str):
        s["operation"] = s["operation"].lower().replace("+", "_")
        # ``MLFF_TRAIN`` was a training *operation* that implied the trainer engine.
        if s["operation"] in ("mlff_train", "mlip_train"):
            s["engine"] = "mlip-train"
            s["operation"] = "mlip_train"

    _normalize_nms_keys(s)

    # sample_type{method, parameters} → sample{method, …}; legacy sample method /
    # key names (integer/high_energy/energy_window, num_structures/energy/…) are
    # rewritten to the v2 vocabulary (min/max/boltzmann, count/window_kcalmol)
    # whether they arrive via the old `sample_type` block or a direct `sample`.
    if "sample_type" in s:
        if "sample" not in s:
            logger.warning("`sample_type` is deprecated; use `sample`")
            s["sample"] = _flatten_sample_type(s["sample_type"])
        s.pop("sample_type")
    # Normalize whatever `sample` we now have (from sample_type, or a direct
    # block, possibly using legacy method/key names) to the v2 vocabulary.
    if isinstance(s.get("sample"), dict):
        s["sample"] = _normalize_sample_block(s["sample"])

    return s


def _flatten_sample_type(sample_type: Any) -> Any:
    """Flatten a legacy ``sample_type`` ``{method, parameters: {…}}`` into a flat dict."""
    if not isinstance(sample_type, dict):
        return sample_type
    flat: dict[str, Any] = {"method": sample_type.get("method")}
    flat.update(sample_type.get("parameters") or {})
    for k, v in sample_type.items():  # carry through any non-nested extras
        if k not in ("method", "parameters"):
            flat.setdefault(k, v)
    return flat


def _normalize_sample_block(sample: Any) -> Any:
    """Rewrite a flat ``sample`` dict's legacy method/key names to the v2 vocabulary.

    Idempotent: a current-vocabulary block (``min`` / ``max`` / ``boltzmann`` with
    ``count`` / ``window_kcalmol`` / ``percent_cumulative``) passes through unchanged.
    Logs one deprecation warning per legacy method or key actually rewritten.
    """
    if not isinstance(sample, dict):
        return sample
    out: dict[str, Any] = {}
    method = sample.get("method")
    if isinstance(method, str) and method in _SAMPLE_METHOD_RENAMES:
        new_method = _SAMPLE_METHOD_RENAMES[method]
        logger.warning("sample method `%s` is deprecated; use `%s`", method, new_method)
        out["method"] = new_method
    elif method is not None:
        out["method"] = method
    for raw_key, v in sample.items():
        key = str(raw_key)
        if key in ("method", "unit"):  # energy_window unit (kcal/mol) is implicit now
            continue
        new_key = _SAMPLE_KEY_RENAMES.get(key, key)
        if new_key != key:
            logger.warning("sample key `%s` is deprecated; use `%s`", key, new_key)
        out.setdefault(new_key, v)
    return out


# ---------------------------------------------------------------------------
# Top-level configuration
# ---------------------------------------------------------------------------


class Config(BaseModel):
    """Top-level YAML config."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_yaml(cls, data: Any) -> Any:
        """Rewrite legacy (v1.3.1 / ``mlff``-named) keys before validation."""
        return _normalize_legacy(data) if isinstance(data, dict) else data

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
    executables: dict[str, str] = Field(default_factory=dict)
    """Global tool-name → binary-path map for external-binary engines (e.g.
    ``{"orca": "/opt/orca/orca"}``). Set once and shared by every step using
    that engine. Importable backends (mlip, pyscf, …) are installed as extras
    and need no entry here; conda/module activation belongs in the SLURM header."""
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

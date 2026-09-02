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

import json
import logging
import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal, Self, TypeAlias

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


class _WindowedSample(_SampleBase):
    """Shared shape of the two extremum filters: keep N of them, or a window around one.

    ``min`` and ``max`` differ in *which* end they keep and in the floor on ``count`` —
    everything else, including "exactly one selector", is one rule, and one rule written per
    variant is one that can be tightened on a single copy. The message reads the
    discriminator so the shared version still names the method the user wrote.
    """

    method: str
    count: int | None = None
    window_kcalmol: float | None = Field(None, gt=0)

    @model_validator(mode="after")
    def _exactly_one_selector(self) -> Self:
        """Require exactly one of ``count`` / ``window_kcalmol``."""
        if (self.count is None) == (self.window_kcalmol is None):
            raise ValueError(f"{self.method}: set exactly one of 'count' or 'window_kcalmol'")
        return self


class MinSample(_WindowedSample):
    """Keep the lowest-energy structures.

    Set **exactly one** selector: ``count`` keeps the N lowest (``0`` = keep all);
    ``window_kcalmol`` keeps every structure within that window of the minimum.
    """

    method: Literal["min"]
    count: int | None = Field(None, ge=0)
    """``0`` is meaningful here — :func:`chemrefine.filtering._filter_min` reads it as
    "keep everything" — which is why the floor differs from :class:`MaxSample`'s."""


class MaxSample(_WindowedSample):
    """Keep the highest-energy structures (e.g. for PES sampling).

    Set **exactly one** selector: ``count`` keeps the N highest;
    ``window_kcalmol`` keeps every structure within that window of the maximum.
    """

    method: Literal["max"]
    count: int | None = Field(None, ge=1)
    """``1`` is the floor: ``max`` has no "keep everything" spelling, and ``0`` would slice
    the survivor set empty rather than mean anything."""


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
    """Opt-in normal-mode sampling — honored only for an NMS-capable engine
    (:class:`~chemrefine.engines.api.NmsCapableEngine`; the generated engine table's
    ``NMS`` column says which)."""

    on_failure: Literal["stop", "skip", "best"] = "stop"
    """What to do when some structures fail this step (job error / no valid output,
    or NMS-unresolved): ``stop`` (default) halts the pipeline after caching the
    step's successes, ``skip`` drops the failures and keeps the successes, ``best``
    keeps every structure using the best geometry obtained for a failed one (else
    its submitted input). The default is ``stop`` so failures are never silently
    dropped — opt into ``skip``/``best`` per step when that is what you want."""

    @property
    def halts_on_failure(self) -> bool:
        """Whether this step's failures stop the run.

        The two predicates below spell what ``on_failure`` *means* to a caller, so the modules
        that act on it ask a question instead of matching a string. Only ``stop`` answers yes
        to either, but the two are different questions — one is about the run ending, the
        other about work still owed — and a policy added later could answer them differently.
        Compared literally at each site, that difference has nowhere to live and every site
        has to be found and re-read to know which meaning it wanted.
        """
        return self.on_failure == "stop"

    @property
    def leaves_failures_pending(self) -> bool:
        """Whether this step's ledgered failures are still owed a re-attempt.

        ``skip`` and ``best`` resolve their failures when the policy is applied, so their
        ledger entries are a record rather than a queue; ``resume`` and ``rerun-errors``
        re-attempt only what is pending. See :meth:`halts_on_failure` for why this is a
        second predicate rather than the same one.
        """
        return self.on_failure == "stop"

    @field_validator("options")
    @classmethod
    def _reject_unrepresentable_options(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Refuse an option value the step's cache key cannot be computed from.

        Options are folded into the cache keys by ``json.dumps``
        (:meth:`chemrefine.cache.StepKey.of`); this probe mirrors that encoder —
        ``sort_keys`` included, since a nested dict with mixed-type keys fails in the
        sort rather than in the encoding. Without it the first thing to meet such a
        value is the fingerprint itself, and what reaches the user is a bare
        ``TypeError`` traceback from inside the cache — outside the exit-code contract
        every other config mistake honours — three layers from the YAML that caused it.

        The likeliest trigger is YAML's own typing, not exotic input: an unquoted
        ``2024-01-01`` parses to ``datetime.date``, which JSON has no spelling for.
        Refused here, at the boundary, the failure names the key and the fix.
        """
        for key, value in v.items():
            try:
                json.dumps(value, sort_keys=True)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"option {key!r} has a value the cache key cannot be computed from "
                    f"({e}); quote it in the YAML so it stays a plain string"
                ) from e
        return v

    @field_validator("operation")
    @classmethod
    def _reject_unsafe_operation(cls, v: str | None) -> str | None:
        """Hold ``operation`` to the same shell rule as the paths and ``executables``.

        This value is interpolated raw into the runlog header's ``cat <<EOF`` — the same
        unquoted heredoc :meth:`Config._reject_unsafe_executables` exists for, and unquotable
        for the same reason (``$(hostname)`` / ``${SLURM_JOB_ID:-$$}`` must still expand). So an
        ``operation: opt_sp$(id -un)`` is a command substitution the job would run, and a
        stray ``$`` corrupts the runlog without saying so.

        Its two neighbours in that header are covered by construction — ``engine`` must be a
        registry key, ``name`` is matched against :data:`_NAME_RE` — which is why the rule is
        keyed to "every config value that reaches generated bash" rather than to a list of
        fields: a value protected some other way is easy to mistake for one that needs no
        protection at all.

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

    def effective_charge(self, default: int) -> int:
        """This step's charge: its own override, else the config-wide ``default``.

        The one spelling of the fallback, shared by the context builder
        (:func:`chemrefine.step.build_context`), the preflight walk
        (:func:`chemrefine.engines.api.preflight_steps`) and ``chemrefine validate`` —
        three readers deciding "unset means the config's value" separately is how two
        of them come to disagree about which species a step runs.
        """
        return self.charge if self.charge is not None else default

    def effective_multiplicity(self, default: int) -> int:
        """This step's multiplicity: its own override, else the config-wide ``default``."""
        return self.multiplicity if self.multiplicity is not None else default

    def matches(self, key: str | int) -> bool:
        """Return True if ``key`` (a CLI argument) targets this step.

        ``isdecimal``, not ``isdigit``: the two disagree on the Unicode ``No`` category —
        ``"²".isdigit()`` is ``True`` and ``int("²")`` raises — and this predicate exists
        only to guard that ``int``. A target of ``"²"`` from the CLI, an agent's
        ``start_run`` or the GUI therefore left a raw ``ValueError``, past the module's
        contract that every failure is a ``ChemRefineError`` carrying an exit code.
        """
        if isinstance(key, int):
            return key == self.step
        if key.isdecimal():
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
    about which column it belongs in. Keyed to a list of fields instead, the rule misses every
    value that reaches bash by another route.

    ==========================  ===============================================================
    Value                       How it is protected
    ==========================  ===============================================================
    ``template_dir`` /          this rule — interpolated into ``export WORK_DIR=…``
    ``output_dir`` /
    ``scratch_dir``
    ``executables``             this rule — embedded raw in the runlog heredoc; the
                                engine-documented root paths (qchem's ``qc``/``qcaux``)
                                are additionally shell-quoted where their run block
                                exports them
    ``operation``               this rule — same heredoc
    ``tensor_folder``           this rule (via :class:`~chemrefine.engines.pyscf.options.\
PyscfExtOptOptions`, the subclass that declares the field) — reaches ``cp -r "…"``, and \
bash substitutes *inside* double quotes
    ``engine``                  safe by construction — must be a registry key
    ``step.name``               safe by construction — matched against :data:`_NAME_RE`
    ``step`` / ``cores``        safe by construction — integers
    ``structure_id``            safe by construction — minted by :mod:`chemrefine.ids`
    ``output_globs``            safe by construction — engine-declared constants (a \
``ClassVar``, or a property over trainer declarations); never config
    ``job_name`` (script)       safe by construction — the input path's stem (minted by
                                :mod:`chemrefine.ids` under the validated ``output_dir``),
                                or the ``_NAME_RE``-checked step label plus a literal
                                ``_array`` suffix
    ``template_path`` /         safe by construction — the header resolves under the
    ``script_path`` (script)    validated ``template_dir``; the script is minted beside
                                the validated input path
    ==========================  ===============================================================

    Public, not underscored, because the engine option models import it: a knob that reaches
    bash must be able to reach the rule, wherever it is declared.
    """
    bad = {c for c in _SHELL_UNSAFE if c in text}
    if "\n" in text or "\r" in text:
        bad.add("newline")
    # A tab breaks a different contract than the quoting characters above: the job-array
    # manifest is tab-delimited (`IFS=$'\t' read -r INP OUT SID` in the generated array
    # script), and every field embeds these paths — so a tab in one shifts every field
    # after it, and each task silently reads the wrong input or writes the wrong place.
    if "\t" in text:
        bad.add("tab")
    # A comma is the tab rule's sibling, one delimiter over: the array path hands each
    # chunk its manifest through `sbatch --export=ALL,CR_MANIFEST=<path>`, and sbatch
    # splits `--export` on commas — so a comma in the path truncates `CR_MANIFEST` and
    # every task of the array reads a manifest that does not exist. Refused here, like
    # the tab, rather than special-cased at the one call site that breaks today: these
    # values reach bash by more routes than any list of call sites stays honest about.
    if "," in text:
        bad.add("comma")
    if bad:
        raise ValueError(
            f"{what} {text!r} contains {sorted(bad)}, which cannot be safely embedded "
            f"in the generated SLURM script; {fix}"
        )


def shell_unsafe_after_resolution(config: Config) -> str | None:
    """Why this config's *resolved* directories cannot be interpolated into bash, or ``None``.

    :func:`reject_shell_unsafe` runs as a field validator, on the paths **as written**. That
    is not where they end up: :func:`resolve_relative_paths` anchors a relative path to the
    config file's own directory through ``model_copy``, which by design runs no validators —
    so every character the rule refuses can be smuggled in through a *parent directory name*
    instead of through the YAML.

    It is a real bypass of the boundary ``docs/internals/security.md`` states, and it was
    reproduced rather than reasoned about: a config containing nothing but
    ``output_dir: ./outputs``, placed in a directory named ``$(touch /tmp/MARKER)``, yields
    ``export OUTPUT_DIR="…/$(touch /tmp/MARKER)/outputs"`` in the generated script — and bash
    performs command substitution inside double quotes, so running the job created the file.
    The directory name is the whole payload; the user's YAML is innocent.

    Asked of the resolved config, therefore, and by both loaders — :func:`load_config` raises,
    :func:`chemrefine.validate.validate_config_text` reports — for the same reason the rule
    exists at all: a value that reaches generated bash is refused before a script is written,
    not after one runs.

    Unlike the whitespace rule (which belongs to ORCA's input format and lives in the ORCA
    input writer), this one is engine-independent: every engine's job script exports these
    three paths, so there is nothing to scope it to.
    """
    for name in ("template_dir", "output_dir", "scratch_dir"):
        value = getattr(config, name)
        if value is None:
            continue
        try:
            reject_shell_unsafe(
                str(value), what=f"resolved {name}", fix="rename the directory it sits in"
            )
        except ValueError as e:
            return str(e)
    return None


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
    budget natively via the array's ``%limit``, and a 10⁴-structure step is
    one submission instead of 10⁴. Steps beyond the per-array task cap are
    split into chunks, and each chunk gets a *share* of the budget
    (``max_cores // PAL // chunks``) rather than all of it — they are all
    queued at once, so anything else would multiply the budget by the number
    of chunks. The tail of a multi-chunk step therefore runs below
    ``max_cores`` once its siblings drain. Per-structure outputs, runlogs,
    and the failure ledger are identical to the per-job path. Ignored when
    running locally (no ``sbatch`` on PATH)."""
    job_timeout_seconds: float | None = Field(None, gt=0)
    """How long a step may go with **nothing finishing**, in seconds.

    ``None`` (default) waits indefinitely, which is the right thing under SLURM: the
    scheduler already enforces the partition's own time limit and will kill the job itself.
    Set it when nothing else will — a local ``dispatch: local`` run, or a cluster where a
    job can sit in ``PD`` forever — so a stuck batch fails with
    :class:`~chemrefine.errors.ThrottleTimeoutError` (exit code 8) instead of blocking the
    pipeline with no diagnostic.

    It is a **stall** deadline, not a budget for the step: the clock restarts every time a job
    completes, so a batch that keeps draining never trips it however long the whole batch
    takes. Bounding the total instead would mean a healthy multi-hour step failing on a
    timeout set to catch a stuck one — and it would have to be re-tuned every time a step grew.

    The same meaning on every path — the per-job throttler
    (:meth:`chemrefine.throttle.Throttler.wait_for_completion`) and the job-array wait
    (:func:`chemrefine.slurm.wait_for_jobs`) share one
    :class:`~chemrefine.throttle.StallDeadline` — so a step's dispatch mode never changes what
    the number means. On the array path progress is counted in *tasks*, not in arrays: a
    single ``sbatch --array`` covers up to 1000 structures and stays one job id until its last
    task exits, so anything coarser would make this a total-runtime bound there and nowhere
    else. It covers *waiting*, not compute: set it above the longest single job you expect,
    not above the step."""
    dispatch: Dispatch = "auto"
    """How jobs are executed. ``auto`` (default) submits via ``sbatch`` when it
    is on PATH and runs the generated scripts locally via ``bash`` otherwise.
    ``local`` forces the local runner even when an ``sbatch`` binary exists
    (e.g. a workstation with SLURM client tools but no reachable cluster);
    ``slurm`` requires ``sbatch`` and fails fast when it is missing instead of
    silently running locally."""
    executables: dict[str, str] = Field(default_factory=dict)
    """Global tool-name → path map for external-binary engines (e.g.
    ``{"orca": "/opt/orca/orca"}``). Set once and shared by every step using
    that engine. Importable backends (mlip, pyscf, …) are installed as extras
    and need no entry here; conda/module activation belongs in the SLURM header.

    An engine may document keys of its own beyond its binary — the qchem engine reads
    ``qc`` and ``qcaux`` as *install-root* paths and exports them as ``QC``/``QCAUX`` in
    its run block. They live in this map rather than in ``step.options`` because they are
    facts about the machine, not about a step: one install serves every qchem step, exactly
    as one ``orca`` binary does. Every value here is held to the same shell-safety rule and
    warned about when it names an absent path, whatever kind of path it is."""
    steps: list[StepConfig]

    @field_validator("template_dir", "output_dir", "scratch_dir")
    @classmethod
    def _reject_shell_metacharacters(cls, v: Path | None) -> Path | None:
        """Reject directory paths carrying characters that break the generated bash.

        These three paths are interpolated into the generated SLURM script (see
        :func:`chemrefine.slurm.script._run_body_lines`), which exports them inside
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


def resolve_relative_paths(cfg: Config, *, base: Path) -> Config:
    """Resolve a config's relative paths against ``base`` (the config file's dir).

    ChemRefine resolves ``template_dir`` / ``output_dir`` / ``scratch_dir`` /
    ``input`` relative to the **config file's** location, not the process working
    directory, so a config is portable: ``chemrefine run sub/proj/input.yaml``
    from anywhere finds ``sub/proj/templates`` and writes ``sub/proj/outputs``.
    Absolute paths pass through unchanged.

    A step's ``options`` get the same treatment for the keys in
    :data:`_STEP_OPTION_PATHS`, and they need it more than the rest: an option's value
    reaches a *job*, and a job runs neither where the config sits nor where the user stood —
    the generated script copies its input into a scratch ``$WORK_DIR`` and runs there. A
    relative ``model_path`` would resolve against that scratch directory and simply not be
    found. Naming the model a previous step produced (``./outputs/step2/train/train.model``)
    is the obvious thing to write, so it has to work from anywhere.

    Public, not underscored, because :func:`chemrefine.validate.validate_config_text`
    resolves the same way: it is the non-raising twin of :func:`load_config`, and a report
    that judged paths against a different directory than the run would is a report about a
    different config.
    """
    updates: dict[str, Path | list[StepConfig]] = {}
    if not cfg.template_dir.is_absolute():
        updates["template_dir"] = base / cfg.template_dir
    if not cfg.output_dir.is_absolute():
        updates["output_dir"] = base / cfg.output_dir
    if cfg.scratch_dir is not None and not cfg.scratch_dir.is_absolute():
        updates["scratch_dir"] = base / cfg.scratch_dir
    if cfg.input is not None and not cfg.input.is_absolute():
        updates["input"] = base / cfg.input
    steps = [_resolve_step_option_paths(step, base=base) for step in cfg.steps]
    if any(new is not old for new, old in zip(steps, cfg.steps, strict=True)):
        updates["steps"] = steps
    return cfg.model_copy(update=updates) if updates else cfg


_STEP_OPTION_PATHS = ("model_path",)
"""``step.options`` keys whose value is a filesystem path, resolved like the config's own.

A deliberately short list rather than "anything that looks like a path": an option is
free-form text and most values are not paths at all, so guessing would rewrite strings that
merely resemble one. Adding a knob here is the cost of introducing a path-valued option, and
it is one line."""


def _resolve_step_option_paths(step: StepConfig, *, base: Path) -> StepConfig:
    """Make a step's path-valued options absolute against ``base``; return the step."""
    options = step.options or {}
    rewritten = {
        key: str((base / value).resolve())
        for key in _STEP_OPTION_PATHS
        if isinstance(value := options.get(key), str) and value and not Path(value).is_absolute()
    }
    if not rewritten:
        return step
    return step.model_copy(update={"options": {**options, **rewritten}})


def load_config(path: str | Path) -> Config:
    """Load and validate a ChemRefine YAML config file.

    Relative ``template_dir`` / ``output_dir`` / ``scratch_dir`` / ``input``
    paths are resolved against the config file's own directory (see
    :func:`resolve_relative_paths`), so the file is portable regardless of the
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
        # `model_validate`, not `Config(**raw)`: a non-string key (YAML 1.1's unquoted
        # `on:` parses to a boolean) made the splat raise a bare TypeError past this
        # handler; pydantic itself turns it into the ValidationError caught here.
        cfg = Config.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(f"invalid config {p}:\n{e}") from e
    resolved = resolve_relative_paths(cfg, base=p.parent.resolve())
    # Re-asked after resolution: the field validator saw the paths as written, and a parent
    # directory name can carry every character it refuses. See `shell_unsafe_after_resolution`.
    if problem := shell_unsafe_after_resolution(resolved):
        raise ConfigError(f"invalid config {p}:\n{problem}")
    return resolved

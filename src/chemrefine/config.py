"""Pydantic models + YAML loader for the ChemRefine config.

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
    executables: { <program>: /path/to/program }   # for the engines that run one

    steps:
      - step: 1
        name: screen        # optional human-readable label
        engine: <engine>    # `chemrefine engines` lists them
        operation: opt_sp
        options: { ... }    # the engine's own knobs
        sample: { method: boltzmann, percent_cumulative: 99 }
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal, Self, TypeAlias, TypeGuard, assert_never, get_args

import yaml
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from chemrefine import config_legacy, io
from chemrefine.errors import ConfigError
from chemrefine.quantities import DEFAULT_TEMPERATURE_K

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


# ---------------------------------------------------------------------------
# Sample (filter) configuration — one model per method, discriminated union
# ---------------------------------------------------------------------------


EnergyType: TypeAlias = Literal["electronic", "gibbs", "enthalpy", "electronic_zero_point"]
"""Which energy a step ranks, filters and reports on — the one declaration of the vocabulary.

A type's short label (:func:`energy_label`) and the :class:`~chemrefine.state.Structure`
attribute holding its value (:func:`chemrefine.filtering.ranking_energy`) are exhaustive
matches over this, so a type added here without either fails type-checking, not a run."""


def energy_label(energy_type: EnergyType) -> str:
    """The short label printed beside ``energy_type``'s value — also the alias a config may use."""
    match energy_type:
        case "electronic":
            return "E"
        case "gibbs":
            return "G"
        case "enthalpy":
            return "H"
        case "electronic_zero_point":
            return "E_ZPE"
        case _:
            assert_never(energy_type)


#: Short alias → canonical ``energy_type`` value (case-insensitive): each type's label.
_ENERGY_TYPE_ALIASES = {energy_label(t).lower(): t for t in get_args(EnergyType)}


class _SampleBase(BaseModel):
    """Common fields shared by every sample method."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    by_parent: bool = False
    """Apply the filter within each parent-ID group instead of globally."""

    temperature_k: float = Field(DEFAULT_TEMPERATURE_K, gt=0, allow_inf_nan=False)
    """Temperature used by Boltzmann-style filters (K). Finite, like every float knob."""

    energy_type: EnergyType = "electronic"
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
    window_kcalmol: float | None = Field(None, gt=0, allow_inf_nan=False)

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
# Normal-mode-sampling knobs
# ---------------------------------------------------------------------------


class NmsKnobs(BaseModel):
    """The normal-mode-sampling knobs a step's ``options`` may carry — the schema half.

    Declared here rather than in :mod:`chemrefine.nms` because two readers share one
    ``options`` mapping when ``nms: true``: the engine's own options model and the NMS
    coordinator. The config is the one place that knows both, so it is where a step can be
    asked for the engine's share (:meth:`StepConfig.engine_options`) — and it cannot import
    the coordinator, which imports the engine subsystem. :class:`chemrefine.nms.NmsOptions`
    extends this with the behaviour (how a reading splits into the cache key), so every
    reader of the knobs still reads one model.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    target: Literal["minimum", "ts", "random"] = "minimum"
    """``minimum`` removes all imaginary modes; ``ts`` keeps the reaction coordinate
    and removes the rest; ``random`` displaces along random modes (exploration)."""

    displacement_value: float = Field(1.0, gt=0, allow_inf_nan=False)
    """Magnitude (Å) of the ± displacement along each selected mode.

    Positive and finite, refused before the run starts like the other float knobs
    (:func:`chemrefine.nms.read_options`): ``0`` submits a whole round-2 batch of children
    identical to their parents and then reports every parent unresolved, and ``.nan`` /
    ``.inf`` put a non-number into every child geometry the batch runs — the config
    boundary is the one place that can say so by name."""

    num_random_displacements: int = Field(1, ge=1)
    """``random`` only: how many modes to draw."""

    ts_mode_index: int | None = Field(None, ge=0)
    """``ts`` only: explicit reaction-coordinate mode index. ``None`` ⇒ the
    largest-magnitude imaginary mode.

    Non-negative, because no mode has a negative index. Which modes are imaginary is known
    only once round 1 has run, so an index is held to its structure there
    (:func:`chemrefine.nms._reaction_coordinate`); one no structure could satisfy is refused
    here, before that round is paid for."""

    seed: int = Field(42, ge=0)
    """Deterministic seed for ``random`` mode selection.

    Non-negative because that is what the generator behind it takes: a negative would
    reach ``numpy.random.default_rng`` as a bare ``ValueError`` at the first fan-out — after
    round 1 has run and been paid for, and outside the exit-code contract."""

    @classmethod
    def from_raw(cls, raw: dict[str, Any] | None) -> Self:
        """Validate the NMS subset of a ``step.options`` dict (ignoring other keys)."""
        raw = raw or {}
        known = {k: raw[k] for k in cls.model_fields if k in raw}
        return cls(**known)


# ---------------------------------------------------------------------------
# Per-step configuration
# ---------------------------------------------------------------------------

FailurePolicy: TypeAlias = Literal["stop", "skip", "best"]
"""What a step does with the structures that fail it — :attr:`StepConfig.on_failure`'s
vocabulary. Its meaning is read as three questions, never as a string match:
:meth:`StepConfig.halts_on_failure`, :meth:`StepConfig.leaves_failures_pending` and
:func:`policy_stores_backfills`."""


def policy_stores_backfills(policy: FailurePolicy) -> bool:
    """Whether results finalized under ``policy`` carry backfilled failures.

    The one question the cache asks of a policy. ``stop`` and ``skip`` persist the
    successes alone, so a document written under either serves the other; ``best`` writes
    the backfilled failures into the same document, so a change across that line would hand
    the user the previous policy's survivor set with the fingerprint still matching. The
    wildcard arm holds only :func:`typing.assert_never`, so a policy added to
    :data:`FailurePolicy` fails type-checking here until it answers, where a literal
    ``== "best"`` would have classed it with ``stop``/``skip`` and nothing would have said so.

    A function rather than only a property because the cache asks it of the *stored*
    policy too (:func:`chemrefine.step._policy_conflict`); :meth:`StepConfig.stores_backfills`
    asks it for a config.
    """
    match policy:
        case "best":
            return True
        case "stop" | "skip":
            return False
        case _:
            assert_never(policy)


def is_failure_policy(value: str) -> TypeGuard[FailurePolicy]:
    """Whether ``value`` names a policy this version knows.

    A cache document records the policy its results were finalized under as text, and a
    document written by another version may name one this version cannot interpret.
    """
    return value in get_args(FailurePolicy)


def fold_engine_name(name: str) -> str:
    """The one spelling of an engine name: its lower case, so ``engine:`` is read in any case.

    Public because the registry holds to it as well as the config: :class:`StepConfig`
    folds its ``engine`` through it as a step is read, and
    :func:`chemrefine.engines.api.register` refuses a name it would change, which no config
    could look up. Case only: a separator belongs to the name (``-`` and ``_`` spell
    different names).
    """
    return name.lower()


def _read_home(value: object) -> object:
    """A typed path's leading ``~`` read as its home (:func:`chemrefine.io.expand_home`).

    Run before a value becomes a ``Path``, whose normalisation turns ``./~/x`` into ``~/x``,
    so the rule is decided on the text as typed. A relative path stays relative — the loader
    anchors it to the config file's directory — and a value that is not text passes through.
    A ``~name`` the host knows no home for is a ``ValueError``, which pydantic reports at
    the key that holds it.
    """
    if not isinstance(value, str):
        return value
    try:
        return io.expand_home(value)
    except ConfigError as e:
        raise ValueError(str(e)) from e


USER_PATH = BeforeValidator(_read_home)
"""``Annotated`` metadata for a key that names a path a person types: its ``~`` is a home.

Declared on the field, so every key that holds a path says so where it is declared — the
config's directories, seed, header and ``executables``, a step's template and header, and
an engine option naming an interpreter or a directory — and :data:`STEP_OPTION_PATHS`
reads the same way for the step options the loader resolves."""


class StepConfig(BaseModel):
    """One stage of the pipeline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int = Field(..., ge=1)
    """Canonical 1-based step number — drives directory naming and ordering."""

    name: str | None = None
    """Optional human label. Appears in logs/CSV and may target CLI subcommands."""

    engine: str
    """Engine key looked up in :data:`chemrefine.engines.api.ENGINES`, read in any case
    (:func:`fold_engine_name`)."""

    operation: str | None = None
    """The step's operation, as its engine defines it: one of the vocabulary the engine
    declares (:class:`~chemrefine.engines.api.OperationsDeclaring`; ``chemrefine engines
    --json`` lists each), or a free label for an engine that declares none. Read in any case
    with ``+`` for ``_`` (``OPT+SP`` is ``opt_sp``). **Optional**: when omitted, an engine
    that can inspect its input (:class:`~chemrefine.engines.api.OperationInferring`) infers
    the run type from it; an explicit value always wins, and an engine that declares its
    vocabulary refuses any other before the run starts. Provide it when inspection can't
    decide."""

    template: Annotated[str | None, USER_PATH] = None
    """Engine input template (basename relative to ``template_dir`` if not absolute)."""

    slurm_template: Annotated[str | None, USER_PATH] = None
    """Per-step SLURM header override (basename in ``template_dir``); falls back to
    ``Config.slurm_template`` when ``None``. A GPU-capable engine with
    ``options.device: cuda`` (or ``options.gpu``) auto-picks ``cuda.slurm.header``
    unless this is set explicitly."""

    charge: int | None = None
    """Per-step charge override; falls back to ``Config.charge`` when ``None``."""

    multiplicity: int | None = Field(None, ge=1)
    """Per-step multiplicity override; falls back to ``Config.multiplicity`` when ``None``.

    Held to the same floor as the workflow value: a ``0`` here would reach every engine's
    input as the workflow value does, with ``chemrefine validate`` saying OK, and every job
    of the step would die at the program's own input parse — after the steps before it
    have run."""

    options: dict[str, Any] = Field(default_factory=dict)
    """Engine-specific knobs (model name, basis, device, etc.). Engine decides what to read."""

    sample: SampleConfig | None = None
    """How to filter survivors at the end of the step. ``None`` = keep all."""

    nms: bool = False
    """Opt-in normal-mode sampling — honored only for an NMS-capable engine
    (:class:`~chemrefine.engines.api.NmsCapableEngine`; the generated engine table's
    ``NMS`` column says which)."""

    on_failure: FailurePolicy = "stop"
    """What to do when some structures fail this step (job error / no valid output,
    or NMS-unresolved): ``stop`` (default) halts the pipeline after caching the
    step's successes, ``skip`` drops the failures and keeps the successes, ``best``
    keeps every structure using the best geometry obtained for a failed one (else
    its submitted input). The default is ``stop`` so failures are never silently
    dropped — opt into ``skip``/``best`` per step when that is what you want."""

    @property
    def halts_on_failure(self) -> bool:
        """Whether this step's failures stop the run.

        The three predicates here spell what ``on_failure`` *means* to a caller, so the
        modules that act on it ask a question instead of matching a string. Only ``stop``
        answers yes to the first two, but they are different questions — one is about the run
        ending, the other about work still owed — and a policy added later could answer them
        differently. Compared literally at each site, that difference has nowhere to live and
        every site has to be found and re-read to know which meaning it wanted.
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

    @property
    def stores_backfills(self) -> bool:
        """Whether this step's cached results carry backfilled failures.

        :func:`policy_stores_backfills` for this step's policy — the question the cache asks
        before serving a document to a config whose policy changed.
        """
        return policy_stores_backfills(self.on_failure)

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

    @field_validator("options")
    @classmethod
    def _read_home_in_option_paths(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Read a leading ``~`` in each path-valued option (:data:`STEP_OPTION_PATHS`) as home.

        The same reading :data:`USER_PATH` gives a declared field, applied to the keys of
        this free mapping that name a path, so the loader anchors what the user meant; a
        refusal names the option, as the row's location stops at the mapping.
        """
        read = dict(v)
        for key in STEP_OPTION_PATHS:
            if key in read:
                try:
                    read[key] = _read_home(read[key])
                except ValueError as e:
                    raise ValueError(f"option {key!r}: {e}") from e
        return read

    @field_validator("engine", mode="before")
    @classmethod
    def _fold_engine(cls, v: object) -> object:
        """One spelling per engine: any case, folded by :func:`fold_engine_name`.

        Folded once, as the step is read, from a file or from code, so every reader of the
        field (the registry lookup, the row key, the manifest, the step record, the runlog
        header) sees the name the engine registered, and none folds again. A non-string is
        left to the field's own type check.
        """
        return fold_engine_name(v) if isinstance(v, str) else v

    @field_validator("operation", mode="before")
    @classmethod
    def _fold_operation(cls, v: object) -> object:
        """One spelling per operation: any case, and ``+`` for ``_`` (``OPT+SP`` is ``opt_sp``).

        Folded once, as the step is read, so every reader of the field (the row key, the
        manifest, the step record, the output parser, the vocabulary check) sees the
        spelling the engines declare, and none folds again. A label, on an engine that
        declares no vocabulary, is folded the same way.
        """
        return v.lower().replace("+", "_") if isinstance(v, str) else v

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

        Only the metacharacters are refused, not the vocabulary: ``OPT+SP`` (folded to
        ``opt_sp`` by :meth:`_fold_operation` before this runs) and any engine's own
        operation name stay legal.
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
        if not _NAME_RE.fullmatch(v):
            # The whole string: under `match` the pattern's `$` also matches before a final
            # newline, and a YAML block scalar's `screen\n` would name the step's directory.
            raise ValueError(
                "step name must contain only letters, digits, underscores, and hyphens"
            )
        if v.startswith("-"):
            # Every step-targeted invocation puts the name on an argv — the CLI's and
            # `start_run`'s — where a leading hyphen parses as an option: exit 2 into
            # an unwatched log, after a pid has been returned.
            raise ValueError("step name must not start with a hyphen (argv reads it as an option)")
        if v.isdecimal():
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

    def engine_options(self) -> dict[str, Any]:
        """``options`` minus the NMS knobs when ``nms`` is on — the engine's own share.

        Two declared readers share the one mapping: the engine's options model and, with
        ``nms: true``, :class:`NmsKnobs`. Handed the whole dict, a strict engine model
        (``extra="forbid"``) would refuse every NMS knob as a stranger, so only the strict
        engines could not sample with a knob set.
        The partition is made here because the config is the one place that knows both
        readers; :func:`chemrefine.validate.undeclared_options` draws the same line for its
        warning. A key that is neither reader's stays in the engine's share, so a typo of an
        NMS knob is still refused by a strict engine and still warned about by ``validate``.
        """
        if not self.nms:
            return dict(self.options)
        return {k: v for k, v in self.options.items() if k not in NmsKnobs.model_fields}

    def matches(self, key: str | int) -> bool:
        """Return True if ``key`` (a CLI argument) targets this step.

        ``isdecimal``, not ``isdigit``: the two disagree on the Unicode ``No`` category —
        ``"²".isdigit()`` is ``True`` and ``int("²")`` raises — and this predicate exists
        only to guard that ``int``. A target of ``"²"`` from the CLI, an agent's
        ``start_run`` or the GUI would otherwise leave a raw ``ValueError``, past the module's
        contract that every failure is a ``ChemRefineError`` carrying an exit code.
        """
        if isinstance(key, int):
            return key == self.step
        if key.isdecimal():
            try:
                return int(key) == self.step
            except ValueError:  # past sys.get_int_max_str_digits(): no step has that number
                return False
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
    ``template_dir`` /          this rule — interpolated into ``export WORK_DIR=…``; asked
    ``output_dir`` /            at load as read and as anchored
    ``scratch_dir``             (:func:`resolve_relative_paths`), and again at render on
                                the *resolved* path bash receives
                                (:mod:`chemrefine.slurm.script`)
    ``executables``             this rule — embedded raw in the runlog heredoc; an engine
                                that exports a root path in its run block also
                                shell-quotes it there
    ``operation``               this rule — same heredoc; the declared value at load, and
                                the one the step resolves (declared or inferred) where
                                :func:`chemrefine.job_log.bash_header` writes it
    ``tensor_folder``           this rule (from the field validator of the options model
                                that declares it, which is why the rule is public) —
                                reaches ``cp -r "…"``, and bash substitutes *inside*
                                double quotes
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
    ``slurm_header``            safe by construction — the user's own cluster header,
                                read from under the validated ``template_dir`` and copied
                                line for line, never interpolated
    ``script_path`` (script)    safe by construction — minted beside the validated input
                                path
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
    # No comma rule: the manifest path travels as the array script's argument
    # (:func:`chemrefine.slurm.dispatch.submit_array`), a channel that reserves no
    # character, and no other site these values reach reserves one.
    if bad:
        raise ValueError(
            f"{what} {text!r} contains {sorted(bad)}, which cannot be safely embedded "
            f"in the generated SLURM script; {fix}"
        )


class Config(BaseModel):
    """Top-level YAML config."""

    # `revalidate_instances`: `model_validate` given a Config re-checks it rather than handing
    # it back — how a copy made with `model_copy`, which runs no validators, is held to the
    # model's rules: the anchored paths (`resolve_relative_paths`) and a budget override
    # (`with_budgets`).
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_yaml(cls, data: Any) -> Any:
        """Rewrite v1.3.1's keys before validation."""
        return config_legacy.normalize(data) if isinstance(data, dict) else data

    template_dir: Annotated[Path, USER_PATH] = Path("./templates")
    scratch_dir: Annotated[Path | None, USER_PATH] = None
    """Fast-storage base for per-calculation working directories.

    Leave unset (``None``) and ChemRefine auto-derives a
    ``_work_<jobid>_<ts>_<rand>`` subdirectory under ``output_dir`` for each
    calculation — fine for a laptop / local development.

    On HPC, set it to the **compute node's fast local disk** (e.g.
    ``/scratch/<user>``, node-local NVMe/SSD) — distinct from the shared network
    filesystem (Lustre/NFS) that holds your home/project. The path is taken as
    written: ``$USER`` is not expanded, and a ``$`` is refused like every shell
    metacharacter. The calculation's heavy
    I/O then runs on that fast local disk instead of hammering the shared FS, and
    the generated SLURM script's exit trap copies the result files back to
    ``output_dir`` when the job ends.
    """
    output_dir: Annotated[Path, USER_PATH] = Path("./outputs")
    input: Annotated[Path | None, USER_PATH] = None
    """Initial structure source: ``.xyz`` file, ``.csv`` of SMILES, or directory of xyz."""

    charge: int = 0
    multiplicity: int = Field(1, ge=1)
    max_cores: int = Field(4, ge=1)
    max_gpus: int | None = Field(None, ge=0)
    """GPU budget for concurrent local jobs. ``None`` (default) auto-resolves:
    unlimited under SLURM (the scheduler places GPUs via ``--gres``) and the
    detected device count locally (``nvidia-smi -L``), so a single-GPU desktop
    serialises CUDA jobs while CPU jobs still parallelise under ``max_cores``."""
    slurm_template: Annotated[str, USER_PATH] = "cpu.slurm.header"
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
    job_timeout_seconds: float | None = Field(None, gt=0, allow_inf_nan=False)
    """How long a step may go with **nothing finishing**, in seconds.

    ``None`` (default) waits indefinitely, which is the right thing under SLURM: the
    scheduler already enforces the partition's own time limit and will kill the job itself.
    ``null`` is the one spelling of that — ``.inf`` is refused, like every non-finite
    value of every float knob.
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
    executables: dict[str, Annotated[str, USER_PATH]] = Field(default_factory=dict)
    """Global program-key → path map for the engines that run an external program. Each
    engine declares the keys it reads (:class:`~chemrefine.engines.api.ExecutablesDeclaring`;
    ``chemrefine engines --json`` lists them), and an entry is set once and shared by every
    step that runs that program. Importable backends are installed as extras and need no
    entry here; conda/module activation belongs in the SLURM header.

    A key may name an *install root* rather than a binary — a directory the engine exports
    in its run block. Both kinds live in this map rather than in ``step.options`` because
    they are facts about the machine, not about a step: one install serves every step that
    uses it. Every value here is held to the same shell-safety rule and warned about when it
    names an absent path, whatever kind of path it is."""
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
        Where the binary is *run*, the engine shell-quotes it. But the runlog header embeds
        it raw inside ``cat <<EOF`` — a heredoc that must stay unquoted so ``$(hostname)``,
        ``$WORK_DIR`` and ``${SLURM_JOB_ID:-$$}`` expand — so
        ``executables: {tool: /opt/tool-$(id -un)/tool}`` is a command substitution the job
        would run, and an innocent ``$`` in a path corrupts the runlog without saying so.

        Quoting the header is not the fix: the heredoc's expansion is load-bearing for
        the fields around it. Refusing the character at the boundary is, and it is the
        rule this config already applies to every other value that reaches bash.

        Spaces stay legal — the run site quotes them correctly and ``/opt/my tool/tool``
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

    def missing_executables(self) -> list[tuple[str, str]]:
        """``(tool, warning)`` for every explicit ``executables`` path absent on this host.

        The one home for the finding and its wording, said by :func:`load_config`'s log line
        for a run **and** by :mod:`chemrefine.validate`'s warning row — a log line alone
        would reach nobody watching ``chemrefine validate``, the GUI, or an MCP client, which
        all read the report and would be told ``ok`` with no warning. A warning, never a
        refusal: on HPC a ``module load`` inside the job often provides the binary, so the
        login node reading the YAML legitimately can't see it, while an explicit path absent
        here is almost always a typo. A bare command name (no path separator) is not on the
        list — it resolves on the executing host at submit time.

        A path the filesystem will not answer for — one under a directory this account
        cannot search — is on the list with the system's own reason: it may well be there,
        and the job's account is the one that has to reach it.
        """
        missing: list[tuple[str, str]] = []
        for tool, value in self.executables.items():
            if os.sep not in value:
                continue
            path = Path(value)
            why = io.unreachable(path)
            if why is None and not io.absent(path):
                continue
            problem = (
                "does not exist on this host"
                if why is None
                else f"cannot be reached on this host ({why})"
            )
            missing.append(
                (
                    tool,
                    f"executable {value!r} for {tool!r} {problem}; "
                    f"ignore if a module load provides it inside the job",
                )
            )
        return missing

    def default_input(self) -> Path:
        """The seed a run reads when ``input`` names none: ``step1.xyz`` beside the templates.

        Spelled here for the run that reads it (:func:`chemrefine.pipeline.bootstrap`) and
        for :mod:`chemrefine.validate`, which reports a seed that is not there under either
        of its names.
        """
        return self.template_dir / "step1.xyz"

    def step_dir(self, step_cfg: StepConfig) -> Path:
        """Return the absolute output directory for ``step_cfg``."""
        return self.output_dir / step_cfg.dir_name()

    def with_budgets(self, *, max_cores: int | None = None, max_gpus: int | None = None) -> Config:
        """A copy with the run's budgets replaced; ``None`` keeps the config's own.

        The budgets are the one thing a run overrides — ``--maxcores`` / ``--maxgpus`` and
        the agent's ``start_run``; every other field comes from the YAML, through the
        loader that resolves its paths. ``model_copy`` runs no validators, so the copy is
        handed to the model to validate: an override meets every rule a YAML value meets —
        its field's floor (``max_cores: ge=1``, ``max_gpus: ge=0``) and any rule the model
        declares — with no second statement of either, and a bad budget is the same
        :class:`ConfigError` a bad YAML value is.
        """
        budgets = {
            name: value
            for name, value in (("max_cores", max_cores), ("max_gpus", max_gpus))
            if value is not None
        }
        if not budgets:
            return self
        try:
            return type(self).model_validate(self.model_copy(update=budgets))
        except ValidationError as e:
            # A field's error names its own budget; a model-level rule's names every one given.
            given = ", ".join(f"{name} {value!r}" for name, value in budgets.items())
            detail = "; ".join(
                f"{err['loc'][0]} {err['input']!r}: {err['msg']}"
                if err["loc"]
                else f"{given}: {err['msg']}"
                for err in e.errors()
            )
            raise ConfigError(detail) from e

    def find_step(self, key: object) -> StepConfig | None:
        """Return the step matching ``key`` (a number or a name), or ``None``.

        ``None`` means no step matches. A ``key`` that is neither a number nor a name — a
        JSON float, list or boolean off the GUI's or an agent's wire — is not a selector
        at all and is refused as a :class:`~chemrefine.errors.ConfigError`: left to
        :meth:`StepConfig.matches` it would raise a bare ``AttributeError`` (``isdecimal``
        on a float), past the contract that every failure carries an exit code, and a JSON
        ``true`` — an ``int`` to :func:`isinstance` — would select step 1. Typed
        ``object`` because this is the one funnel every wire selector passes through.
        """
        if isinstance(key, bool) or not isinstance(key, (int, str)):
            raise ConfigError(f"step selector {key!r} is neither a step number nor a name")
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
    Absolute paths pass through unchanged — a leading ``~`` is one by now, read as the
    user's home when the model was built (:data:`USER_PATH`).

    A step's ``options`` get the same treatment for the keys in
    :data:`STEP_OPTION_PATHS`, and they need it more than the rest: an option's value
    reaches a *job*, and a job runs neither where the config sits nor where the user stood —
    the generated script copies its input into a scratch ``$WORK_DIR`` and runs there. A
    relative ``model_path`` would resolve against that scratch directory and simply not be
    found. Naming the model a previous step produced (``./outputs/step2/train/train.model``)
    is the obvious thing to write, so it has to work from anywhere.

    The anchored copy is validated by the model, which saw only the paths as read: a parent
    directory name can carry every character the shell rule refuses (``output_dir:
    ./outputs`` in a directory named ``$(…)`` exports ``…/$(…)/outputs``, and bash
    substitutes inside double quotes), and an absolute ``scratch_dir`` can name the very
    directory a relative ``output_dir`` anchors to. Raises pydantic's
    :class:`~pydantic.ValidationError` for each loader to word. A symlink whose *target*
    carries such a character still passes — this sees the path with its symlinks intact —
    and is refused where the path becomes bash (:mod:`chemrefine.slurm.script`).

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
    return type(cfg).model_validate(cfg.model_copy(update=updates)) if updates else cfg


STEP_OPTION_PATHS = ("model_path", "solvation_weights")
"""``step.options`` keys whose value is a filesystem path, resolved like the config's own.

A deliberately short list rather than "anything that looks like a path": an option is
free-form text and most values are not paths at all, so guessing would rewrite strings that
merely resemble one. Adding a knob here is the cost of introducing a path-valued option, and
it is one line.

Public, not underscored, because the cache reads it too. :func:`chemrefine.step.derive_step_key`
keys such an option by its basename — the resolved value is absolute, and an absolute
string inside a cache row key would tie a finished tree to the directory it ran in — and
:func:`chemrefine.cache.option_file_digests` pins the named file's bytes under the same
name. One list decides what the loader resolves, what the key must not spell out, and
which files it pins; a knob absent from it is text to all three, whatever it happens to
name on disk."""


def _resolve_step_option_paths(step: StepConfig, *, base: Path) -> StepConfig:
    """Make a step's path-valued options absolute against ``base``; return the step."""
    options = step.options or {}
    rewritten = {
        key: str((base / value).resolve())
        for key in STEP_OPTION_PATHS
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
    process working directory; a leading ``~`` in any path is the user's home
    (:data:`USER_PATH`).

    Raises :class:`ConfigError` for any malformed file, unknown top-level
    field, or per-step validation failure. The exception message is the
    raw Pydantic error text — already actionable for the user.
    """
    p = Path(path)
    text = io.read_utf8(p, what="config")
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ConfigError(f"malformed YAML in {p}: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"config at {p} is not a YAML mapping")
    try:
        # `model_validate`, not `Config(**raw)`: a non-string key (YAML 1.1's unquoted
        # `on:` parses to a boolean) would make the splat raise a bare TypeError past this
        # handler; pydantic itself turns it into the ValidationError caught here.
        resolved = resolve_relative_paths(Config.model_validate(raw), base=p.parent.resolve())
    except ValidationError as e:
        raise ConfigError(f"invalid config {p}:\n{e}") from e
    # Said here, once per load, rather than by a model validator, which runs again on every
    # copy the model re-checks.
    for _tool, warning in resolved.missing_executables():
        logger.warning("%s", warning)
    return resolved

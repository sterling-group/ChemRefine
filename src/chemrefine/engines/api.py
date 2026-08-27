"""The engine plugin contract + the ``ENGINES`` registry — the public face of ``engines/``.

This is the one module the flat pipeline imports from the engine subsystem: the
:class:`CalculationEngine` / :class:`NmsCapableEngine` / :class:`ProvisionableEngine` Protocols,
the :class:`JobExecutable` provision contract, the DTOs (:class:`ParsedResult`,
:class:`NmsInputInfo`, :class:`BackendRequirement`), and the registry (:data:`ENGINES` /
:func:`register` / :func:`get_engine`). Importing :mod:`chemrefine.engines` registers every
bundled engine, so the orchestrator looks engines up by name and never imports a concrete
engine module.

chemrefine **declares** a small contract; an engine **provides** it. Reusable building blocks
(underscored, never edited to add an engine) live in the subsystem: :class:`._job.JobEngine`
(the per-structure lifecycle), :mod:`._execution` (the scheduler), :class:`._script.ScriptEngine`
(the user-Python-script kind), :mod:`._backend_server` (the ExtOpt server). Plugins are the
bare-named packages under ``engines/``.

Adding a new engine — pick a *kind* and provide the requested pieces
---------------------------------------------------------------------

============================  ==============================  =====================================
Kind                          Base                            The engine provides
============================  ==============================  =====================================
Per-structure program         :class:`._job.JobEngine`        ``build_input`` / ``run_block`` /
(own input format, like ORCA)                                 ``parse_one`` / ``pal`` / ``gpus``
                                                              + the ClassVars
User Python script            :class:`._script.ScriptEngine`  usually only ``_template_vars``
ORCA-driven gradients         :class:`.orca.extopt.engine.    ``backend`` / ``wrapper_filename`` /
                              ExtOptOrcaEngine`               ``options_cls`` / ``calculator_cls``
One job, one product          :class:`CalculationEngine` +    ``prepare`` / ``submit`` / ``parse`` /
(mlip-train)                  :class:`ArtifactEngine`         ``artifact`` (+ ``JobExecutable`` to
                                                              run through the scheduler)
============================  ==============================  =====================================

* **Capabilities** — never a flag, always a Protocol detected via ``isinstance``. NMS:
  implement :class:`NmsCapableEngine`'s ``nms_input_info`` hook and populate
  ``imaginary_freqs`` / ``normal_modes`` on each parsed ``Structure`` (the generic coordinator
  is :mod:`chemrefine.nms`). Managed backend envs: implement :class:`ProvisionableEngine`'s
  ``backend_requirement`` (the generic provisioner is :mod:`chemrefine.engines._provision`).
* **YAML knobs** — a Pydantic model in ``engines/<name>/options.py`` subclassing
  :class:`~chemrefine.engines._options.EngineOptions`; read it in the primitives and declare
  it as ``options_cls`` (the :class:`OptionsDeclaring` capability), so second readers —
  provisioning, schema introspection — resolve the knobs through the engine's own model.
* **Register** — import the class in ``engines/<name>/__init__.py``; the bare-named package is
  **auto-discovered** when :mod:`chemrefine.engines` loads (registration is a side effect), so
  nothing outside the new package changes. Legacy YAML spellings map to the canonical name in
  :func:`chemrefine.config_legacy.normalize`.
* **Resources** — an external binary reads its path from ``ctx.executables.get("<name>")``; an
  importable backend ships as a ``pip install chemrefine[<name>]`` extra (imported lazily).
* **Tests** go in ``tests/test_engines_<name>*.py``, and every engine ships a trimmed
  real-output contract fixture under ``tests/data/engines/<name>/`` (enforced by
  ``test_every_registered_engine_ships_a_contract_case``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from chemrefine.config import StepConfig
from chemrefine.engines._options import EngineOptions
from chemrefine.errors import EngineNotFoundError
from chemrefine.state import JobBatch, StepContext, StepInputs, StepResults

# Part of the contract this module publishes, not an implementation detail of it: an engine
# returns one from `run_block`, so it reads the type from the same place it reads the
# Protocol. The redundant alias is how a re-export is spelled explicitly, which
# `no_implicit_reexport` requires — importing it for our own annotations would not say that
# the engines importing it from here are meant to.
from chemrefine.state import JobTriple as JobTriple
from chemrefine.state import RunBlock as RunBlock


@runtime_checkable
class CalculationEngine(Protocol):
    """Structural contract every engine must satisfy.

    Three methods, and that is the whole of it: :meth:`prepare` → :meth:`submit` (which
    blocks until the jobs finish) → :meth:`parse`. An engine turns a step's specification
    into a calculation and the calculation's output back into structures. How a run is
    resumed, what a cache key is, whether the template changed since yesterday — none of
    that is an engine's business, which is why nothing here mentions caching.

    Everything else is a *capability*, detected via ``isinstance`` so there is no flag to
    keep in sync: :class:`TemplateDriven` (reads a per-step template),
    :class:`NmsCapableEngine`, :class:`ProvisionableEngine`, :class:`StructureArtifacts`.
    Per-structure *job* engines get the three methods from
    :class:`chemrefine.engines._job.JobEngine` and supply only primitives.
    """

    name: ClassVar[str]
    """The canonical YAML ``engine:`` spelling, on the class.

    A ``ClassVar``, not an instance attribute, because that is what every engine declares and
    what :func:`register` requires: the gate runs on the class without constructing anything,
    so an engine's ``__init__`` never runs at import of the package that defines it. Declared
    as a plain ``name: str``, the Protocol said an *instance* variable — which no engine has,
    so ``type[Engine]`` did not satisfy it and any function annotating this Protocol had to
    take ``object`` instead. That is a typing lie in the load-bearing direction: it is the
    contract every consumer narrows from.
    """

    def prepare(self, ctx: StepContext) -> StepInputs:
        """Write engine-specific input files for this step's seed structures."""
        ...

    def submit(self, inputs: StepInputs, ctx: StepContext) -> JobBatch:
        """Run the prepared inputs (locally or via SLURM); block until all finish."""
        ...

    def parse(self, inputs: StepInputs, ctx: StepContext) -> StepResults:
        """Parse each output file into a :class:`~chemrefine.state.Structure`."""
        ...


class CompletionSink(Protocol):
    """What a scheduler tells when one job finishes, and asks for more work.

    The seam that lets :func:`chemrefine.engines._execution.run_batch` parse and re-run as it
    goes without knowing anything about chemistry: it hands over a finished job and enqueues
    whatever comes back. Everything about *why* a follow-up exists — parsing, classification,
    the one-attempt budget, lineage — belongs to the implementation
    (:class:`chemrefine.lifecycle._QueueSink`).

    Deliberately **not** ``runtime_checkable``: nothing tests it with ``isinstance``, and the
    engine capability that *is* tested is :class:`StreamingSubmit`.
    """

    def on_complete(self, job: JobTriple) -> tuple[JobTriple, ...]:
        """Handle one finished job; return follow-up jobs to enqueue (usually empty)."""
        ...


class _NullSink:
    """A sink that wants to hear nothing and asks for nothing back."""

    def on_complete(self, job: JobTriple) -> tuple[JobTriple, ...]:
        """Discard the completion; there is no follow-up work."""
        return ()


NULL_SINK: CompletionSink = _NullSink()
"""The sink for a caller that only wants the batch run.

``run_batch`` takes a sink always, never ``None``, and this is what "no sink" means. That is
what lets there be **one** submission loop: a queue driven by a sink that never adds to it
submits exactly what it was given and drains it, which is the whole of what the old
fixed-batch path did. Keeping the two apart would mean two loops answering the same questions
about the same budget, differing only in whether anyone was listening.
"""


@runtime_checkable
class StreamingSubmit(CalculationEngine, Protocol):
    """An engine that can report each job's completion as it happens.

    A capability, detected with ``isinstance`` like every other one here. It is a **separate
    method** rather than an optional argument to :meth:`CalculationEngine.submit` because
    ``runtime_checkable`` only checks that a method *exists*, not its signature — an optional
    parameter would make every engine answer yes and the narrowing would mean nothing.

    Extending :class:`CalculationEngine` is what keeps the check honest in the other
    direction: a satisfying engine must also have ``name`` / ``prepare`` / ``submit`` /
    ``parse``, so a bare object with one convenient method cannot pass.
    """

    def submit_streaming(
        self, inputs: StepInputs, ctx: StepContext, sink: CompletionSink
    ) -> JobBatch:
        """Run the prepared inputs, calling ``sink.on_complete`` as each job finishes."""
        ...


def sweep(inputs: StepInputs, sink: CompletionSink) -> tuple[JobTriple, ...]:
    """Hand every job of an already-drained batch to ``sink``; return its follow-ups.

    What a scheduler that cannot report completions *individually* does instead — the SLURM
    job-array path and :func:`chemrefine.lifecycle._drain`. One function rather than one per
    caller: they are the same three lines, and two copies of "drive the sink over a finished
    batch" would be two places for a follow-up to go missing.
    """
    return tuple(f for job in inputs.files for f in sink.on_complete(job))


@runtime_checkable
class TemplateDriven(CalculationEngine, Protocol):
    """An engine whose step input is rendered from a per-step template.

    **Declarations only — no methods.** This says *which file the engine reads*, not
    anything it does differently, so it costs an implementer two ClassVars and no code.
    :func:`chemrefine.step.build_context` reads them once per step and puts the resolved
    path on :attr:`~chemrefine.state.StepContext.template`; everything downstream — the
    engine rendering it, the cache digesting it — reads that field.

    A capability rather than part of :class:`CalculationEngine`, because being
    template-driven is genuinely optional: ``mlip-train`` reads a template but writes no
    per-structure inputs, and an engine that fabricates or passes structures through reads
    none at all.

    ``template_suffix`` means the template's extension and only that. Declared on
    :class:`~chemrefine.engines._job.JobEngine` instead, it would have to mean the
    per-structure input's extension too, and an engine with one and not the other could not
    declare it at all. ``JobEngine`` reuses it for artifact naming — not by coincidence, but
    because that artifact *is* a copy of the template and shares its extension.
    """

    template_suffix: ClassVar[str]
    """Extension of this engine's step template (``inp`` / ``py``), without the dot."""

    label: ClassVar[str]
    """Human name for this engine's inputs, used in "…​ template not found" errors."""


@runtime_checkable
class WhitespacePathIntolerant(Protocol):
    """An engine whose generated input cannot express a path containing whitespace.

    **Declarations only**, like :class:`TemplateDriven` — one ClassVar carrying the reason,
    so a caller can say *why* without importing the engine that knows. It exists because the
    constraint is real for exactly one family and invisible everywhere else: ORCA reads each
    geometry through ``* xyzfile <path>``, a whitespace-delimited field it does not treat as
    quotable, and execs an ExtOpt wrapper through ``sh``. Every other shipped engine is
    unaffected — Q-Chem inlines the geometry into ``$molecule``, the script engines
    substitute a path the template quotes, and the generated bash quotes everything it
    interpolates.

    A capability rather than a rule in :mod:`chemrefine.config`, and that placement is the
    whole point. Held at config load it made *validity depend on where a project sits on
    disk*: a relative ``output_dir`` inherits the config file's directory, so an mlip-only
    workflow under ``~/My Drive`` — which runs perfectly well — was refused, and this
    repository's own example suite went red whenever the checkout path contained a space.
    The engine that cannot express the path is the one that should refuse it, and it refuses
    the *resolved* path it is about to write, which is also what closes the symlinked-parent
    case that a check on the configured value cannot see.

    :mod:`chemrefine.validate` detects this with ``isinstance`` to warn early — a warning,
    not an error, because the config is well-formed and only this engine family cannot run
    under that path.
    """

    whitespace_path_reason: ClassVar[str]
    """Why this engine cannot take a path with whitespace — quoted verbatim in the refusal."""


@runtime_checkable
class OptionsDeclaring(Protocol):
    """An engine that declares the Pydantic model validating its ``step.options``.

    Declarations only, like :class:`TemplateDriven` — one ClassVar naming the
    :class:`~chemrefine.engines._options.EngineOptions` subclass the engine itself reads
    its knobs through. A *second* reader of those knobs — provisioning's
    ``backend_python`` (:func:`chemrefine.engines._provision._backend_python`), the GPU
    demand helper (:func:`chemrefine.engines._job.gpus_from_options`), schema
    introspection — resolves aliases and defaults through the same model the engine
    will, so two readers of one knob cannot disagree.

    A capability and a **claim**, not boilerplate: declaring a model asserts the engine
    reads its fields, and the cross-engine invariants hold every declaring engine's
    ``gpus()`` to its model's ``device``. That is why ORCA declares nothing — its knobs
    live in the step template, its ``options:`` dict is read only by NMS, and a model
    here would advertise a ``device`` knob the engine ignores. A consumer meeting a
    non-declaring engine reports "no engine options" (introspection) or falls back to
    the base :class:`~chemrefine.engines._options.EngineOptions` for the shared knobs
    (provisioning), rather than inventing a schema the engine does not honour.
    """

    options_cls: ClassVar[type[EngineOptions]]
    """The model validating this engine's ``step.options`` — the engine's one reader."""


@runtime_checkable
class OperationsDeclaring(Protocol):
    """An engine that declares which ``operation:`` values it interprets.

    Declarations only, like :class:`TemplateDriven`. The config schema keeps
    ``operation`` a free string because engines interpret it themselves; this ClassVar
    is how an engine *publishes* its vocabulary so introspection (and the GUI's
    dropdown) can offer exactly the values that engine will act on — the ORCA family
    declares its parser dispatch's own set, a script engine that treats the field as a
    label declares nothing, and a future engine with operations of its own declares
    them here without touching any consumer.
    """

    operations: ClassVar[tuple[str, ...]]
    """The ``operation:`` spellings this engine interprets, canonical only."""


@dataclass(frozen=True)
class NmsInputInfo:
    """What an engine's configured input does, for the generic NMS coordinator.

    ``is_transition_state`` drives the default NMS target (``ts`` vs ``minimum``);
    ``computes_frequencies`` gates NMS (a step that won't produce frequencies cannot be
    resolved). Both are read from the engine's own input (ORCA: its template keywords).
    """

    is_transition_state: bool
    computes_frequencies: bool


@dataclass(frozen=True)
class ParsedResult:
    """One structure extracted from an engine's output, before lineage is assigned.

    The DTO a :class:`JobEngine`'s ``parse_one`` returns; the assembler
    (:func:`chemrefine.engines._job.build_structures`) turns a step's
    ``ParsedResult``s into :class:`~chemrefine.state.Structure` objects with IDs +
    parents. ``terminated_normally`` / ``converged`` are run-status flags (``None`` when the
    engine doesn't report them, e.g. sidecar ensemble frames); a structure is a
    *failure* only when one is explicitly ``False``. The thermochemistry + frequency
    fields are populated only by a frequency run: ``imaginary_freqs`` maps a mode index to
    its frequency (cm⁻¹) — ``None`` = no frequency table at all (distinct from ``{}`` = a
    table with zero imaginary modes) — and ``normal_modes`` is the displacement tensor NMS
    displaces along.

    ``frequencies`` is the whole mode table in the same index space, of which
    ``imaginary_freqs`` is a subset. NMS only ever needed the imaginary ones, so for a long
    time only those were kept and every real mode's frequency was thrown away at the parse
    boundary — leaving anything that named a mode afterwards (a viewer's mode list,
    :func:`chemrefine.agent_tools.analyze_mode`) with an index and no cm⁻¹ to put beside it.
    """

    symbols: tuple[str, ...]
    positions: NDArray[np.float64]
    energy_hartree: float
    forces_ev_per_a: NDArray[np.float64] | None
    converged: bool | None = None
    terminated_normally: bool | None = None
    gibbs_hartree: float | None = None
    enthalpy_hartree: float | None = None
    energy_zpe_hartree: float | None = None
    imaginary_freqs: dict[int, float] | None = None
    frequencies: dict[int, float] | None = None
    normal_modes: NDArray[np.float64] | None = None


@runtime_checkable
class JobExecutable(Protocol):
    """The primitives :func:`chemrefine.engines._execution.run_batch` needs from an engine.

    The narrow provision surface a :class:`~chemrefine.engines._job.JobEngine` exposes so
    the flat scheduler can run one job per structure without knowing the engine's type —
    Interface Segregation: ``run_batch`` depends on the members below and not the whole
    engine. ``JobEngine`` satisfies it structurally. The members are listed rather than
    counted, so a new one does not date the sentence above it.
    """

    @property
    def output_globs(self) -> tuple[str, ...]:
        """Loose files to copy back out of the job's scratch directory.

        A read-only property declaration rather than a ``ClassVar``, so an engine may
        *compute* the answer at read time — ``mlip-train`` derives it from its trainer
        registry, which is only populated after the engine's class body has run. A plain
        class attribute satisfies it just the same, and that is what every other engine
        declares. Read on instances only, without a :class:`~chemrefine.state.
        StepContext` — which is why it cannot be exact per step the way
        :meth:`output_dirs` can.
        """
        ...

    def run_block(self, ctx: StepContext, inp_path: Path, out_path: Path) -> RunBlock:
        """The bash that runs one job inside ``$WORK_DIR``, plus any teardown it needs."""
        ...

    def pal(self, ctx: StepContext) -> int:
        """Per-job core count (PAL) before the scheduler clamps it to ``max_cores``."""
        ...

    def slurm_layout(self, ctx: StepContext) -> tuple[int, int]:
        """How this job's cores are spelled to SLURM: ``(ntasks, cpus_per_task)``.

        The same core count means different directives to different programs: MPI ranks
        (ORCA) are ``(pal, 1)``, one threaded process (Q-Chem's ``-nt``) is ``(1, pal)``,
        and a hybrid MPI+OpenMP job is ``(ranks, threads)``. The scheduler charges the
        product against ``max_cores`` and writes both directives, so the allocation and
        the program's own idea of its parallelism cannot disagree.
        """
        ...

    def gpus(self, ctx: StepContext) -> int:
        """GPUs one job needs (``0`` = CPU)."""
        ...

    def memory_mb(self, ctx: StepContext) -> int | None:
        """Total MB one job requires, or ``None`` when the engine declares nothing.

        Read from the step's own input where the program reads it — ORCA's ``%maxcore``,
        Q-Chem's ``mem_total`` — and already carrying the engine's headroom rule, so the
        script builder can compare it against the header's allocation whole. ``None``
        leaves the header's memory policy untouched, which is what every engine did before
        this existed.
        """
        ...

    def output_dirs(self, ctx: StepContext) -> tuple[str, ...]:
        """Scratch sub-directories to copy back wholesale."""
        ...

    def extra_header_fields(self, ctx: StepContext) -> tuple[tuple[str, object], ...]:
        """Engine-specific ``(key, value)`` rows for the runlog header."""
        ...


@runtime_checkable
class StructureArtifacts(Protocol):
    """An engine whose per-structure input and output live at paths the caller can compute.

    Separate from :class:`CalculationEngine` because it is a different claim: an engine can
    produce structures without every one of them having a file of its own — a training step
    prepares no per-structure inputs at all. And separate from :class:`JobExecutable`, whose
    members are about *how* a job runs (cores, GPUs, the bash it executes) rather than where
    its files land.

    :class:`NmsCapableEngine` requires it because the two-round algorithm has to find a
    round-2 child's output without being told the engine's file extensions.
    """

    def artifact_paths(self, ctx: StepContext, structure_id: str) -> tuple[Path, Path]:
        """This structure's ``(input, output)`` paths under ``ctx.step_dir``."""
        ...


@runtime_checkable
class NmsCapableEngine(CalculationEngine, StructureArtifacts, Protocol):
    """An engine that supports normal-mode sampling, via one input-introspection hook.

    The two-round NMS algorithm — displacement, round-2 submission, resolution, retry — is
    engine-independent and lives in :mod:`chemrefine.nms`, which drives any engine through this
    hook plus the standard lifecycle. The *output* half is no hook at all: the engine's
    :meth:`parse` already carries ``imaginary_freqs`` + ``normal_modes`` on each
    :class:`~chemrefine.state.Structure` (parsed in the same single pass as energy/geometry),
    so NMS reads them off the structures it already holds. A new NMS-capable engine implements
    ``nms_input_info`` and :meth:`~StructureArtifacts.artifact_paths` — which
    :class:`~chemrefine.engines._job.JobEngine` already provides — and populates those two
    structure fields; capability is detected with ``isinstance``.

    ``normal_modes`` is indexed with the trivial translation/rotation modes first: NMS skips
    the leading six (five for a linear molecule) when drawing at random, so an engine
    populating the tensor must use that ordering.
    """

    def nms_input_info(self, ctx: StepContext) -> NmsInputInfo:
        """Introspect this step's configured input (TS search? computes frequencies?)."""
        ...


@runtime_checkable
class ArtifactEngine(CalculationEngine, Protocol):
    """An engine whose product is one **artifact**, and whose structures pass straight through.

    Training is the shape this exists for: a step that submits a single job over the whole
    prior ensemble, writes a model, and hands the *same* structures to the next step. Nothing
    about that fits the per-structure lifecycle — there is no input per structure to prepare,
    no output per structure to parse, and no lineage to assign — so an artifact step runs
    through :func:`chemrefine.step._run_artifact_step` instead of
    :func:`~chemrefine.step._run_full_step`.

    A capability detected via ``isinstance``, like every other one here, and one hook for the
    same reason :class:`NmsCapableEngine` has one: ``prepare`` / ``submit`` / ``parse`` are
    already :class:`CalculationEngine`'s, and the only thing the orchestrator cannot work out
    for itself is *where the product is*.

    That hook is what makes the step's outcome decidable. A job that leaves the queue having
    written nothing is indistinguishable, to the scheduler, from one that succeeded — so the
    existence of :meth:`artifact` is the success test, and its absence raises
    :class:`~chemrefine.errors.JobFailureError` **before** any cache is written. No cache is
    the whole point: ``resume`` then re-runs the step rather than serving a model that was
    never produced.
    """

    def run_dir(self, ctx: StepContext) -> Path:
        """The directory this step's single job runs in, under ``ctx.step_dir``.

        Declared rather than assumed, because the orchestrator needs it and cannot derive it.
        :func:`chemrefine.step._run_artifact_step` moves the previous run aside before
        re-executing, and that archive is what makes :meth:`artifact` a usable success test:
        ``artifact.exists()`` cannot tell this run's product from the last one's, so a re-run
        whose job dies having written nothing would otherwise find the *previous* model,
        digest it into the sidecar, and cache it under the **new** fingerprint — a run that is
        internally consistent and describes a training that never happened.

        A constant named in ``step.py`` would answer for whichever engine happens to use that
        directory and be silently wrong for any other: its run directory would go unarchived,
        and the guard would pass while protecting nothing. A capability the orchestrator acts
        on belongs in the contract, like every other one here.
        """
        ...

    def artifact(self, ctx: StepContext) -> Path:
        """Where this step's product lives once its job has run.

        Called both to decide success and — by ``rebuild-cache`` — to adopt a product whose
        run finished before the driver died, so it must be derivable from ``ctx`` alone
        rather than from anything the submission returned.

        May sit *inside* :meth:`run_dir` rather than directly in it — FAIRChem's is at
        ``checkpoints/final/inference_ckpt.pt`` under a directory it names itself — which is
        why the two are separate questions.
        """
        ...


@dataclass(frozen=True)
class BackendRequirement:
    """The environment a step's compute backend needs, for the provisioner.

    ``extra`` is the pip extra that provides the backend — ``chemrefine[<extra>]`` installs
    it; ``import_name`` is the backend's top-level module, probed to decide whether the
    backend is usable at all. Which *directory* the extra installs into is
    :func:`chemrefine.engines._provision.backend_env_path`'s to decide, not a field here:
    the CLI provisions from a bare extra name and has no requirement object to read.
    """

    extra: str
    import_name: str


@runtime_checkable
class ProvisionableEngine(CalculationEngine, Protocol):
    """An engine whose compute backend can live in its own managed environment.

    The MLIP libraries' torch/e3nn trees conflict, so one Python process can host only one
    backend family; a *provisionable* engine names the environment it needs
    (:class:`BackendRequirement`) and the generic provisioner
    (:mod:`chemrefine.engines._provision`) checks it before any job submits and launches the
    step's Python from the matching managed env. Like NMS, this is a capability detected via
    ``isinstance`` — ORCA (a binary, not a Python backend) simply doesn't implement it.
    Both methods are required for the capability to be detected.
    """

    def backend_requirement(self, options: Mapping[str, Any] | None) -> BackendRequirement:
        """The backend env this step needs, derived from its ``step.options``."""
        ...

    def backend_extras(self) -> frozenset[str]:
        """Every extra this engine can require (drives ``chemrefine backends`` listing)."""
        ...


@runtime_checkable
class PreflightChecking(Protocol):
    """An engine that can vet a step's configuration before the run starts.

    The refusals an engine would otherwise make in ``prepare`` — a required knob left
    unset, an option combination its backend cannot serve — fire there only when the
    step's own turn comes, which for a late step is after every earlier one has been
    computed and paid for. This hook is the sanctioned earlier moment:
    :func:`preflight_steps` calls it for every submittable step before anything runs
    (the walk that already checks backend envs), and ``chemrefine validate`` reports
    the same refusals without running anything. An engine keeps making its own checks
    in ``prepare`` too — the recovery paths that skip the preflight still deserve them.

    A capability detected via ``isinstance`` like every other one here, and an opt-in
    deliberately, never a generic strict pass over every options model: the direct
    script engines read their options leniently by documented design (a ``step{N}.py``
    template may carry knobs no model declares), so only an engine that *owns* a
    fail-fast refusal declares the hook.
    """

    def check_step(self, step_cfg: StepConfig, *, charge: int, multiplicity: int) -> None:
        """Raise :class:`~chemrefine.errors.ConfigError` if this step cannot run as configured.

        ``charge`` and ``multiplicity`` are the step's **effective** values — the
        config defaults with any per-step override applied — because a refusal may
        hinge on them: an open-shell step asking for a closed-shell-only extraction is
        decidable from the config alone, but only from the resolved species.
        """
        ...


ENGINES: dict[str, type[CalculationEngine]] = {}
"""Registry mapping the YAML ``engine:`` string to a concrete engine class.

Only **canonical** names live here. Old spellings (``mlff*``, ``dft``) are
rewritten to canonical names by the config normalizer
(:func:`chemrefine.config_legacy.normalize`) — the single place that knows the
legacy vocabulary — before any lookup, so the registry stays alias-free.
"""


def contract_members(protocol: type) -> frozenset[str]:
    """Every member a Protocol declares — its methods plus its annotated attributes.

    Derived rather than restated, so the gate below cannot come to disagree with the contract
    it enforces. Written out by hand instead of read from ``__protocol_attrs__`` because that
    is a 3.12 addition and ``requires-python`` is ``>=3.11``; the two are asserted equal by
    ``test_engines_base.py`` wherever the interpreter offers both.
    """
    return frozenset(n for n in dir(protocol) if not n.startswith("_")) | frozenset(
        getattr(protocol, "__annotations__", {})
    )


_CONTRACT_MEMBERS = contract_members(CalculationEngine)


def register(name: str) -> Callable[[type[CalculationEngine]], type[CalculationEngine]]:
    """Decorator: register ``cls`` under ``name`` in :data:`ENGINES`, if it qualifies.

    **The decorator is the gate**, in the shape
    :meth:`chemrefine.engines.mlip.registry.MlipLibrary.trainer` already sets one subsystem
    over: a class that cannot serve as an engine is refused *here*, at its own decorator line
    during discovery, naming what is missing — rather than at the first step that submits, by
    which time ``run_step`` has built a context, derived a cache key and created a directory.
    Three ways a class can fail to qualify, each with its own reason:

    * **It does not satisfy the contract.** ``ENGINES`` is annotated
      ``dict[str, type[CalculationEngine]]`` and :func:`get_engine` hands what it holds to the
      pipeline as one. The signature above is what makes that claim checkable statically —
      typed over ``type``, the decorator would accept anything, since ``type`` is ``type[Any]``
      and mypy has nothing to compare a decorated class against. The runtime check below is the
      other half, for a plugin whose author does not run the type checker.
    * **It inherits the Protocol.** :class:`CalculationEngine` is ``runtime_checkable``, and a
      subclass of it inherits every method as an ellipsis body returning ``None`` — so
      ``isinstance`` says yes, ``prepare`` returns ``None``, and every structural check in the
      codebase passes. A structural contract used as a base defeats the checks that stand in
      for this gate, which is why it is refused rather than merely discouraged.
    * **It leaves a declaration the machinery reads unset.** The engine bases declare ClassVars
      with no default, and no ``ABCMeta`` machinery watches those: ``abstractmethod`` covers the
      *methods* only. A base names its own in ``required_declarations``, which is the
      ``required = [...]`` list ``@trainer`` checks, one subsystem over. Unset, ``output_suffix``
      surfaces as a bare ``AttributeError`` inside ``prepare``, and ``template_suffix`` is
      worse: the engine simply stops satisfying :class:`TemplateDriven`, and the user is told
      their template does not exist while it sits on disk.

    The check runs on the class, never on an instance: constructing one to interrogate it would
    make an engine's ``__init__`` run at import of the package that defines it.
    """

    def decorator(cls: type[CalculationEngine]) -> type[CalculationEngine]:
        if name in ENGINES and ENGINES[name] is not cls:
            raise ValueError(f"engine {name!r} is already registered to {ENGINES[name]!r}")
        if CalculationEngine in getattr(cls, "__mro__", ()):
            raise TypeError(
                f"engine {name!r}: {cls.__name__} inherits CalculationEngine, which is a "
                f"structural contract — inheriting it supplies every method as a no-op "
                f"returning None, so isinstance would pass a class that does nothing. "
                f"Satisfy it structurally, or subclass a base such as JobEngine."
            )
        missing = sorted(m for m in _CONTRACT_MEMBERS if not hasattr(cls, m))
        if missing:
            raise TypeError(
                f"engine {name!r}: {cls.__name__} does not satisfy CalculationEngine — "
                f"missing {missing}. Members are looked for on the class, so declare them "
                f"there: an attribute bound in __init__ is not visible to this check, which "
                f"runs without constructing anything. See chemrefine.engines.api."
            )
        undeclared = sorted(
            d for d in getattr(cls, "required_declarations", ()) if not hasattr(cls, d)
        )
        if undeclared:
            raise TypeError(
                f"engine {name!r}: {cls.__name__} is missing declaration(s) {undeclared} — "
                f"the machinery reads them; see the base it inherits from."
            )
        ENGINES[name] = cls
        return cls

    return decorator


def get_engine(name: str) -> CalculationEngine:
    """Look up an engine by its canonical name and return a fresh instance.

    Raises :class:`~chemrefine.errors.EngineNotFoundError` if ``name`` isn't
    registered. (Legacy spellings are normalized at config-parse time, so the
    name reaching here is already canonical.)
    """
    engine_cls = ENGINES.get(name)
    if engine_cls is None:
        raise EngineNotFoundError(f"unknown engine {name!r}; registered: {sorted(ENGINES)}")
    return engine_cls()


def preflight_steps(steps: Sequence[StepConfig], *, charge: int, multiplicity: int) -> None:
    """Ask every step's engine to vet its configuration before anything runs.

    Called from :func:`chemrefine.pipeline.run` beside ``preflight_backends``, over the
    same submittable steps and for the same reason: a refusal decidable from the config
    alone must not wait for the failing step's own turn — in a pipeline that spends
    days computing labels before a training step, that is the difference between a
    typo caught in seconds and one caught on Thursday. Engines without the capability
    have nothing to check and are not asked.

    ``charge`` / ``multiplicity`` are the config-wide defaults; each step's own
    override is applied here (:meth:`~chemrefine.config.StepConfig.effective_charge`),
    so a hook always sees the effective values its step would run with — the same
    resolution :func:`chemrefine.step.build_context` performs for the run itself.
    """
    for step_cfg in steps:
        engine = get_engine(step_cfg.engine)
        if isinstance(engine, PreflightChecking):
            engine.check_step(
                step_cfg,
                charge=step_cfg.effective_charge(charge),
                multiplicity=step_cfg.effective_multiplicity(multiplicity),
            )

"""Frozen runtime types passed between pipeline stages.

These are the values that flow from one step to the next and between the
engine lifecycle stages (prepare → submit → wait → parse → sample) —
what a step was given, what it produced, and what it has to say about a
structure that did not succeed (:class:`FailureKind`, :class:`Failure`,
:class:`FailureRecord`). Keeping them frozen and small forces callers to
thread state explicitly rather than mutating shared state on a god class.

Vocabulary only: deciding what a failure *means* for a step is
:mod:`chemrefine.lifecycle`, and persisting the ledger is
:mod:`chemrefine.cache`. Both import from here, so neither has to import
the other to name an outcome.

``StepConfig`` lives in :mod:`chemrefine.config` and is imported here for
:class:`StepContext`, which carries a step's own specification alongside the
state it runs over. That makes this module a Pydantic importer too — seventeen
modules import it, so it is not a leaf on cost, only on direction: it depends on
the configuration vocabulary and on nothing above it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np
from ase import Atoms
from numpy.typing import NDArray

from chemrefine.config import Dispatch, StepConfig


@dataclass(frozen=True)
class Structure:
    """A single molecular geometry with its computed energy and forces.

    Energy is in Hartree (engine-native); forces are in eV/Å (ASE-native).
    ``None`` values mean the field has not been populated yet — e.g. a
    seed structure before its first calculation.

    ``parent_id`` is the ID of the previous-step structure this one was
    derived from — ``None`` for seed structures. Filtering by parent
    reads this field directly instead of parsing a hyphen-encoded
    lineage out of ``id``.

    ``forces_ev_per_a`` is a numpy array, which ``frozen=True`` cannot make
    immutable on its own — so ``__post_init__`` clears its write flag. The
    array is shared by reference across the pipeline, so a stray write raises
    :class:`ValueError` at the point of the mistake rather than silently
    changing a structure another step already holds.
    """

    id: str
    atoms: Atoms
    """The geometry — shared by reference, so **copy before deriving a mutated one**.

    ``frozen=True`` cannot reach inside an :class:`~ase.Atoms`, and unlike the two array
    fields below its position buffer cannot have its write flag cleared — ASE's own code
    writes through it. The protection is therefore a rule rather than a flag: any code
    that moves atoms or attaches labels works on ``struct.atoms.copy()``, never in place
    (see :func:`chemrefine.nms._displaced` and the trainer dataset writers).
    :func:`chemrefine.cache.structure_digest` hashes these positions into every downstream
    step's cache key, so an in-place write would not crash anything — it would silently
    re-fingerprint work that was already done. Each copy site carries a test asserting
    the source structure comes through untouched."""
    parent_id: str | None = None
    energy_hartree: float | None = None
    forces_ev_per_a: NDArray[np.float64] | None = None
    converged: bool | None = None
    """Did the run converge (SCF + geometry)? ``None`` when the engine
    doesn't report it — treated as 'not a failure signal'."""
    terminated_normally: bool | None = None
    """Did the program terminate normally — i.e. exit cleanly? ``None`` when the
    engine doesn't report it (the script engines never do).

    ``True`` means **success**, which is why the name carries ``_normally``: a bare
    ``terminated`` reads as "the job was killed" and inverts the flag that decides whether a
    calculation counts. It matches ORCA's own banner (``ORCA TERMINATED NORMALLY``) and
    :attr:`FailureKind.NOT_TERMINATED_NORMALLY`'s wording.

    A structure is a *failure* only when a flag is explicitly ``False`` (see
    :func:`chemrefine.lifecycle.succeeded`)."""
    gibbs_hartree: float | None = None
    """Gibbs free energy (Hartree) from a frequency calc; ``None`` when no
    thermochemistry was computed. Used by ``sample.energy_type: gibbs``."""
    enthalpy_hartree: float | None = None
    """Total enthalpy (Hartree) from a frequency calc; ``None`` when none.
    Used by ``sample.energy_type: enthalpy``."""
    energy_zpe_hartree: float | None = None
    """Electronic energy + zero-point correction (Hartree); ``None`` when no
    thermochemistry was computed. Used by ``sample.energy_type: electronic_zero_point``."""
    imaginary_freqs: dict[int, float] | None = None
    """Imaginary normal modes (mode index → cm⁻¹) from a frequency calc, read in the same
    pass as energy/geometry; ``None`` = no frequency table (distinct from ``{}`` = a verified
    minimum). The engine-independent NMS coordinator reads this off the structure."""
    normal_modes: NDArray[np.float64] | None = None
    """Normal-mode displacement tensor ``(n_atoms, 3, n_modes)`` from a frequency calc; ``None``
    when absent. A **transient** artifact used by NMS to displace along imaginary modes — it is
    *not* persisted to the cache (an active NMS run always re-parses), so a cache-reloaded
    structure carries ``None``."""
    resolved_from: str | None = None
    """ID of the displaced child whose calculation this structure's artifacts came from, for a
    structure resolved by normal-mode sampling; ``None`` for every other structure.

    A resolved parent keeps its own ID, and the winning child's files are promoted to the
    parent's canonical basenames — so this is the only record of which of the ± children
    actually produced them. ``random`` sampling and a parent already at its target leave it
    ``None``: no promotion happened."""

    def __post_init__(self) -> None:
        """Make the array fields as read-only as the dataclass claims to be.

        ``frozen=True`` stops the *attribute* being rebound but says nothing about the
        contents of an ndarray it points at, and these arrays are shared by reference —
        every structure carried between steps hands out the same buffer. Clearing the write
        flag turns a stray in-place write into a ``ValueError`` where it happens, rather than
        a value that quietly changes under a step that already holds it.
        """
        for field_name in ("forces_ev_per_a", "normal_modes"):
            array = getattr(self, field_name)
            if array is not None:
                array.setflags(write=False)


@dataclass(frozen=True)
class PipelineState:
    """Survivors carried between steps.

    ``structures`` is the filtered set produced by the previous step (or
    the bootstrap seed for step 1). Truthiness reflects "have anything
    to refine."
    """

    structures: tuple[Structure, ...] = field(default_factory=tuple)

    def __bool__(self) -> bool:
        return len(self.structures) > 0

    def __len__(self) -> int:
        return len(self.structures)


@dataclass(frozen=True)
class StepContext:
    """Per-step input bundle handed to ``engine.prepare`` / ``engine.parse``.

    The engine receives this instead of the full :class:`Config` so its
    surface stays focused on what one step needs.
    """

    step_cfg: StepConfig
    step_dir: Path
    template_dir: Path
    template: Path | None
    """Where this step's input template lives; ``None`` for an engine that reads none.

    Part of the step's *specification* — change the template and the step must re-run — so it
    belongs on the bundle that carries the specification, resolved once by
    :func:`chemrefine.step.build_context`. An ORCA step alone has five readers (``prepare``,
    ``pal``, run-type detection, the NMS probe, the cache), and
    :func:`~chemrefine.engines.orca.inspect.inspect_template` caches nothing, so each would
    otherwise re-resolve and re-read the file.

    ``None`` and "a path that is not there" are different states and stay distinguishable:
    ``None`` means the engine is not :class:`~chemrefine.engines.api.TemplateDriven`, while a
    missing file means one was named and is absent. :func:`chemrefine.cache.template_digest`
    treats both as "contributes no digest"; :func:`chemrefine.ids.require_template` refuses
    both, naming the path when there is one.
    """
    scratch_dir: Path | None
    """``None`` means the SLURM script auto-derives a per-calc work dir under ``step_dir``."""
    prev_state: PipelineState
    charge: int
    multiplicity: int
    max_cores: int
    slurm_template: str
    executables: dict[str, str] = field(default_factory=dict)
    """Tool-name → binary-path map (from ``Config.executables``); an engine
    reads its own key, e.g. ``executables.get("orca", "orca")``."""
    max_gpus: int | None = None
    """Configured GPU budget (``Config.max_gpus``); ``None`` = auto-resolve at
    submit time (unlimited under SLURM, detected device count locally)."""
    slurm_array: bool = False
    """Submit this step as SLURM job array(s) (``Config.slurm_array``);
    ignored when running locally."""
    job_timeout_seconds: float | None = None
    """Wall-clock deadline for this step's jobs (``Config.job_timeout_seconds``);
    ``None`` waits indefinitely."""
    dispatch: Dispatch = "auto"
    """Job dispatch mode (``Config.dispatch``): auto / local / slurm."""


JobTriple = tuple[Path, Path, str]
"""One prepared job: ``(input_path, output_path, structure_id)``.

Named here rather than in :mod:`chemrefine.engines.api`, which is where the scheduler
contracts live, because that module imports *this* one — an alias declared there could not be
used by :class:`StepInputs` itself, and the spelling would stay duplicated at the site that
defines the shape.
"""


@dataclass(frozen=True)
class StepInputs:
    """Engine-prepared inputs for one step's job batch.

    ``files`` is an ordered tuple of :data:`JobTriple`. Order matches the seed
    structures' order so parse results can be aligned back to parents.
    """

    files: tuple[JobTriple, ...]


@dataclass(frozen=True)
class RunBlock:
    """The bash one engine contributes to a job script — split into what runs and what cleans up.

    Two fields rather than one string, so teardown is *data* the infra layer places rather
    than bash an engine emits. :func:`chemrefine.slurm.script._run_body_lines` installs a
    single ``EXIT`` trap that copies results back, copies each ``output_dirs`` entry back,
    tears down scratch and writes the runlog footer; ``cleanup`` is interpolated inside that
    handler.

    The split matters because bash keeps one handler per signal. An engine that wrote its own
    ``trap … EXIT`` would replace the whole of that — and the loss is quiet, because ORCA
    redirects its ``.out`` straight to ``$OUTPUT_DIR``, so parsing still succeeds while
    ``.gbw``/``.hess`` stay in scratch and every job leaks its ``$WORK_DIR``. With teardown as
    a field, an engine has no reason to trap at all.

    Note what this does and does not guarantee. The handler is armed *before* the body runs —
    it has to be, or a failure inside the body would clean up nothing — so an engine that
    wrote ``trap … EXIT`` into ``body`` anyway would still displace it. The type removes the
    motive; the rule is enforced by
    ``test_no_engine_emits_a_trap_of_its_own`` over every registered engine, and its
    consequence by ``test_the_assembled_script_still_runs_its_exit_handler``, which runs the
    composed script.
    """

    body: str
    """Bash run inside ``$WORK_DIR`` after the runlog header, in place of the calculation."""

    cleanup: str = ""
    """Bash run inside the script's own ``EXIT`` handler, before anything is copied back.

    For teardown that must happen however the job ends — the ExtOpt engines stop their gradient
    server here. Runs with ``set +e``, so failing cleanup cannot abort the copy-back."""


@dataclass(frozen=True)
class JobBatch:
    """Opaque handle returned by ``engine.submit`` and consumed by ``engine.wait``.

    ``jobs`` maps each input file path to its job identifier (a SLURM
    job ID, a local-runner PID, or whatever the engine's submitter
    produces). The wait step polls this mapping.

    **Opaque** is the operative word: a structure re-run in the same batch reuses its input
    path, so the mapping holds that structure's *latest* attempt and there is no longer one
    entry per prepared job. Nothing reads it back — treat it as a receipt, not an index.
    """

    jobs: dict[Path, str]


# ---------------------------------------------------------------------------
# Failure vocabulary — what a step says about a structure that did not succeed
# ---------------------------------------------------------------------------


class FailureKind(StrEnum):
    """Why a structure failed — the classification recovery branches on.

    A closed vocabulary rather than free text: ``retry_unconverged``, ``_resubmit_failed``
    and ``reattempt_nms`` each route on *which* kind of failure this is, so the comparison
    has to survive an edit to the wording a user reads. Matching on the message itself made
    rewording it — a docs change, to all appearances — disable a recovery path.

    The values *are* that wording, so the ledger on disk and the log lines stay readable;
    what the code compares is the name.
    """

    MISSING_OUTPUT = "output missing"
    """The job produced no output file at all — crashed, killed, never started."""

    UNPARSEABLE = "unparseable"
    """An output exists but the engine could not read it (truncated, corrupt)."""

    NOT_TERMINATED_NORMALLY = "did not terminate normally"
    """The program ran but did not exit cleanly.

    Named for the negation of :attr:`Structure.terminated_normally`, in full. Dropping the
    adverb reads as "did not terminate" — a job still running, or one whose exit nobody
    minded — which is the opposite of what it records. The field itself carries the word
    for the same reason."""

    NOT_CONVERGED = "did not converge"
    """It finished, but the SCF or the geometry did not converge. The only kind
    that is retried from its best geometry — resubmitting the identical input for
    any of the others would just fail the same way."""

    UNRESOLVED_NMS = "NMS: target stationary point not reached"
    """Normal-mode sampling could not reach the requested stationary point."""

    FAILED = "failed"
    """The engine set a failure flag with no more specific name here."""


@dataclass(frozen=True)
class Failure:
    """One failed structure: its id, why, and the best geometry obtained (if any)."""

    sid: str
    kind: FailureKind
    best: Structure | None
    detail: str = ""
    """Extra context for kinds that have some (the parser message for
    ``UNPARSEABLE``); empty otherwise."""

    @property
    def reason(self) -> str:
        """The human-readable reason, as written to the ledger and the logs."""
        return f"{self.kind.value}: {self.detail}" if self.detail else self.kind.value


@dataclass(frozen=True)
class FailureRecord:
    """A ledger entry — one failed structure, as persisted to ``failed_jobs.json``.

    The recovery paths read this back to decide what to re-attempt, so it is a typed
    record rather than a bare dict indexed with string literals at four call sites.
    """

    structure_id: str
    kind: FailureKind
    reason: str

    @classmethod
    def of(cls, failure: Failure) -> FailureRecord:
        """The ledger entry for an in-flight :class:`Failure`."""
        return cls(structure_id=failure.sid, kind=failure.kind, reason=failure.reason)

    def to_json(self) -> dict[str, str]:
        """Serialize for ``failed_jobs.json``."""
        return {"structure_id": self.structure_id, "kind": self.kind.value, "reason": self.reason}

    @classmethod
    def from_json(cls, data: dict[str, str]) -> FailureRecord:
        """Rebuild from a ``failed_jobs.json`` entry."""
        return cls(
            structure_id=data["structure_id"],
            kind=FailureKind(data["kind"]),
            reason=data.get("reason", ""),
        )


@dataclass(frozen=True)
class StepResults:
    """What ``engine.parse`` produces — one entry per output file."""

    structures: tuple[Structure, ...]

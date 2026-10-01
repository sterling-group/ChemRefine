"""Hierarchical structure-ID allocation and canonical filenames.

ChemRefine tracks every conformer through the pipeline with a string ID
that records its lineage. The seed structures of step 1 get plain integer
IDs (``"0"``, ``"1"``, ...). Whenever a step expands one parent into
multiple children (e.g. a GOAT ensemble), the children inherit a
hyphen-suffixed ID: ``"0-1"`` means "child 1 of parent 0",
``"0-1-2"`` means "child 2 of that branch", and so on. The lineage
itself is carried by :attr:`chemrefine.state.Structure.parent_id`; the
hyphenated form here is just the display convention used for filenames
and grep-friendliness.

The functions here own three concerns:

1. Allocate IDs for new children (:func:`allocate_child_ids`) —
   engines compute their per-parent fan-out and mint child IDs here.
2. Build the canonical per-structure artifact paths
   (:func:`structure_artifact_path`, :func:`input_geometry_path`) — each
   structure gets its own ``step_dir/{ID}/`` directory holding
   ``step{N}_{ID}.{ext}`` (and ``step{N}_{ID}_inp.xyz`` for the input
   geometry) — and spell one relative to its step (:func:`step_relative`).
   (IDs travel in the step manifest, never re-parsed out of
   filenames — note that NMS child IDs like ``0_m5_pos`` contain letters,
   so a filename is not a reliable place to recover an ID from.)
3. Resolve a step's input *template* path (:func:`step_template_path`,
   :func:`default_template_name`) and read it, or refuse a step that has none or cannot
   be read (:func:`require_template`) — the ``step{N}.{ext}`` template-naming
   convention, kept here beside the artifact-path convention so every
   canonical ChemRefine filename lives in one module. Whatever consumes the text asks
   :func:`require_template` (the t=0 walk, validate, the engine rendering or inspecting
   it); the cache never does, because it digests bytes and must answer for an absent
   template.
4. Name the output tree's bookkeeping: a step's ``_cache/`` (:func:`cache_dir`), the
   cumulative report every step appends to (:func:`steps_report_path`), and where a run
   started for an agent or the GUI logs (:func:`agent_runs_dir`) — each spelled once, so
   a reader cannot look for a name its writer no longer uses.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from chemrefine import io
from chemrefine.errors import ConfigError

_ATTEMPT_DIR_RE = re.compile(r"attempt(\d+)$")

TRAINING_ID = "train"
"""The job id an artifact step's single job runs under, in place of a structure id.

A training job is one job over the whole ensemble rather than one per structure, but it wants
the same on-disk shape — its own directory under the step dir, holding its config, its script,
its runlog and its product — so it takes an id here like everything else. Minted in this
module because that is what makes it safe to interpolate into generated bash: every value that
reaches a job script is either quoted at the point of use or comes from here.

It is not a structure id and never collides with one: structure ids are digits and hyphens
(:func:`allocate_child_ids`), so no ensemble can produce this name.
"""


def allocate_child_ids(parents: Sequence[str], fanouts: Sequence[int]) -> list[str]:
    """Allocate persistent IDs for one step's children given parent IDs and per-parent fanout.

    Rules:

    * ``fanout == 0`` — the parent contributes nothing.
    * ``fanout == 1`` — the child inherits the parent ID unchanged.
    * ``fanout >= 2`` — each child gets ``"{parent}-{i}"`` for ``i`` in ``0..fanout-1``.
    """
    if len(parents) != len(fanouts):
        raise ValueError(
            f"parents and fanouts must have the same length: got {len(parents)} vs {len(fanouts)}"
        )
    children: list[str] = []
    for parent, fanout in zip(parents, fanouts, strict=True):
        if fanout < 0:
            raise ValueError(f"fanout must be >= 0; got {fanout} for parent {parent!r}")
        if fanout == 0:
            continue
        if fanout == 1:
            children.append(str(parent))
        else:
            for i in range(fanout):
                children.append(f"{parent}-{i}")
    return children


def structure_stem(step: int, structure_id: str) -> str:
    """The basename every one of a structure's artifacts is built on: ``step{N}_{id}``.

    The one place that spells the convention, so a layout change touches this line and the
    paths below follow. :func:`chemrefine.attempts.promote` swaps one stem for another when
    it renames a child's files to its parent's.
    """
    return f"step{step}_{structure_id}"


def structure_artifact_path(step_dir: Path, step: int, structure_id: str, ext: str) -> Path:
    """Canonical per-structure artifact path.

    Each structure gets its **own directory** under the step dir, so a
    calculation's files (input, output, optimized geometry, whatever the
    engine copies back from scratch) sit together and never collide across
    structures: ``step_dir/{structure_id}/step{step}_{structure_id}.{ext}``.
    Keeping the naming convention here means a layout change touches one file.
    """
    return step_dir / structure_id / f"{structure_stem(step, structure_id)}.{ext}"


def step_relative(path: Path, step_dir: Path) -> str:
    """How a step's file is spelled wherever it is named: relative to the step directory.

    Everything about a tree is addressed relatively — ``Config.step_dir`` derives from
    ``output_dir``, which resolves against the config file's own directory, and no key holds
    a path — so a tree copied off a cluster or moved to another directory keeps working
    everywhere. A file spelled absolute would name the machine the step ran on, and
    ``rebuild-cache``, a ledgered ``rerun`` or a failure reason on a copy would point at paths
    that are not there. Every shipped engine writes its jobs under the step directory, so this
    is the normal spelling (``0/step1_0.out``); a file elsewhere stays absolute.
    """
    return str(path.relative_to(step_dir)) if path.is_relative_to(step_dir) else str(path)


def step_ensemble_path(step_dir: Path, step: int) -> Path:
    """The step's multi-frame ensemble XYZ — every parsed result, pre-filter.

    Lives at the step dir's root rather than in a structure directory because it describes
    the whole step. It cannot collide with a structure's artifacts: those live inside per-id
    subdirectories, and no minted id (digits, hyphens, NMS ``_m…`` suffixes,
    :data:`TRAINING_ID`) is ever ``ensemble``.
    """
    return step_dir / f"step{step}_ensemble.xyz"


def step_survivors_path(step_dir: Path, step: int) -> Path:
    """The step's multi-frame survivors XYZ — what the ``sample:`` filter kept.

    The sibling of :func:`step_ensemble_path` for the post-filter set that feeds the next
    step; the pair is what makes a filter's effect visible as geometry rather than only as
    a row count in ``steps.csv``.
    """
    return step_dir / f"step{step}_survivors.xyz"


def result_record_path(job_dir: Path, step: int, structure_id: str) -> Path:
    """Canonical parsed-result record path for one structure of a job.

    Takes the **job** directory (an output file's parent), not the step dir:
    fan-out children of an ensemble job have no directory of their own, so
    their records land beside the sidecar that produced them
    (``step1/0/step1_0-3.result.json``), and NMS round-2 children get
    ``attemptK/<child>/…`` for free. Distinct from the script engines'
    native ``step{N}_{id}.json`` output.
    """
    return job_dir / f"{structure_stem(step, structure_id)}.result.json"


def cache_dir(step_dir: Path) -> Path:
    """A step's ``_cache/``: its results, manifest, failure ledger and lease ledger."""
    return step_dir / "_cache"


def steps_report_path(output_dir: Path) -> Path:
    """The cumulative per-step report every step appends to."""
    return output_dir / "steps.csv"


def agent_runs_dir(output_dir: Path) -> Path:
    """Where a run started for an agent or the GUI writes its log."""
    return output_dir / "agent_runs"


def input_geometry_path(step_dir: Path, step: int, structure_id: str) -> Path:
    """Path of a structure's **input** geometry, distinct from any output xyz.

    Named ``step{step}_{structure_id}_inp.xyz`` (an ``_inp`` stem) so it never
    shares a name with an engine's *output* geometry (an engine may write its
    optimized geometry to ``step{step}_{structure_id}.xyz``); both survive the
    copy-back into the structure's directory.
    """
    return step_dir / structure_id / f"{structure_stem(step, structure_id)}_inp.xyz"


def _attempt_numbers(structure_dir: Path) -> list[tuple[int, Path]]:
    """Every ``attempt<n>/`` under a structure's dir as ``(n, path)``, unordered.

    Shared by the next-free and most-recent lookups so one scan defines what counts.
    A non-matching ``attempt*`` entry — a stray file, say — is ignored.
    """
    return [
        (int(m.group(1)), d)
        for d in structure_dir.glob("attempt*")
        if d.is_dir() and (m := _ATTEMPT_DIR_RE.fullmatch(d.name))
    ]


def next_attempt_dir(structure_dir: Path) -> Path:
    """Return the next free ``attemptK/`` sub-directory under a structure's dir.

    ``K`` is one past the highest existing ``attempt<n>`` (``attempt1`` if none), so
    a re-run — or a manually-added ``attempt2/`` — never collides with, or is blocked
    by, an existing one. This is the shared "attempt" primitive of the unified
    resolution model: the convergence retry moves a failed attempt here, and NMS
    archives a structure's exploration here. The directory is **not** created here
    (path only); a non-matching ``attempt*`` entry (e.g. a stray file) is ignored.
    """
    numbered = _attempt_numbers(structure_dir)
    return structure_dir / f"attempt{max((n for n, _ in numbered), default=0) + 1}"


def is_attempt_dir(path: Path) -> bool:
    """Whether ``path`` is an ``attempt<n>/`` directory of the resolution model.

    The membership test behind the two lookups above, so "what counts as an attempt" is
    decided in one place. :func:`chemrefine.attempts.seal` asks it to
    know what *not* to move: everything else in a structure dir is that attempt's output and
    goes with it, but folding one attempt into another would lose a run's history.
    """
    return path.is_dir() and _ATTEMPT_DIR_RE.fullmatch(path.name) is not None


def attempt_dirs_newest_first(structure_dir: Path) -> list[Path]:
    """Every existing ``attemptK/`` under a structure's dir, highest number first.

    **The one ordering of a structure's attempts.** Attempt order is the number in the
    name — :func:`next_attempt_dir` mints them ascending — and nothing else: a modification
    time is copy order on a tree that was ``cp -r``'d off a cluster, which is the very tree
    the readers of this list exist for. :func:`latest_attempt_dir` is its first element;
    :func:`chemrefine.agent_tools._locate_output` walks it when the canonical output is gone.
    """
    numbered = _attempt_numbers(structure_dir)
    return [path for _n, path in sorted(numbered, reverse=True)]


def latest_attempt_dir(structure_dir: Path) -> Path | None:
    """Return the highest-numbered existing ``attemptK/`` under a structure's dir.

    The read-side counterpart of :func:`next_attempt_dir`: ``rebuild-cache`` reads a
    structure's most recent attempt (e.g. an NMS exploration) from here. ``None`` when
    no ``attempt<n>/`` exists.
    """
    attempts = attempt_dirs_newest_first(structure_dir)
    return attempts[0] if attempts else None


def default_template_name(step: int, suffix: str) -> str:
    """Return the default per-step template basename ``step{step}.{suffix}``.

    The single source of truth for the ``step{N}.<ext>`` template-naming
    convention; every engine resolves its template through here so a future
    layout change touches one place.
    """
    return f"step{step}.{suffix}"


def step_template_path(
    template_dir: Path,
    step: int,
    *,
    template: str | None,
    suffix: str,
) -> Path:
    """Where a step's input template lives — the step's override, else the default name.

    Naming only: the file need not exist. Two different questions are asked about a step
    template and they belong to different callers, so they are two functions. *Where is it*
    is this one, and :func:`chemrefine.step.build_context` asks it once per step so the
    answer travels on the :class:`~chemrefine.state.StepContext` instead of being re-derived.
    *Read it, or refuse* is :func:`require_template`, and only a consumer of the text asks
    that — the cache, which merely digests it, must not, or a step whose template is absent
    could never be re-fingerprinted at all.
    """
    return template_dir / (template or default_template_name(step, suffix))


@dataclass(frozen=True)
class StepTemplate:
    """A step's input template as :func:`require_template` read it: where it is, and its text.

    Read once and handed on, so a consumer renders or inspects the text the read accepted;
    the path stays beside it for what is relative to the template and for a refusal to name
    the file.
    """

    path: Path
    text: str


def require_template(template: Path | None, *, label: str) -> StepTemplate:
    """Read ``template``, or raise :class:`~chemrefine.errors.ConfigError` if it is unusable.

    The one home for "this engine cannot run without its input". ``None`` means the engine
    declared no template at all (it is not
    :class:`~chemrefine.engines.api.TemplateDriven`). Otherwise the template is read by the
    reader every user file goes through (:func:`chemrefine.io.read_utf8`), which refuses one
    that is missing — the likeliest error of a first run — unreadable, or not UTF-8 text, so
    the run's t=0 walk refuses the template its ``prepare`` would. Returns what it read, so a
    consumer hands the text on instead of reading the file again.
    """
    if template is None:
        raise ConfigError(f"{label} needs an input template, but this engine declares none")
    return StepTemplate(template, io.read_utf8(template, what=f"{label} template"))

"""Engine-independent normal-mode sampling (the unified "attempt" model).

NMS is a generic capability, not an engine feature. This module owns the whole
two-round algorithm — read each structure's imaginary modes, displace along them,
re-optimise the ± children, resolve to a stationary point — and drives the compute
engine *only* through :class:`chemrefine.engines.api.NmsCapableEngine` (its one
``nms_input_info`` hook plus the standard lifecycle). The frequency *values* it reads come
off each ``Structure`` (``imaginary_freqs`` / ``normal_modes``, parsed in the same pass as
geometry), so it imports no engine package and never re-parses an output.

Unified "attempt" model (shared with the on_failure convergence retry): a structure
lives at ``stepN/<id>/``; resolving it is an *attempt* whose displaced ± re-opts run
under ``stepN/<id>/attemptK/<child>/`` and whose **winner stays at the canonical
``stepN/<id>/`` with the id unchanged** — for ``minimum``/``ts`` the single best
resolved geometry is written back there, so a structure keeps its identity and never
spawns duplicate minima. ``random`` is the exception: with no resolution gate it is
pure exploration, so it fans out to new child structures.

The displacement maths + :class:`NmsOptions` here are pure and side-effect-free; the
coordinator (`run_nms` / `rebuild_nms` / `reattempt_nms`) does the I/O, reusing the
shared retry helper in :mod:`chemrefine.lifecycle` for unconverged children.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from ase import Atoms
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from chemrefine import attempts, cache, filtering, io, lifecycle
from chemrefine.config import StepConfig
from chemrefine.engines.api import NmsCapableEngine
from chemrefine.errors import CacheError, ConfigError
from chemrefine.ids import (
    latest_attempt_dir,
    next_attempt_dir,
    structure_artifact_path,
)
from chemrefine.state import (
    Failure,
    FailureKind,
    PipelineState,
    StepContext,
    StepInputs,
    StepResults,
    Structure,
)

logger = logging.getLogger(__name__)

_TRIVIAL_MODES = 6
"""Leading modes of the ``normal_modes`` tensor that are not vibrations.

A frequency table lists all ``3N`` modes with the translations and rotations first — six of
them for a non-linear molecule, five for a linear one. ``random`` sampling skips them, since
displacing along a translation just moves the molecule and re-computes the same energy.

Six for every molecule, deliberately. Distinguishing the linear case means deciding whether a
geometry is collinear to some tolerance, and being wrong in the permissive direction spends a
job on a translated copy. Being wrong the conservative way — which this is — costs a linear
molecule one candidate vibration out of a random draw. `minimum` and `ts` are unaffected:
they displace along the modes the parse flagged imaginary, whatever their index."""


# ---------------------------------------------------------------------------
# Options + pure displacement maths (engine-independent)
# ---------------------------------------------------------------------------


class NmsOptions(BaseModel):
    """Validated normal-mode-sampling knobs (from ``step.options``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target: Literal["minimum", "ts", "random"] = "minimum"
    """``minimum`` removes all imaginary modes; ``ts`` keeps the reaction coordinate
    and removes the rest; ``random`` displaces along random modes (exploration)."""

    displacement_value: float = 1.0
    """Magnitude (Å) of the ± displacement along each selected mode."""

    num_random_displacements: int = Field(1, ge=1)
    """``random`` only: how many modes to draw."""

    ts_mode_index: int | None = None
    """``ts`` only: explicit reaction-coordinate mode index. ``None`` ⇒ the
    largest-magnitude imaginary mode."""

    seed: int = 42
    """Deterministic seed for ``random`` mode selection."""

    @classmethod
    def from_raw(cls, raw: dict[str, Any] | None) -> NmsOptions:
        """Validate the NMS subset of a ``step.options`` dict (ignoring other keys)."""
        raw = raw or {}
        known = {k: raw[k] for k in cls.model_fields if k in raw}
        return cls(**known)


def target_imaginary_count(opts: NmsOptions) -> int | None:
    """Imaginary-mode count that means *resolved* for this target.

    ``minimum`` → 0, ``ts`` → 1, ``random`` → ``None`` (no resolution gate; random
    sampling is exploration, not stationary-point cleanup).
    """
    if opts.target == "minimum":
        return 0
    if opts.target == "ts":
        return 1
    return None


def displace_along_mode(
    positions_angstrom: NDArray[np.float64],
    mode_vector: NDArray[np.float64],
    *,
    displacement: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(positions + d·v, positions - d·v)`` (same shape as the inputs)."""
    if positions_angstrom.shape != mode_vector.shape:
        raise ValueError(
            f"position / mode shape mismatch: {positions_angstrom.shape} vs {mode_vector.shape}"
        )
    pos = positions_angstrom + displacement * mode_vector
    neg = positions_angstrom - displacement * mode_vector
    return pos, neg


def rng_for(structure_id: str, seed: int) -> np.random.Generator:
    """A per-structure RNG, so ``random`` mode selection cannot depend on loop position.

    Seeding from the structure id makes the draw a pure function of ``(seed, id)``. One
    generator drawn across the whole loop would instead make each structure's modes depend
    on how many structures preceded it, and ``run_nms`` and ``rebuild_nms`` do not visit
    the same set — only the rebuild skips a structure with no ``attemptK/`` on disk. A
    single skip would shift the stream for everything after it, so ``rebuild-cache`` would
    re-derive children that were never computed and report a resolved structure as
    unresolved.
    """
    return np.random.default_rng([seed, *structure_id.encode()])


def _reaction_coordinate(imag_freqs: dict[int, float], opts: NmsOptions) -> int:
    """The mode to *keep* for a TS: ``ts_mode_index`` or the most-imaginary.

    An explicit index must name a mode that is actually imaginary. It cannot be checked at
    config time — which modes are imaginary is a property of each structure's frequency
    calculation, not of the YAML — so it is checked here, against the structure at hand.

    The check matters because the failure is otherwise silent and inverted: the exclusion
    is written as a filter (``if i != rc``), so an index naming no imaginary mode simply
    excludes nothing, and NMS displaces along *every* imaginary mode — including the
    reaction coordinate the setting exists to preserve. The children then come back as
    minima, fail the ``target == 1`` test, and the whole step reports "target stationary
    point not reached" after paying for the full round-2 batch: a chemistry message for an
    off-by-one.
    """
    if opts.ts_mode_index is None:
        return max(imag_freqs, key=lambda i: abs(imag_freqs[i]))
    if opts.ts_mode_index not in imag_freqs:
        raise ConfigError(
            f"nms ts_mode_index={opts.ts_mode_index} is not an imaginary mode of this "
            f"structure; its imaginary modes are {sorted(imag_freqs)}. Name one of those, "
            f"or drop ts_mode_index to keep the most imaginary mode."
        )
    return opts.ts_mode_index


def _selected_modes(
    imag_freqs: dict[int, float], n_modes: int, opts: NmsOptions, rng: np.random.Generator
) -> list[int]:
    """Mode indices to displace along, per ``target``."""
    if opts.target == "random":
        candidates = list(range(_TRIVIAL_MODES, n_modes))
        if not candidates:
            return []
        k = min(opts.num_random_displacements, len(candidates))
        return sorted(int(i) for i in rng.choice(candidates, size=k, replace=False))
    if not imag_freqs:
        return []
    if opts.target == "ts":
        rc = _reaction_coordinate(imag_freqs, opts)
        return [i for i in sorted(imag_freqs) if i != rc]
    return sorted(imag_freqs)  # minimum: every imaginary mode


def select_displacements(
    struct: Structure,
    imag_freqs: dict[int, float],
    modes: NDArray[np.float64],
    opts: NmsOptions,
    rng: np.random.Generator,
) -> list[tuple[str, NDArray[np.float64]]]:
    """Return ``[(suffix, displaced_positions), ...]`` child geometries.

    One ``±`` pair per selected mode; ``suffix`` (e.g. ``m5_pos``) makes the child id
    ``{parent}_{suffix}`` unique and traceable to its mode.
    """
    positions = struct.atoms.get_positions()
    n_modes = modes.shape[2]
    out: list[tuple[str, NDArray[np.float64]]] = []
    for idx in _selected_modes(imag_freqs, n_modes, opts, rng):
        if idx >= n_modes:
            logger.warning(
                "NMS %s: mode %d outside the %d-mode tensor; skipping", struct.id, idx, n_modes
            )
            continue
        mode = modes[:, :, idx]
        if mode.shape != positions.shape:
            logger.warning("NMS %s: mode %d shape mismatch; skipping", struct.id, idx)
            continue
        pos, neg = displace_along_mode(positions, mode, displacement=opts.displacement_value)
        out.append((f"m{idx}_pos", pos))
        out.append((f"m{idx}_neg", neg))
    return out


@dataclass(frozen=True)
class NmsResolution:
    """Outcome of NMS resolution: the resolved survivors plus unresolved failures.

    The coordinators below return this; the step lifecycle then applies the ``on_failure``
    policy to ``failures`` exactly as for a plain step, so NMS reuses the same handling.
    """

    survivors: tuple[Structure, ...]
    failures: tuple[Failure, ...]


# ---------------------------------------------------------------------------
# The two-round coordinator (drives the engine through NmsCapableEngine)
# ---------------------------------------------------------------------------


def _resolved_options(engine: NmsCapableEngine, ctx: StepContext) -> NmsOptions:
    """NMS options with ``target`` resolved (F1): explicit wins, else inferred.

    When the step doesn't set ``options.target``, derive it from the engine's input —
    a TS search targets a ``ts`` (keep one imaginary mode), anything else a ``minimum``.
    """
    raw = ctx.step_cfg.options or {}
    opts = NmsOptions.from_raw(raw)
    if "target" not in raw:
        is_ts = engine.nms_input_info(ctx).is_transition_state
        opts = opts.model_copy(update={"target": "ts" if is_ts else "minimum"})
    return opts


def _displaced(structure: Structure, positions: NDArray[np.float64]) -> Atoms:
    """A copy of ``structure``'s atoms moved to ``positions`` (a displaced child geometry)."""
    atoms: Atoms = structure.atoms.copy()
    atoms.set_positions(positions)
    return atoms


def _energy_attr(step_cfg: StepConfig) -> str:
    """The :class:`~chemrefine.state.Structure` attribute this step ranks structures by.

    The step's own ``sample.energy_type``, through :data:`chemrefine.filtering.ENERGY_ATTR` —
    the single home of that mapping. A step with no ``sample`` filter has declared no
    preference, so electronic energy, matching what
    :func:`chemrefine.pipeline._write_step_csv` reports in the same situation.
    """
    sample = step_cfg.sample
    return filtering.ENERGY_ATTR["electronic" if sample is None else sample.energy_type]


def _best(structures: list[Structure], fallback: Structure, energy_attr: str) -> Structure:
    """The lowest-``energy_attr`` structure (``None`` sorts last), or ``fallback`` if empty.

    ``energy_attr`` is the step's own ranking energy (see :func:`_energy_attr`), not always the
    electronic one. When several round-2 children reach the target, the one carried forward has
    to be the one the step's filter would have kept — a TS step sampling on ``gibbs`` picked its
    winner on electronic energy and could therefore promote a child that the very next filter
    would have discarded.
    """
    if not structures:
        return fallback
    return min(
        structures,
        key=lambda s: (getattr(s, energy_attr) is None, getattr(s, energy_attr) or 0.0),
    )


def _is_resolved(child: Structure, target: int | None) -> bool:
    """Whether a round-2 child reached the target (terminated normally + matching imaginary count).

    ``target is None`` (random) accepts any normally-terminated child. A child whose parse found no
    frequency table (``imaginary_freqs is None``) is never resolved — without freq evidence,
    zero imaginary modes can't be claimed (only counted-and-zero is a verified minimum). The
    frequencies were parsed onto the child in the same pass as its geometry, so this is a
    field read — no second parse of the ``.out``.
    """
    if child.terminated_normally is False:
        return False
    if target is None:
        return True
    return child.imaginary_freqs is not None and len(child.imaginary_freqs) == target


def _children_of(
    structure: Structure, displacements: list[tuple[str, NDArray[np.float64]]]
) -> list[Structure]:
    """Build the displaced ± child structures for one round-1 structure."""
    return [
        Structure(
            id=f"{structure.id}_{suffix}",
            atoms=_displaced(structure, positions),
            parent_id=structure.id,
        )
        for suffix, positions in displacements
    ]


class _AttemptMode(Protocol):
    """How one parent's attempt is reached: located, populated, and concluded.

    :func:`run_nms` and :func:`rebuild_nms` differ in exactly these three answers and in
    nothing else, so :func:`_resolve_all` is the only resolution loop and they agree by
    construction. Two loops making the same twelve decisions agree only as long as every
    edit touches both.

    A Protocol rather than a flag or a record of callables: ``install`` is behaviour, and
    :mod:`chemrefine.engines.api` sets the rule that capabilities are never a flag.

    **Private, and never a parameter of the public functions.** The same choice exists
    upstream as :meth:`chemrefine.step.StepMode.may_submit`, so a mode a caller could
    *pass* would be a second vocabulary for it — another pair to keep in step. ``run_nms``
    and ``rebuild_nms`` each name their own mode tautologically; nobody chooses one.
    """

    def attempt_dir(self, structure_dir: Path) -> Path | None:
        """The attempt directory to use, or ``None`` when there is none to work with."""
        ...

    def obtain(
        self,
        engine: NmsCapableEngine,
        children: list[Structure],
        ctx: StepContext,
        attempt: Path,
    ) -> list[Structure]:
        """The round-2 children's parsed results."""
        ...

    def install(
        self,
        survivor: Structure,
        source_id: str,
        parent_id: str,
        ctx: StepContext,
        attempt: Path,
    ) -> None:
        """Put the winner's calculation at the parent's canonical path, if that applies."""
        ...


class _RunAttempt:
    """Submit round 2 into a fresh attempt and promote the winner."""

    def attempt_dir(self, structure_dir: Path) -> Path | None:
        """A fresh ``attemptK/`` — never ``None``, so the shared guard's other half is moot."""
        return next_attempt_dir(structure_dir)

    def obtain(
        self,
        engine: NmsCapableEngine,
        children: list[Structure],
        ctx: StepContext,
        attempt: Path,
    ) -> list[Structure]:
        """Submit + parse the displaced children under ``attempt``; retry unconverged ones."""
        child_ctx = replace(ctx, step_dir=attempt)
        succ, fail = lifecycle.submit_and_parse(engine, child_ctx, children)
        succ, _fail = lifecycle.retry_unconverged(engine, child_ctx, succ, fail)
        return succ

    def install(
        self,
        survivor: Structure,
        source_id: str,
        parent_id: str,
        ctx: StepContext,
        attempt: Path,
    ) -> None:
        """Promote the winning child to the parent's canonical basenames."""
        _install_winner(survivor, source_id, parent_id, ctx, attempt)


class _RebuildAttempt:
    """Re-read the latest attempt from disk, changing nothing (``rebuild-cache``)."""

    def attempt_dir(self, structure_dir: Path) -> Path | None:
        """The most recent ``attemptK/``, or ``None`` for children re-derivable but never run."""
        return latest_attempt_dir(structure_dir)

    def obtain(
        self,
        engine: NmsCapableEngine,
        children: list[Structure],
        ctx: StepContext,
        attempt: Path,
    ) -> list[Structure]:
        """Parse already-on-disk child outputs under ``attempt`` — no submission.

        The engine says where its files are; a child whose output is absent is simply left
        out, so its parent stays unresolved rather than the rebuild failing.
        """
        child_ctx = replace(
            ctx, step_dir=attempt, prev_state=PipelineState(structures=tuple(children))
        )
        paths = ((c.id, engine.artifact_paths(child_ctx, c.id)) for c in children)
        files = tuple((inp, out, cid) for cid, (inp, out) in paths if out.is_file())
        succ, _fail = lifecycle.parse_with_failures(engine, StepInputs(files=files), child_ctx)
        return succ

    def install(
        self,
        survivor: Structure,
        source_id: str,
        parent_id: str,
        ctx: StepContext,
        attempt: Path,
    ) -> None:
        """Nothing: a rebuild must not rewrite the outputs it was asked to read.

        Not an omission but the invariant, held by the type rather than by a caller
        remembering it. The winner's artifacts are already at the canonical path, put there
        by the run that promoted them; re-promoting would rewrite the very records a later
        comparison is meant to trust.
        """


_RESOLUTION_FILE = "resolution.json"
"""Sidecar naming the child an attempt resolved to, written inside that ``attemptK/``."""


def _write_resolution(attempt_dir: Path, winner_id: str) -> None:
    """Record which child this attempt resolved to.

    Written last, after the winner's artifacts are in place, so the claim never outlives the
    files it describes.
    """
    cache.write_json(attempt_dir / _RESOLUTION_FILE, {"resolved_from": winner_id})


def _read_resolution(structure_dir: Path) -> str | None:
    """The child a structure's most recent attempt resolved to, or ``None`` if none did.

    Promotion leaves the winner's output at the canonical path, so re-parsing a resolved
    structure yields one already at its target — which is the answer, but not *which child
    produced it*. That is only on disk, here.
    """
    attempt = latest_attempt_dir(structure_dir)
    if attempt is None:
        return None
    record = cache.read_json(attempt / _RESOLUTION_FILE, None, label="NMS resolution")
    return None if record is None else str(record["resolved_from"])


def _select_survivors(
    resolved: list[Structure],
    round2: list[Structure],
    parent: Structure,
    step_cfg: StepConfig,
    target: int | None,
) -> tuple[list[Structure], list[Failure]]:
    """Decide what a parent's round-2 children amount to. Pure — reads no disk, writes none.

    ``random`` (``target is None``) keeps every resolved child as a fan-out structure;
    ``minimum``/``ts`` collapse to the single best resolved geometry **at the parent's
    canonical id**, recording which child that was in
    :attr:`~chemrefine.state.Structure.resolved_from`. A parent with nothing resolved becomes
    one :class:`~chemrefine.state.Failure` carrying the best geometry it did obtain.

    "Best" is by the step's own ranking energy throughout — see :func:`_energy_attr`.

    Takes the step config rather than the whole :class:`~chemrefine.state.StepContext`
    because the ranking energy is all it needs, and a signature that asks for the
    scheduler's directories, dispatch mode and core budget makes the purity claim above
    something a reader has to verify rather than read.
    """
    energy_attr = _energy_attr(step_cfg)
    if target is None:  # random: the children are the (fan-out) results
        if resolved:
            return resolved, []
        return [], [
            Failure(parent.id, FailureKind.UNRESOLVED_NMS, _best(round2, parent, energy_attr))
        ]
    if not resolved:
        return [], [
            Failure(parent.id, FailureKind.UNRESOLVED_NMS, _best(round2, parent, energy_attr))
        ]
    winner = _best(resolved, parent, energy_attr)
    survivor = replace(winner, id=parent.id, parent_id=parent.parent_id, resolved_from=winner.id)
    return [survivor], []


def _install_winner(
    survivor: Structure, source_id: str, parent_id: str, ctx: StepContext, attempt_dir: Path
) -> None:
    """Put the chosen child's calculation at the parent's canonical path.

    ``source_id`` is the child :func:`_select_survivors` chose, taken from the survivor's
    :attr:`~chemrefine.state.Structure.resolved_from` — so there is one answer to "which
    child won" and the disk follows it rather than re-deciding.

    Archive first, promote second: the copy lands on the basenames round 1 still occupies,
    so moving those aside is what stops one calculation overwriting another. The resolution
    sidecar is written last, after the artifacts it describes are in place.
    """
    step = ctx.step_cfg.step
    attempts.seal(ctx.step_dir / parent_id, attempt_dir)
    attempts.promote(attempt_dir, step=step, source_id=source_id, target_id=parent_id)
    # Re-stamp the geometry file with its provenance: the same coordinates the promoted
    # `.out` reports — `survivor.atoms` was parsed from it — now saying how it was reached.
    io.write_single_xyz(
        survivor.atoms,
        structure_artifact_path(ctx.step_dir, step, parent_id, "xyz"),
        comment=f"NMS-resolved {parent_id}",
    )
    _write_resolution(attempt_dir, source_id)


def _already_at_target(structure: Structure, target: int | None) -> bool:
    """Whether this structure needs no displacement at all.

    ``target is None`` (``random``) never short-circuits — it is exploration, not cleanup —
    and a structure whose parse found no frequency table cannot claim to be anywhere.
    """
    return (
        target is not None
        and structure.imaginary_freqs is not None
        and len(structure.imaginary_freqs) == target
    )


def _passthrough(structure: Structure, step_dir: Path) -> Structure:
    """A structure already at its target, carried through at the canonical place.

    That is either an untouched round-1 result or one this step already resolved; the
    attempt's sidecar is what tells the two apart — which is the one thing this needs a
    directory for, so that is what it asks for.
    """
    return replace(
        structure,
        converged=True,
        resolved_from=_read_resolution(step_dir / structure.id),
    )


def _children_for(structure: Structure, opts: NmsOptions) -> list[Structure]:
    """The displaced ± children to try for one structure; empty when there are none.

    Empty means the structure cannot be resolved without new information — the parse found no
    normal-mode tensor, or no mode qualified for displacement under this target. Both
    coordinators ask this one question, so neither can decide it differently from the other.
    """
    if structure.normal_modes is None:
        logger.warning("NMS %s: no normal-mode tensor; cannot displace (unresolved)", structure.id)
        return []
    return _children_of(
        structure,
        select_displacements(
            structure,
            structure.imaginary_freqs or {},
            structure.normal_modes,
            opts,
            rng_for(structure.id, opts.seed),
        ),
    )


def _resolve_all(
    engine: NmsCapableEngine,
    round1: StepResults,
    round1_failures: list[Failure] | tuple[Failure, ...],
    ctx: StepContext,
    mode: _AttemptMode,
) -> NmsResolution:
    """Resolve every round-1 survivor through ``mode``. The one NMS loop.

    For each round-1 structure: if it is already at the target it passes through at the
    canonical place; otherwise its ± displaced children are obtained under
    ``stepN/<id>/attemptK/`` and the best resolved geometry becomes the survivor at the
    canonical id (``random`` fans out instead). Round-1 jobs that already failed are carried
    through unchanged.

    The attempt is located before the children are checked, which is safe because both
    lookups are pure on a directory that does not exist yet. That ordering lets one guard
    cover both modes: a run's ``attempt`` is never ``None``, so for it the second disjunct
    is dead, and for a rebuild it is the "re-derivable but never ran" case.
    """
    opts = _resolved_options(engine, ctx)
    target = target_imaginary_count(opts)
    survivors: list[Structure] = []
    failures: list[Failure] = list(round1_failures)
    for s in round1.structures:
        if _already_at_target(s, target):
            survivors.append(_passthrough(s, ctx.step_dir))
            continue
        children = _children_for(s, opts)
        attempt = mode.attempt_dir(ctx.step_dir / s.id)
        if not children or attempt is None:
            failures.append(Failure(s.id, FailureKind.UNRESOLVED_NMS, s))
            continue
        round2 = mode.obtain(engine, children, ctx, attempt)
        resolved = [c for c in round2 if _is_resolved(c, target)]
        s_surv, s_fail = _select_survivors(resolved, round2, s, ctx.step_cfg, target)
        # A promoted winner is the only case with a `resolved_from`: `random` fans out and
        # an unresolved parent has no survivor, and neither installs anything.
        winner_source = s_surv[0].resolved_from if s_surv else None
        if winner_source is not None:
            mode.install(s_surv[0], winner_source, s.id, ctx, attempt)
        survivors.extend(s_surv)
        failures.extend(s_fail)
    return NmsResolution(tuple(survivors), tuple(failures))


def run_nms(
    engine: NmsCapableEngine,
    round1: StepResults,
    round1_failures: list[Failure] | tuple[Failure, ...],
    ctx: StepContext,
) -> NmsResolution:
    """Resolve each round-1 survivor to its stationary point (submits round-2).

    Displaced children run under a fresh ``stepN/<id>/attemptK/`` and the winner is promoted
    to the parent's canonical basenames.
    """
    return _resolve_all(engine, round1, round1_failures, ctx, _RunAttempt())


def rebuild_nms(
    engine: NmsCapableEngine,
    round1: StepResults,
    round1_failures: list[Failure] | tuple[Failure, ...],
    ctx: StepContext,
) -> NmsResolution:
    """Re-resolve NMS from outputs already on disk — no submission (``rebuild-cache``).

    Re-derives the (deterministic) displaced children and parses their existing round-2
    outputs from the structure's latest ``attemptK/``; a child with no output on disk is
    simply absent, so its parent stays unresolved, and nothing on disk is rewritten.
    """
    return _resolve_all(engine, round1, round1_failures, ctx, _RebuildAttempt())


def reattempt_nms(
    engine: NmsCapableEngine,
    ctx: StepContext,
    step_cfg: StepConfig,
    cached: cache.StepCache,
    key: cache.StepKey,
) -> StepResults:
    """Re-attempt only the ledgered-unresolved NMS parents, reusing round-1.

    The still-valid resolved survivors from the old cache are kept; the failed parents'
    round-1 outputs are re-parsed (missing ones resubmitted, unconverged ones retried),
    NMS round-2 is re-run for them, and the merged result is re-cached.

    **Why it archives and regenerates nothing on entry**, where
    :func:`chemrefine.step._resubmit_failed` does both before resubmitting. Reusing round 1 is
    the point of this path, and archiving it up front would defeat that. Two facts make the
    omission safe rather than lucky:

    * Only ``MISSING_OUTPUT`` ids are resubmitted. Archiving exists to stop a re-executed job
      re-reading the previous run's output as if it were its own, and a structure with no
      output has nothing to re-read.
    * :func:`chemrefine.cache.reuse_fingerprint` covers ``template_digest``, so an edited
      template changes the key gating this path — a stale input can never be resubmitted from
      the manifest.

    Round 1 *is* archived later, for a resolved parent, by
    :func:`_install_winner` sealing it into the attempt its children ran in. That is the
    resolution's doing, not this function's, and it happens only once there is a winner to put
    at the canonical path.
    """
    manifest = cache.load_manifest(ctx.step_dir)
    if manifest is None:
        raise CacheError(f"step {step_cfg.step}: cannot re-attempt NMS — no manifest on disk")
    failed = cache.load_failure_records(ctx.step_dir)
    failed_ids = {f.structure_id for f in failed}
    missing_ids = {f.structure_id for f in failed if f.kind is FailureKind.MISSING_OUTPUT}

    failed_manifest = StepInputs(files=tuple(f for f in manifest.files if f[2] in failed_ids))
    missing_inputs = StepInputs(
        files=tuple(f for f in failed_manifest.files if f[2] in missing_ids)
    )
    if missing_inputs.files:
        logger.info(
            "step %d: NMS re-attempt resubmitting %d missing round-1 job(s)",
            step_cfg.step,
            len(missing_inputs.files),
        )
        engine.submit(missing_inputs, ctx)

    r1_succ, r1_fail = lifecycle.parse_and_record(engine, failed_manifest, ctx)
    r1_succ, r1_fail = lifecycle.retry_unconverged(engine, ctx, r1_succ, r1_fail)
    reattempt = run_nms(engine, StepResults(structures=tuple(r1_succ)), r1_fail, ctx)
    kept = tuple(
        s
        for s in cached.results.structures
        if s.id not in failed_ids and s.parent_id not in failed_ids
    )
    return lifecycle.finalize(
        engine,
        ctx,
        step_cfg,
        key,
        list(kept + reattempt.survivors),
        list(reattempt.failures),
    )

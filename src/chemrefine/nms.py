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
shared retry helper in :mod:`chemrefine.step_failures` for unconverged children.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
from ase import Atoms
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from chemrefine import cache, filtering, io, step_failures
from chemrefine.config import StepConfig
from chemrefine.engines.api import NmsCapableEngine
from chemrefine.errors import CacheError
from chemrefine.ids import latest_attempt_dir, next_attempt_dir, structure_artifact_path
from chemrefine.state import PipelineState, StepContext, StepInputs, StepResults, Structure

logger = logging.getLogger(__name__)

# ORCA-style frequency tables print 6 (5 for linear) trivial translation/rotation
# modes at low index; ``random`` sampling skips them.
_TRIVIAL_MODES = 6


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
    """A per-structure RNG, so ``random`` mode selection can't depend on loop position.

    One generator shared across the structure loop made each structure's draw depend on
    how many structures preceded it — and :func:`run_nms` and :func:`rebuild_nms` skip on
    *different* conditions (only the rebuild skips a structure with no ``attemptK/`` on
    disk). One skip shifted the stream for every structure after it, so ``rebuild-cache``
    re-derived children that were never computed, found no outputs for them, and reported
    a resolved structure as unresolved.

    Seeding from the structure id makes the draw a pure function of ``(seed, id)``: the
    two coordinators agree whatever either one skips.
    """
    return np.random.default_rng([seed, *structure_id.encode()])


def _reaction_coordinate(imag_freqs: dict[int, float], opts: NmsOptions) -> int:
    """The mode to *keep* for a TS: ``ts_mode_index`` or the most-imaginary."""
    if opts.ts_mode_index is not None:
        return opts.ts_mode_index
    return max(imag_freqs, key=lambda i: abs(imag_freqs[i]))


def _selected_modes(
    imag_freqs: dict[int, float], n_modes: int, opts: NmsOptions, rng: np.random.Generator
) -> list[int]:
    """Mode indices to displace along, per ``target``."""
    if opts.target == "random":
        lo = min(_TRIVIAL_MODES, n_modes)
        candidates = list(range(lo, n_modes)) or list(range(n_modes))
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


def _run_round_two(
    engine: NmsCapableEngine, children: list[Structure], ctx: StepContext, attempt_dir: Path
) -> list[Structure]:
    """Submit + parse the displaced children under ``attempt_dir``; retry unconverged ones."""
    child_ctx = replace(
        ctx, step_dir=attempt_dir, prev_state=PipelineState(structures=tuple(children))
    )
    inputs = engine.prepare(child_ctx)
    engine.submit(inputs, child_ctx)
    succ, fail = step_failures.parse_with_failures(engine, inputs, child_ctx)
    succ, _fail = step_failures.retry_unconverged(engine, child_ctx, succ, fail)
    return succ


def _parse_round_two(
    engine: NmsCapableEngine, children: list[Structure], ctx: StepContext, attempt_dir: Path
) -> list[Structure]:
    """Parse already-on-disk child outputs under ``attempt_dir`` (rebuild — no submit)."""
    step = ctx.step_cfg.step
    present = StepInputs(
        files=tuple(
            (
                structure_artifact_path(attempt_dir, step, c.id, "inp"),
                structure_artifact_path(attempt_dir, step, c.id, "out"),
                c.id,
            )
            for c in children
            if structure_artifact_path(attempt_dir, step, c.id, "out").is_file()
        )
    )
    child_ctx = replace(
        ctx, step_dir=attempt_dir, prev_state=PipelineState(structures=tuple(children))
    )
    succ, _fail = step_failures.parse_with_failures(engine, present, child_ctx)
    return succ


def _accept(
    resolved: list[Structure],
    round2: list[Structure],
    parent: Structure,
    ctx: StepContext,
    target: int | None,
    *,
    write_winner: bool,
) -> tuple[list[Structure], list[step_failures.Failure]]:
    """Turn a parent's resolved children into the unified survivor(s) + failure.

    ``random`` (``target is None``) keeps every resolved child as a fan-out structure;
    ``minimum``/``ts`` collapse to the single best resolved geometry **at the parent's
    canonical id** (winner geometry written back when ``write_winner``). A parent with
    nothing resolved becomes one ``Failure`` carrying its best geometry obtained.

    "Best" is by the step's own ranking energy throughout — see :func:`_energy_attr`.
    """
    energy_attr = _energy_attr(ctx.step_cfg)
    if target is None:  # random: the children are the (fan-out) results
        if resolved:
            return resolved, []
        return [], [
            step_failures.Failure(
                parent.id,
                step_failures.FailureKind.UNRESOLVED_NMS,
                _best(round2, parent, energy_attr),
            )
        ]
    if resolved:
        winner = _best(resolved, parent, energy_attr)
        if write_winner:
            io.write_single_xyz(
                winner.atoms,
                structure_artifact_path(ctx.step_dir, ctx.step_cfg.step, parent.id, "xyz"),
                comment=f"NMS-resolved {parent.id}",
            )
        return [replace(winner, id=parent.id, parent_id=parent.parent_id)], []
    return [], [
        step_failures.Failure(
            parent.id, step_failures.FailureKind.UNRESOLVED_NMS, _best(round2, parent, energy_attr)
        )
    ]


def run_nms(
    engine: NmsCapableEngine,
    round1: StepResults,
    round1_failures: list[step_failures.Failure] | tuple[step_failures.Failure, ...],
    ctx: StepContext,
    step_cfg: StepConfig,
) -> step_failures.NmsResolution:
    """Resolve each round-1 survivor to its stationary point (submits round-2).

    For each round-1 structure: if it is already at the target it passes through at the
    canonical place; otherwise its ± displaced children run under ``stepN/<id>/attemptK/``
    and the best resolved geometry becomes the survivor at the canonical id (``random``
    fans out instead). Round-1 jobs that already failed (``round1_failures``) are carried
    through unchanged.
    """
    opts = _resolved_options(engine, ctx)
    target = target_imaginary_count(opts)
    survivors: list[Structure] = []
    failures: list[step_failures.Failure] = list(round1_failures)
    for s in round1.structures:
        if (
            target is not None
            and s.imaginary_freqs is not None
            and len(s.imaginary_freqs) == target
        ):
            survivors.append(replace(s, converged=True))  # already at the target, in place
            continue
        if s.normal_modes is None:
            logger.warning("NMS %s: no normal-mode tensor; cannot displace (unresolved)", s.id)
            failures.append(
                step_failures.Failure(s.id, step_failures.FailureKind.UNRESOLVED_NMS, s)
            )
            continue
        children = _children_of(
            s,
            select_displacements(
                s, s.imaginary_freqs or {}, s.normal_modes, opts, rng_for(s.id, opts.seed)
            ),
        )
        if not children:
            failures.append(
                step_failures.Failure(s.id, step_failures.FailureKind.UNRESOLVED_NMS, s)
            )
            continue
        attempt = next_attempt_dir(ctx.step_dir / s.id)
        round2 = _run_round_two(engine, children, ctx, attempt)
        resolved = [c for c in round2 if _is_resolved(c, target)]
        s_surv, s_fail = _accept(resolved, round2, s, ctx, target, write_winner=True)
        survivors.extend(s_surv)
        failures.extend(s_fail)
    return step_failures.NmsResolution(tuple(survivors), tuple(failures))


def rebuild_nms(
    engine: NmsCapableEngine,
    round1: StepResults,
    round1_failures: list[step_failures.Failure] | tuple[step_failures.Failure, ...],
    ctx: StepContext,
    step_cfg: StepConfig,
) -> step_failures.NmsResolution:
    """Re-resolve NMS from outputs already on disk — no submission (``rebuild-cache``).

    Re-derives the (deterministic) displaced children and parses their existing round-2
    outputs from the structure's latest ``attemptK/``; a child with no output on disk is
    simply absent, so its parent stays unresolved.
    """
    opts = _resolved_options(engine, ctx)
    target = target_imaginary_count(opts)
    survivors: list[Structure] = []
    failures: list[step_failures.Failure] = list(round1_failures)
    for s in round1.structures:
        if (
            target is not None
            and s.imaginary_freqs is not None
            and len(s.imaginary_freqs) == target
        ):
            survivors.append(replace(s, converged=True))
            continue
        attempt = latest_attempt_dir(ctx.step_dir / s.id)
        if s.normal_modes is None or attempt is None:
            failures.append(
                step_failures.Failure(s.id, step_failures.FailureKind.UNRESOLVED_NMS, s)
            )
            continue
        children = _children_of(
            s,
            select_displacements(
                s, s.imaginary_freqs or {}, s.normal_modes, opts, rng_for(s.id, opts.seed)
            ),
        )
        round2 = _parse_round_two(engine, children, ctx, attempt)
        resolved = [c for c in round2 if _is_resolved(c, target)]
        s_surv, s_fail = _accept(resolved, round2, s, ctx, target, write_winner=False)
        survivors.extend(s_surv)
        failures.extend(s_fail)
    return step_failures.NmsResolution(tuple(survivors), tuple(failures))


def reattempt_nms(
    engine: NmsCapableEngine,
    ctx: StepContext,
    step_cfg: StepConfig,
    cached: cache.StepCache,
    parent_ids: tuple[str, ...],
) -> StepResults:
    """Re-attempt only the ledgered-unresolved NMS parents, reusing round-1.

    The still-valid resolved survivors from the old cache are kept; the failed parents'
    round-1 outputs are re-parsed (missing ones resubmitted, unconverged ones retried),
    NMS round-2 is re-run for them, and the merged result is re-cached. Mirrors
    :func:`chemrefine.step._resubmit_failed` for the two-round case.
    """
    manifest = cache.load_manifest(ctx.step_dir)
    if manifest is None:
        raise CacheError(f"step {step_cfg.step}: cannot re-attempt NMS — no manifest on disk")
    failed = step_failures.load_failure_records(ctx.step_dir)
    failed_ids = {f.structure_id for f in failed}
    missing_ids = {
        f.structure_id for f in failed if f.kind is step_failures.FailureKind.MISSING_OUTPUT
    }

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

    r1_succ, r1_fail = step_failures.parse_with_failures(engine, failed_manifest, ctx)
    r1_succ, r1_fail = step_failures.retry_unconverged(engine, ctx, r1_succ, r1_fail)
    reattempt = run_nms(engine, StepResults(structures=tuple(r1_succ)), r1_fail, ctx, step_cfg)
    kept = tuple(
        s
        for s in cached.results.structures
        if s.id not in failed_ids and s.parent_id not in failed_ids
    )
    return step_failures.finalize(
        engine,
        ctx,
        step_cfg,
        parent_ids,
        list(kept + reattempt.survivors),
        list(reattempt.failures),
    )

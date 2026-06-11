"""Normal-mode sampling: target-aware displacement of imaginary modes.

These are the **pure** helpers the ORCA engine's two-round NMS uses
(:meth:`chemrefine.engines.orca.engine.OrcaEngine.normal_mode_sample`):
round 1 runs an ``opt+freq``; this module turns each structure's parsed
imaginary modes + normal-mode tensor into displaced child geometries
(per the ``target``); round 2 re-optimises them and keeps the ones that
reach the target stationary point. The round-2 submission and the
resolution check live in the engine — everything here is side-effect-free.

Targets:

* ``minimum`` — displace ± along **every** imaginary mode (remove all).
* ``ts`` — keep the reaction-coordinate mode (``ts_mode_index`` or the
  largest-magnitude imaginary), displace ± along every **other** imaginary
  mode (remove the spurious extras, keep the first-order saddle).
* ``random`` — displace ± along ``num_random_displacements`` modes drawn
  from all normal modes (broad exploration).
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from chemrefine.state import Structure

logger = logging.getLogger(__name__)

# ORCA prints 6 (5 for linear) trivial translation/rotation modes at low
# index; ``random`` sampling skips them.
_TRIVIAL_MODES = 6


class NmsOptions(BaseModel):
    """Validated normal-mode-sampling knobs (from ``step.options``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target: Literal["minimum", "ts", "random"] = "minimum"
    """``minimum`` removes all imaginary modes; ``ts`` keeps the reaction
    coordinate and removes the rest; ``random`` displaces along random modes."""

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

    ``minimum`` → 0, ``ts`` → 1, ``random`` → ``None`` (no resolution gate;
    random sampling is exploration, not stationary-point cleanup).
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

    One ``±`` pair per selected mode; ``suffix`` (e.g. ``m5_pos``) makes the
    child id ``{parent}_{suffix}`` unique and traceable to its mode.
    """
    positions = struct.atoms.get_positions()
    n_modes = modes.shape[2]
    out: list[tuple[str, NDArray[np.float64]]] = []
    for idx in _selected_modes(imag_freqs, n_modes, opts, rng):
        if idx >= n_modes:
            logger.warning(
                "NMS %s: mode %d outside the %d-mode tensor; skipping",
                struct.id,
                idx,
                n_modes,
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

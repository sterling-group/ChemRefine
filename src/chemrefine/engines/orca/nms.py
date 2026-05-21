"""Normal-mode sampling for ORCA frequency outputs.

After a frequency calculation, each structure produces a normal-mode
tensor and (if any vibrational instabilities exist) a set of imaginary
frequencies. NMS displaces each structure along its least-imaginary
mode in ±directions; the next pipeline step re-optimises both
displaced copies and keeps whichever survives with one imaginary
frequency removed.

Ported from v3 :class:`OrcaInterface.normal_mode_sampling`, but the
v4 shape is a single :func:`normal_mode_sample` that takes the
already-parsed :class:`StepResults` and expands it. SLURM submission
of the displaced inputs happens in the next step, not here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from chemrefine.engines.orca import frequencies
from chemrefine.state import StepContext, StepResults, Structure

logger = logging.getLogger(__name__)


def displace_along_mode(
    positions_angstrom: NDArray[np.float64],
    mode_vector: NDArray[np.float64],
    *,
    displacement: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(positions + d * v, positions - d * v)``.

    Both arrays carry the same shape as ``positions_angstrom``.
    ``mode_vector`` must match that shape — typically a single mode
    slice ``tensor[:, :, mode_idx]`` from
    :func:`parse_normal_modes_tensor`.
    """
    if positions_angstrom.shape != mode_vector.shape:
        raise ValueError(
            f"position / mode shape mismatch: {positions_angstrom.shape} vs {mode_vector.shape}"
        )
    pos = positions_angstrom + displacement * mode_vector
    neg = positions_angstrom - displacement * mode_vector
    return pos, neg


def _least_imaginary_mode(imag_freqs: dict[int, float]) -> int:
    """Return the index of the imaginary mode closest to zero (least imaginary)."""
    return min(imag_freqs.items(), key=lambda kv: abs(kv[1]))[0]


def normal_mode_sample(results: StepResults, ctx: StepContext) -> StepResults:
    """Expand each structure into ± displaced copies along its least-imaginary mode.

    Structures whose frequency output is missing, has no imaginary
    modes, or lacks a printable normal-mode tensor are skipped with a
    warning — NMS is a best-effort expansion, not a hard requirement.

    The displacement magnitude defaults to ``1.0`` (Ångström) and is
    overridable via ``step.options.displacement_value`` in the YAML.
    """
    options = ctx.step_cfg.options or {}
    displacement = float(options.get("displacement_value", 1.0))
    expanded: list[Structure] = []
    for struct in results.structures:
        out_path = (
            ctx.step_dir
            / f"step{ctx.step_cfg.step}_structure_{struct.id}.out"
        )
        pair = _expand_one(struct, out_path, displacement=displacement)
        if pair is None:
            continue
        expanded.extend(pair)
    return StepResults(structures=tuple(expanded))


def _expand_one(
    struct: Structure, out_path: Path, *, displacement: float
) -> tuple[Structure, Structure] | None:
    """Try to expand ``struct`` into a (pos, neg) displaced pair.

    Returns ``None`` (with a warning logged) when the frequency output
    is missing, has no imaginary modes, or lacks a parseable
    normal-mode tensor.
    """
    if not out_path.is_file():
        logger.warning("NMS skip %s: frequency output not found at %s", struct.id, out_path)
        return None
    imag = frequencies.parse_imaginary_frequencies(out_path)
    if not imag:
        logger.warning("NMS skip %s: no imaginary modes in %s", struct.id, out_path)
        return None
    try:
        tensor = frequencies.parse_normal_modes_tensor(
            out_path, num_atoms=len(struct.atoms)
        )
    except ValueError as e:
        logger.warning("NMS skip %s: %s", struct.id, e)
        return None
    mode_idx = _least_imaginary_mode(imag)
    if mode_idx >= tensor.shape[2]:
        logger.warning(
            "NMS skip %s: imaginary mode index %d outside tensor (%d modes)",
            struct.id, mode_idx, tensor.shape[2],
        )
        return None
    pos_xyz, neg_xyz = displace_along_mode(
        struct.atoms.get_positions(),
        tensor[:, :, mode_idx],
        displacement=displacement,
    )
    pos_atoms = struct.atoms.copy()
    pos_atoms.set_positions(pos_xyz)
    neg_atoms = struct.atoms.copy()
    neg_atoms.set_positions(neg_xyz)
    return (
        Structure(id=f"{struct.id}_pos", atoms=pos_atoms),
        Structure(id=f"{struct.id}_neg", atoms=neg_atoms),
    )

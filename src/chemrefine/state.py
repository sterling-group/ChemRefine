"""Frozen runtime types passed between pipeline stages.

These dataclasses are the values that flow from one step to the next and
between the engine lifecycle stages (prepare → submit → wait → parse →
sample). Keeping them frozen and small forces callers to thread state
explicitly rather than mutating shared state on a god class.

``StepConfig`` lives in :mod:`chemrefine.config` to avoid pulling Pydantic
into this module — it is type-hinted as a forward reference where needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - typing only
    from chemrefine.config import StepConfig


@dataclass(frozen=True)
class Structure:
    """A single molecular geometry with its computed energy and forces.

    Energy is in Hartree (engine-native); forces are in eV/Å (ASE-native).
    ``None`` values mean the field has not been populated yet — e.g. a
    seed structure before its first calculation.
    """

    id: str
    atoms: Atoms
    energy_hartree: float | None = None
    forces_eV_per_A: NDArray[np.float64] | None = None


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
    scratch_dir: Path
    prev_state: PipelineState
    charge: int
    multiplicity: int


@dataclass(frozen=True)
class StepInputs:
    """Engine-prepared inputs for one step's job batch.

    ``files`` is an ordered tuple of ``(input_path, output_path,
    structure_id)`` triples. Order matches the seed structures' order so
    parse results can be aligned back to parents.
    """

    files: tuple[tuple[Path, Path, str], ...]

    def input_paths(self) -> tuple[Path, ...]:
        """Return just the input file paths in order."""
        return tuple(t[0] for t in self.files)

    def output_paths(self) -> tuple[Path, ...]:
        """Return just the output file paths in order."""
        return tuple(t[1] for t in self.files)

    def structure_ids(self) -> tuple[str, ...]:
        """Return the structure IDs in order."""
        return tuple(t[2] for t in self.files)


@dataclass(frozen=True)
class JobBatch:
    """Opaque handle returned by ``engine.submit`` and consumed by ``engine.wait``.

    ``jobs`` maps each input file path to its job identifier (a SLURM
    job ID, a local-runner PID, or whatever the engine's submitter
    produces). The wait step polls this mapping.
    """

    jobs: dict[Path, str]


@dataclass(frozen=True)
class StepResults:
    """What ``engine.parse`` produces — one entry per output file."""

    structures: tuple[Structure, ...]

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

import numpy as np
from ase import Atoms
from numpy.typing import NDArray

from chemrefine.config import StepConfig


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

    ``forces_ev_per_a`` is a numpy array, which the ``frozen=True``
    contract cannot enforce as immutable. Callers must treat it as
    read-only — the array is shared by reference across the pipeline.
    """

    id: str
    atoms: Atoms
    parent_id: str | None = None
    energy_hartree: float | None = None
    forces_ev_per_a: NDArray[np.float64] | None = None
    converged: bool | None = None
    """Did the run converge (SCF + geometry)? ``None`` when the engine
    doesn't report it — treated as 'not a failure signal'."""
    terminated: bool | None = None
    """Did the program terminate normally? ``None`` when the engine doesn't
    report it. A structure is a *failure* only when a flag is explicitly
    ``False`` (see :func:`chemrefine.step_failures.succeeded`)."""
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


@dataclass(frozen=True)
class StepInputs:
    """Engine-prepared inputs for one step's job batch.

    ``files`` is an ordered tuple of ``(input_path, output_path,
    structure_id)`` triples. Order matches the seed structures' order so
    parse results can be aligned back to parents.
    """

    files: tuple[tuple[Path, Path, str], ...]


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

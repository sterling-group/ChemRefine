"""Pydantic validator for PySCF engine YAML options.

``step.options`` is a free-form ``dict[str, Any]`` so the orchestrator
stays engine-agnostic; backend-specific validation lives here.
:class:`PyscfOptions` covers both the SCF knobs (method, xc, basis,
df, gpu) and the active-space tensor-extraction knobs (save_tensors,
localized, tensor_folder).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PyscfOptions(BaseModel):
    """Validated knobs for the PySCF backend."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: Literal["dft", "hf"] = "dft"
    """Electronic-structure method to run."""

    xc: str = "pbe"
    """Exchange-correlation functional (DFT only)."""

    basis: str = "def2-svp"
    """Orbital basis-set name (passed verbatim to :class:`pyscf.gto.Mole`)."""

    df: bool = False
    """Enable density fitting / RI."""

    gpu: bool = False
    """Attempt :mod:`gpu4pyscf` if installed."""

    save_tensors: bool = False
    """Extract one- + two-electron tensors after the SCF and persist to ``.npz``."""

    localized: bool = False
    """Boys-localize occupied / virtual orbitals before tensor extraction."""

    tensor_folder: str = Field("tensors", min_length=1)
    """Output directory for ``save_tensors`` ``.npz`` files.

    When ``save_tensors`` is set this **must be an absolute path**: the
    calculator runs server-side with ``cwd = $WORK_DIR`` (the per-job scratch,
    removed when the job ends), so a relative path would write the tensors into
    scratch and they would be silently deleted — point it at a directory that
    persists past the run (e.g. under the step's output directory). When
    ``save_tensors`` is off the value is unused, so the relative default stands.
    """

    @field_validator("tensor_folder")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        """``tensor_folder`` must be non-empty (no implicit cwd writes)."""
        if not v.strip():
            raise ValueError("tensor_folder must be a non-empty string")
        return v

    @model_validator(mode="after")
    def _tensor_folder_absolute_when_saving(self) -> PyscfOptions:
        """A relative ``tensor_folder`` with ``save_tensors`` would lose the dumps.

        The calculator writes under ``cwd = $WORK_DIR`` (scratch, deleted at job
        end), and the engine's ``output_globs`` don't copy ``.npz`` back — so a
        relative folder silently discards the tensors. Fail fast at config load
        instead, naming the fix.
        """
        if self.save_tensors and not Path(self.tensor_folder).is_absolute():
            raise ValueError(
                f"tensor_folder must be an absolute path when save_tensors is set; "
                f"got {self.tensor_folder!r}. A relative path resolves under the "
                "per-job scratch ($WORK_DIR), which is deleted when the job ends — "
                "the tensors would be lost. Point it at a persistent directory."
            )
        return self

    @classmethod
    def from_raw(cls, raw: dict[str, Any] | None) -> PyscfOptions:
        """Validate a raw ``step.options`` dict; an empty dict yields defaults."""
        return cls(**(raw or {}))

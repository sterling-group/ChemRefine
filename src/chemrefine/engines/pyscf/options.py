"""Pydantic validator for PySCF engine YAML options.

``step.options`` is a free-form ``dict[str, Any]`` so the orchestrator
stays engine-agnostic; backend-specific validation lives here.
:class:`PyscfOptions` covers both the SCF knobs (method, xc, basis,
df, gpu) and the active-space tensor-extraction knobs (save_tensors,
localized, tensor_folder).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


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

    A relative path is resolved against the server's working directory
    (``$WORK_DIR``, the per-job scratch), which is removed when the job
    finishes — pass an **absolute** path (e.g. under the step's output
    directory) when the tensors must persist past the run.
    """

    @field_validator("tensor_folder")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        """``tensor_folder`` must be non-empty (no implicit cwd writes)."""
        if not v.strip():
            raise ValueError("tensor_folder must be a non-empty string")
        return v

    @classmethod
    def from_raw(cls, raw: dict[str, Any] | None) -> PyscfOptions:
        """Validate a raw ``step.options`` dict; an empty dict yields defaults."""
        return cls(**(raw or {}))

"""Pydantic validator for PySCF engine YAML options.

``step.options`` is a free-form ``dict[str, Any]`` so the orchestrator
stays engine-agnostic; backend-specific validation lives here.
:class:`PyscfOptions` covers both the SCF knobs (method, xc, basis,
df, gpu, device) and the active-space tensor-extraction knobs
(save_tensors, localized, tensor_folder).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from chemrefine.engines._options import EngineOptions


class PyscfOptions(EngineOptions):
    """Validated knobs for the PySCF backend (``device`` comes from :class:`EngineOptions`)."""

    method: Literal["dft", "hf"] = "dft"
    """Electronic-structure method to run."""

    xc: str = "pbe"
    """Exchange-correlation functional. **Required (no silent default) when
    ``method: dft``** — the YAML must name it (see :meth:`from_raw`)."""

    basis: str = Field("def2-svp", min_length=1)
    """Orbital basis-set name (passed verbatim to :class:`pyscf.gto.Mole`).
    **Required (no silent default)** — the YAML must name it (see
    :meth:`from_raw`). The field keeps a value only so the server CLI and
    programmatic callers can construct an instance."""

    df: bool = True
    """Enable density fitting / RI. Defaults **on** — DF is a large speed-up at
    negligible accuracy cost for the gradient-server use case."""

    gpu: bool = False
    """Attempt :mod:`gpu4pyscf` if installed. When omitted it is derived from
    ``device`` (``cuda`` ⇒ ``True``); set it explicitly to override. The SCF
    falls back to CPU if gpu4pyscf can't initialise."""

    save_tensors: bool = False
    """Extract one- + two-electron tensors after the SCF and persist to ``.npz``."""

    localized: bool = False
    """Boys-localize occupied / virtual orbitals before tensor extraction."""

    tensor_folder: str = Field("tensors", min_length=1)
    """Output directory for ``save_tensors`` ``.npz`` files.

    A **relative** path (the default ``tensors``) is fine: the calculator writes
    under ``cwd = $WORK_DIR`` (scratch) and the engine copies the directory back
    into the structure's own output dir on exit, so each structure's tensors land
    in ``outputs/stepN/<id>/tensors/`` with no collisions. An absolute path writes
    (and persists) directly at that location instead.
    """

    @model_validator(mode="before")
    @classmethod
    def _derive_gpu_from_device(cls, data: Any) -> Any:
        """Default ``gpu`` from ``device`` when ``gpu`` isn't given (``cuda`` ⇒ ``True``).

        The fallback reads ``device``'s own field default rather than repeating the
        literal: spelling it twice is what let this derivation keep saying ``cuda``
        after the field default moved, so an unset ``device`` would have derived
        ``gpu: true`` while the scheduler booked a CPU job.
        """
        if isinstance(data, dict) and "gpu" not in data:
            default_device = cls.model_fields["device"].default
            device = str(data.get("device", default_device)).lower()
            data = {**data, "gpu": device == "cuda"}
        return data

    @field_validator("tensor_folder")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        """``tensor_folder`` must be non-empty (no implicit cwd writes)."""
        if not v.strip():
            raise ValueError("tensor_folder must be a non-empty string")
        return v

    @classmethod
    def from_raw(cls, raw: dict[str, Any] | None) -> PyscfOptions:
        """Validate a raw ``step.options`` dict from the YAML.

        ``basis`` must be named explicitly (no silent default), and ``xc`` must
        be named when ``method`` is ``dft`` — a misconfigured PySCF step fails
        fast rather than running with a surprise level of theory. (The model
        fields keep values only so the server CLI / programmatic callers can
        still construct an instance.)
        """
        raw = raw or {}
        opts = cls(**raw)
        if "basis" not in raw:
            raise ValueError("pyscf: 'basis' is required (name the basis set explicitly)")
        if opts.method == "dft" and "xc" not in raw:
            raise ValueError("pyscf: 'xc' is required when method is 'dft'")
        return opts

"""Pydantic validator for PySCF engine YAML options.

``step.options`` is a free-form ``dict[str, Any]`` so the orchestrator
stays engine-agnostic; backend-specific validation lives here.
:class:`PyscfOptions` covers both the SCF knobs (method, xc, basis,
df, gpu, device) and the active-space tensor-extraction knobs
(save_tensors, localized, tensor_folder).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from chemrefine.config import reject_shell_unsafe
from chemrefine.engines._options import EngineOptions
from chemrefine.errors import ConfigError


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

    strict_scf: bool = True
    """Refuse to serve a gradient from an SCF that did not converge.

    On by default because the alternative is silent: PySCF returns the last iterate rather
    than raising, so ORCA would take its next optimisation step on a gradient computed from
    a non-stationary density, and the ``.out`` it writes reports *ORCA's* geometry
    convergence — which says nothing about the backend's SCF. The structure then ranks and
    filters against correctly-converged siblings with nothing marking it.

    Set ``false`` only if you knowingly want the loose behaviour (a deliberately truncated
    SCF, a scan where a few points are expected not to settle); the energy is then whatever
    the last iteration produced."""

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
        literal. Spelled twice, the two drift the first time the field default moves: an
        unset ``device`` would derive ``gpu: true`` here while the scheduler books a CPU
        job from the model.
        """
        if isinstance(data, dict) and "gpu" not in data:
            default_device = cls.model_fields["device"].default
            device = str(data.get("device", default_device)).lower()
            data = {**data, "gpu": device == "cuda"}
        return data

    @field_validator("tensor_folder")
    @classmethod
    def _non_empty_and_shell_safe(cls, v: str) -> str:
        """Non-empty (no implicit cwd writes), and safe to interpolate into generated bash.

        This is the one engine option that reaches the job script: a relative
        ``tensor_folder`` becomes an ``output_dirs`` entry, which the on-exit handler copies
        with ``cp -r "<tensor_folder>" "$OUTPUT_DIR/"``. The double quotes there are **not**
        protection — bash performs command substitution inside them — so
        ``tensor_folder: 'tensors$(...)'`` runs that command when the job does.

        Held to :func:`chemrefine.config.reject_shell_unsafe`, the same rule as the
        directory paths, ``executables`` and ``operation``, rather than a copy of it. The
        rule belongs to "every config value that reaches generated bash", not to a list of
        fields — a knob declared in an engine's own options model reaches bash just as
        surely as one declared in the top-level config.
        """
        if not v.strip():
            raise ValueError("tensor_folder must be a non-empty string")
        reject_shell_unsafe(v, what="tensor_folder", fix="rename the folder")
        return v

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any] | None) -> Self:
        """Validate a raw ``step.options`` dict from the YAML.

        ``basis`` must be named explicitly (no silent default), and ``xc`` must
        be named when ``method`` is ``dft`` — a misconfigured PySCF step fails
        fast rather than running with a surprise level of theory. (The model
        fields keep values only so the server CLI / programmatic callers can
        still construct an instance.)

        Validation is delegated to the base rather than calling ``cls`` directly, so a
        pydantic error still becomes a :class:`~chemrefine.errors.ConfigError` and a typoed
        knob stays inside the CLI's exit-code contract. The extra requirements below are the
        only thing this override adds.
        """
        raw = raw or {}
        opts = super().from_raw(raw)
        if "basis" not in raw:
            raise ConfigError("pyscf: 'basis' is required (name the basis set explicitly)")
        if opts.method == "dft" and "xc" not in raw:
            raise ConfigError("pyscf: 'xc' is required when method is 'dft'")
        return opts

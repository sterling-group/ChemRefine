"""Pydantic validators for the PySCF engines' YAML options.

``step.options`` is a free-form ``dict[str, Any]`` so the orchestrator stays engine-agnostic;
backend-specific validation lives here. Two models, because the engines in this package share
a backend and not a set of knobs:

* :class:`PyscfOptions` — what both read: the SCF selection (``method`` / ``xc`` / ``basis``)
  and the GPU request, on top of :class:`~chemrefine.engines._options.EngineOptions`.
* :class:`PyscfExtOptOptions` — those, plus the knobs only the gradient *server* acts on.

An engine declares the model whose fields it reads. Declared together, the direct engine
accepts knobs it has no way to honour — and because ``accepted_names()`` reports them as
declared, ``chemrefine validate`` cannot warn about them either, so naming one is silence in
both directions. The same split :class:`~chemrefine.engines.mlip.options.MlipOptions` and
:class:`~chemrefine.engines.mlip.options.MlipTrainOptions` make, so an inference step cannot
accept a training knob.
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

    xc: str | None = None
    """Exchange-correlation functional — **required when ``method: dft``**, with no
    silent default anywhere (see :meth:`require_level_of_theory`)."""

    basis: str | None = Field(None, min_length=1)
    """Orbital basis-set name (passed verbatim to :class:`pyscf.gto.Mole`) —
    **required**, with no silent default anywhere: every engine makes the user name
    the level of theory (see :meth:`require_level_of_theory`)."""

    df: bool = True
    """Enable density fitting / RI, on both engines. Defaults **on** — a large speed-up at
    negligible accuracy cost.

    Shared rather than server-only because it shapes the SCF, like ``method`` / ``xc`` /
    ``basis``: the ExtOpt server passes it to :func:`~chemrefine.engines.pyscf._runtime.run_dft`
    and a direct template reads it as ``$DF``. What separates the two models is work the user
    has no code of their own to do — post-SCF extraction on a path where ORCA drives and there
    is no ``step{N}.py`` — and density fitting is not that.
    """

    gpu: bool = False
    """Attempt :mod:`gpu4pyscf` if installed. When omitted it is derived from
    ``device`` (``cuda`` ⇒ ``True``); set it explicitly to override. The SCF
    falls back to CPU if gpu4pyscf can't initialise."""

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

    @classmethod
    def require_level_of_theory(cls, raw: Mapping[str, Any] | None) -> str:
        """Refuse options that name no explicit level of theory; return the named basis.

        Returning the value the check proved is what lets a caller that computes with
        it (the ExtOpt calculator's ``calc``) take it already narrowed to ``str`` —
        the alternative was an ``assert`` re-stating what this just established.

        ``basis`` must be named, and ``xc`` when ``method`` is ``dft`` — every other
        engine makes the user say the level of theory (ORCA's lives in the template's
        ``!`` line, Q-Chem's in ``$rem``), so a silent ``def2-svp``/``pbe`` would make
        pyscf the one exception, and the wrong kind of exception: a level of theory the
        user never chose. The fields default to ``None`` — a bare instance is
        constructible (the server's CLI parser and the lockstep-held calculator build
        one) but nothing runs without the values named: this rule, the strict read,
        and the calculator's own ``calc`` all refuse.

        One home, two callers, because each engine reaches the requirement through
        different reads: :meth:`from_raw` (the ExtOpt server's strict read) enforces it
        inside validation, while the direct engine's preflight
        (:meth:`~chemrefine.engines.pyscf.engine.PyscfEngine.check_step`) reads leniently
        — a ``step{N}.py`` template may carry knobs no model declares, so ``from_raw``'s
        ``extra="forbid"`` cannot be its gate — and asks this instead. The whole check reads
        through the lenient model — never off the raw dict — so ``basis: null`` and an
        absent key are one fact, and the raw-read invariant holds.
        """
        opts = cls.from_raw_lenient(raw or {})
        if opts.basis is None:
            raise ConfigError("pyscf: 'basis' is required (name the basis set explicitly)")
        if opts.method == "dft" and opts.xc is None:
            raise ConfigError("pyscf: 'xc' is required when method is 'dft'")
        return opts.basis

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any] | None) -> Self:
        """Validate a raw ``step.options`` dict from the YAML — the strict read.

        The level-of-theory requirement (:meth:`require_level_of_theory`) plus the base's
        strict validation: ``pyscf-extopt`` calls this (its options configure a server),
        and the direct engine makes the same refusal at preflight through the shared
        classmethod, so the engines cannot disagree about what a runnable step names.

        Validation is delegated to the base rather than calling ``cls`` directly, so a
        pydantic error still becomes a :class:`~chemrefine.errors.ConfigError` and a typoed
        knob stays inside the CLI's exit-code contract.
        """
        raw = raw or {}
        cls.require_level_of_theory(raw)
        return super().from_raw(raw)


class PyscfExtOptOptions(PyscfOptions):
    """The SCF knobs, plus the ones only the gradient server reads.

    ``pyscf`` renders a user ``step{N}.py`` and reaches its options through template
    placeholders; ``pyscf-extopt`` builds a long-running server from the whole set. The knobs
    below are that difference, and ``strict_scf`` is why it matters: a non-converged SCF is
    refused on the ExtOpt path for the reason its own docstring gives, and the direct path has
    no channel to refuse it with — a script reports what its output contract declares.
    """

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

"""Shared base for per-engine YAML option models.

``step.options`` is a free-form ``dict[str, Any]`` so the orchestrator stays
engine-agnostic; each engine validates its own subset with a small Pydantic model.
:class:`EngineOptions` carries the knobs every compute engine shares — the ``device``
selector and the ``frozen`` / ``extra="forbid"`` config — so an engine's options model only
adds its own fields. It also lets the ExtOpt base type its ``options_cls`` ClassVar.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Self

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError

from chemrefine.errors import ConfigError


def refuse_template_knobs(extra: Mapping[str, Any]) -> None:
    """Refuse ``extra`` on an engine that renders no ``step{N}.py`` for it to reach.

    The direct script engines declare ``extra`` as the bag for a template's own knobs. An
    engine that renders no template reads nothing from it, and a knob nothing reads is the
    silent no-op the declared-key rule exists to catch — so the models of those engines
    refuse it, through this one rule, rather than accept and ignore it. Raised as a
    ``ValueError`` because it runs inside a pydantic validator, which reports it as the
    model's own validation error.
    """
    if extra:
        raise ValueError("`extra` is for a stepN.py template; this engine renders none")


class EngineOptions(BaseModel):
    """Base for an engine's validated ``step.options`` model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    device: Literal["cuda", "cpu"] = "cpu"
    """Compute device for this engine (``cuda`` ⇒ GPU, ``cpu`` ⇒ CPU).

    Defaults to ``cpu`` because this value is read by two layers that must agree: the
    engine renders it into the step's script (``$DEVICE``), and the scheduler derives
    the step's GPU demand and SLURM header from it
    (:attr:`gpu_demand`, below). ``cpu`` is the floor that
    always runs; requesting a GPU is one line of YAML, whereas a wrong ``cuda``
    default schedules a CPU job whose script then asks for a device it wasn't given.
    """

    backend_python: str | None = None
    """Explicit Python interpreter for this step's compute backend (escape hatch).

    Normally unset: the provisioner (:mod:`chemrefine.engines._provision`) resolves the
    backend's managed env by *name* — no paths in the YAML. Set this only to force a
    specific interpreter (e.g. a hand-built env the provisioner doesn't manage)."""

    cores: int = Field(1, ge=1)
    """Per-structure core budget. Lives here because it is what the scheduler asks
    every job engine for (``JobEngine.pal``), not something one backend invented."""

    @property
    def gpu_demand(self) -> int:
        """GPUs this step's options ask the scheduler for: one for ``device: cuda``, else none.

        The scheduler's read of ``device`` — the header it picks, the budget it charges, the
        device it assigns — belongs to the model beside the field, not to a helper that peeks
        at subclass fields by name: a model with another way to ask overrides this (PySCF's
        ``gpu``), and :meth:`chemrefine.engines._job.JobEngine.gpus` asks every declaring
        engine's model the same question. Read through the model rather than off the raw
        dict because the two disagree about what "unset" means: ``options.get("device", "")``
        yielded no GPU while the field default once said ``cuda``, so a step that named no
        device rendered ``$DEVICE=cuda`` into its script while being scheduled as a CPU job
        on the CPU header — bypassing the GPU budget and
        :meth:`~chemrefine.throttle.Throttler.assign_device`, so concurrent local steps piled
        onto device 0. One reader, one default.
        """
        return 1 if self.device == "cuda" else 0

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any] | None) -> Self:
        """Validate a raw ``step.options`` dict; empty/``None`` yields defaults.

        Strict — ``extra="forbid"`` means a typoed knob fails the step rather than
        being silently ignored. Subclasses override to add engine-specific
        required-field checks (e.g. PySCF's ``basis`` / ``xc``).
        """
        raw = raw or {}
        cls._reject_ambiguous_spellings(raw)
        return cls._validate(raw)

    @classmethod
    def _spellings_by_field(cls) -> dict[str, set[str]]:
        """Every YAML spelling this model accepts, grouped by the field it sets."""
        grouped: dict[str, set[str]] = {}
        for name, field in cls.model_fields.items():
            names = {name}
            alias = field.validation_alias
            if isinstance(alias, str):
                names.add(alias)
            elif isinstance(alias, AliasChoices):
                names.update(c for c in alias.choices if isinstance(c, str))
            grouped[name] = names
        return grouped

    @classmethod
    def accepted_names(cls) -> set[str]:
        """Every YAML spelling this model accepts — field names plus their aliases.

        Public, not underscored, because :mod:`chemrefine.validate` asks it which option
        keys a step declares: the warning about an undeclared key is only as good as the
        list it is checked against, and that list belongs to the model that defines it.
        """
        return {name for names in cls._spellings_by_field().values() for name in names}

    @classmethod
    def _reject_ambiguous_spellings(cls, raw: Mapping[str, Any]) -> None:
        """Refuse a step that sets two spellings of the same knob.

        Aliases exist so each backend reads naturally (``task`` for ``task_name``,
        ``model``/``size`` for ``model_name``), which means a step *can* name one field
        twice. Pydantic rejects that on its own, but as ``extra="forbid"`` on whichever
        spelling it did not pick — a message naming the wrong problem — and only on the
        strict path, which would leave the two readers of the same options disagreeing about
        whether such a step is valid at all.

        Raising here, before validation, makes both paths agree and says which knob is
        doubled.
        """
        for field, names in cls._spellings_by_field().items():
            present = sorted(name for name in names if name in raw)
            if len(present) > 1:
                raise ConfigError(
                    f"options set {' and '.join(repr(n) for n in present)}, which are "
                    f"spellings of the same knob ({field!r}); keep one"
                )

    @classmethod
    def from_raw_lenient(cls, raw: Mapping[str, Any] | None) -> Self:
        """Validate only the keys this model knows, ignoring any others.

        For the one place strictness would be wrong: a direct ``step{N}.py`` template
        may carry knobs of its own that no engine model declares, and rendering it
        must not fail over them. Reading the raw dict by hand instead means re-spelling
        the alias rules (``model`` / ``size`` for ``model_name``) next to the model that
        already declares them.
        """
        raw = raw or {}
        cls._reject_ambiguous_spellings(raw)
        accepted = cls.accepted_names()
        return cls._validate({k: v for k, v in raw.items() if k in accepted})

    @classmethod
    def _validate(cls, data: Mapping[str, Any]) -> Self:
        """Build the model, reporting a bad knob as a :class:`ConfigError`.

        An invalid ``step.options`` value is a config error and must exit with the code
        :mod:`chemrefine.errors` documents for one. The CLI catches ``ChemRefineError``, so a
        pydantic ``ValidationError`` allowed to escape would leave that contract and reach the
        user as a traceback rather than a message.
        """
        try:
            return cls(**data)
        except ValidationError as e:
            raise ConfigError(
                f"invalid {cls.__name__.removesuffix('Options').lower()} options:\n{e}"
            ) from e


class ExtOptOptions(EngineOptions):
    """The knobs every ExtOpt engine's options model carries on top of its backend's.

    An ExtOpt step is an ORCA run whose gradients come from a server the engine starts, and
    the bridge ORCA invokes per geometry is that server's one client — so the knobs of that
    call belong to the ExtOpt family, not to any backend. The backend models mix this in
    (:class:`~chemrefine.engines.mlip.options.MlipExtOptOptions`,
    :class:`~chemrefine.engines.pyscf.options.PyscfExtOptOptions`), which is what lets the
    ExtOpt base type its ``options_cls`` to it and read the knob without a cast.
    """

    gradient_timeout_seconds: float = Field(600.0, gt=0, allow_inf_nan=False)
    """How long the bridge waits on one gradient before it gives the geometry up.

    A bound on a single ``/calculate`` call, not on the optimisation. The default is
    generous for a potential and tight for a DFT gradient on a large system — raise it when
    a healthy server takes longer than that per geometry. On expiry the step records a
    timeout that names this knob, not an unreachable server.

    Finite as well as positive. ``.inf`` — an ordinary spelling of "no timeout" — passed
    ``gt=0``, rode into the wrapper as ``--timeout inf``, and reached the socket as a
    timeout Python cannot represent (``OverflowError``, none of the failures the bridge
    classifies), so every geometry step died in the wrapper with a traceback naming
    neither the knob nor the value. There is no unbounded setting: raise the number."""

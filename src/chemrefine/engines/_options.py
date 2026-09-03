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


class EngineOptions(BaseModel):
    """Base for an engine's validated ``step.options`` model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    device: Literal["cuda", "cpu"] = "cpu"
    """Compute device for this engine (``cuda`` ⇒ GPU, ``cpu`` ⇒ CPU).

    Defaults to ``cpu`` because this value is read by two layers that must agree: the
    engine renders it into the step's script (``$DEVICE``), and the scheduler derives
    the step's GPU demand and SLURM header from it
    (:func:`gpus_from_options`, below). ``cpu`` is the floor that
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


def gpus_from_options(
    options: dict[str, object] | None,
    options_cls: type[EngineOptions] = EngineOptions,
) -> int:
    """1 if the step's **validated** options request a GPU, else 0.

    Reads through ``options_cls`` rather than off the raw dict, because the raw dict and
    the model disagree about what "unset" means: ``options.get("device", "")`` yielded no
    GPU while ``EngineOptions.device`` defaulted to ``cuda``, so a step that named no
    device rendered ``$DEVICE=cuda`` into its script while being scheduled as a CPU job on
    the CPU header — and it bypassed both the GPU budget and
    :meth:`~chemrefine.throttle.Throttler.assign_device`, so concurrent local steps piled
    onto device 0. One reader, one default.

    ``options_cls`` is the engine's own model, so a backend that expresses the request
    differently is honoured without this helper knowing about it: PySCF's ``gpu`` (try
    gpu4pyscf) is a :class:`~chemrefine.engines.pyscf.options.PyscfOptions` field derived
    from ``device``, and ``getattr`` picks it up for engines that declare it.

    Lives beside the ``device`` field it reads, and is re-exported by
    :mod:`chemrefine.engines.api` for the orchestrator: ``validate`` and the scaffold's
    header choice read the GPU demand too, and the public face is the one module the flat
    pipeline imports from this subsystem.
    """
    opts = options_cls.from_raw_lenient(options)
    return 1 if opts.device == "cuda" or bool(getattr(opts, "gpu", False)) else 0

"""Shared base for per-engine YAML option models.

``step.options`` is a free-form ``dict[str, Any]`` so the orchestrator stays
engine-agnostic; each engine validates its own subset with a small Pydantic model.
:class:`EngineOptions` carries the knobs every compute engine shares — the ``device``
selector and the ``frozen`` / ``extra="forbid"`` config — so an engine's options model only
adds its own fields. It also lets the ExtOpt base type its ``options_cls`` ClassVar.
"""

from __future__ import annotations

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
    (:func:`chemrefine.engines._job.gpus_from_options`). ``cpu`` is the floor that
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
    def from_raw(cls, raw: dict[str, Any] | None) -> Self:
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
    def _accepted_names(cls) -> set[str]:
        """Every YAML spelling this model accepts — field names plus their aliases."""
        return {name for names in cls._spellings_by_field().values() for name in names}

    @classmethod
    def _reject_ambiguous_spellings(cls, raw: dict[str, Any]) -> None:
        """Refuse a step that sets two spellings of the same knob.

        Aliases exist so each backend reads naturally (``task`` for ``task_name``,
        ``model``/``size`` for ``model_name``), which means a step *can* name one field
        twice. Pydantic already rejects that, but as ``extra="forbid"`` on whichever
        spelling it did not pick — a message that names the wrong problem — and only on
        the strict path, so the two readers of the same options disagreed about whether
        such a step was valid at all.

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
    def from_raw_lenient(cls, raw: dict[str, Any] | None) -> Self:
        """Validate only the keys this model knows, ignoring any others.

        For the one place strictness would be wrong: a direct ``step{N}.py`` template
        may carry knobs of its own that no engine model declares, and rendering it
        must not fail over them. Reading the raw dict by hand instead is what led to
        the alias rules (``model`` / ``size`` for ``model_name``) being spelled out a
        second time, by hand, next to the model that already declared them.
        """
        raw = raw or {}
        cls._reject_ambiguous_spellings(raw)
        accepted = cls._accepted_names()
        return cls._validate({k: v for k, v in raw.items() if k in accepted})

    @classmethod
    def _validate(cls, data: dict[str, Any]) -> Self:
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

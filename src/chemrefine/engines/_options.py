"""Shared base for per-engine YAML option models.

``step.options`` is a free-form ``dict[str, Any]`` so the orchestrator stays
engine-agnostic; each engine validates its own subset with a small Pydantic model.
:class:`EngineOptions` carries the knobs every compute engine shares — the ``device``
selector and the ``frozen`` / ``extra="forbid"`` config — so an engine's options model only
adds its own fields.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal, Self

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError

from chemrefine.config import USER_PATH
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


class Unkeyed:
    """Marks an options field the cache key leaves out: ``Annotated[int, UNKEYED]``.

    For a field that chooses where or how a job runs rather than what it computes — the
    interpreter, the core count, how long to wait on one call. A key that held one would tie
    a finished tree to the machine and the budget it ran with: raising the core count, or
    the timeout a failure named, would recompute every finished row on ``resume``, and
    ``rebuild-cache`` would refuse the tree as another configuration's
    (:func:`chemrefine.step.derive_step_key`). Declared on the field, so a model that adds
    such a knob marks it where it declares it.
    """


UNKEYED = Unkeyed()
"""The :class:`Unkeyed` marker an options field carries (:meth:`EngineOptions.unkeyed`)."""


class KeyedWhenSet:
    """Marks an options field the cache key leaves out while it is ``None``.

    For a knob added to a model whose steps already have finished trees, and which changes
    nothing a step computes while it is unset. A key that held its ``None`` would move every
    such step's key the day the knob ships, so ``resume`` would recompute finished trees whose
    results the knob cannot have changed, and ``rebuild-cache`` would refuse them as another
    configuration's. Once set, the field is keyed like any other.
    """


KEYED_WHEN_SET = KeyedWhenSet()
"""The :class:`KeyedWhenSet` marker an options field carries.

Read by :meth:`EngineOptions.key_exclusions`."""


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

    backend_python: Annotated[str | None, USER_PATH, UNKEYED] = None
    """Explicit Python interpreter for this step's compute backend (escape hatch).

    Normally unset: the provisioner (:mod:`chemrefine.engines._provision`) resolves the
    backend's managed env by *name* — no paths in the YAML. Set this only to force a
    specific interpreter (e.g. a hand-built env the provisioner doesn't manage)."""

    cores: Annotated[int, UNKEYED] = Field(1, ge=1)
    """Per-structure core budget. Lives here because it is what the scheduler asks
    every job engine for (``JobEngine.pal``), not something one backend invented."""

    @classmethod
    def unkeyed(cls) -> set[str]:
        """The fields marked :data:`UNKEYED`, which the cache key leaves out."""
        return {name for name, field in cls.model_fields.items() if UNKEYED in field.metadata}

    def key_exclusions(self) -> set[str]:
        """What the cache key leaves out of these options: :meth:`unkeyed`, and each unset
        :data:`KEYED_WHEN_SET` field."""
        unset = {
            name
            for name, field in type(self).model_fields.items()
            if KEYED_WHEN_SET in field.metadata and getattr(self, name) is None
        }
        return self.unkeyed() | unset

    @property
    def gpu_demand(self) -> int:
        """GPUs this step's options ask the scheduler for: one for ``device: cuda``, else none.

        The scheduler's read of ``device`` — the header it picks, the budget it charges, the
        device it assigns — belongs to the model beside the field, not to a helper that peeks
        at subclass fields by name: a model with another way to ask overrides this (a GPU
        switch of its own), and :meth:`chemrefine.engines._job.JobEngine.gpus` asks every declaring
        engine's model the same question. Read through the model rather than off the raw
        dict because a second reader with its own idea of "unset" can disagree with the
        field default: a step that named no device could render ``$DEVICE=cuda`` into its
        script while being scheduled as a CPU job on the CPU header — bypassing the GPU
        budget and :meth:`~chemrefine.throttle.Throttler.assign_device`, so concurrent local
        steps would pile onto device 0. One reader, one default.
        """
        return 1 if self.device == "cuda" else 0

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any] | None) -> Self:
        """Validate a raw ``step.options`` dict; empty/``None`` yields defaults.

        Strict — ``extra="forbid"`` means a typoed knob fails the step rather than
        being silently ignored. Subclasses override to add engine-specific
        required-field checks (a level of theory the user must name, say).
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

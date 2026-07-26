"""Shared base for per-engine YAML option models.

``step.options`` is a free-form ``dict[str, Any]`` so the orchestrator stays
engine-agnostic; each engine validates its own subset with a small Pydantic model.
:class:`EngineOptions` carries the knobs every compute engine shares — the ``device``
selector and the ``frozen`` / ``extra="forbid"`` config — so an engine's options model only
adds its own fields. It also lets the ExtOpt base type its ``options_cls`` ClassVar.
"""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class EngineOptions(BaseModel):
    """Base for an engine's validated ``step.options`` model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    device: Literal["cuda", "cpu"] = "cuda"
    """Compute device for this engine (``cuda`` ⇒ GPU, ``cpu`` ⇒ CPU)."""

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
        return cls(**(raw or {}))

    @classmethod
    def _accepted_names(cls) -> set[str]:
        """Every YAML spelling this model accepts — field names plus their aliases."""
        names: set[str] = set()
        for name, field in cls.model_fields.items():
            names.add(name)
            alias = field.validation_alias
            if isinstance(alias, str):
                names.add(alias)
            elif isinstance(alias, AliasChoices):
                names.update(c for c in alias.choices if isinstance(c, str))
        return names

    @classmethod
    def from_raw_lenient(cls, raw: dict[str, Any] | None) -> Self:
        """Validate only the keys this model knows, ignoring any others.

        For the one place strictness would be wrong: a direct ``step{N}.py`` template
        may carry knobs of its own that no engine model declares, and rendering it
        must not fail over them. Reading the raw dict by hand instead is what led to
        the alias rules (``model`` / ``size`` for ``model_name``) being spelled out a
        second time, by hand, next to the model that already declared them.
        """
        accepted = cls._accepted_names()
        return cls(**{k: v for k, v in (raw or {}).items() if k in accepted})

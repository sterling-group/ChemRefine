"""Shared base for per-engine YAML option models.

``step.options`` is a free-form ``dict[str, Any]`` so the orchestrator stays
engine-agnostic; each engine validates its own subset with a small Pydantic model.
:class:`EngineOptions` carries the knobs every compute engine shares — the ``device``
selector and the ``frozen`` / ``extra="forbid"`` config — so an engine's options model only
adds its own fields. It also lets the ExtOpt base type its ``options_cls`` ClassVar.
"""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict


class EngineOptions(BaseModel):
    """Base for an engine's validated ``step.options`` model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    device: Literal["cuda", "cpu"] = "cuda"
    """Compute device for this engine (``cuda`` ⇒ GPU, ``cpu`` ⇒ CPU)."""

    @classmethod
    def from_raw(cls, raw: dict[str, Any] | None) -> Self:
        """Validate a raw ``step.options`` dict; empty/``None`` yields defaults.

        Subclasses override to add engine-specific required-field checks (e.g. PySCF's
        ``basis`` / ``xc``).
        """
        return cls(**(raw or {}))

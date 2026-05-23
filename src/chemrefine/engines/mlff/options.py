"""Pydantic validator for MLFF engine YAML options.

Mirrors :class:`chemrefine.engines.pyscf.options.PyscfOptions` so a
new engine implementer adds an ``options.py`` per backend by the
same pattern. Both the direct (:class:`~chemrefine.engines.mlff.engine.MlffEngine`)
and ExtOpt (:class:`~chemrefine.engines.mlff.extopt_engine.MlffExtOptEngine`)
engines route ``step.options`` through this validator instead of
reading the raw YAML dict.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class MlffOptions(BaseModel):
    """Validated knobs for the MLFF backend."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_name: str = "uma-s-1"
    """Pretrained model identifier (e.g. UMA, MACE-MP, CHGNet, …)."""

    task_name: str = "omol"
    """Backend task / family selector consumed by :class:`MlffCalculator`."""

    model_path: str | None = None
    """Optional custom MACE checkpoint path (overrides ``model_name``)."""

    device: Literal["cuda", "cpu"] = "cuda"
    """Compute device for the MLFF model."""

    cores: int = Field(1, ge=1)
    """Per-structure core budget (passed to the throttler when applicable)."""

    @classmethod
    def from_raw(cls, raw: dict[str, Any] | None) -> MlffOptions:
        """Validate a raw ``step.options`` dict; empty/None yields defaults."""
        return cls(**(raw or {}))

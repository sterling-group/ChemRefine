"""SevenNet backend."""

from __future__ import annotations

from typing import Any

from chemrefine.engines.mlip.calculator import register_backend


@register_backend("sevenn")
def _build_sevenn(*, model_name: str = "", device: str = "cuda", **_: Any) -> Any:
    """SevenNet potential (``task_name: sevenn``); ``model_name`` is the 7net id."""
    from sevenn.calculator import SevenNetCalculator

    return SevenNetCalculator(model=model_name, device=device)

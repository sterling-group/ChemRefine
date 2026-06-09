"""CHGNet backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chemrefine.engines.mlip.calculator import optional_backend, register_backend


@register_backend("chgnet")
def _build_chgnet(
    *, model_path: str | Path | None = None, device: str = "cuda", **_: Any
) -> Any:
    """CHGNet universal potential (a local checkpoint via ``model_path``).

    Note: ``model_path`` reaches here only if a caller selects ``chgnet``
    explicitly with a checkpoint; the YAML ``model_path`` shortcut routes to
    ``custom_mace`` (matching main). The canonical import is
    ``from chgnet.model import CHGNet, CHGNetCalculator``.
    """
    with optional_backend(package="chgnet", extra="mlip-chgnet"):
        from chgnet.model import CHGNet, CHGNetCalculator

    model = CHGNet.load(str(model_path)) if model_path else CHGNet.load()
    return CHGNetCalculator(model=model, use_device=device)

"""CHGNet backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chemrefine.engines.mlip.registry import MlipLibrary

CHGNET = MlipLibrary(extra="mlip-chgnet", package="chgnet", import_name="chgnet")
"""The one declaration of what provides this library."""


@CHGNET.calculator("chgnet")
def _build_chgnet(*, model_path: str | Path | None = None, device: str = "cuda", **_: Any) -> Any:
    """CHGNet universal potential (a local checkpoint via ``model_path``).

    ``model_path`` reaches here whenever a step names ``task_name: chgnet`` with one, which is
    the same rule every library follows: the task names the library, and the path only says
    where its weights come from. The canonical import is
    ``from chgnet.model import CHGNet, CHGNetCalculator``.
    """
    from chgnet.model import CHGNet, CHGNetCalculator

    model = CHGNet.load(str(model_path)) if model_path else CHGNet.load()
    return CHGNetCalculator(model=model, use_device=device)

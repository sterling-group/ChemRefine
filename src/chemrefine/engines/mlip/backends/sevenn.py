"""SevenNet backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chemrefine.engines.mlip.registry import MlipLibrary

SEVENN = MlipLibrary(extra="mlip-sevenn", package="sevenn", import_name="sevenn")
"""The one declaration of what provides this library."""


@SEVENN.calculator("sevenn")
def _build_sevenn(
    *,
    model_name: str = "",
    device: str = "cuda",
    model_path: str | Path | None = None,
    **_: Any,
) -> Any:
    """SevenNet potential (``task_name: sevenn``); the weights come from name or path.

    ``model_path`` is honoured with SevenNet's own loader, like every builder's
    (:mod:`chemrefine.engines.mlip.registry`'s "three knobs" rule): ``SevenNetCalculator``'s
    ``model`` argument is typed ``str | Path`` — "Name of pretrained models … or path to the
    checkpoint" — and its resolution checks the filesystem before trying release names. The
    existence check runs before the library is imported, so a mistyped checkpoint is reported
    as the configuration mistake it is rather than as a loader traceback that names neither
    the step nor the option.
    """
    weights: str | Path = model_name
    if model_path is not None:
        weights = Path(model_path)
        if not weights.is_file():
            raise FileNotFoundError(f"SevenNet checkpoint not found: {weights}")

    from sevenn.calculator import SevenNetCalculator

    return SevenNetCalculator(model=weights, device=device)

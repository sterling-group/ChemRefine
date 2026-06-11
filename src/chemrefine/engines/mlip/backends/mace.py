"""MACE backends: the foundation models (off / mp / omol) + local checkpoints.

``task_name`` (``mace_off``/``mace_mp``/``mace_omol``) selects the variant; the
MACE ``model`` arg — the size (``small``/``medium``/``large``) or a named model —
is the ``model_name`` weights. Empty ``model_name`` falls back to each helper's
own default. A local checkpoint goes through ``custom_mace`` (auto-selected when
``model_path`` is set) and loads via :class:`mace.calculators.MACECalculator`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chemrefine.engines.mlip.calculator import optional_backend, register_backend


@register_backend("mace_off")
def _build_mace_off(*, model_name: str = "", device: str = "cuda", **_: Any) -> Any:
    """MACE-OFF foundation model (organic molecules); ``model_name`` = the size."""
    with optional_backend(package="mace-torch", extra="mlip-mace"):
        from mace.calculators import mace_off

    return mace_off(model=model_name or None, device=device)


@register_backend("mace_mp")
def _build_mace_mp(*, model_name: str = "", device: str = "cuda", **_: Any) -> Any:
    """MACE-MP foundation model (Materials Project); ``model_name`` = the size/model."""
    with optional_backend(package="mace-torch", extra="mlip-mace"):
        from mace.calculators import mace_mp

    return mace_mp(model=model_name or None, device=device)


@register_backend("mace_omol")
def _build_mace_omol(*, model_name: str = "", device: str = "cuda", **_: Any) -> Any:
    """MACE-OMOL foundation model (charge/spin embeddings); ``model_name`` = the size."""
    with optional_backend(package="mace-torch", extra="mlip-mace"):
        from mace.calculators import mace_omol

    return mace_omol(model=model_name or None, device=device)


@register_backend("custom_mace")
def _build_custom_mace(*, model_path: str | Path, device: str = "cuda", **_: Any) -> Any:
    """User-supplied MACE checkpoint at ``model_path``."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"custom MACE model not found: {path}")
    with optional_backend(package="mace-torch", extra="mlip-mace"):
        from mace.calculators import MACECalculator

    return MACECalculator(model_paths=str(path), device=device)

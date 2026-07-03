"""ORB backend (Orbital Materials) — a *separate* library, not SevenNet.

``model_name`` names a loader in :mod:`orb_models.forcefield.pretrained`
(e.g. ``orb_v3_conservative_inf_omat``, ``orb_v2``). The loader returns either
a model or a ``(model, atoms_adapter)`` tuple depending on the orb-models
version; both are handled. Requires ``pip install orb-models``.
"""

from __future__ import annotations

from typing import Any

from chemrefine.engines.mlip.calculator import register_backend


@register_backend("orb", extra="mlip-orb", package="orb-models", import_name="orb_models")
def _build_orb(*, model_name: str = "", device: str = "cuda", **_: Any) -> Any:
    """ORB potential; ``model_name`` picks a loader from ``orb_models...pretrained``."""
    from orb_models.forcefield import pretrained

    try:  # v3 layout
        from orb_models.forcefield.inference.calculator import ORBCalculator
    except ImportError:  # older layout
        from orb_models.forcefield.calculator import ORBCalculator

    loader = getattr(pretrained, model_name, None)
    if loader is None:
        raise ValueError(
            f"unknown ORB model {model_name!r}; pick a loader from "
            "orb_models.forcefield.pretrained (e.g. 'orb_v3_conservative_inf_omat')"
        )
    loaded = loader(device=device)
    orbff = loaded[0] if isinstance(loaded, tuple) else loaded
    return ORBCalculator(orbff, device=device)

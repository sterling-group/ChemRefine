"""ORB backend (Orbital Materials) — a *separate* library, not SevenNet.

``model_name`` names a loader in :mod:`orb_models.forcefield.pretrained`
(e.g. ``orb_v3_conservative_inf_omat``, ``orb_v2``). The loader returns either
a model or a ``(model, atoms_adapter)`` tuple depending on the orb-models
version; both are handled. Requires ``pip install orb-models``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chemrefine.engines.mlip.registry import MlipLibrary
from chemrefine.errors import ConfigError

ORB = MlipLibrary(extra="mlip-orb", package="orb-models", import_name="orb_models")
"""The one declaration of what provides this library."""


@ORB.calculator("orb")
def _build_orb(
    *,
    model_name: str = "",
    device: str = "cuda",
    model_path: str | Path | None = None,
    **_: Any,
) -> Any:
    """ORB potential; ``model_name`` picks a loader from ``orb_models...pretrained``.

    ``model_path`` is honoured with ORB's own loader, like every builder's
    (:mod:`chemrefine.engines.mlip.registry`'s "three knobs" rule): the pretrained loaders
    take ``weights_path``, defaulting to the release URL and accepting a local file. The
    loader is still selected by ``model_name`` — a checkpoint carries weights, not an
    architecture, so the loader that built it must be named alongside it, exactly as the
    library that trained it must. An older loader without the keyword is reported as the
    version limitation it is (:class:`~chemrefine.errors.ConfigError`), not left as a
    ``TypeError`` naming neither the step nor the option.
    """
    weights: Path | None = None
    if model_path is not None:
        weights = Path(model_path)
        if not weights.is_file():
            raise FileNotFoundError(f"ORB checkpoint not found: {weights}")

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
    if weights is None:
        loaded = loader(device=device)
    else:
        try:
            loaded = loader(weights_path=str(weights), device=device)
        except TypeError as e:
            raise ConfigError(
                f"this orb-models version's {model_name!r} loader takes no local "
                f"weights_path, so model_path cannot be honoured; upgrade orb-models "
                f"or drop model_path to run the named release"
            ) from e
    orbff = loaded[0] if isinstance(loaded, tuple) else loaded
    return ORBCalculator(orbff, device=device)

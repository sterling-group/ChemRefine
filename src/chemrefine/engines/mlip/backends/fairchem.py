"""FAIRChem backend (UMA / eSEN): one builder, every task head.

FAIRChem is *one family with many checkpoints and a fixed set of task heads*.
``model_name`` selects the checkpoint (any ``uma-…`` or ``esen-…`` id in
``pretrained_mlip.available_models``); ``task_name`` *is* the head and keys the
registry, so it is passed straight through to
:class:`~fairchem.core.FAIRChemCalculator`. The one builder is registered under
each of the 7 heads, so ``task_name: omat`` builds the materials head — not a
hardcoded ``omol``.
"""

from __future__ import annotations

from typing import Any

from chemrefine.engines.mlip.calculator import optional_backend, register_backend

#: Default checkpoint when ``model_name`` is unset — UMA-1.2, the latest small UMA
#: model (fastest while still SOTA on most benchmarks); ships with ``fairchem-core>=2.18``.
_DEFAULT_MODEL = "uma-s-1p2"


@register_backend("omol")  # molecules & polymers (needs charge + spin)
@register_backend("omat")  # inorganic materials
@register_backend("odac")  # MOFs for direct air capture
@register_backend("oc20")  # heterogeneous catalysis
@register_backend("oc22")  # oxide catalysts
@register_backend("oc25")  # electrolyte interfaces
@register_backend("omc")  # molecular crystals
def _build_fairchem(*, task_name: str, model_name: str = "", device: str = "cuda", **_: Any) -> Any:
    """FAIRChem (UMA/eSEN): ``task_name`` is the head, ``model_name`` the checkpoint."""
    with optional_backend(package="fairchem-core", extra="mlip-fairchem"):
        from fairchem.core import FAIRChemCalculator, pretrained_mlip

    predictor = pretrained_mlip.get_predict_unit(
        model_name=model_name or _DEFAULT_MODEL, device=device
    )
    return FAIRChemCalculator(predictor, task_name=task_name)

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

from chemrefine.engines.mlip.calculator import register_backend

# One declaration per module: the extra that provides this backend, the pip distribution
# named in the actionable import error, and the module the provisioner probes.
_EXTRA = "mlip-fairchem"
_PACKAGE = "fairchem-core"
_IMPORT = "fairchem"

#: Default checkpoint when ``model_name`` is unset — UMA-1.2, the latest small UMA
#: model (fastest while still SOTA on most benchmarks); ships with ``fairchem-core>=2.18``.
_DEFAULT_MODEL = "uma-s-1p2"


@register_backend("omol", extra=_EXTRA, package=_PACKAGE, import_name=_IMPORT)  # molecules
@register_backend("omat", extra=_EXTRA, package=_PACKAGE, import_name=_IMPORT)  # materials
@register_backend("odac", extra=_EXTRA, package=_PACKAGE, import_name=_IMPORT)  # DAC MOFs
@register_backend("oc20", extra=_EXTRA, package=_PACKAGE, import_name=_IMPORT)  # catalysis
@register_backend("oc22", extra=_EXTRA, package=_PACKAGE, import_name=_IMPORT)  # oxides
@register_backend("oc25", extra=_EXTRA, package=_PACKAGE, import_name=_IMPORT)  # electrolytes
@register_backend("omc", extra=_EXTRA, package=_PACKAGE, import_name=_IMPORT)  # mol. crystals
def _build_fairchem(*, task_name: str, model_name: str = "", device: str = "cuda", **_: Any) -> Any:
    """FAIRChem (UMA/eSEN): ``task_name`` is the head, ``model_name`` the checkpoint."""
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    predictor = pretrained_mlip.get_predict_unit(
        model_name=model_name or _DEFAULT_MODEL, device=device
    )
    return FAIRChemCalculator(predictor, task_name=task_name)

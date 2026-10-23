"""What every MLIP engine needs from the managed-env provisioner.

The backend a step runs on is a property of the **library**, not of the engine kind: ``mlip``
(in-process inference), ``mlip-extopt`` (a gradient server ORCA talks to) and ``mlip-train``
(fine-tuning) all resolve the same library from the same ``task_name`` selection, and all
need it in its own environment because the MLIP dependency trees conflict and cannot share
one.

Declared once, here, and mixed into each engine — rather than repeated per engine, where the
three are free to drift apart.

``mlip-train`` mixes this in like the other two, and overrides one method: a training step
needs the library it selected to be *trainable*, not merely installable. Asking that here
means ``preflight_backends`` refuses an untrainable ``task_name`` before any step submits,
rather than at the training step itself — in a pipeline that spends days computing labels
first, that is the difference between a typo caught in seconds and one caught on Thursday.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from chemrefine.engines.api import BackendRequirement
from chemrefine.engines.mlip.options import MlipOptions
from chemrefine.engines.mlip.registry import registered_extras, requirement_from_options
from chemrefine.errors import ConfigError

logger = logging.getLogger(__name__)


class MlipBackend:
    """Mixin: this engine's compute backend is an MLIP library, resolved by ``task_name``.

    Satisfies :class:`~chemrefine.engines.api.ProvisionableEngine` for any MLIP engine, so
    ``preflight_backends`` checks it before a run submits anything and the step launches from
    the matching managed env.
    """

    def backend_requirement(self, options: Mapping[str, Any] | None) -> BackendRequirement:
        """The env this step needs, from its task/model selection.

        Through :func:`~chemrefine.engines.mlip.registry.requirement_from_options`, which
        reads the selection through the options model — so the env the preflight demands is
        always the one the step would actually load — and refuses a ``model_name`` its
        library does not release.
        """
        return requirement_from_options(options)

    def backend_extras(self) -> frozenset[str]:
        """Every extra a registered MLIP library declares."""
        return registered_extras()

    def resolve_options(
        self,
        options: Mapping[str, Any],
        launcher: str,
        previous: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """``solvation_model: auto``, chosen in the step's own environment; else nothing.

        Satisfies :class:`~chemrefine.engines.api.OptionsResolving`. Each model ``auto``
        passed over is logged with where its weights were looked for, so a run that falls
        back says why it did.

        A step whose last run chose a model ``auto`` prefers to the one chosen now (it ran
        on ``moe``, and the MoE weights are gone) is refused rather than recomputed: the
        fallback would replace that run's results with ones from a lesser model. Choosing a
        better model than last time is allowed, and recomputes the step with it.
        """
        from chemrefine.engines.mlip.solvation import AUTO, solvation_library
        from chemrefine.engines.mlip.solvation._choose import choose_in_environment

        opts = MlipOptions.from_raw_lenient(dict(options))
        if not (opts.solvation and opts.solvation_model == AUTO):
            return {}
        chosen, skipped = choose_in_environment(launcher, opts.solvation, opts.solvation_weights)
        before = (previous or {}).get("solvation_model")
        order = solvation_library(opts.solvation).auto
        if before in order and chosen in order and order.index(before) < order.index(chosen):
            raise ConfigError(
                f"solvation_model: {AUTO} chose {before} when this step last ran, and "
                f"{before} is not available now ({'; '.join(skipped)}). Recomputing would "
                f"replace those results with {chosen}'s. Restore the {before} weights, set "
                f"`solvation_model: {chosen}` to recompute on {chosen} deliberately, or "
                f"`chemrefine rerun` the step."
            )
        for why in skipped:
            logger.warning("solvation_model: %s passed over %s", AUTO, why)
        return {"solvation_model": chosen}

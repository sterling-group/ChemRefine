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

from collections.abc import Mapping
from typing import Any

from chemrefine.engines.api import BackendRequirement
from chemrefine.engines.mlip.registry import registered_extras, requirement_from_options


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
        always the one the step would actually load.
        """
        return requirement_from_options(options)

    def backend_extras(self) -> frozenset[str]:
        """Every extra a registered MLIP library declares."""
        return registered_extras()

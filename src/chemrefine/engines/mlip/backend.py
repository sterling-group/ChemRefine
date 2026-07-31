"""What every MLIP engine needs from the managed-env provisioner.

The backend a step runs on is a property of the **backend family**, not of the engine kind:
``mlip`` (in-process inference) and ``mlip-extopt`` (a gradient server ORCA talks to) resolve
the same library from the same ``task_name``/``model_path`` selection, and both need it in its
own environment because the MLIP dependency trees conflict and cannot share one.

Declared once, here, and mixed into each engine — rather than repeated per engine, which is
how the two carried identical copies.

``mlip-train`` **does not use this yet, and adopting it is not a one-line change.** It is not
a :class:`~chemrefine.engines.api.ProvisionableEngine` at all today: it emits a bare
``mace_run_train`` and hopes it is on ``PATH``, where every inference engine resolves its
interpreter through :func:`chemrefine.engines._provision.launcher_for`. Mixing this in without
also routing the launch through the provisioner would be worse than leaving it — the preflight
would demand an env the training job then does not use. Both halves belong to the same piece of
work, which is also what a second trainer (FAIRChem) needs before it can exist at all, since it
cannot share MACE's environment.
"""

from __future__ import annotations

from typing import Any

from chemrefine.engines.api import BackendRequirement
from chemrefine.engines.mlip.calculator import registered_extras, requirement_from_options


class MlipBackend:
    """Mixin: this engine's compute backend is an MLIP library, resolved by ``task_name``.

    Satisfies :class:`~chemrefine.engines.api.ProvisionableEngine` for any MLIP engine, so
    ``preflight_backends`` checks it before a run submits anything and the step launches from
    the matching managed env.
    """

    def backend_requirement(self, options: dict[str, Any] | None) -> BackendRequirement:
        """The env this step needs, from its task/model selection.

        Through :func:`~chemrefine.engines.mlip.calculator.requirement_from_options`, which
        reads the selection through the options model — so the env the preflight demands is
        always the one the calculator would actually load.
        """
        return requirement_from_options(options)

    def backend_extras(self) -> frozenset[str]:
        """Every extra a registered MLIP backend declares."""
        return registered_extras()

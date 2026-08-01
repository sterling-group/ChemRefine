"""What every PySCF engine needs from the managed-env provisioner.

The backend a step runs on is a property of the **backend family**, not of the engine kind:
``pyscf`` (a rendered script) and ``pyscf-extopt`` (a gradient server ORCA talks to) both need
PySCF, whatever their options say. Declared once, here, and mixed into each engine — rather
than repeated per engine, where the two are free to drift apart.

Constant where :class:`~chemrefine.engines.mlip.backend.MlipBackend` derives: PySCF is one
library, so there is nothing to select. That difference is a property of the backends, not an
inconsistency between the mixins.
"""

from __future__ import annotations

from typing import Any

from chemrefine.engines.api import BackendRequirement

_PYSCF = BackendRequirement(extra="pyscf", import_name="pyscf")


class PyscfBackend:
    """Mixin: this engine's compute backend is PySCF.

    Satisfies :class:`~chemrefine.engines.api.ProvisionableEngine`, so ``preflight_backends``
    checks it before a run submits anything and the step launches from the matching managed
    env (which is also where ``gpu4pyscf`` lives, when the step asks for a GPU).
    """

    def backend_requirement(self, options: dict[str, Any] | None) -> BackendRequirement:
        """PySCF, whatever the options say."""
        return _PYSCF

    def backend_extras(self) -> frozenset[str]:
        """The one extra this engine can require."""
        return frozenset({_PYSCF.extra})

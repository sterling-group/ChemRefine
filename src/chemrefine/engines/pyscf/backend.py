"""What every PySCF engine needs from the managed-env provisioner.

The backend a step runs on is a property of the **backend family**, not of the engine kind:
``pyscf`` (a rendered script) and ``pyscf-extopt`` (a gradient server ORCA talks to) both need
PySCF, whatever their options say. Declared once, here, and mixed into each engine — rather
than repeated per engine, where the two are free to drift apart.

Derived from the step's options, like :class:`~chemrefine.engines.mlip.backend.MlipBackend`:
PySCF is one library, but it has two stacks, and a step that asks for a GPU needs the one
with ``gpu4pyscf`` in it. Both install into a single env — see
:func:`chemrefine.engines._provision.backend_env_path`.
"""

from __future__ import annotations

from typing import Any

from chemrefine.engines.api import BackendRequirement
from chemrefine.engines.pyscf.options import PyscfOptions

_PYSCF = BackendRequirement(extra="pyscf", import_name="pyscf")

_PYSCF_GPU = BackendRequirement(extra="pyscf-gpu", import_name="gpu4pyscf")
"""The GPU stack. ``import_name`` is the load-bearing half.

Probed for ``gpu4pyscf`` rather than ``pyscf``, a step that asks for a GPU now fails at
preflight, by name, on an env that has only the CPU stack. Probed for ``pyscf`` — as it was
when this requirement did not exist — the step started, :func:`
chemrefine.engines.pyscf._runtime._build_scf` caught the missing import, fell back to CPU,
and recorded why in the ExtOpt *server* log: a run that succeeds on the wrong hardware and
says so nowhere the user is looking."""


class PyscfBackend:
    """Mixin: this engine's compute backend is PySCF.

    Satisfies :class:`~chemrefine.engines.api.ProvisionableEngine`, so ``preflight_backends``
    checks it before a run submits anything and the step launches from the matching managed
    env.
    """

    def backend_requirement(self, options: dict[str, Any] | None) -> BackendRequirement:
        """PySCF — the GPU stack when the step asks for one.

        Read through :class:`~chemrefine.engines.pyscf.options.PyscfOptions` rather than off
        the raw dict, because ``gpu`` is *derived* from ``device`` when it is not given
        (``_derive_gpu_from_device``): a second reader would decide differently for
        ``device: cuda`` alone, and demand the wrong stack.
        """
        return _PYSCF_GPU if PyscfOptions.from_raw_lenient(options or {}).gpu else _PYSCF

    def backend_extras(self) -> frozenset[str]:
        """Both stacks — either is installable, and they share one env."""
        return frozenset({_PYSCF.extra, _PYSCF_GPU.extra})

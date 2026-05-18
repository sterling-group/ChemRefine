"""Engine plugin layer.

Importing this package self-registers every bundled engine into
:data:`chemrefine.engines.base.ENGINES`. The orchestrator imports
:mod:`chemrefine.engines` once and then looks up engines by name from
that dict — it never imports a concrete engine module directly.

New engines join the registry by listing their package below; each
engine's own ``__init__`` registers it via the
:func:`~chemrefine.engines.base.register` decorator at import time.
"""

from chemrefine.engines import _fake, mlff, orca, pyscf

__all__ = ["_fake", "mlff", "orca", "pyscf"]

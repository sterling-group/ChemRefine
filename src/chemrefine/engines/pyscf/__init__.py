"""PySCF engine — importing this module registers ``pyscf`` and ``pyscf-direct``.

Two engines ship in this package:

* ``PyscfEngine`` (registered as ``"pyscf"``) — drives ORCA's external
  optimizer protocol with a PySCF (or gpu4pyscf) gradient server.
* ``PyscfDirectEngine`` (registered as ``"pyscf-direct"``) — runs PySCF
  in-process, no ORCA, for quick single-point evaluations.

Both engines are ported from the unmerged ``origin/pyscf`` PR
(commits ``334e85c`` + ``df4600b``). The server / client / direct-mode
internals carry TODO placeholders until a real-run regression is set up.
"""

from chemrefine.engines.pyscf.direct import PyscfDirectEngine
from chemrefine.engines.pyscf.engine import PyscfEngine

__all__ = ["PyscfDirectEngine", "PyscfEngine"]

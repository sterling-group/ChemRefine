"""PySCF engine — importing this module registers ``pyscf`` and ``pyscf-direct``.

Two engines ship in this package:

* ``PyscfEngine`` (registered as ``"pyscf"``) — drives ORCA's external
  optimizer protocol with a PySCF (or gpu4pyscf) gradient server. The
  HTTP / ExtOpt plumbing lives in :mod:`chemrefine.engines._extopt`;
  only the SCF + gradient adapter lives here as :mod:`.extopt_calc`.
* ``PyscfDirectEngine`` (registered as ``"pyscf-direct"``) — runs PySCF
  in-process, no ORCA, for quick single-point evaluations.

The actual SCF / gradient body in :class:`PyscfExtOptCalculator.calc`
lands in B6 (ported from
``origin/codex/add-function-to-save-tensor-integrals``); until then it
raises :class:`NotImplementedError` so a misconfigured ``engine: pyscf``
step fails loudly rather than silently.
"""

from chemrefine.engines.pyscf.direct import PyscfDirectEngine
from chemrefine.engines.pyscf.engine import PyscfEngine
from chemrefine.engines.pyscf.extopt_calc import PyscfExtOptCalculator

__all__ = ["PyscfDirectEngine", "PyscfEngine", "PyscfExtOptCalculator"]

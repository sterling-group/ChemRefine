"""PySCF engine — importing this module registers ``pyscf`` and ``pyscf-extopt``.

Two engines ship in this package:

* ``PyscfEngine`` (registered as ``"pyscf"``) — template-driven direct
  PySCF. The user supplies ``step{N}.py``; ChemRefine renders one per
  structure and runs it via SLURM (auto-falls back to local bash).
* ``PyscfExtOptEngine`` (registered as ``"pyscf-extopt"``) — drives ORCA's
  external optimizer protocol with a PySCF (or gpu4pyscf) gradient
  server. The HTTP / ExtOpt plumbing lives in
  :mod:`chemrefine.engines._extopt`; only the SCF + gradient adapter
  lives here as :mod:`.extopt_calc`.
"""

from chemrefine.engines.pyscf.engine import PyscfEngine
from chemrefine.engines.pyscf.extopt_calc import PyscfExtOptCalculator
from chemrefine.engines.pyscf.extopt_engine import PyscfExtOptEngine

__all__ = ["PyscfEngine", "PyscfExtOptCalculator", "PyscfExtOptEngine"]

"""Flask/Waitress HTTP server exposing PySCF energies + gradients.

TODO: port from the unmerged ``origin/pyscf`` PR. The reference is
``src/chemrefine/pyscf_server.py`` on that branch. Highlights:

* CLI flags: ``--bind``, ``--nthreads``, ``--log-file``,
  ``--default-method`` (``dft``/``hf``), ``--default-xc``,
  ``--default-basis``, ``--default-df``, ``--default-gpu``.
* Route ``POST /calculate`` accepts an optional ``settings`` block in
  the payload that overrides the per-process defaults.
* Builds a :class:`pyscf.gto.Mole` from atom types + Å coordinates;
  selects ``RKS`` for closed-shell (spin == 0) and ``UKS`` otherwise.
* When ``--gpu`` is set, lazy-imports ``gpu4pyscf.dft.RKS``/``UKS`` and
  uses those instead.

This module exists today only so the ORCA-driven engine (``%method ...
ProgExt ...``) can reference ``python -m chemrefine.engines.pyscf.server``
in its run-block — see :class:`PyscfEngine`.
"""

from __future__ import annotations


def main() -> int:
    """Server entry point (placeholder)."""
    raise NotImplementedError(
        "PySCF server not yet ported — see TODO in engines/pyscf/server.py"
    )

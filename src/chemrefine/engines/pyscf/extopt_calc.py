"""PySCF backend for the shared ExtOpt server (placeholder until B6).

The :class:`PyscfExtOptCalculator` shape is fixed in B1 so the shared
server can route ``--backend pyscf`` requests; the actual
:meth:`calc` body — the PySCF / gpu4pyscf SCF + gradient — lands in
B6 once the :mod:`chemrefine.engines.pyscf._runtime` helpers are
ported from
``origin/codex/add-function-to-save-tensor-integrals``.
"""

from __future__ import annotations

import argparse

from chemrefine.engines._extopt.base import (
    BaseExtOptCalculator,
    CalculationData,
)


class PyscfExtOptCalculator(BaseExtOptCalculator):
    """ExtOpt-side adapter for PySCF / gpu4pyscf gradient calls."""

    name = "pyscf"

    def __init__(
        self,
        *,
        method: str = "dft",
        xc: str = "pbe",
        basis: str = "def2-svp",
        df: bool = False,
        gpu: bool = False,
    ) -> None:
        self.method = method
        self.xc = xc
        self.basis = basis
        self.df = df
        self.gpu = gpu

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> PyscfExtOptCalculator:
        """Read the PySCF-relevant fields off the shared server CLI namespace."""
        return cls(
            method=args.method,
            xc=args.xc,
            basis=args.basis,
            df=args.df,
            gpu=args.gpu,
        )

    def calc(
        self, data: CalculationData
    ) -> tuple[float, list[list[float]]]:
        """Compute ``(energy_hartree, gradient_hartree_per_bohr)`` — body lands in B6."""
        raise NotImplementedError(
            "PySCF ExtOpt body not yet ported — see B6 in the v4 plan"
        )

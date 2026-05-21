"""MLFF backend for the shared ExtOpt server.

Wraps an :class:`MlffCalculator` (the ASE-style backend dispatcher in
``calculator.py``) and exposes it as a :class:`BaseExtOptCalculator` so
the shared ExtOpt server can serve MLFF gradients alongside PySCF
gradients without duplicating any HTTP / file-I/O glue.
"""

from __future__ import annotations

import argparse

from ase import Atoms

from chemrefine.constants import BOHR_TO_ANGSTROM, HARTREE_TO_EV
from chemrefine.engines._extopt.base import (
    BaseExtOptCalculator,
    CalculationData,
)
from chemrefine.engines.mlff.calculator import MlffCalculator


class MlffExtOptCalculator(BaseExtOptCalculator):
    """ExtOpt-side adapter for any MLFF backend ``MlffCalculator`` supports."""

    name = "mlff"

    def __init__(
        self,
        *,
        model_name: str | None,
        task_name: str,
        device: str,
        model_path: str | None = None,
    ) -> None:
        self._calculator = MlffCalculator(
            model_name=model_name or "",
            task_name=task_name,
            device=device,
            model_path=model_path,
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> MlffExtOptCalculator:
        """Read the MLFF-relevant fields off the shared server CLI namespace."""
        return cls(
            model_name=args.model,
            task_name=args.task_name,
            device=args.device,
            model_path=args.model_path,
        )

    def calc(
        self, data: CalculationData
    ) -> tuple[float, list[list[float]]]:
        """Score one geometry, return ``(energy_hartree, gradient_hartree_per_bohr)``."""
        atoms = Atoms(
            symbols=list(data.symbols),
            positions=data.positions_angstrom,
        )
        energy_ev, gradient_ev_per_a = self._calculator.single_point(atoms)
        energy_hartree = energy_ev / HARTREE_TO_EV
        gradient_hartree_per_bohr = [
            [component * BOHR_TO_ANGSTROM / HARTREE_TO_EV for component in row]
            for row in gradient_ev_per_a
        ]
        return energy_hartree, gradient_hartree_per_bohr

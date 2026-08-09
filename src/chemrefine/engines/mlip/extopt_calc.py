"""MLIP backend for the shared ExtOpt server.

Wraps an :class:`MlipCalculator` (the ASE-style backend dispatcher in
``calculator.py``) and exposes it as a :class:`ComputeBackend` so
the shared ExtOpt server can serve MLIP gradients alongside PySCF
gradients without duplicating any HTTP / file-I/O glue.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
from ase import Atoms

from chemrefine.engines._backend_server.base import (
    CalculationData,
    ComputeBackend,
    tokens_from_options,
)
from chemrefine.engines.mlip.calculator import MlipCalculator
from chemrefine.engines.mlip.options import MlipOptions
from chemrefine.quantities import convert

# YAML key → CLI flag mapping. YAML uses ``model_name`` / ``task_name`` /
# ``model_path`` (Pydantic-friendly underscores); the CLI uses kebab-case
# ``--model`` / ``--task-name`` / ``--model-path`` to match what argparse's
# ``dest`` rewrite expects.
_KEY_VALUE_FLAGS: tuple[tuple[str, str], ...] = (
    ("model_name", "--model"),
    ("task_name", "--task-name"),
    ("device", "--device"),
    ("model_path", "--model-path"),
)


class MlipExtOptCalculator(ComputeBackend):
    """ExtOpt-side adapter for any MLIP backend ``MlipCalculator`` supports."""

    name = "mlip"

    def __init__(
        self,
        *,
        model_name: str | None,
        task_name: str,
        device: str,
        model_path: str | None = None,
    ) -> None:
        self._calculator = MlipCalculator(
            model_name=model_name or "",
            task_name=task_name,
            device=device,
            model_path=model_path,
        )

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Register MLIP flags on a shared server / client parser.

        Defaults mirror :class:`MlipOptions`; the Pydantic model stays
        the canonical name + default source.
        """
        defaults = MlipOptions()
        parser.add_argument(
            "--model",
            default=defaults.model_name,
            help="MLIP model weights (a MACE size or a FAIRChem/SevenNet/ORB id)",
        )
        parser.add_argument(
            "--task-name",
            default=defaults.task_name,
            help="MLIP method / head — selects the backend (omol, mace_off, …)",
        )
        parser.add_argument(
            "--device",
            default=defaults.device,
            choices=["cuda", "cpu"],
            help="Compute device for the MLIP model",
        )
        parser.add_argument(
            "--model-path",
            default=None,
            help="Local checkpoint to load instead of --model, with the --task-name library",
        )

    @classmethod
    def server_cli_from_options(cls, options: dict[str, Any]) -> list[str]:
        """Translate validated YAML options into ``--flag value`` tokens (kebab-mapped)."""
        return tokens_from_options(options, value_flags=_KEY_VALUE_FLAGS)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> MlipExtOptCalculator:
        """Read the MLIP-relevant fields off the shared server CLI namespace."""
        return cls(
            model_name=args.model,
            task_name=args.task_name,
            device=args.device,
            model_path=args.model_path,
        )

    def calc(self, data: CalculationData) -> tuple[float, list[list[float]]]:
        """Score one geometry, return ``(energy_hartree, gradient_hartree_per_bohr)``."""
        atoms = Atoms(
            symbols=list(data.symbols),
            positions=data.positions_angstrom,
        )
        # The FAIRChem ``omol`` head reads charge + spin from atoms.info (other
        # backends/heads ignore them); pass ORCA's values through so charged /
        # open-shell molecules aren't silently treated as neutral singlets.
        atoms.info["charge"] = data.charge
        atoms.info["spin"] = data.multiplicity
        energy_ev, gradient_ev_per_a = self._calculator.single_point(atoms)
        energy_hartree = convert(energy_ev, "ev", "hartree")
        # convert() is typed to return ArrayLike for array input; asarray is a
        # no-copy pass-through on the ndarray it actually produces here.
        gradient_hartree_per_bohr = np.asarray(
            convert(gradient_ev_per_a, "ev/angstrom", "hartree/bohr")
        ).tolist()
        return energy_hartree, gradient_hartree_per_bohr

"""MLIP backend for the shared ExtOpt server.

Wraps an :class:`MlipCalculator` (the ASE-style backend dispatcher in
``calculator.py``) and exposes it as a :class:`ComputeBackend` so
the shared ExtOpt server can serve MLIP gradients alongside PySCF
gradients without duplicating any HTTP / file-I/O glue.
"""

from __future__ import annotations

import argparse
from typing import Any

from ase import Atoms

from chemrefine.engines._backend_server.base import (
    CalculationData,
    ComputeBackend,
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
            "--model", default=defaults.model_name,
            help="MLIP model weights (a MACE size or a FAIRChem/SevenNet/ORB id)",
        )
        parser.add_argument(
            "--task-name", default=defaults.task_name,
            help="MLIP method / head — selects the backend (omol, mace_off, …)",
        )
        parser.add_argument(
            "--device", default=defaults.device, choices=["cuda", "cpu"],
            help="Compute device for the MLIP model",
        )
        parser.add_argument(
            "--model-path", default=None,
            help="Custom MACE checkpoint path (selects the custom_mace backend)",
        )

    @classmethod
    def settings_from_args(cls, args: argparse.Namespace) -> dict[str, Any]:
        """MLIP has no per-call client knobs today — return empty dict."""
        return {}

    @classmethod
    def server_cli_from_options(cls, options: dict[str, Any]) -> list[str]:
        """Translate validated YAML options into a list of ``--flag value`` tokens.

        Falsy values are omitted so the engine's ``run_block`` only
        emits flags the user explicitly set.
        """
        tokens: list[str] = []
        for yaml_key, cli_flag in _KEY_VALUE_FLAGS:
            value = options.get(yaml_key)
            if value:
                tokens.extend([cli_flag, str(value)])
        return tokens

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> MlipExtOptCalculator:
        """Read the MLIP-relevant fields off the shared server CLI namespace."""
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
        # The FAIRChem ``omol`` head reads charge + spin from atoms.info (other
        # backends/heads ignore them); pass ORCA's values through so charged /
        # open-shell molecules aren't silently treated as neutral singlets.
        atoms.info["charge"] = data.charge
        atoms.info["spin"] = data.multiplicity
        energy_ev, gradient_ev_per_a = self._calculator.single_point(atoms)
        energy_hartree = convert(energy_ev, "ev", "hartree")
        gradient_hartree_per_bohr = convert(
            gradient_ev_per_a, "ev/angstrom", "hartree/bohr"
        ).tolist()
        return energy_hartree, gradient_hartree_per_bohr

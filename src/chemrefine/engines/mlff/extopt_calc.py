"""MLFF backend for the shared ExtOpt server.

Wraps an :class:`MlffCalculator` (the ASE-style backend dispatcher in
``calculator.py``) and exposes it as a :class:`BaseExtOptCalculator` so
the shared ExtOpt server can serve MLFF gradients alongside PySCF
gradients without duplicating any HTTP / file-I/O glue.
"""

from __future__ import annotations

import argparse
from typing import Any

from ase import Atoms

from chemrefine.engines._extopt.base import (
    BaseExtOptCalculator,
    CalculationData,
)
from chemrefine.engines.mlff.calculator import MlffCalculator
from chemrefine.engines.mlff.options import MlffOptions
from chemrefine.quantities import BOHR_TO_ANGSTROM, HARTREE_TO_EV

# YAML key → CLI flag mapping. YAML uses ``model_name`` / ``task_name`` /
# ``model_path`` (Pydantic-friendly underscores); the CLI uses kebab-case
# ``--model`` / ``--task-name`` / ``--model-path`` to match what
# argparse's ``dest`` rewrite expects.
_KEY_VALUE_FLAGS: tuple[tuple[str, str], ...] = (
    ("model_name", "--model"),
    ("task_name", "--task-name"),
    ("device", "--device"),
    ("model_path", "--model-path"),
)


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
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Register MLFF flags on a shared server / client parser.

        Defaults mirror :class:`MlffOptions`; the Pydantic model stays
        the canonical name + default source.
        """
        defaults = MlffOptions()
        parser.add_argument(
            "--model", default=None,
            help="MLFF pretrained model name (overrides default)",
        )
        parser.add_argument(
            "--task-name", default=defaults.task_name,
            help="MLFF task name / family selector",
        )
        parser.add_argument(
            "--device", default=defaults.device, choices=["cuda", "cpu"],
            help="Compute device for the MLFF model",
        )
        parser.add_argument(
            "--model-path", default=None,
            help="Custom MACE checkpoint path (overrides --model)",
        )

    @classmethod
    def settings_from_args(cls, args: argparse.Namespace) -> dict[str, Any]:
        """MLFF has no per-call client knobs today — return empty dict."""
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

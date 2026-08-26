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
from chemrefine.engines.mlip.options import CALCULATOR_KNOBS, MlipOptions
from chemrefine.quantities import convert

_FLAG_OVERRIDES = {"model_name": "--model"}
"""The one knob whose CLI spelling is not its kebab-cased field name.

``--model`` predates the rewrite and is pinned by the engine-rendered server command in
recorded runs; the rest follow the rule below, so a knob added to
:data:`~chemrefine.engines.mlip.options.CALCULATOR_KNOBS` grows a flag with no edit here."""

_KEY_VALUE_FLAGS: tuple[tuple[str, str], ...] = tuple(
    (name, _FLAG_OVERRIDES.get(name, "--" + name.replace("_", "-"))) for name in CALCULATOR_KNOBS
)
"""YAML key → CLI flag, derived from the knob list — kebab-case plus the one override."""

_FLAG_HELP = {
    "model_name": "MLIP model weights (a MACE size or a FAIRChem/SevenNet/ORB id)",
    "task_name": "MLIP method / head — selects the backend (omol, mace_off, …)",
    "device": "Compute device for the MLIP model",
    "model_path": "Local checkpoint to load instead of --model, with the --task-name library",
}
"""CLI presentation only; names and defaults come from the options model."""


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
        """Register MLIP flags on a shared server / client parser — one loop, no table.

        The flags come from the knob list, the defaults from the model's own fields, so
        this parser cannot come to disagree with :class:`MlipOptions` about either — the
        old hand-written quadruple restated both, including a ``--model-path`` default
        the model already declared.
        """
        for name, flag in _KEY_VALUE_FLAGS:
            kwargs: dict[str, Any] = {
                "default": MlipOptions.model_fields[name].default,
                # `.get`, so the claim above holds: a knob added to CALCULATOR_KNOBS grows a
                # flag here with no edit. Subscripting would make a knob without a help entry a
                # KeyError at server startup instead.
                "help": _FLAG_HELP.get(name, f"MLIP {name.replace('_', ' ')}"),
            }
            if name == "device":
                kwargs["choices"] = ["cuda", "cpu"]
            parser.add_argument(flag, **kwargs)

    @classmethod
    def server_cli_from_options(cls, options: dict[str, Any]) -> list[str]:
        """Translate validated YAML options into ``--flag value`` tokens (kebab-mapped)."""
        return tokens_from_options(options, value_flags=_KEY_VALUE_FLAGS)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> MlipExtOptCalculator:
        """Read the MLIP-relevant fields off the shared server CLI namespace.

        The same loop that added the flags reads them back — argparse's ``dest`` is the
        flag with dashes as underscores, so the pairs carry both directions.
        """
        values = {
            name: getattr(args, flag.lstrip("-").replace("-", "_"))
            for name, flag in _KEY_VALUE_FLAGS
        }
        return cls(**values)

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

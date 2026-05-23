"""PySCF backend for the shared ExtOpt server.

Wraps :mod:`chemrefine.engines.pyscf._runtime` so the shared ExtOpt
server can serve PySCF gradients via the same
:class:`BaseExtOptCalculator` contract MLFF uses. Optional active-space
tensor extraction is gated on the per-call ``settings['save_tensors']``
flag (which the wrapper-script CLI flips on with ``--save-tensors``).

The SCF + gradient body needs PySCF + (optionally) gpu4pyscf
installed; tests under :file:`tests/test_engines_pyscf_extopt_calc.py`
mock those imports so the call graph can be exercised in CI.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from chemrefine.engines._extopt.base import (
    BaseExtOptCalculator,
    CalculationData,
)
from chemrefine.engines.pyscf import _runtime
from chemrefine.engines.pyscf.options import PyscfOptions

logger = logging.getLogger(__name__)

# CLI flag names. The Pydantic ``PyscfOptions`` model owns the *defaults*;
# this tuple lists which fields are exposed on the ExtOpt CLI surface
# (server + client + engine run_block) so the three callers stay in lockstep.
_KEY_VALUE_FLAGS: tuple[str, ...] = ("method", "xc", "basis")
_BOOL_FLAGS: tuple[str, ...] = ("df", "gpu")


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
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Register PySCF flags on a shared server / client parser.

        Defaults mirror :class:`PyscfOptions`; the Pydantic model stays
        the canonical name + default source. Adding a knob here means
        also adding it to ``PyscfOptions`` (or vice-versa) — the
        ``_KEY_VALUE_FLAGS`` / ``_BOOL_FLAGS`` tuples gate which knobs
        are CLI-exposed.
        """
        defaults = PyscfOptions()
        parser.add_argument(
            "--method", default=defaults.method, choices=["dft", "hf"],
            help="SCF method (dft | hf)",
        )
        parser.add_argument(
            "--xc", default=defaults.xc,
            help="DFT exchange-correlation functional",
        )
        parser.add_argument(
            "--basis", default=defaults.basis,
            help="Orbital basis set",
        )
        parser.add_argument(
            "--df", action="store_true",
            help="Enable density fitting / RI",
        )
        parser.add_argument(
            "--gpu", action="store_true",
            help="Attempt gpu4pyscf if installed",
        )

    @classmethod
    def settings_from_args(cls, args: argparse.Namespace) -> dict[str, Any]:
        """Pack per-call PySCF knobs into the wrapper-script POST payload."""
        return {
            "method": args.method,
            "xc": args.xc,
            "basis": args.basis,
            "df": bool(args.df),
            "gpu": bool(args.gpu),
        }

    @classmethod
    def server_cli_from_options(cls, options: dict[str, Any]) -> list[str]:
        """Translate validated YAML options into a list of ``--flag value`` tokens.

        Falsy values (``None``, empty string, ``False``) are omitted so
        the engine's ``run_block`` only emits flags the user explicitly
        set.
        """
        tokens: list[str] = []
        for key in _KEY_VALUE_FLAGS:
            value = options.get(key)
            if value:
                tokens.extend([f"--{key}", str(value)])
        for flag in _BOOL_FLAGS:
            if options.get(flag):
                tokens.append(f"--{flag}")
        return tokens

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
        """Run the SCF + (optional) gradient, return ``(energy, gradient)`` in atomic units."""
        # Per-call overrides from the wrapper-script POST payload; missing
        # keys fall back to the per-process defaults set at construction.
        per_call = PyscfOptions.from_raw(
            {
                "method": data.settings.get("method", self.method),
                "xc": data.settings.get("xc", self.xc),
                "basis": data.settings.get("basis", self.basis),
                "df": bool(data.settings.get("df", self.df)),
                "gpu": bool(data.settings.get("gpu", self.gpu)),
                "save_tensors": bool(data.settings.get("save_tensors", False)),
                "localized": bool(data.settings.get("localized", False)),
                "tensor_folder": data.settings.get("tensor_folder", "tensors"),
            }
        )

        mol = _runtime.build_mol(
            symbols=data.symbols,
            positions_angstrom=data.positions_angstrom,
            charge=data.charge,
            multiplicity=data.multiplicity,
            basis=per_call.basis,
        )
        energy, gradient, meta, mf = _runtime.run_dft(
            mol,
            method=per_call.method,
            xc=per_call.xc,
            use_df=per_call.df,
            want_gpu=per_call.gpu,
            nthreads=data.nthreads,
            dograd=data.dograd,
        )
        logger.info(
            "PySCF calc: E=%.10f Eh converged=%s gpu=%s t=%.3fs",
            energy, meta["converged"], meta["gpu_used"], meta["elapsed_seconds"],
        )

        if per_call.save_tensors:
            tag = data.settings.get("tag") or meta.get("tag") or "untagged"
            nuc, h1, h2 = _runtime.get_active_space_tensors(
                mol, mf, localized=per_call.localized
            )
            target = Path(per_call.tensor_folder) / f"{tag}.npz"
            _runtime.save_tensors(path=target, nuc=nuc, h1=h1, h2=h2)
            logger.info("PySCF tensors saved: %s", target)

        return energy, gradient

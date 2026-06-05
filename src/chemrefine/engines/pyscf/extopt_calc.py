"""PySCF backend for the shared ExtOpt server.

Wraps :mod:`chemrefine.engines.pyscf._runtime` so the shared ExtOpt
server can serve PySCF gradients via the same
:class:`ComputeBackend` contract MLIP uses. **Single channel:** the SCF
knobs (method / xc / basis / df / gpu / tensor settings) are baked into
the calculator once, at server construction, from the step's YAML
options — the wrapper and the per-call POST carry nothing. Optional
active-space tensor extraction is gated on the server-constructed
``save_tensors`` flag; only the per-call correlation ``tag`` rides the
request (so dumps don't overwrite each other).

The SCF + gradient body needs PySCF + (optionally) gpu4pyscf
installed; tests under :file:`tests/test_engines_pyscf_extopt_calc.py`
mock those imports so the call graph can be exercised in CI.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from chemrefine.engines._backend_server.base import (
    CalculationData,
    ComputeBackend,
)
from chemrefine.engines.pyscf import _runtime
from chemrefine.engines.pyscf.options import PyscfOptions

logger = logging.getLogger(__name__)

# CLI flag names. The Pydantic ``PyscfOptions`` model owns the *defaults*;
# this tuple lists which fields are exposed on the ExtOpt CLI surface
# (server + client + engine run_block) so the three callers stay in lockstep.
# Flag spelling == YAML key == Pydantic field (underscores), so the generic
# ``--{key}`` token builder below needs no per-flag special-casing.
_KEY_VALUE_FLAGS: tuple[str, ...] = ("method", "xc", "basis", "tensor_folder")
_BOOL_FLAGS: tuple[str, ...] = ("df", "gpu", "save_tensors", "localized")


class PyscfExtOptCalculator(ComputeBackend):
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
        save_tensors: bool = False,
        localized: bool = False,
        tensor_folder: str = "tensors",
    ) -> None:
        self.method = method
        self.xc = xc
        self.basis = basis
        self.df = df
        self.gpu = gpu
        self.save_tensors = save_tensors
        self.localized = localized
        self.tensor_folder = tensor_folder

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
        parser.add_argument(
            "--save_tensors", action="store_true",
            help="Dump active-space 1e/2e MO tensors after the SCF",
        )
        parser.add_argument(
            "--localized", action="store_true",
            help="Boys-localize occupied / virtual orbitals before tensor extraction",
        )
        parser.add_argument(
            "--tensor_folder", default=defaults.tensor_folder,
            help="Directory (relative to $WORK_DIR) for save_tensors .npz output",
        )

    @classmethod
    def settings_from_args(cls, args: argparse.Namespace) -> dict[str, Any]:
        """No per-call client knobs — PySCF is single-channel like MLIP.

        The calculator is constructed from the server CLI (:meth:`from_args`),
        so the wrapper's POST carries no settings; the server still injects the
        correlation ``tag`` on its own.
        """
        return {}

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
            save_tensors=args.save_tensors,
            localized=args.localized,
            tensor_folder=args.tensor_folder,
        )

    def calc(
        self, data: CalculationData
    ) -> tuple[float, list[list[float]]]:
        """Run the SCF + (optional) gradient, return ``(energy, gradient)`` in atomic units.

        Single channel: the SCF knobs come from this instance (built once on the
        server from the step's YAML options); only the per-call correlation
        ``tag`` is read off the request, to key ``save_tensors`` dumps.
        """
        mol = _runtime.build_mol(
            symbols=data.symbols,
            positions_angstrom=data.positions_angstrom,
            charge=data.charge,
            multiplicity=data.multiplicity,
            basis=self.basis,
        )
        energy, gradient, meta, mf = _runtime.run_dft(
            mol,
            method=self.method,
            xc=self.xc,
            use_df=self.df,
            want_gpu=self.gpu,
            nthreads=data.nthreads,
            dograd=data.dograd,
        )
        logger.info(
            "PySCF calc: E=%.10f Eh converged=%s gpu=%s t=%.3fs",
            energy, meta["converged"], meta["gpu_used"], meta["elapsed_seconds"],
        )

        if self.save_tensors:
            # ``tag`` is the per-call correlation id the bridge derives from the
            # ``.extinp.tmp`` stem (one file per ORCA geometry step); the server
            # injects it into ``settings`` so dumps don't overwrite each other.
            tag = data.settings.get("tag") or "untagged"
            nuc, h1, h2 = _runtime.get_active_space_tensors(
                mol, mf, localized=self.localized
            )
            target = Path(self.tensor_folder) / f"{tag}.npz"
            _runtime.save_tensors(path=target, nuc=nuc, h1=h1, h2=h2)
            logger.info("PySCF tensors saved: %s", target)

        return energy, gradient

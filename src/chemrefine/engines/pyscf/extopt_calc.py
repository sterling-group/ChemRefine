"""PySCF backend for the shared ExtOpt server.

Wraps :mod:`chemrefine.engines.pyscf._runtime` so the shared ExtOpt
server can serve PySCF gradients via the same
:class:`BaseExtOptCalculator` contract MLFF uses. Optional active-space
tensor extraction is gated on the per-call ``settings['save_tensors']``
flag (which the wrapper-script CLI flips on with ``--save-tensors``).

Ported from
``origin/codex/add-function-to-save-tensor-integrals:src/chemrefine/pyscf_server.py``.
The actual SCF + gradient body needs PySCF + (optionally) gpu4pyscf
installed; tests under :file:`tests/test_engines_pyscf_extopt_calc.py`
mock those imports so the call graph can be exercised in CI.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from chemrefine.engines._extopt.base import (
    BaseExtOptCalculator,
    CalculationData,
)
from chemrefine.engines.pyscf import _runtime
from chemrefine.engines.pyscf.options import PyscfOptions

logger = logging.getLogger(__name__)


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

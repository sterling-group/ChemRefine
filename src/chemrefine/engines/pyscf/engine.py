"""ORCA-driven PySCF engine.

Mirrors :class:`MlffEngine` — same shared ExtOpt server, different
backend choice (``--backend pyscf``) and a different per-step CLI
that selects method / xc / basis / df / gpu. The actual PySCF SCF +
gradient lands in B6 inside :class:`PyscfExtOptCalculator.calc`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from chemrefine.engines.base import register
from chemrefine.engines.mlff.engine import (
    _build_extopt_run_block,
    _server_command,
)
from chemrefine.engines.orca.engine import OrcaEngine
from chemrefine.state import StepContext

logger = logging.getLogger(__name__)


@register("pyscf")
class PyscfEngine(OrcaEngine):
    """ORCA optimisation backed by a PySCF gradient server."""

    name = "pyscf"
    supports_nms = False

    # -- ORCA input customisation -----------------------------------------

    def _extra_blocks(self, ctx: StepContext) -> str:
        """Emit a ``%method ProgExt "<wrapper>"`` block + per-call settings."""
        wrapper = self._wrapper_path(ctx)
        ext_params = self._ext_params(ctx)
        return (
            "%method\n"
            f'  ProgExt "{wrapper}"\n'
            f'  Ext_Params "{ext_params}"\n'
            "end"
        )

    # -- SLURM customisation ----------------------------------------------

    def _run_block(self, ctx: StepContext, inp_path: Path, out_path: Path) -> str:
        """Spin up the ExtOpt server (pyscf backend), run ORCA, clean up."""
        options = ctx.step_cfg.options or {}
        extra: list[tuple[str, str]] = []
        method = options.get("method")
        if method:
            extra.append(("--method", str(method)))
        xc = options.get("xc")
        if xc:
            extra.append(("--xc", str(xc)))
        basis = options.get("basis")
        if basis:
            extra.append(("--basis", str(basis)))
        # Boolean flags rendered as bare ``--df`` / ``--gpu``
        bool_flags = [
            f"--{flag}" for flag in ("df", "gpu") if options.get(flag)
        ]

        server_cmd = _server_command(backend="pyscf", extra_flags=extra)
        if bool_flags:
            server_cmd = f"{server_cmd} {' '.join(bool_flags)}"

        return _build_extopt_run_block(
            server_cmd=server_cmd,
            orca_executable=ctx.orca_executable,
            inp_name=inp_path.name,
            out_name=out_path.name,
        )

    # -- Helpers -----------------------------------------------------------

    def _wrapper_path(self, ctx: StepContext) -> Path:
        """Path of the per-step ``ProgExt`` wrapper script."""
        return (ctx.step_dir / "pyscf_extopt.sh").resolve()

    def _ext_params(self, ctx: StepContext) -> str:
        """Build the ``Ext_Params`` string passed to the wrapper script.

        These flags reach the shared client (``_extopt.client``) which
        forwards them as the ``settings`` block in each ``/calculate``
        POST so per-call overrides work without restarting the server.
        """
        options = ctx.step_cfg.options or {}
        parts: list[str] = []
        for flag in ("method", "xc", "basis"):
            value = options.get(flag)
            if value is not None:
                parts.append(f"--{flag} {value}")
        for flag in ("df", "gpu"):
            if options.get(flag):
                parts.append(f"--{flag}")
        return " ".join(parts)

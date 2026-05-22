"""ORCA-driven PySCF engine.

Mirrors :class:`MlffEngine` — same shared ExtOpt server, different
backend choice (``--backend pyscf``) and a different per-step CLI
that selects method / xc / basis / df / gpu. The actual SCF +
gradient is computed by :class:`PyscfExtOptCalculator`.
"""

from __future__ import annotations

import logging

from chemrefine.engines._extopt import run_block
from chemrefine.engines._extopt.orca_engine import ExtOptOrcaEngine
from chemrefine.engines.base import register
from chemrefine.state import StepContext

logger = logging.getLogger(__name__)


@register("pyscf")
class PyscfEngine(ExtOptOrcaEngine):
    """ORCA optimisation backed by a PySCF gradient server."""

    name = "pyscf"
    wrapper_filename = "pyscf_extopt.sh"

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

    def _server_cmd(self, ctx: StepContext) -> str:
        """Build the ``python -m ..._extopt.server --backend pyscf ...`` command."""
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

        server_cmd = run_block._server_command(backend="pyscf", extra_flags=extra)
        if bool_flags:
            server_cmd = f"{server_cmd} {' '.join(bool_flags)}"
        return server_cmd

    # -- Helpers -----------------------------------------------------------

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

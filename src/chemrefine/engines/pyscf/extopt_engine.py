"""ORCA-driven PySCF engine (ORCA optimises, PySCF provides gradients).

Mirrors :class:`MlffExtOptEngine` — same shared ExtOpt server, different
backend choice (``--backend pyscf``) and a different per-step CLI
that selects method / xc / basis / df / gpu. The actual SCF +
gradient is computed by :class:`PyscfExtOptCalculator`.
"""

from __future__ import annotations

import logging

from chemrefine.engines._extopt import run_block
from chemrefine.engines._extopt.orca_engine import ExtOptOrcaEngine
from chemrefine.engines.base import register
from chemrefine.engines.pyscf.extopt_calc import PyscfExtOptCalculator
from chemrefine.state import StepContext

logger = logging.getLogger(__name__)


@register("pyscf-extopt")
class PyscfExtOptEngine(ExtOptOrcaEngine):
    """ORCA optimisation backed by a PySCF gradient server."""

    name = "pyscf-extopt"
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
        """Build the ``python -m ..._extopt.server --backend pyscf ...`` command.

        Delegates CLI generation to
        :meth:`PyscfExtOptCalculator.server_cli_from_options` so the
        flag list lives in exactly one module.
        """
        tokens = PyscfExtOptCalculator.server_cli_from_options(
            ctx.step_cfg.options or {}
        )
        return run_block._server_command(backend="pyscf", extra_tokens=tokens)

    # -- Helpers -----------------------------------------------------------

    def _ext_params(self, ctx: StepContext) -> str:
        """Build the ``Ext_Params`` string passed to the wrapper script.

        These flags reach the shared client (``_extopt.client``) which
        forwards them as the ``settings`` block in each ``/calculate``
        POST so per-call overrides work without restarting the server.
        """
        tokens = PyscfExtOptCalculator.server_cli_from_options(
            ctx.step_cfg.options or {}
        )
        return " ".join(tokens)

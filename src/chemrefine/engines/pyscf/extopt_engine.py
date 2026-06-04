"""ORCA-driven PySCF engine (ORCA optimises, PySCF provides gradients).

Mirrors :class:`MlipExtOptEngine` — same shared ExtOpt server, different
backend choice (``--backend pyscf``) and a different per-step CLI
that selects method / xc / basis / df / gpu. The actual SCF +
gradient is computed by :class:`PyscfExtOptCalculator`.
"""

from __future__ import annotations

import logging

from chemrefine.engines.base import register
from chemrefine.engines.orca.extopt import run_block
from chemrefine.engines.orca.extopt.engine import ExtOptOrcaEngine
from chemrefine.engines.pyscf.extopt_calc import PyscfExtOptCalculator
from chemrefine.state import StepContext

logger = logging.getLogger(__name__)


@register("pyscf-extopt")
class PyscfExtOptEngine(ExtOptOrcaEngine):
    """ORCA optimisation backed by a PySCF gradient server."""

    name = "pyscf-extopt"
    backend = "pyscf"
    wrapper_filename = "pyscf_extopt.sh"

    # -- ORCA input customisation -----------------------------------------

    def _extra_blocks(self, ctx: StepContext) -> str:
        """Emit a ``%method ProgExt "<wrapper>"`` block.

        The PySCF method / xc / basis selection is baked into the wrapper's
        client invocation (see :meth:`_wrapper_extra_args`), so ORCA needs no
        ``Ext_Params`` — the wrapper takes only the ``.extinp.tmp`` ORCA hands
        it. This matches :class:`MlipExtOptEngine`'s single-channel design.
        """
        wrapper = self._wrapper_path(ctx)
        return f'%method\n  ProgExt "{wrapper}"\nend'

    # -- SLURM customisation ----------------------------------------------

    def _option_tokens(self, ctx: StepContext) -> list[str]:
        """The PySCF CLI flag tokens for this step's options — built in one place.

        Both consumers (the server command and the wrapper's client args)
        derive from here so the flag list lives in exactly one module
        (:meth:`PyscfExtOptCalculator.server_cli_from_options`).
        """
        return PyscfExtOptCalculator.server_cli_from_options(ctx.step_cfg.options or {})

    def _server_cmd(self, ctx: StepContext) -> str:
        """Build the ``python -m ..._backend_server.server --backend pyscf ...`` command."""
        return run_block._server_command(
            backend=self.backend, extra_tokens=self._option_tokens(ctx)
        )

    def _wrapper_extra_args(self, ctx: StepContext) -> str:
        """Bake this step's PySCF flags into the wrapper's client invocation.

        The client's argparse defaults would otherwise post default settings
        that override the server's real construction, so the wrapper must
        carry the step's actual method / xc / basis.
        """
        return " ".join(self._option_tokens(ctx))

"""ORCA-driven MLFF engine (ORCA optimises, MLFF provides gradients).

This engine reuses the entire ORCA submission pipeline (input writer,
SLURM, output parser) and only differs in two ways:

1. The generated ORCA ``.inp`` carries a ``%method ... ProgExt ... end``
   block pointing at a wrapper script. ORCA invokes that wrapper at each
   optimization step; the wrapper relays the geometry to a long-running
   ExtOpt HTTP server and pipes the resulting energy + gradient back as
   ``.engrad``.
2. The SLURM ``run_block`` (assembled by
   :class:`~chemrefine.engines.orca.extopt.engine.ExtOptOrcaEngine`)
   starts the shared :mod:`chemrefine.engines._backend_server.server` with
   ``--backend mlff`` before ORCA, probes ``/healthz`` until the server
   is ready, and tears it down on exit.

Backend selection / model name are read from ``step.options`` in the
YAML via :class:`~chemrefine.engines.mlff.options.MlffOptions`.
"""

from __future__ import annotations

import logging

from chemrefine.engines.base import register
from chemrefine.engines.mlff.extopt_calc import MlffExtOptCalculator
from chemrefine.engines.mlff.options import MlffOptions
from chemrefine.engines.orca.extopt import run_block
from chemrefine.engines.orca.extopt.engine import ExtOptOrcaEngine
from chemrefine.state import StepContext

logger = logging.getLogger(__name__)


@register("mlff-extopt")
class MlffExtOptEngine(ExtOptOrcaEngine):
    """ORCA optimisation backed by an MLFF gradient server."""

    name = "mlff-extopt"
    backend = "mlff"
    wrapper_filename = "mlff_extopt.sh"

    # -- ORCA input customisation -----------------------------------------

    def _extra_blocks(self, ctx: StepContext) -> str:
        """Emit a ``%method ProgExt "<wrapper>"`` block tied to this step."""
        wrapper = self._wrapper_path(ctx)
        return (
            "%method\n"
            f'  ProgExt "{wrapper}"\n'
            "end"
        )

    # -- SLURM customisation ----------------------------------------------

    def _server_cmd(self, ctx: StepContext) -> str:
        """Build the ``python -m ..._backend_server.server --backend mlff ...`` command.

        Validates ``ctx.step_cfg.options`` through :class:`MlffOptions`
        first (so the engine fails fast on unknown YAML keys), then
        delegates CLI generation to
        :meth:`MlffExtOptCalculator.server_cli_from_options`.
        """
        options = MlffOptions.from_raw(ctx.step_cfg.options).model_dump()
        tokens = MlffExtOptCalculator.server_cli_from_options(options)
        return run_block._server_command(backend=self.backend, extra_tokens=tokens)

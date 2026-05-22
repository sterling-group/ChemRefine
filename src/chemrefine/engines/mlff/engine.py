"""ORCA-driven MLFF engine.

This engine reuses the entire ORCA submission pipeline (input writer,
SLURM, output parser) and only differs in two ways:

1. The generated ORCA ``.inp`` carries a ``%method ... ProgExt ... end``
   block pointing at a wrapper script. ORCA invokes that wrapper at each
   optimization step; the wrapper relays the geometry to a long-running
   ExtOpt HTTP server and pipes the resulting energy + gradient back as
   ``.engrad``.
2. The SLURM ``run_block`` (assembled by
   :class:`~chemrefine.engines._extopt.orca_engine.ExtOptOrcaEngine`)
   starts the shared :mod:`chemrefine.engines._extopt.server` with
   ``--backend mlff`` before ORCA, probes ``/healthz`` until the server
   is ready, and tears it down on exit.

Backend selection / model name are read from ``step.options`` in the
YAML.
"""

from __future__ import annotations

import logging

from chemrefine.engines._extopt import run_block
from chemrefine.engines._extopt.orca_engine import ExtOptOrcaEngine
from chemrefine.engines.base import register
from chemrefine.state import StepContext

logger = logging.getLogger(__name__)


@register("mlff")
class MlffEngine(ExtOptOrcaEngine):
    """ORCA optimisation backed by an MLFF gradient server."""

    name = "mlff"
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
        """Build the ``python -m ..._extopt.server --backend mlff ...`` command."""
        options = ctx.step_cfg.options or {}
        model_name = options.get("model_name") or options.get("model") or "uma-s-1"
        task_name = options.get("task_name") or options.get("task") or "omol"
        device = options.get("device", "cuda")

        return run_block._server_command(
            backend="mlff",
            extra_flags=[
                ("--model", model_name),
                ("--task-name", task_name),
                ("--device", device),
            ],
        )

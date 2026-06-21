"""ORCA-driven PySCF engine (ORCA optimises, PySCF provides gradients).

Mirrors :class:`MlipExtOptEngine` — same shared ExtOpt server, different
backend choice (``--backend pyscf``) and a different per-step CLI
that selects method / xc / basis / df / gpu. The actual SCF +
gradient is computed by :class:`PyscfExtOptCalculator`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

from chemrefine.engines.base import register
from chemrefine.engines.orca.extopt import run_block
from chemrefine.engines.orca.extopt.engine import ExtOptOrcaEngine
from chemrefine.engines.pyscf.extopt_calc import PyscfExtOptCalculator
from chemrefine.engines.pyscf.options import PyscfOptions
from chemrefine.state import StepContext

logger = logging.getLogger(__name__)


@register("pyscf-extopt")
class PyscfExtOptEngine(ExtOptOrcaEngine):
    """ORCA optimisation backed by a PySCF gradient server."""

    name: ClassVar[str] = "pyscf-extopt"
    backend: ClassVar[str] = "pyscf"
    wrapper_filename: ClassVar[str] = "pyscf_extopt.sh"

    # -- ORCA input customisation -----------------------------------------

    def _extra_blocks(self, ctx: StepContext) -> str:
        """Emit a ``%method ProgExt "<wrapper>"`` block.

        The PySCF method / xc / basis selection lives only on the **server**
        (built once from the step's YAML options, see :meth:`_server_cmd`), so
        ORCA needs no ``Ext_Params`` and the wrapper carries no per-call args —
        it takes only the ``.extinp.tmp`` ORCA hands it. Single channel, exactly
        like :class:`MlipExtOptEngine`.
        """
        wrapper = self._wrapper_path(ctx)
        return f'%method\n  ProgExt "{wrapper}"\nend'

    # -- SLURM customisation ----------------------------------------------

    def _server_cmd(self, ctx: StepContext) -> str:
        """Build the ``python -m ..._backend_server.server --backend pyscf ...`` command.

        Validates ``ctx.step_cfg.options`` through :class:`PyscfOptions` first
        (so a typoed knob — ``basis_set:`` for ``basis:`` — fails the step
        instead of silently running with the default), then bakes the method /
        xc / basis / df / gpu / tensor knobs into the server construction (the
        single source of truth); the wrapper and per-call POST stay empty (see
        :class:`PyscfExtOptCalculator`).
        """
        options = PyscfOptions.from_raw(ctx.step_cfg.options).model_dump()
        tokens = PyscfExtOptCalculator.server_cli_from_options(options)
        return run_block._server_command(backend=self.backend, extra_tokens=tokens)

    def _output_dirs(self, ctx: StepContext) -> tuple[str, ...]:
        """Copy the ``save_tensors`` output directory back into the structure dir.

        The server writes ``<tensor_folder>/<tag>.npz`` under ``$WORK_DIR``; a
        relative ``tensor_folder`` is copied back so the tensors persist beside
        the structure's other artifacts. An absolute ``tensor_folder`` already
        persists at its own location, so nothing extra is copied.
        """
        opts = PyscfOptions.from_raw(ctx.step_cfg.options)
        if opts.save_tensors and not Path(opts.tensor_folder).is_absolute():
            return (opts.tensor_folder,)
        return ()

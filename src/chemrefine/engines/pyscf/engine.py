"""ORCA-driven PySCF engine.

Subclasses :class:`OrcaEngine` exactly the way :class:`MlffEngine` does
— the only differences are the ``%method`` block contents (points at a
PySCF wrapper script + ``--method/--xc/--basis`` flags) and the SLURM
``run_block`` (starts ``chemrefine.engines.pyscf.server`` instead of
the MLFF one).

Method, basis, exchange-correlation functional, and the
density-fitting / GPU toggles are read from the step's
``options:`` block in the YAML.
"""

from __future__ import annotations

import logging
import shlex
from pathlib import Path

from chemrefine.engines.base import register
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
        """Emit a ``%method ProgExt "<wrapper>"`` block pointing to the PySCF wrapper."""
        wrapper = self._wrapper_path(ctx)
        bind = self._bind_address(ctx)
        ext_params = self._ext_params(ctx, bind)
        return (
            "%method\n"
            f'  ProgExt "{wrapper}"\n'
            f'  Ext_Params "{ext_params}"\n'
            "end"
        )

    # -- SLURM customisation ----------------------------------------------

    def _run_block(self, ctx: StepContext, inp_path: Path, out_path: Path) -> str:
        """Spin up the PySCF server, run ORCA, then tear the server down."""
        options = ctx.step_cfg.options or {}
        bind = self._bind_address(ctx)
        log_file = options.get("server_log", "pyscf_server.log")

        cmd = (
            "python -m chemrefine.engines.pyscf.server"
            f" --bind {shlex.quote(str(bind))}"
            f" --log-file {shlex.quote(str(log_file))}"
        )
        method = options.get("method", "dft")
        if method:
            cmd += f" --default-method {shlex.quote(str(method))}"
        xc = options.get("xc")
        if xc:
            cmd += f" --default-xc {shlex.quote(str(xc))}"
        basis = options.get("basis")
        if basis:
            cmd += f" --default-basis {shlex.quote(str(basis))}"
        if options.get("df"):
            cmd += " --default-df"
        if options.get("gpu"):
            cmd += " --default-gpu"

        return (
            "# Start PySCF server in background\n"
            f"{cmd} > $OUTPUT_DIR/pyscf_server.log 2>&1 &\n"
            "SERVER_PID=$!\n"
            "trap 'kill $SERVER_PID 2>/dev/null' EXIT\n"
            "sleep 10\n"
            "export OMP_NUM_THREADS=1\n"
            f"{ctx.orca_executable} {inp_path.name} > $OUTPUT_DIR/{out_path.name}\n"
            "kill $SERVER_PID 2>/dev/null || true"
        )

    # -- Helpers -----------------------------------------------------------

    def _wrapper_path(self, ctx: StepContext) -> Path:
        """Path to the ExtOpt wrapper script for the PySCF server."""
        return (ctx.step_dir / "pyscf_extopt.sh").resolve()

    def _bind_address(self, ctx: StepContext) -> str:
        """Resolve the ``host:port`` the PySCF server will bind to."""
        options = ctx.step_cfg.options or {}
        return options.get("bind", "127.0.0.1:8889")

    def _ext_params(self, ctx: StepContext, bind: str) -> str:
        """Build the ``Ext_Params`` string passed to the wrapper script."""
        options = ctx.step_cfg.options or {}
        parts = [f"--bind {bind}"]
        for flag in ("method", "xc", "basis"):
            value = options.get(flag)
            if value is not None:
                parts.append(f"--{flag} {value}")
        if options.get("df"):
            parts.append("--df")
        if options.get("gpu"):
            parts.append("--gpu")
        return " ".join(parts)

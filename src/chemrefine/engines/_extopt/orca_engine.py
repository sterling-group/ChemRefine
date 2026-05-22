"""Shared base class for ORCA-driven ExtOpt engines (MLFF, PySCF, …).

The MLFF and PySCF engines both run an external HTTP server for the
gradient evaluation and use ORCA itself as the optimizer. They differ
only in what server they spin up and which ``%method`` block ORCA gets.
This base class wires up the common bash ``run_block`` and the per-step
wrapper-path convention so subclasses only implement ``_server_cmd``
and declare a ``wrapper_filename``.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import ClassVar

from chemrefine.engines._extopt import run_block
from chemrefine.engines.orca.engine import OrcaEngine
from chemrefine.state import StepContext


class ExtOptOrcaEngine(OrcaEngine):
    """ORCA driven by an ExtOpt HTTP server.

    Subclasses set ``wrapper_filename`` (the per-step ``ProgExt`` script
    name) and implement :meth:`_server_cmd` to return the shell command
    that launches their backend's server. The base wires the shared
    SLURM ``run_block`` and the per-step wrapper-path lookup.
    """

    supports_nms: ClassVar[bool] = False
    wrapper_filename: ClassVar[str]

    @abstractmethod
    def _server_cmd(self, ctx: StepContext) -> str:
        """Return the shell command that launches this backend's ExtOpt server."""

    def _run_block(self, ctx: StepContext, inp_path: Path, out_path: Path) -> str:
        """Combine the engine's server command with the shared lifecycle bash."""
        return run_block._build_extopt_run_block(
            server_cmd=self._server_cmd(ctx),
            orca_executable=ctx.orca_executable,
            inp_name=inp_path.name,
            out_name=out_path.name,
        )

    def _wrapper_path(self, ctx: StepContext) -> Path:
        """Path of the per-step ``ProgExt`` wrapper script."""
        return (ctx.step_dir / self.wrapper_filename).resolve()

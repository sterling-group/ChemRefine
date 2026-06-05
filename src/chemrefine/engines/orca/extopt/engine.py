"""Shared base class for ORCA-driven ExtOpt engines (MLIP, PySCF, …).

The MLIP and PySCF engines both run an external HTTP server for the
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

from chemrefine.engines._backend_server.base import SERVER_URL_FILENAME
from chemrefine.engines.orca.engine import OrcaEngine
from chemrefine.engines.orca.extopt import protocol, run_block
from chemrefine.state import StepContext, StepInputs


class ExtOptOrcaEngine(OrcaEngine):
    """ORCA driven by an ExtOpt HTTP server.

    Subclasses set ``backend`` (the registry key the server / client load,
    e.g. ``"mlip"``) and ``wrapper_filename`` (the per-step ``ProgExt``
    script name), and implement :meth:`_server_cmd` to return the shell
    command that launches their backend's server. The base wires the
    shared SLURM ``run_block``, the per-step wrapper-path lookup, and the
    per-step generation of that wrapper.
    """

    supports_nms: ClassVar[bool] = False
    backend: ClassVar[str]
    wrapper_filename: ClassVar[str]

    @abstractmethod
    def _server_cmd(self, ctx: StepContext) -> str:
        """Return the shell command that launches this backend's ExtOpt server."""

    def prepare(self, ctx: StepContext) -> StepInputs:
        """Write the per-structure ORCA ``.inp`` files plus the ``ProgExt`` wrapper.

        :class:`~chemrefine.engines.orca.engine.OrcaEngine.prepare` writes the
        ``.inp`` files whose ``%method`` block points at
        :meth:`_wrapper_path`; this override then materialises that wrapper so
        ORCA finds it at optimisation time. The wrapper reads the sidecar URL
        file the server writes to ``$WORK_DIR`` and relays each call to the
        shared ExtOpt client.
        """
        inputs = super().prepare(ctx)
        protocol.write_wrapper_script(
            path=self._wrapper_path(ctx),
            backend=self.backend,
            url_file=f"$WORK_DIR/{SERVER_URL_FILENAME}",
            extra_args=self._wrapper_extra_args(ctx),
        )
        return inputs

    def _wrapper_extra_args(self, ctx: StepContext) -> str:
        """Per-call backend flags baked into the wrapper's client invocation.

        Empty for every shipped backend: MLIP and PySCF are both single-channel
        (the calculator is constructed once on the server from the step's YAML
        options, so the wrapper carries no per-call flags). The hook stays as an
        extension point for a future backend that genuinely needs per-geometry
        knobs in the POST.
        """
        return ""

    def _run_block(self, ctx: StepContext, inp_path: Path, out_path: Path) -> str:
        """Combine the engine's server command with the shared lifecycle bash."""
        return run_block._build_extopt_run_block(
            server_cmd=self._server_cmd(ctx),
            orca_executable=ctx.executables.get("orca", "orca"),
            inp_name=inp_path.name,
            out_name=out_path.name,
        )

    def _wrapper_path(self, ctx: StepContext) -> Path:
        """Path of the per-step ``ProgExt`` wrapper script."""
        return (ctx.step_dir / self.wrapper_filename).resolve()

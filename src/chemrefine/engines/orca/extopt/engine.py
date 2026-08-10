"""Shared base class for ORCA-driven ExtOpt engines (MLIP, PySCF, …).

The MLIP and PySCF ExtOpt engines both run an external HTTP server for the gradient
evaluation and use ORCA itself as the optimizer. They differ only in *which* server they
spin up and which option/calculator classes describe it — so this base owns everything
shared (the ``%method ProgExt`` block, the server-launch ``run_block``, the per-step wrapper,
the ``_server_cmd`` template) and a concrete subclass declares just four ClassVars:
``backend`` / ``wrapper_filename`` / ``options_cls`` / ``calculator_cls``.

ExtOpt engines are **NMS-capable**: they inherit ORCA's ``nms_input_info`` hook and its
frequency-carrying parse, because ORCA computes the Hessian numerically over the backend's
gradients — so a ``Freq`` template yields real frequencies on each parsed structure.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from chemrefine.engines import _provision
from chemrefine.engines._backend_server.base import SERVER_URL_FILENAME, ComputeBackend
from chemrefine.engines._job import gpus_from_options
from chemrefine.engines._options import EngineOptions
from chemrefine.engines.orca.engine import OrcaEngine
from chemrefine.engines.orca.extopt import protocol, run_block
from chemrefine.state import RunBlock, StepContext, StepInputs


class ExtOptOrcaEngine(OrcaEngine):
    """ORCA driven by an ExtOpt HTTP server; subclasses declare four ClassVars."""

    backend: ClassVar[str]
    wrapper_filename: ClassVar[str]
    options_cls: ClassVar[type[EngineOptions]]
    calculator_cls: ClassVar[type[ComputeBackend]]

    def gpus(self, ctx: StepContext) -> int:
        """A GPU when the backend's **validated** options request one; else CPU.

        Read through :attr:`options_cls` — the same model :meth:`_server_cmd` validates —
        so the step's GPU demand, its SLURM header, and the calculator the server builds
        can never disagree about what ``device`` was asked for.
        """
        return gpus_from_options(ctx.step_cfg.options, self.options_cls)

    def _extra_blocks(self, ctx: StepContext) -> str:
        """Emit the ``%method ProgExt "<wrapper>"`` block tying ORCA to this step's wrapper."""
        return f'%method\n  ProgExt "{self._wrapper_path(ctx)}"\nend'

    def _server_cmd(self, ctx: StepContext) -> str:
        """Build the ``<python> -m ..._backend_server.server --backend <name> …`` command.

        Validates ``ctx.step_cfg.options`` through ``options_cls`` (so the step fails fast on a
        typoed/unknown YAML knob), then asks ``calculator_cls`` to turn the validated options
        into the matching server CLI tokens — the single source of truth for the backend's
        configuration. The wrapper and per-call POST carry nothing (single-channel). The
        server's interpreter comes from the provisioner (managed backend env when one exists),
        so conflicting backends can serve side by side in one run.
        """
        validated = self.options_cls.from_raw(ctx.step_cfg.options)
        tokens = self.calculator_cls.server_cli_from_options(validated.model_dump())
        interpreter = _provision.launcher_for(self, ctx.step_cfg.options)
        return run_block.server_command(
            backend=self.backend, extra_tokens=tokens, interpreter=interpreter
        )

    def prepare(self, ctx: StepContext) -> StepInputs:
        """Write the per-structure ORCA ``.inp`` files plus the ``ProgExt`` wrapper.

        :class:`~chemrefine.engines.orca.engine.OrcaEngine.prepare` writes the
        ``.inp`` files whose ``%method`` block points at :meth:`_wrapper_path`; this override
        then materialises that wrapper so ORCA finds it at optimisation time. The wrapper reads
        the sidecar URL file the server writes to ``$WORK_DIR`` and relays each call to the
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

    def run_block(self, ctx: StepContext, inp_path: Path, out_path: Path) -> RunBlock:
        """Combine the engine's server command with the shared lifecycle bash.

        The ORCA invocation itself comes from :meth:`OrcaEngine.orca_command`, so this
        path runs the identically-quoted command the plain ORCA path does.

        This is the engine that made :class:`~chemrefine.engines.api.RunBlock` two fields:
        its gradient server has to be stopped however the job ends, and expressing that as a
        ``trap`` inside the body replaced the script's own ``EXIT`` handler. Now the teardown
        is returned as ``cleanup`` and the infra layer places it.
        """
        return run_block._build_extopt_run_block(
            server_cmd=self._server_cmd(ctx),
            orca_command=self.orca_command(ctx, inp_path.name, out_path.name),
            pal=self.pal(ctx),
        )

    def _wrapper_path(self, ctx: StepContext) -> Path:
        """Path of the per-step ``ProgExt`` wrapper script."""
        return (ctx.step_dir / self.wrapper_filename).resolve()

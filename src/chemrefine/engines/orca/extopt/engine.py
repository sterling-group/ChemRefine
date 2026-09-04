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

from chemrefine.config import StepConfig
from chemrefine.engines import _provision
from chemrefine.engines._backend_server.base import SERVER_URL_FILENAME, ComputeBackend
from chemrefine.engines._options import EngineOptions
from chemrefine.engines.orca import input as orca_input
from chemrefine.engines.orca.engine import OrcaEngine
from chemrefine.engines.orca.extopt import protocol, run_block
from chemrefine.state import RunBlock, StepContext, StepInputs


class ExtOptOrcaEngine(OrcaEngine):
    """ORCA driven by an ExtOpt HTTP server; subclasses declare four ClassVars."""

    required_declarations: ClassVar[tuple[str, ...]] = (
        *OrcaEngine.required_declarations,
        "backend",
        "wrapper_filename",
        "options_cls",
        "calculator_cls",
    )
    """ORCA's own declarations, plus the ClassVars this kind adds — see
    :attr:`chemrefine.engines._job.JobEngine.required_declarations`. Unset, they surface as an
    ``AttributeError`` from inside ``_server_cmd`` or ``prepare``, in a job rather than at
    import."""

    backend: ClassVar[str]
    wrapper_filename: ClassVar[str]
    options_cls: ClassVar[type[EngineOptions]]
    calculator_cls: ClassVar[type[ComputeBackend]]
    preflight_refuses: ClassVar[str] = (
        "a typoed or missing server knob: these options configure a gradient server, so the "
        "strict read the run block makes is made up front, before earlier steps are paid for"
    )

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Refuse a subclass whose ``calculator_cls`` cannot serve.

        :class:`~chemrefine.engines._backend_server.base.ComputeBackend` is an ABC, so
        an incomplete backend cannot be *instantiated* — but nothing instantiates one
        before the server calls ``from_args`` inside the job, and abstract classmethods
        stay callable on the class. Checked here because it is the first moment the
        declaration exists; left to the run, the failure surfaces as a ``TypeError``
        while building the job script or a 500 per geometry whose actual cause is
        "this class implements nothing". ``__abstractmethods__`` is Python's own ledger
        of what is missing; ``required_declarations`` covers the bare ClassVars no
        ``ABCMeta`` machinery sees — the :class:`~chemrefine.engines._job.JobEngine`
        idiom, read from the backend's own declaration.
        """
        super().__init_subclass__(**kwargs)
        backend = cls.__dict__.get("calculator_cls")
        if backend is None:  # an intermediate base; `register` catches one that never declares
            return
        missing = sorted(getattr(backend, "__abstractmethods__", ()))
        if missing:
            raise TypeError(
                f"{cls.__name__}: calculator_cls {backend.__name__} leaves {missing} "
                f"abstract — every ComputeBackend hook must be implemented before an "
                f"engine can serve it."
            )
        undeclared = sorted(
            d for d in getattr(backend, "required_declarations", ()) if not hasattr(backend, d)
        )
        if undeclared:
            raise TypeError(
                f"{cls.__name__}: calculator_cls {backend.__name__} is missing "
                f"declaration(s) {undeclared} — the machinery reads them; see ComputeBackend."
            )

    def check_step(self, step_cfg: StepConfig, *, charge: int, multiplicity: int) -> None:
        """Validate the step's options through this engine's own model, up front.

        The same strict read :meth:`_server_cmd` makes when the job script is built —
        hoisted onto :class:`~chemrefine.engines.api.PreflightChecking` so a typoed
        knob (``extra="forbid"``) or a missing required one (PySCF's ``basis``/``xc``)
        is refused before any earlier step is paid for, rather than at this step's own
        turn. On the base because every ExtOpt engine validates strictly — its options
        configure a server, not a template a user's own code reads — so each subclass
        gets the preflight for the price of its existing ``options_cls`` declaration; a
        subclass with a refusal of its own extends rather than replaces.

        Extends rather than replaces *upward* too: the ``super()`` call is ORCA's own
        operation-vocabulary refusal, which these engines inherit because they parse the
        same outputs — held family-wide by
        ``test_every_orca_family_engine_refuses_an_unknown_operation_up_front``.
        """
        super().check_step(step_cfg, charge=charge, multiplicity=multiplicity)
        self.options_cls.from_raw(step_cfg.options)

    def _extra_blocks(self, ctx: StepContext) -> str:
        """Emit the ``%method ProgExt "<wrapper>"`` block tying ORCA to this step's wrapper.

        The wrapper path is checked as well as the geometry path, even though both are
        ``step_dir``-derived and one check would catch today's trees: they are written by
        two different functions, and the failure mode differs (ORCA truncates the geometry
        field; it hands this one to ``sh``). Each emitter owning its own guard is what keeps
        that true if either path ever stops sharing a root.
        """
        wrapper = self._wrapper_path(ctx)
        orca_input.require_whitespace_free(wrapper, what="the ExtOpt wrapper path")
        return f'%method\n  ProgExt "{wrapper}"\nend'

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
        # The granted cores, not the template's ask: the `.inp` rewrite and the SLURM
        # directives both clamp to max_cores, and the server's thread export must say the
        # same number — the raw pal() let a `%pal` above the budget thread past what the
        # throttler charges (see the invariant stated in run_block.py beside the export).
        ntasks, cpus_per_task = self.slurm_layout(ctx)
        return run_block.build_extopt_run_block(
            server_cmd=self._server_cmd(ctx),
            orca_command=self.orca_command(ctx, inp_path.name, out_path.name),
            pal=ntasks * cpus_per_task,
        )

    def _wrapper_path(self, ctx: StepContext) -> Path:
        """Path of the per-step ``ProgExt`` wrapper script."""
        return (ctx.step_dir / self.wrapper_filename).resolve()

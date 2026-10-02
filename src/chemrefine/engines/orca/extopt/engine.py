"""Shared base class for the ORCA-driven ExtOpt engines.

Every ExtOpt engine runs an external HTTP server for the gradient evaluation and uses ORCA
itself as the optimizer. They differ only in *which* server they spin up and which
option/calculator classes describe it — so this base owns everything
shared (the ``%method ProgExt`` block, the server-launch ``run_block``, the per-step wrapper,
the ``_server_cmd`` template) and a concrete subclass declares just four ClassVars:
``backend`` / ``wrapper_filename`` / ``options_cls`` / ``calculator_cls``.

ExtOpt engines are **NMS-capable**: they inherit ORCA's ``nms_input_info`` hook and its
frequency-carrying parse, because ORCA computes the Hessian numerically over the backend's
gradients — so a ``NumFreq`` template yields real frequencies on each parsed structure. It is
also why every strict read here takes :meth:`~chemrefine.config.StepConfig.engine_options`
rather than the raw ``options``: with ``nms: true`` the NMS knobs share that mapping, and a
model with ``extra="forbid"`` handed the whole of it refused them as strangers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, ClassVar

from pydantic import Field

from chemrefine.config import StepConfig
from chemrefine.engines import _provision
from chemrefine.engines._backend_server.base import SERVER_URL_FILENAME, ComputeBackend
from chemrefine.engines._options import UNKEYED, EngineOptions
from chemrefine.engines.orca import input as orca_input
from chemrefine.engines.orca import inspect
from chemrefine.engines.orca.engine import OrcaEngine
from chemrefine.engines.orca.extopt import protocol, run_block
from chemrefine.errors import ConfigError
from chemrefine.ids import StepTemplate
from chemrefine.state import RunBlock, StepContext, StepInputs


class ExtOptOptions(EngineOptions):
    """The knobs every ExtOpt engine's options model carries on top of its backend's.

    An ExtOpt step is an ORCA run whose gradients come from a server the engine starts, and
    the bridge ORCA invokes per geometry is that server's one client — so the knobs of that
    call belong to the ExtOpt family, not to any backend. Each ExtOpt engine's options model
    mixes this in, which is what lets :class:`ExtOptOrcaEngine` type its ``options_cls`` to
    it and read the knob without a cast.
    """

    gradient_timeout_seconds: Annotated[float, UNKEYED] = Field(600.0, gt=0, allow_inf_nan=False)
    """How long the bridge waits on one gradient before it gives the geometry up.

    A bound on a single ``/calculate`` call, not on the optimisation. The default is
    generous for a potential and tight for a DFT gradient on a large system — raise it when
    a healthy server takes longer than that per geometry. On expiry the step records a
    timeout that names this knob, not an unreachable server.

    Finite as well as positive. ``.inf`` — an ordinary spelling of "no timeout" — passes
    ``gt=0``, and would ride into the wrapper as ``--timeout inf`` and reach the socket as
    a timeout Python cannot represent (``OverflowError``, none of the failures the bridge
    classifies), so every geometry step would die in the wrapper with a traceback naming
    neither the knob nor the value. There is no unbounded setting: raise the number."""


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
    options_cls: ClassVar[type[ExtOptOptions]]
    calculator_cls: ClassVar[type[ComputeBackend]]
    preflight_refuses: ClassVar[str] = (
        "a typoed or missing server knob — these options configure a gradient server, so the "
        "strict read the run block makes is made up front, before earlier steps are paid "
        "for — and a template that does not name `ExtOpt`, whose step ORCA would compute "
        "itself while the server sat idle"
    )
    template_starter: ClassVar[str] = (
        "# ORCA ExtOpt starter: `! ExtOpt` has ORCA take every energy and gradient from the\n"
        "# backend of this step; ChemRefine adds the %method ProgExt block and each geometry.\n"
        "! ExtOpt Opt\n"
        "%pal nprocs 4 end\n"
    )
    """What ``chemrefine scaffold`` writes for a missing ``stepN.inp``. ORCA computes a
    step itself unless its ``!`` line names ``ExtOpt``, so ORCA's own starter would leave
    the gradient server idle. Declared on the base, so every ExtOpt engine gets it. No
    ``%maxcore``: the job's memory is the server's, so the header's memory policy stands."""
    frequency_request: ClassVar[str] = (
        "`NumFreq` among the keywords, or `NumFreq true` in a %freq block — the backend "
        "serves gradients, which ORCA differentiates numerically"
    )
    """Satisfies :class:`~chemrefine.engines.api.NmsCapableEngine`: the numerical Hessian,
    the one an ExtOpt job can compute, since its backend serves energies and gradients
    only."""

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
        knob (``extra="forbid"``) or a missing required one (a level of theory the backend
        requires) is refused before any earlier step is paid for, rather than at this step's
        own turn. On the base because every ExtOpt engine validates strictly — its options
        configure a server, not a template a user's own code reads — so each subclass
        gets the preflight for the price of its existing ``options_cls`` declaration; a
        subclass with a refusal of its own extends rather than replaces.

        Read over the engine's share of the options, not the whole mapping: with
        ``nms: true`` the NMS knobs live in the same dict, and a strict read of the whole
        would refuse every one of them, so an engine the table marks NMS-capable could
        sample only with each knob at its default. A key neither reader declares is still
        refused here.
        """
        self.options_cls.from_raw(step_cfg.engine_options())

    def check_template(self, ctx: StepContext, template: StepTemplate) -> None:
        """Refuse a template whose ``!`` line does not name ``ExtOpt``.

        Without it ORCA computes the step with its own method — whatever else the keywords
        name — while the server this engine starts serves nothing, and the results would be
        recorded as this backend's. Satisfies
        :class:`~chemrefine.engines.api.TemplateChecking`, so the run's t=0 walk and
        ``chemrefine validate`` refuse it before anything runs, and ``prepare`` asks again of
        its own read of the template.
        """
        if not inspect.inspect_template(template.text).has_extopt:
            raise ConfigError(
                f"step {ctx.step_cfg.step}: the template's `!` line does not name `ExtOpt`, so "
                f"ORCA would compute the step with its own method while the {self.backend} "
                f"server served nothing. Add it (e.g. `! ExtOpt Opt`), or run the step with "
                f"`engine: orca` if ORCA's own method is what you want."
            )

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

    def _server_cmd(self, ctx: StepContext, interpreter: str | None = None) -> str:
        """Build the ``<python> -m ..._backend_server.server --backend <name> …`` command.

        Validates the engine's share of ``ctx.step_cfg.options`` through ``options_cls`` (so
        the step fails fast on a typoed/unknown YAML knob; the NMS knobs are the sampler's,
        see :meth:`check_step`), then asks ``calculator_cls`` to turn the validated options
        into the matching server CLI tokens — the single source of truth for the backend's
        configuration. The wrapper and per-call POST carry nothing (single-channel). The
        server's interpreter comes from the provisioner (managed backend env when one exists),
        so conflicting backends can serve side by side in one run; :meth:`run_block`
        resolves it once and passes it in, since the readiness probe runs under it too.
        """
        validated = self.options_cls.from_raw(ctx.step_cfg.engine_options())
        tokens = self.calculator_cls.server_cli_from_options(validated.model_dump())
        if interpreter is None:
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
        """Per-call client flags baked into the wrapper's bridge invocation.

        One flag for every shipped backend: the client's own ``--timeout``, from the step's
        ``gradient_timeout_seconds`` — how long the bridge waits on one ``/calculate`` call
        before it gives the geometry up. Read over the engine's share of the options like
        every strict read in this family. The backends themselves are single-channel (the
        calculator is built once on the server from the step's YAML options), so nothing
        backend-specific rides here; the hook stays the extension point for a backend that
        genuinely needs per-geometry knobs in the POST.
        """
        opts = self.options_cls.from_raw(ctx.step_cfg.engine_options())
        return f"--timeout {opts.gradient_timeout_seconds:g}"

    def run_block(self, ctx: StepContext, inp_path: Path, out_path: Path) -> RunBlock:
        """Combine the engine's server command with the shared lifecycle bash.

        The ORCA invocation itself comes from :meth:`OrcaEngine.orca_command`, so this
        path runs the identically-quoted command the plain ORCA path does.

        Its gradient server has to be stopped however the job ends, and a ``trap`` inside
        the body would replace the script's own ``EXIT`` handler, so the teardown is returned
        as the :class:`~chemrefine.engines.api.RunBlock`'s ``cleanup`` and the infra layer
        places it.
        """
        # The granted cores, not the template's ask: the `.inp` rewrite and the SLURM
        # directives both clamp to max_cores, and the server's thread export must say the
        # same number — the raw pal() let a `%pal` above the budget thread past what the
        # throttler charges (see the invariant stated in run_block.py beside the export).
        ntasks, cpus_per_task = self.slurm_layout(ctx)
        interpreter = _provision.launcher_for(self, ctx.step_cfg.options)
        return run_block.build_extopt_run_block(
            server_cmd=self._server_cmd(ctx, interpreter),
            orca_command=self.orca_command(ctx, inp_path.name, out_path.name),
            pal=ntasks * cpus_per_task,
            interpreter=interpreter,
        )

    def single_node(self, ctx: StepContext) -> bool:
        """``True``: the MPI ranks need the tasks, and the gradient server needs them on one node.

        ORCA in ``ProgExt`` mode still runs the ``%pal`` MPI processes of its input — it
        reports "running with N parallel MPI-processes" and waits on the wrapper from
        them — while the server this engine starts threads the same N in one process on
        the same allocation. Neither spelling of N cores alone is right: N one-CPU tasks
        may be granted across nodes, which one server process cannot use, and one task
        with N CPUs offers ORCA a single MPI slot. So the ranks' spelling stays and the
        script pins the node.
        """
        return True

    def _wrapper_path(self, ctx: StepContext) -> Path:
        """Path of the per-step ``ProgExt`` wrapper script."""
        return (ctx.step_dir / self.wrapper_filename).resolve()

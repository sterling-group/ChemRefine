"""``ScriptEngine`` — the engine kind whose per-structure input is a user Python script.

Both ``pyscf`` and ``mlip`` (direct) take a user-supplied ``step{N}.py``, render it per
structure, run each rendered script through the SLURM-or-local machinery, and parse the JSON
its appended footer writes. A thin :class:`~chemrefine.engines._job.JobEngine`: it supplies
only the primitives — render the ``.py`` (:mod:`chemrefine.engines._script.render`), parse the
``.json`` (:mod:`chemrefine.engines._script.output`), the run command, the core budget — while
the base owns ``prepare`` / ``submit`` / ``parse``. Concrete engines set ``name`` + ``label``
and (optionally) override ``_vars_from``.

This is the ``.py``-format sibling of :mod:`chemrefine.engines.orca.input` /
:mod:`chemrefine.engines.orca.output` — the script kind's input writer + output reader.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import ClassVar, Generic, TypeVar, cast

from chemrefine.engines import _provision
from chemrefine.engines._job import JobEngine
from chemrefine.engines._options import EngineOptions, gpus_from_options
from chemrefine.engines._script import output as script_output
from chemrefine.engines._script import render as script_render
from chemrefine.engines._script.contract import SCRIPT_OUTPUT, OutputField
from chemrefine.engines.api import ParsedResult, RunBlock
from chemrefine.state import StepContext

OptsT = TypeVar("OptsT", bound=EngineOptions)


class ScriptEngine(JobEngine, Generic[OptsT]):
    """A ``JobEngine`` whose per-structure input is a user ``step{N}.py`` run with ``python``.

    Generic in its options model so a subclass declares ``ScriptEngine[MlipOptions]`` and
    reads its own fields by name in :meth:`_vars_from`, type-checked.
    """

    name: ClassVar[str]
    label: ClassVar[str]
    template_suffix: ClassVar[str] = "py"
    output_suffix: ClassVar[str] = "json"
    output_globs: ClassVar[tuple[str, ...]] = ("*.json", "*.xyz")
    options_cls: ClassVar[type[EngineOptions]] = EngineOptions
    """This engine's validated ``step.options`` model — the single reader of those knobs.

    Subclasses point it at their own model so ``pal`` / ``gpus`` and the template
    placeholders all resolve the same defaults; the ExtOpt engines declare the same
    ClassVar for the same reason."""

    output_fields: ClassVar[tuple[OutputField, ...]] = SCRIPT_OUTPUT
    """What this engine's ``step{N}.py`` may report back — the output side's ``_vars_from``.

    The input seam lets an engine choose which options reach the template; this is the same
    choice for the return trip, and the two are the whole of what a script engine varies. A
    subclass reporting more than the shared set extends the tuple in its own module::

        output_fields = (*SCRIPT_OUTPUT, OutputField("gibbs_hartree", "gibbs_hartree"))

    and the generated footer, the finiteness sweep, the JSON mapping and the scaffold's
    starter comment all follow from it — no building block edited, which is what
    ``docs/developer/adding-an-engine.md`` promises for every engine kind.

    Declared here rather than passed per call because it is a property of the engine, not of
    a step: the writer and the reader are two processes on two machines, and they have to
    agree without talking."""

    # -- input -------------------------------------------------------------

    def build_input(
        self,
        *,
        xyz_path: Path,
        template_path: Path,
        input_path: Path,
        output_path: Path,
        ctx: StepContext,
    ) -> None:
        """Render one ``step{N}.py`` script that writes its results to ``output_path``."""
        script_render.build_input(
            xyz_path=xyz_path,
            template_path=template_path,
            output_path=input_path,
            output_json_path=output_path,
            charge=ctx.charge,
            multiplicity=ctx.multiplicity,
            extra_vars=self._template_vars(ctx),
            fields=self.output_fields,
        )

    def _template_vars(self, ctx: StepContext) -> dict[str, object]:
        """Extra ``$VAR`` substitutions for the script, from this step's validated options.

        **Final on purpose** — subclasses override :meth:`_vars_from`, which is handed the
        already-validated options. Reading them is the part that must not vary: leniently
        (a template may carry knobs no engine model declares, and rendering must not fail
        over them) and always through :attr:`options_cls`, never off the raw dict. An engine
        spelling that out for itself is how a literal default ends up beside a model that
        declares a different one — a split nothing detects until it produces a wrong job.
        """
        # ``options_cls`` is a ClassVar of the base type, so the parameter it produces is
        # narrowed here — once, in the one place that reads it — rather than by each
        # subclass asserting its way back to the type it already declared.
        opts = cast(OptsT, self.options_cls.from_raw_lenient(ctx.step_cfg.options))
        return self._vars_from(opts)

    def _vars_from(self, opts: OptsT) -> dict[str, object]:
        """Which placeholders this engine exposes, from its validated options (default: none).

        Subclasses read ``opts``' fields by name; the reading of them is the base's job.
        """
        return {}

    # -- run ---------------------------------------------------------------

    def pal(self, ctx: StepContext) -> int:
        """Direct scripts take their core count from ``options.cores`` (default 1).

        Read through this engine's :attr:`options_cls`, which declares and bounds the
        field, rather than off the raw dict.
        """
        return self.options_cls.from_raw_lenient(ctx.step_cfg.options).cores

    def gpus(self, ctx: StepContext) -> int:
        """A GPU if this engine's validated options request one; else CPU."""
        return gpus_from_options(ctx.step_cfg.options, self.options_cls)

    def run_block(self, ctx: StepContext, inp_path: Path, out_path: Path) -> RunBlock:
        """Run the rendered Python script inside ``$WORK_DIR``, capped to its core budget.

        Script engines (pyscf / mlip direct) are OpenMP/MKL/torch-threaded with no MPI, so the
        thread count is pinned to what the job is actually *granted* — the
        :meth:`slurm_layout` product, which is ``options.cores`` clamped to ``max_cores``,
        the same number the SLURM directives request and the throttler charges. The raw
        :meth:`pal` here let a step asking for more than the budget export the ask rather
        than the grant: charged ``max_cores``, threading ``cores`` — the exact
        oversubscription this export exists to prevent, on every local run. (ORCA is the
        opposite — MPI ranks, ``OMP=1``.) The script's interpreter comes from the
        provisioner (a managed backend env when one exists), so conflicting backends can run
        side by side in one pipeline.
        """
        ntasks, cpus_per_task = self.slurm_layout(ctx)
        cores = ntasks * cpus_per_task
        interpreter = _provision.launcher_for(self, ctx.step_cfg.options)
        return RunBlock(
            body=f"export OMP_NUM_THREADS={cores}\n"
            f"export MKL_NUM_THREADS={cores}\n"
            f"export OPENBLAS_NUM_THREADS={cores}\n"
            f"{shlex.quote(interpreter)} {inp_path.name}"
        )

    # -- parse -------------------------------------------------------------

    def parse_one(
        self, output_path: Path, structure_id: str, ctx: StepContext
    ) -> list[ParsedResult]:
        """Read one output JSON into a single ``ParsedResult`` (seed geometry as fallback).

        Read through :attr:`output_fields`, the same contract :meth:`build_input` rendered the
        footer from — so what the script was told it could write and what the driver reads
        back cannot come apart.
        """
        seed = ctx.prev_state.by_id.get(structure_id)
        return script_output.parse_output(
            output_path,
            label=self.label,
            fallback=seed.atoms if seed is not None else None,
            fields=self.output_fields,
        )

"""``ScriptEngine`` — the engine kind whose per-structure input is a user Python script.

Both ``pyscf`` and ``mlip`` (direct) take a user-supplied ``step{N}.py``, render it per
structure, run each rendered script through the SLURM-or-local machinery, and parse the JSON
its appended footer writes. A thin :class:`~chemrefine.engines._job.JobEngine`: it supplies
only the primitives — render the ``.py`` (:mod:`chemrefine.engines._script.render`), parse the
``.json`` (:mod:`chemrefine.engines._script.output`), the run command, the core budget — while
the base owns ``prepare`` / ``submit`` / ``parse``. Concrete engines set ``name`` + ``label``
and (optionally) override ``_template_vars``.

This is the ``.py``-format sibling of :mod:`chemrefine.engines.orca.input` /
:mod:`chemrefine.engines.orca.output` — the script kind's input writer + output reader.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import ClassVar

from chemrefine.engines import _provision
from chemrefine.engines._job import JobEngine, gpus_from_options
from chemrefine.engines._options import EngineOptions
from chemrefine.engines._script import output as script_output
from chemrefine.engines._script import render as script_render
from chemrefine.engines.api import ParsedResult, RunBlock
from chemrefine.state import StepContext


class ScriptEngine(JobEngine):
    """A ``JobEngine`` whose per-structure input is a user ``step{N}.py`` run with ``python``."""

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
        )

    def _template_vars(self, ctx: StepContext) -> dict[str, object]:
        """Extra ``$VAR`` substitutions for the script (default: none).

        Subclasses override this to let ``step.options`` drive the rendered script — e.g.
        the MLIP engine injects ``$MODEL_NAME`` / ``$TASK_NAME`` / ``$DEVICE``.
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
        thread count is pinned to the step's ``options.cores`` (= :meth:`pal`) — the real
        oversubscription guard on a laptop where jobs run concurrently, and harmless under
        SLURM. (ORCA is the opposite — MPI ranks, ``OMP=1``.) The script's interpreter comes
        from the provisioner (a managed backend env when one exists), so conflicting backends
        can run side by side in one pipeline.
        """
        cores = self.pal(ctx)
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
        """Read one output JSON into a single ``ParsedResult`` (seed geometry as fallback)."""
        seed = next((s for s in ctx.prev_state.structures if s.id == structure_id), None)
        return script_output.parse_output(
            output_path, label=self.label, fallback=seed.atoms if seed is not None else None
        )

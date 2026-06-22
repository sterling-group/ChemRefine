"""Shared base for direct template-driven engines (``pyscf`` and ``mlip``).

Both render a user-supplied ``step{N}.py`` per structure, run each rendered script
through the SLURM-or-local machinery, and parse the JSON its appended footer writes.
A thin :class:`~chemrefine.engines._batch.BatchEngine`: it supplies only the template
primitives (render the ``.py`` via :mod:`chemrefine.engines._template_render`, parse the
``.json`` via :mod:`chemrefine.engines._template_output`, the run command, the core
budget); the base owns ``prepare`` / ``submit`` / ``parse``. Concrete engines set
``name`` + ``label`` and (optionally) ``_template_vars``.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from chemrefine.engines import _template_output, _template_render
from chemrefine.engines._assemble import ParsedResult
from chemrefine.engines._batch import BatchEngine, gpus_from_device_options
from chemrefine.state import StepContext


class TemplateScriptEngine(BatchEngine):
    """Direct template-driven engine — runs the user's ``step{N}.py`` per structure."""

    name: ClassVar[str]
    label: ClassVar[str]
    template_suffix: ClassVar[str] = "py"
    output_suffix: ClassVar[str] = "json"
    output_globs: ClassVar[tuple[str, ...]] = ("*.json", "*.xyz")

    # -- input -------------------------------------------------------------

    def _build_input(
        self,
        *,
        xyz_path: Path,
        template_path: Path,
        input_path: Path,
        output_path: Path,
        ctx: StepContext,
    ) -> None:
        """Render one ``step{N}.py`` script that writes its results to ``output_path``."""
        _template_render.build_input(
            xyz_path=xyz_path,
            template_path=template_path,
            output_path=input_path,
            output_json_path=output_path,
            charge=ctx.charge,
            multiplicity=ctx.multiplicity,
            extra_vars=self._template_vars(ctx),
        )

    def _template_vars(self, ctx: StepContext) -> dict[str, object]:
        """Extra ``$VAR`` substitutions for the template (default: none).

        Subclasses override this to let ``step.options`` drive the rendered script — e.g.
        the MLIP engine injects ``$MODEL_NAME`` / ``$TASK_NAME`` / ``$DEVICE``.
        """
        return {}

    # -- run ---------------------------------------------------------------

    def _pal(self, ctx: StepContext) -> int:
        """Direct scripts take their core count from ``options.cores`` (default 1)."""
        return int((ctx.step_cfg.options or {}).get("cores", 1))

    def _gpus(self, ctx: StepContext) -> int:
        """A GPU if ``options.device: cuda`` (or a truthy ``gpu``); else CPU."""
        return gpus_from_device_options(ctx.step_cfg.options)

    def _run_block(self, ctx: StepContext, inp_path: Path, out_path: Path) -> str:
        """Run the rendered Python script inside ``$WORK_DIR``, capped to its core budget.

        Template engines (pyscf / mlip direct) are OpenMP/MKL/torch-threaded with no MPI,
        so the thread count is pinned to the step's ``options.cores`` (= :meth:`_pal`) —
        the real oversubscription guard on a laptop where jobs run concurrently, and
        harmless under SLURM. (ORCA is the opposite — MPI ranks, ``OMP=1``.)
        """
        cores = self._pal(ctx)
        return (
            f"export OMP_NUM_THREADS={cores}\n"
            f"export MKL_NUM_THREADS={cores}\n"
            f"export OPENBLAS_NUM_THREADS={cores}\n"
            f"python {inp_path.name}"
        )

    # -- parse -------------------------------------------------------------

    def _parse_one(
        self, output_path: Path, structure_id: str, ctx: StepContext
    ) -> list[ParsedResult]:
        """Read one output JSON into a single ``ParsedResult`` (seed geometry as fallback)."""
        seed = next((s for s in ctx.prev_state.structures if s.id == structure_id), None)
        return _template_output.parse_output(
            output_path, label=self.label, fallback=seed.atoms if seed is not None else None
        )

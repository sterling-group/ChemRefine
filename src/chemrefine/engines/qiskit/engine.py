"""Template-driven ChemRefine engine that launches the modular Qiskit runner."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import ClassVar

from chemrefine.config import StepConfig
from chemrefine.engines._script import ScriptEngine
from chemrefine.engines._script.contract import SCRIPT_OUTPUT, OutputField
from chemrefine.engines._script.render import json_placeholder
from chemrefine.engines.api import register
from chemrefine.engines.qiskit.backend import QiskitBackend
from chemrefine.engines.qiskit.options import QiskitOptions
from chemrefine.engines.qiskit.workflow import validate_options
from chemrefine.state import StepContext, StepInputs


@register("qiskit")
class QiskitEngine(
    QiskitBackend,
    ScriptEngine[QiskitOptions],
):
    """Direct Qiskit Nature electronic-structure engine."""

    name: ClassVar[str] = "qiskit"
    label: ClassVar[str] = "Qiskit Nature"
    options_cls: ClassVar[type[QiskitOptions]] = QiskitOptions
    output_fields: ClassVar[tuple[OutputField, ...]] = (
        *SCRIPT_OUTPUT,
        OutputField("engine_metadata", field=None, finite=False),
    )
    """Keep Qiskit diagnostics in its raw JSON without extending structure records."""
    template_starter: ClassVar[str] = (
        "# Qiskit Nature single-point calculation; scientific choices come from options.\n"
        "$OUTPUT_CONTRACT"
        "import json\n"
        "\n"
        "from chemrefine.engines.qiskit.workflow import run_job\n"
        "\n"
        "result = run_job(\n"
        '    "$XYZ_PATH",\n'
        '    charge=int("$CHARGE"),\n'
        '    multiplicity=int("$MULTIPLICITY"),\n'
        '    options=json.loads("$OPTIONS_JSON"),\n'
        ")\n"
        "\n"
        "energy_hartree = result.energy_hartree\n"
        "engine_metadata = result.metadata\n"
    )
    preflight_refuses: ClassVar[str] = (
        "unknown Qiskit components, invalid component options, or incompatible "
        "algorithm, ansatz and estimator selections"
    )

    def check_step(self, step_cfg: StepConfig, *, charge: int, multiplicity: int) -> None:
        """Validate the existing component graph before any pipeline step submits."""
        validate_options(self.options_cls.from_raw(step_cfg.engine_options()))

    def prepare(self, ctx: StepContext) -> StepInputs:
        """Repeat the preflight checks for recovery callers, then render the inputs."""
        self.check_step(ctx.step_cfg, charge=ctx.charge, multiplicity=ctx.multiplicity)
        return super().prepare(ctx)

    def build_input(
        self,
        *,
        xyz_path: Path,
        template_path: Path,
        input_path: Path,
        output_path: Path,
        ctx: StepContext,
    ) -> None:
        """Render the worker with the scheduler's granted CPU budget.

        Aer explicitly sets ``max_parallel_threads`` from the worker options, so the
        thread environment alone cannot enforce the scheduler's clamp. Give the shared
        renderer a copied context with that same grant; both JSON placeholders then
        agree, while the user's requested options and standalone runner remain unchanged.
        """
        ntasks, cpus_per_task = self.slurm_layout(ctx)
        worker_step = ctx.step_cfg.model_copy(
            update={"options": {**ctx.step_cfg.options, "cores": ntasks * cpus_per_task}}
        )
        super().build_input(
            xyz_path=xyz_path,
            template_path=template_path,
            input_path=input_path,
            output_path=output_path,
            ctx=replace(ctx, step_cfg=worker_step),
        )

    def _vars_from(self, opts: QiskitOptions) -> dict[str, object]:
        """Preserve the original Qiskit placeholder using the shared JSON escaping."""
        validate_options(opts)
        return {"QISKIT_OPTIONS_JSON": json_placeholder(opts.as_job_spec())}

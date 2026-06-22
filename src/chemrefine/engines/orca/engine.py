"""``OrcaEngine`` — the standard DFT engine, driven from ORCA inputs.

A :class:`~chemrefine.engines._batch.BatchEngine`: it supplies only ORCA-specific
primitives — write the ``.inp`` (:mod:`engines.orca.input`), give the run command, and
parse the ``.out`` (:mod:`engines.orca.output`) into ``ParsedResult`` — while the shared
base handles ``prepare`` / ``submit`` / ``parse`` and :mod:`chemrefine.submit` runs the
batch under the budget.

NMS is engine-independent (:mod:`chemrefine.nms`); ORCA supplies its two hooks —
:meth:`nms_input_info` (read the template keywords) and :meth:`read_frequencies` (parse
the ``.out``'s imaginary modes + tensor). The ExtOpt engines subclass this and inherit
them, so they are NMS-capable too (ORCA computes the Hessian numerically over the backend
gradients).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

from chemrefine.engines._assemble import ParsedResult
from chemrefine.engines._batch import BatchEngine
from chemrefine.engines.base import FrequencyData, NmsInputInfo, register
from chemrefine.engines.orca import frequencies, inspect, output
from chemrefine.engines.orca import input as orca_input
from chemrefine.errors import OutputParseError
from chemrefine.ids import structure_artifact_path
from chemrefine.state import StepContext

logger = logging.getLogger(__name__)


@register("orca")
class OrcaEngine(BatchEngine):
    """Standard ORCA DFT engine (also the base for the ExtOpt engines)."""

    name: ClassVar[str] = "orca"
    label: ClassVar[str] = "ORCA"
    template_suffix: ClassVar[str] = "inp"
    output_suffix: ClassVar[str] = "out"
    output_globs: ClassVar[tuple[str, ...]] = ("*.out", "*.xyz", "*.gbw", "*.hess")

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
        """Write one ORCA ``.inp`` (the ``.out`` is a stdout redirect, not declared here)."""
        orca_input.build_input(
            xyz_path=xyz_path,
            template_path=template_path,
            output_path=input_path,
            charge=ctx.charge,
            multiplicity=ctx.multiplicity,
            extra_blocks=self._extra_blocks(ctx),
            # The submit loop clamps the SLURM allocation to max_cores; the .inp must
            # declare the same PAL or ORCA over-spawns MPI ranks.
            max_pal=ctx.max_cores,
        )

    def _extra_blocks(self, ctx: StepContext) -> str:
        """Extra ORCA blocks; MLIP/PySCF ExtOpt override to inject ``%method ProgExt …``."""
        return ""

    # -- run ---------------------------------------------------------------

    def _pal(self, ctx: StepContext) -> int:
        """PAL is a property of the template (one ``%pal`` for the step), read once."""
        return orca_input.parse_pal(self._resolve_template(ctx))

    def _run_block(self, ctx: StepContext, inp_path: Path, out_path: Path) -> str:
        """Engine-specific bash that runs inside ``$WORK_DIR``."""
        orca = ctx.executables.get("orca", "orca")
        return f"export OMP_NUM_THREADS=1\n{orca} {inp_path.name} > $OUTPUT_DIR/{out_path.name}"

    def _extra_header_fields(self, ctx: StepContext) -> tuple[tuple[str, object], ...]:
        """Record which ORCA binary ran in the runlog header."""
        return (("orca_executable", ctx.executables.get("orca", "orca")),)

    # -- parse -------------------------------------------------------------

    def _resolve_operation(self, ctx: StepContext) -> str:
        """Operation for parsing: an explicit ``operation`` wins, else inspect the template.

        The template's ORCA keywords (:mod:`engines.orca.inspect`) pick the parser when
        ``operation`` is omitted.
        """
        if ctx.step_cfg.operation is not None:
            return ctx.step_cfg.operation
        return inspect.inspect_template(self._resolve_template(ctx)).operation

    def _parse_one(
        self, output_path: Path, structure_id: str, ctx: StepContext
    ) -> list[ParsedResult]:
        """Parse one ORCA output — a ``.out`` (1:1) or its ensemble sidecar (fan-out)."""
        operation = self._resolve_operation(ctx)
        if operation.lower().replace("+", "_") in output.TEXT_BASED_OPERATIONS:
            return output.parse_text(
                output_path.read_text(encoding="utf-8", errors="replace"),
                operation,
                src=str(output_path),
            )
        return output.parse_output(output_path, operation)

    # -- nms hooks (the engine-specific half of chemrefine.nms) ------------

    def nms_input_info(self, ctx: StepContext) -> NmsInputInfo:
        """Whether the template runs a TS search and computes frequencies (keyword scan)."""
        run = inspect.inspect_template(self._resolve_template(ctx))
        return NmsInputInfo(is_transition_state=run.is_ts, computes_frequencies=run.has_freq)

    def read_frequencies(
        self, structure_id: str, step_dir: Path, ctx: StepContext
    ) -> FrequencyData:
        """Parse a structure's imaginary frequencies + normal-mode tensor from its ``.out``.

        ``imaginary`` is ``None`` when the output has **no** frequency table at all
        (distinct from ``{}`` = zero imaginary modes); ``modes`` is ``None`` when the
        displacement tensor is absent / unparseable.
        """
        out_path = structure_artifact_path(step_dir, ctx.step_cfg.step, structure_id, "out")
        if not out_path.is_file():
            return FrequencyData(imaginary=None, modes=None)
        text = out_path.read_text(encoding="utf-8", errors="replace")
        if "VIBRATIONAL FREQUENCIES" not in text:
            logger.warning(
                "NMS: %s produced no frequency table — the template must request a "
                "frequency calc (e.g. opt+freq); the structure is left unresolved",
                structure_id,
            )
            return FrequencyData(imaginary=None, modes=None)
        try:
            parsed = output.parse_dft_from_text(text, src=str(out_path))
            n_atoms = len(parsed[0].symbols) if parsed else 0
            modes = (
                frequencies.parse_normal_modes_tensor_from_text(text, num_atoms=n_atoms)
                if n_atoms
                else None
            )
        except (OutputParseError, ValueError):
            modes = None
        return FrequencyData(
            imaginary=frequencies.parse_imaginary_frequencies_from_text(text), modes=modes
        )

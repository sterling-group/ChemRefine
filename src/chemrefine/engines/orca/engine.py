"""``OrcaEngine`` — the standard DFT engine, driven from ORCA inputs.

A :class:`~chemrefine.engines._job.JobEngine`: it supplies only ORCA-specific
primitives — write the ``.inp`` (:mod:`engines.orca.input`), give the run command, and
parse the ``.out`` (:mod:`engines.orca.output`) into ``ParsedResult`` — while the shared
base handles ``prepare`` / ``submit`` / ``parse`` and :mod:`chemrefine.engines._execution`
runs the batch under the budget.

NMS is engine-independent (:mod:`chemrefine.nms`). ORCA supplies only :meth:`nms_input_info`
(read the template keywords); the *frequency values* it provides are carried on each parsed
``Structure`` (``imaginary_freqs`` + ``normal_modes``, filled in the single ``.out`` parse —
see :mod:`engines.orca.output`), so there is no separate frequency-reading hook. The ExtOpt
engines subclass this and are NMS-capable too (ORCA computes the Hessian numerically over the
backend gradients).
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import ClassVar

from chemrefine.engines._job import JobEngine
from chemrefine.engines.api import NmsInputInfo, ParsedResult, RunBlock, register
from chemrefine.engines.orca import input as orca_input
from chemrefine.engines.orca import inspect, output
from chemrefine.ids import require_template
from chemrefine.state import StepContext


@register("orca")
class OrcaEngine(JobEngine):
    """Standard ORCA DFT engine (also the base for the ExtOpt engines)."""

    name: ClassVar[str] = "orca"
    label: ClassVar[str] = "ORCA"
    template_suffix: ClassVar[str] = "inp"
    output_suffix: ClassVar[str] = "out"
    output_globs: ClassVar[tuple[str, ...]] = (
        "*.out",
        "*.xyz",
        "*.gbw",
        "*.hess",
        "*.property.json",
        "*.property.txt",
        "*.opt",
    )
    """Result files copied back out of ``$WORK_DIR``; anything else is scratch.

    ``.property.txt`` is the human-readable twin of the property JSON, and ``.opt`` is the
    optimisation restart file — the one artifact that lets a stalled optimisation be picked
    up where it stopped instead of started over. Both were left behind in scratch.

    Named exactly rather than as ``*.txt``, which would sweep up whatever a user's template
    happens to write. Note ``*.hess`` also matches ORCA's numbered intermediates
    (``<base>.001.hess``, ``.002.hess``, …), so a long TS search brings back one Hessian per
    recompute; they are large and superseded, and keeping them is a deliberate choice."""

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

    def pal(self, ctx: StepContext) -> int:
        """PAL is a property of the template (one ``%pal`` for the step), read once."""
        return inspect.inspect_template(require_template(ctx.template, label=self.label)).pal

    @staticmethod
    def orca_command(ctx: StepContext, inp_name: str, out_name: str) -> str:
        """The quoted ``orca <input> > <output>`` invocation, for every ORCA run block.

        The executable comes from the YAML, and an unquoted path with a space in it
        silently becomes two words while one with a shell metacharacter becomes
        something else entirely. ``$OUTPUT_DIR`` is quoted for the same reason: the
        config validator refuses metacharacters in the directory paths but **not**
        spaces, so ``output_dir: ./my outputs`` would word-split the redirect.

        Shared rather than inlined per engine:
        :class:`~chemrefine.engines.orca.extopt.engine.ExtOptOrcaEngine` overrides
        ``run_block`` and embeds the same value, so one definition is what stops a subclass
        reopening the hole with its own unquoted copy.
        """
        orca = shlex.quote(ctx.executables.get("orca", "orca"))
        return f'{orca} {inp_name} > "$OUTPUT_DIR/{out_name}"'

    def run_block(self, ctx: StepContext, inp_path: Path, out_path: Path) -> RunBlock:
        """Engine-specific bash that runs inside ``$WORK_DIR``; no teardown of its own."""
        return RunBlock(
            body=f"export OMP_NUM_THREADS=1\n{self.orca_command(ctx, inp_path.name, out_path.name)}"
        )

    def extra_header_fields(self, ctx: StepContext) -> tuple[tuple[str, object], ...]:
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
        return inspect.inspect_template(require_template(ctx.template, label=self.label)).operation

    def parse_one(
        self, output_path: Path, structure_id: str, ctx: StepContext
    ) -> list[ParsedResult]:
        """Parse one ORCA output in a single pass — geometry/energy/forces/thermo + frequencies.

        For a single-structure ``.out`` the frequency values (``imaginary_freqs`` /
        ``normal_modes``) ride on the returned ``ParsedResult`` (and thus on the ``Structure``),
        so NMS never re-parses the file.
        """
        operation = self._resolve_operation(ctx)
        if operation.lower().replace("+", "_") in output.TEXT_BASED_OPERATIONS:
            return output.parse_text(
                output_path.read_text(encoding="utf-8", errors="replace"),
                operation,
                src=str(output_path),
            )
        return output.parse_output(output_path, operation)

    # -- nms hook (the only engine-specific half of chemrefine.nms) --------

    def nms_input_info(self, ctx: StepContext) -> NmsInputInfo:
        """Whether the template runs a TS search and computes frequencies (keyword scan)."""
        run = inspect.inspect_template(require_template(ctx.template, label=self.label))
        return NmsInputInfo(is_transition_state=run.is_ts, computes_frequencies=run.has_freq)

"""Template-method base for engines that run one job per structure.

A *batch engine* shares one lifecycle algorithm — prepare one input per structure, run
them under the budget, parse each output back into structures — and differs only in the
per-structure primitives. This base owns the algorithms (:meth:`prepare` / :meth:`parse`
/ :meth:`submit`) and calls hooks for the specifics, so every batch engine has the same
shape: it supplies ``_build_input``, ``_parse_one``, ``_run_block``, ``_pal`` (+ ``_gpus``
when GPU-capable). Submission itself is flat (:func:`chemrefine.submit.run_batch`) — not
an engine responsibility; the engine only declares *how to run one job*, and the flat
scheduler runs the batch.

Engines that aren't per-structure batch jobs (the fake engine, ``mlip-train``) implement
:class:`chemrefine.engines.base.CalculationEngine` directly instead.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import ClassVar

from chemrefine import submit
from chemrefine.engines._assemble import ParsedResult, build_structures
from chemrefine.ids import input_geometry_path, resolve_step_template, structure_artifact_path
from chemrefine.io import write_single_xyz
from chemrefine.state import JobBatch, StepContext, StepInputs, StepResults


def gpus_from_device_options(options: dict[str, object] | None) -> int:
    """1 if the step's options request a GPU (``device: cuda`` or a truthy ``gpu``), else 0.

    The shared device→GPU heuristic for GPU-capable engines (the template engines and the
    ExtOpt engines), so the base / scheduler never reads option semantics themselves.
    """
    options = options or {}
    if str(options.get("device", "")).lower() == "cuda" or options.get("gpu"):
        return 1
    return 0


class BatchEngine:
    """Shared lifecycle for one-job-per-structure engines (ORCA, template-driven)."""

    name: ClassVar[str]
    label: ClassVar[str]
    template_suffix: ClassVar[str]  # input-file extension (e.g. ``inp`` / ``py``)
    output_suffix: ClassVar[str]  # output-file extension (e.g. ``out`` / ``json``)
    output_globs: ClassVar[tuple[str, ...]]

    # -- lifecycle (shared algorithms) -------------------------------------

    def prepare(self, ctx: StepContext) -> StepInputs:
        """Write one input geometry (``_inp.xyz``) + one input file per seed structure."""
        ctx.step_dir.mkdir(parents=True, exist_ok=True)
        template = self._resolve_template(ctx)
        step = ctx.step_cfg.step
        files: list[tuple[Path, Path, str]] = []
        for struct in ctx.prev_state.structures:
            xyz_path = write_single_xyz(
                struct.atoms,
                input_geometry_path(ctx.step_dir, step, struct.id),
                comment=f"step {step} {struct.id} input",
            )
            input_path = structure_artifact_path(
                ctx.step_dir, step, struct.id, self.template_suffix
            )
            output_path = structure_artifact_path(ctx.step_dir, step, struct.id, self.output_suffix)
            self._build_input(
                xyz_path=xyz_path,
                template_path=template,
                input_path=input_path,
                output_path=output_path,
                ctx=ctx,
            )
            files.append((input_path, output_path, struct.id))
        return StepInputs(files=tuple(files))

    def submit(self, inputs: StepInputs, ctx: StepContext) -> JobBatch:
        """Run the prepared inputs as a batch of per-structure jobs (blocks until done)."""
        return submit.run_batch(self, inputs, ctx)

    def parse(self, inputs: StepInputs, ctx: StepContext) -> StepResults:
        """Parse each output, then assemble structures + fan-out lineage centrally."""
        parsed_per_input = [
            (sid, self._parse_one(out_path, sid, ctx)) for _inp, out_path, sid in inputs.files
        ]
        return build_structures(parsed_per_input, ctx.prev_state)

    # -- shared template helpers -------------------------------------------

    def _resolve_template(self, ctx: StepContext) -> Path:
        """Resolve this step's input template (override or ``step{N}.{suffix}``)."""
        return resolve_step_template(
            ctx.template_dir,
            ctx.step_cfg.step,
            template=ctx.step_cfg.template,
            suffix=self.template_suffix,
            label=self.label,
        )

    def input_digest(self, ctx: StepContext) -> str:
        """SHA-1 (16 hex) of the resolved template's bytes; ``""`` if it's missing.

        Folded into the cache fingerprint so editing a template in place re-runs the
        step (see :func:`chemrefine.cache.fingerprint`).
        """
        try:
            template = self._resolve_template(ctx)
        except FileNotFoundError:
            return ""
        return hashlib.sha1(template.read_bytes()).hexdigest()[:16]

    # -- engine primitives (subclass hooks) --------------------------------

    def _build_input(
        self,
        *,
        xyz_path: Path,
        template_path: Path,
        input_path: Path,
        output_path: Path,
        ctx: StepContext,
    ) -> None:
        """Write one structure's engine input file (e.g. the ``.inp`` / ``.py``)."""
        raise NotImplementedError

    def _parse_one(
        self, output_path: Path, structure_id: str, ctx: StepContext
    ) -> list[ParsedResult]:
        """Parse one output into ``ParsedResult``(s) — ≥2 for an ensemble fan-out."""
        raise NotImplementedError

    def _run_block(self, ctx: StepContext, inp_path: Path, out_path: Path) -> str:
        """The engine-specific bash that runs inside ``$WORK_DIR``."""
        raise NotImplementedError

    def _pal(self, ctx: StepContext) -> int:
        """Per-job core count (PAL) before the scheduler clamps it to ``max_cores``."""
        raise NotImplementedError

    def _gpus(self, ctx: StepContext) -> int:
        """GPUs this job needs (default ``0`` = CPU); GPU engines override."""
        return 0

    def _output_dirs(self, ctx: StepContext) -> tuple[str, ...]:
        """Scratch sub-directories to copy back wholesale (default none)."""
        return ()

    def _extra_header_fields(self, ctx: StepContext) -> tuple[tuple[str, object], ...]:
        """Engine-specific ``(key, value)`` rows for the runlog header (default none)."""
        return ()

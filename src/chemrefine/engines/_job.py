"""``JobEngine`` — the base for engines that run one job per structure.

A *job engine* shares one lifecycle algorithm — build one input per structure, run them
under the budget, parse each output back into structures — and differs only in the
per-structure primitives. This base owns the algorithm (:meth:`prepare` / :meth:`submit` /
:meth:`parse`) and calls public hooks for the specifics, so every job engine has the same
shape: it supplies ``build_input``, ``parse_one``, ``run_block``, ``pal`` (+ ``gpus`` when
GPU-capable). Submission is the flat scheduler (:func:`chemrefine.engines._execution.run_batch`)
— not an engine responsibility; the engine only declares *how to run one job*.

This module also owns :func:`build_structures`, the engine-independent assembler that turns a
step's ``ParsedResult``s into :class:`~chemrefine.state.Structure` objects with IDs + lineage
(used by :meth:`JobEngine.parse`).

Engines that aren't per-structure jobs (the fake engine, ``mlip-train``) implement
:class:`chemrefine.engines.api.CalculationEngine` directly instead.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

from ase import Atoms

from chemrefine.engines import _execution
from chemrefine.engines.api import ParsedResult
from chemrefine.ids import (
    allocate_child_ids,
    input_geometry_path,
    resolve_step_template,
    structure_artifact_path,
)
from chemrefine.io import write_single_xyz
from chemrefine.state import (
    JobBatch,
    PipelineState,
    StepContext,
    StepInputs,
    StepResults,
    Structure,
)


def gpus_from_device_options(options: dict[str, object] | None) -> int:
    """1 if the step's options request a GPU (``device: cuda`` or a truthy ``gpu``), else 0.

    The shared device→GPU heuristic for GPU-capable engines (the script engines and the
    ExtOpt engines), so the base / scheduler never reads option semantics themselves.
    """
    options = options or {}
    if str(options.get("device", "")).lower() == "cuda" or options.get("gpu"):
        return 1
    return 0


def build_structures(
    parsed_per_input: Sequence[tuple[str, list[ParsedResult]]],
    prev_state: PipelineState,
) -> StepResults:
    """Assemble parsed results into :class:`Structure` objects with correct lineage.

    Child IDs come from :func:`chemrefine.ids.allocate_child_ids` (1:1 inherits the input's
    ID; a fan-out gets ``{parent}-{i}``). A fan-out child's ``parent_id`` is the input that
    produced it; a 1:1 child inherits the input's own ``parent_id``. Engine-independent — the
    one home for a step's fan-out + ID lineage.
    """
    prev_by_id = {s.id: s for s in prev_state.structures}
    parents = [sid for sid, _ in parsed_per_input]
    fanouts = [len(parsed) for _, parsed in parsed_per_input]
    child_ids = iter(allocate_child_ids(parents, fanouts))

    out: list[Structure] = []
    for sid, parsed in parsed_per_input:
        input_struct = prev_by_id.get(sid)
        is_fanout = len(parsed) > 1
        for ps in parsed:
            child_parent = (
                sid if is_fanout else (input_struct.parent_id if input_struct is not None else None)
            )
            out.append(
                Structure(
                    id=next(child_ids),
                    atoms=Atoms(symbols=list(ps.symbols), positions=ps.positions),
                    parent_id=child_parent,
                    energy_hartree=ps.energy_hartree,
                    forces_ev_per_a=ps.forces_ev_per_a,
                    converged=ps.converged,
                    terminated=ps.terminated,
                    gibbs_hartree=ps.gibbs_hartree,
                    enthalpy_hartree=ps.enthalpy_hartree,
                    energy_zpe_hartree=ps.energy_zpe_hartree,
                    imaginary_freqs=ps.imaginary_freqs,
                    normal_modes=ps.normal_modes,
                )
            )
    return StepResults(structures=tuple(out))


class JobEngine:
    """Shared lifecycle for one-job-per-structure engines (ORCA, the script engines)."""

    name: ClassVar[str]
    label: ClassVar[str]
    template_suffix: ClassVar[str]  # input-file extension (e.g. ``inp`` / ``py``)
    output_suffix: ClassVar[str]  # output-file extension (e.g. ``out`` / ``json``)
    output_globs: ClassVar[tuple[str, ...]]

    # -- lifecycle (shared algorithm) --------------------------------------

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
            self.build_input(
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
        return _execution.run_batch(self, inputs, ctx)

    def parse(self, inputs: StepInputs, ctx: StepContext) -> StepResults:
        """Parse each output, then assemble structures + fan-out lineage centrally."""
        parsed_per_input = [
            (sid, self.parse_one(out_path, sid, ctx)) for _inp, out_path, sid in inputs.files
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

    # -- engine primitives (the public provision surface) ------------------

    def build_input(
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

    def parse_one(
        self, output_path: Path, structure_id: str, ctx: StepContext
    ) -> list[ParsedResult]:
        """Parse one output into ``ParsedResult``(s) — ≥2 for an ensemble fan-out."""
        raise NotImplementedError

    def run_block(self, ctx: StepContext, inp_path: Path, out_path: Path) -> str:
        """The engine-specific bash that runs inside ``$WORK_DIR``."""
        raise NotImplementedError

    def pal(self, ctx: StepContext) -> int:
        """Per-job core count (PAL) before the scheduler clamps it to ``max_cores``."""
        raise NotImplementedError

    def gpus(self, ctx: StepContext) -> int:
        """GPUs this job needs (default ``0`` = CPU); GPU engines override."""
        return 0

    def output_dirs(self, ctx: StepContext) -> tuple[str, ...]:
        """Scratch sub-directories to copy back wholesale (default none)."""
        return ()

    def extra_header_fields(self, ctx: StepContext) -> tuple[tuple[str, object], ...]:
        """Engine-specific ``(key, value)`` rows for the runlog header (default none)."""
        return ()

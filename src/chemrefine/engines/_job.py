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

import abc
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

from ase import Atoms

from chemrefine.engines import _execution
from chemrefine.engines._options import EngineOptions
from chemrefine.engines.api import CompletionSink, ParsedResult, RunBlock
from chemrefine.ids import (
    allocate_child_ids,
    input_geometry_path,
    require_template,
    structure_artifact_path,
)
from chemrefine.io import write_single_xyz
from chemrefine.state import (
    JobBatch,
    JobTriple,
    PipelineState,
    StepContext,
    StepInputs,
    StepResults,
    Structure,
)


def gpus_from_options(
    options: dict[str, object] | None,
    options_cls: type[EngineOptions] = EngineOptions,
) -> int:
    """1 if the step's **validated** options request a GPU, else 0.

    Reads through ``options_cls`` rather than off the raw dict, because the raw dict and
    the model disagree about what "unset" means: ``options.get("device", "")`` yielded no
    GPU while ``EngineOptions.device`` defaulted to ``cuda``, so a step that named no
    device rendered ``$DEVICE=cuda`` into its script while being scheduled as a CPU job on
    the CPU header — and it bypassed both the GPU budget and
    :meth:`~chemrefine.throttle.Throttler.assign_device`, so concurrent local steps piled
    onto device 0. One reader, one default.

    ``options_cls`` is the engine's own model, so a backend that expresses the request
    differently is honoured without this helper knowing about it: PySCF's ``gpu`` (try
    gpu4pyscf) is a :class:`~chemrefine.engines.pyscf.options.PyscfOptions` field derived
    from ``device``, and ``getattr`` picks it up for engines that declare it.
    """
    opts = options_cls.from_raw_lenient(options)
    return 1 if opts.device == "cuda" or bool(getattr(opts, "gpu", False)) else 0


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
    prev_by_id = prev_state.by_id
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
                    terminated_normally=ps.terminated_normally,
                    gibbs_hartree=ps.gibbs_hartree,
                    enthalpy_hartree=ps.enthalpy_hartree,
                    energy_zpe_hartree=ps.energy_zpe_hartree,
                    imaginary_freqs=ps.imaginary_freqs,
                    frequencies=ps.frequencies,
                    normal_modes=ps.normal_modes,
                )
            )
    return StepResults(structures=tuple(out))


class JobEngine(abc.ABC):
    """Shared lifecycle for one-job-per-structure engines (ORCA, the script engines).

    The four primitives below are abstract, so an incomplete subclass fails at construction —
    which is where :func:`chemrefine.engines.api.get_engine` builds it. Left as runtime
    ``NotImplementedError``, a subclass missing ``parse_one`` still satisfies ``isinstance``,
    still registers, and still submits every job of a step before anything notices.
    """

    required_declarations: ClassVar[tuple[str, ...]] = (
        "name",
        "label",
        "template_suffix",
        "output_suffix",
        "output_globs",
    )
    """The ClassVars below that a concrete engine must define — checked by
    :func:`~chemrefine.engines.api.register` at its decorator line.

    ``abstractmethod`` already fails an incomplete subclass at construction, which is where
    ``get_engine`` builds it — but it watches *methods*. The names below are bare annotations,
    which no ``ABCMeta`` machinery sees: unset, ``output_suffix`` surfaces as a bare
    ``AttributeError`` inside ``prepare``, and ``template_suffix`` stops the engine satisfying
    :class:`~chemrefine.engines.api.TemplateDriven`, so the step reports the user's template as
    missing while it sits on disk. Naming them here is the same move
    :meth:`chemrefine.engines.mlip.registry.MlipLibrary.trainer` makes with its
    ``required = [...]``, and for the same stated reason.

    A base extends the tuple rather than replacing it, so a kind's requirements accumulate
    down the chain (see :class:`~chemrefine.engines.orca.extopt.engine.ExtOptOrcaEngine`)."""

    name: ClassVar[str]
    label: ClassVar[str]
    template_suffix: ClassVar[str]
    """This engine's step-template extension — see :class:`~chemrefine.engines.api.TemplateDriven`.

    Each structure's rendered input takes the same extension, because that artifact *is*
    a copy of the template."""
    output_suffix: ClassVar[str]  # output-file extension (e.g. ``out`` / ``json``)
    output_globs: ClassVar[tuple[str, ...]]

    # -- lifecycle (shared algorithm) --------------------------------------

    def artifact_paths(self, ctx: StepContext, structure_id: str) -> tuple[Path, Path]:
        """This structure's ``(input, output)`` paths, from the engine's two suffixes.

        One derivation, used by :meth:`prepare` when it writes them and by
        :mod:`chemrefine.nms` when it re-reads a round-2 child — so a caller never has to
        know whether this engine writes ``.inp``/``.out`` or ``.py``/``.json``.
        """
        step = ctx.step_cfg.step
        return (
            structure_artifact_path(ctx.step_dir, step, structure_id, self.template_suffix),
            structure_artifact_path(ctx.step_dir, step, structure_id, self.output_suffix),
        )

    def prepare(self, ctx: StepContext) -> StepInputs:
        """Write one input geometry (``_inp.xyz``) + one input file per seed structure."""
        ctx.step_dir.mkdir(parents=True, exist_ok=True)
        template = require_template(ctx.template, label=self.label)
        step = ctx.step_cfg.step
        files: list[JobTriple] = []
        for struct in ctx.prev_state.structures:
            xyz_path = write_single_xyz(
                struct.atoms,
                input_geometry_path(ctx.step_dir, step, struct.id),
                comment=f"step {step} {struct.id} input",
            )
            input_path, output_path = self.artifact_paths(ctx, struct.id)
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

    def submit_streaming(
        self, inputs: StepInputs, ctx: StepContext, sink: CompletionSink
    ) -> JobBatch:
        """:meth:`submit`, reporting each job to ``sink`` the moment it finishes.

        Satisfies :class:`~chemrefine.engines.api.StreamingSubmit` for every job engine at
        once — the scheduler already works job by job, so this is the whole of what an engine
        has to provide for a failed structure's re-run to land in the slot it just freed.
        """
        return _execution.run_batch(self, inputs, ctx, sink=sink)

    def parse(self, inputs: StepInputs, ctx: StepContext) -> StepResults:
        """Parse each output, then assemble structures + fan-out lineage centrally."""
        parsed_per_input = [
            (sid, self.parse_one(out_path, sid, ctx)) for _inp, out_path, sid in inputs.files
        ]
        return build_structures(parsed_per_input, ctx.prev_state)

    # -- engine primitives (the public provision surface) ------------------

    @abc.abstractmethod
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

    @abc.abstractmethod
    def parse_one(
        self, output_path: Path, structure_id: str, ctx: StepContext
    ) -> list[ParsedResult]:
        """Parse one output into ``ParsedResult``(s) — ≥2 for an ensemble fan-out."""

    @abc.abstractmethod
    def run_block(self, ctx: StepContext, inp_path: Path, out_path: Path) -> RunBlock:
        """The bash that runs inside ``$WORK_DIR``, plus any teardown it needs."""

    @abc.abstractmethod
    def pal(self, ctx: StepContext) -> int:
        """Per-job core count (PAL) before the scheduler clamps it to ``max_cores``."""

    def slurm_layout(self, ctx: StepContext) -> tuple[int, int]:
        """The MPI-ranks spelling ``(min(pal, max_cores), 1)`` — every engine's until now.

        One task per core, clamped to the budget exactly as the scheduler always has; the
        clamp lives here rather than in the scheduler so an override cannot silently
        disagree with it. A threaded engine overrides this to ``(1, threads)`` — N tasks
        with one CPU each can be granted across nodes, where a single threaded process can
        only use the first node's share.
        """
        return (min(self.pal(ctx), ctx.max_cores), 1)

    def gpus(self, ctx: StepContext) -> int:
        """GPUs this job needs (default ``0`` = CPU); GPU engines override."""
        return 0

    def memory_mb(self, ctx: StepContext) -> int | None:
        """Total MB one job requires; ``None`` (the default) leaves the header's policy alone.

        An engine whose input declares its memory — ORCA's ``%maxcore``, Q-Chem's
        ``mem_total`` — overrides this with that declaration (plus its own headroom rule),
        and the script builder then extends a header allocation that falls short of it.
        """
        return None

    def output_dirs(self, ctx: StepContext) -> tuple[str, ...]:
        """Scratch sub-directories to copy back wholesale (default none)."""
        return ()

    def extra_header_fields(self, ctx: StepContext) -> tuple[tuple[str, object], ...]:
        """Engine-specific ``(key, value)`` rows for the runlog header (default none)."""
        return ()

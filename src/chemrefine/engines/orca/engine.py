"""``OrcaEngine`` — the standard DFT engine, driven from ORCA inputs.

Implements :class:`~chemrefine.engines.base.CalculationEngine` by
composing the smaller modules in this package:

* :mod:`engines.orca.input` writes the per-structure ``.inp`` files.
* :mod:`chemrefine.slurm` + :mod:`chemrefine.throttle` build and submit
  SLURM scripts under the PAL budget.
* :mod:`engines.orca.output` parses the resulting ``.out`` files into
  :class:`~chemrefine.state.Structure` instances.
* :mod:`engines.orca.nms` runs normal-mode sampling for steps that
  request it.

Submit and wait are intentionally collapsed: :meth:`submit` registers
every job with the throttler and blocks until every job finishes, so
:meth:`wait` is a no-op. The engine instance therefore carries no
state between steps.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import numpy as np
from ase import Atoms
from numpy.typing import NDArray

from chemrefine.engines.base import SlurmBatchEngine, register
from chemrefine.engines.orca import frequencies, inspect, nms, output
from chemrefine.engines.orca import input as orca_input
from chemrefine.errors import ConfigError
from chemrefine.ids import allocate_child_ids, input_geometry_path, structure_artifact_path
from chemrefine.io import write_single_xyz
from chemrefine.state import (
    PipelineState,
    StepContext,
    StepInputs,
    StepResults,
    Structure,
)

logger = logging.getLogger(__name__)


@register("orca")
class OrcaEngine(SlurmBatchEngine):
    """Standard ORCA DFT engine."""

    name: ClassVar[str] = "orca"
    supports_nms: ClassVar[bool] = True
    label: ClassVar[str] = "ORCA"
    template_suffix: ClassVar[str] = "inp"
    output_globs: ClassVar[tuple[str, ...]] = ("*.out", "*.xyz", "*.gbw", "*.hess")

    def __init__(self) -> None:
        # Per-run, per-id caches populated in `parse` (parse-once): the
        # imaginary frequencies and normal-mode tensor of each NMS output,
        # reused by `normal_mode_sample` for displacement + resolution
        # without re-reading the `.out`. ``None`` (vs ``{}``) records that the
        # output had no frequency table at all — "no freq evidence" must never
        # count as "zero imaginary modes" at resolution time.
        self._imag_freqs: dict[str, dict[int, float] | None] = {}
        self._modes: dict[str, NDArray[np.float64] | None] = {}

    # -- prepare -----------------------------------------------------------

    def prepare(self, ctx: StepContext) -> StepInputs:
        """Write one ``.xyz`` + ``.inp`` per seed structure."""
        ctx.step_dir.mkdir(parents=True, exist_ok=True)
        template = self._resolve_template(ctx)
        self._check_nms_freq_requirement(ctx, template)

        files: list[tuple[Path, Path, str]] = []
        step = ctx.step_cfg.step
        for struct in ctx.prev_state.structures:
            xyz_path = write_single_xyz(
                struct.atoms,
                input_geometry_path(ctx.step_dir, step, struct.id),
                comment=f"step {step} {struct.id} input",
            )
            inp_path = structure_artifact_path(ctx.step_dir, step, struct.id, "inp")
            out_path = structure_artifact_path(ctx.step_dir, step, struct.id, "out")
            orca_input.build_input(
                xyz_path=xyz_path,
                template_path=template,
                output_path=inp_path,
                charge=ctx.charge,
                multiplicity=ctx.multiplicity,
                extra_blocks=self._extra_blocks(ctx),
                # The submit loop clamps the SLURM allocation to max_cores; the
                # .inp must declare the same PAL or ORCA over-spawns MPI ranks.
                max_pal=ctx.max_cores,
            )
            files.append((inp_path, out_path, struct.id))
        return StepInputs(files=tuple(files))

    def _extra_blocks(self, ctx: StepContext) -> str:
        """Subclass hook for engines that need extra ORCA blocks (e.g. MLIP ``%method``).

        The base ORCA engine has nothing extra to add; MLIP and PySCF
        override this to inject their ``%method ProgExt …`` block.
        """
        return ""

    def _check_nms_freq_requirement(self, ctx: StepContext, template: Path) -> None:
        """B9: an NMS step must run a frequency calc — reject one whose template has none.

        NMS acts on imaginary modes, so a step with ``nms: true`` whose template
        carries no ``Freq`` keyword can only ever leave every structure unresolved.
        Caught here at prepare time as a :class:`~chemrefine.errors.ConfigError`
        rather than after a wasted round of jobs. An explicit ``operation``
        bypasses the check — it's the user's override for when inspection misreads
        the template.
        """
        if not (ctx.step_cfg.nms and self.supports_nms):
            return
        if ctx.step_cfg.operation is not None:
            return
        if not inspect.inspect_template(template).has_freq:
            raise ConfigError(
                f"step {ctx.step_cfg.step}: `nms: true` needs a frequency calculation, "
                f"but {template.name} has no Freq keyword. Add Freq to the template "
                f"(e.g. `! Opt Freq`), or set `operation` explicitly to override."
            )

    # -- submit (PAL + run_block hooks; the loop lives on SlurmBatchEngine) -

    def _pal(self, ctx: StepContext) -> int:
        """Read PAL once from the step template.

        PAL is a property of the template, not of any individual structure:
        ORCA copies the same ``%pal`` block into every generated ``.inp``, so
        read it once from the template rather than the per-structure copies.
        """
        return orca_input.parse_pal(self._resolve_template(ctx))

    def _run_block(self, ctx: StepContext, inp_path: Path, out_path: Path) -> str:
        """Engine-specific bash that runs inside ``$WORK_DIR``."""
        orca = ctx.executables.get("orca", "orca")
        return f"export OMP_NUM_THREADS=1\n{orca} {inp_path.name} > $OUTPUT_DIR/{out_path.name}"

    def _extra_header_fields(self, ctx: StepContext) -> tuple[tuple[str, object], ...]:
        """Record which ORCA binary ran in the runlog header."""
        return (("orca_executable", ctx.executables.get("orca", "orca")),)

    def _effective_operation(self, ctx: StepContext) -> str:
        """Operation to parse with: the explicit one if set, else inspect the template.

        An explicit ``operation`` always wins; otherwise the run type is inferred
        from the template's ORCA keywords (:mod:`chemrefine.engines.orca.inspect`).
        """
        if ctx.step_cfg.operation is not None:
            return ctx.step_cfg.operation
        return inspect.inspect_template(self._resolve_template(ctx)).operation

    # -- parse -------------------------------------------------------------

    def parse(self, inputs: StepInputs, ctx: StepContext) -> StepResults:
        """Parse each output once into :class:`Structure` instances.

        For ``.out``-based operations the file is read a single time and that
        text yields geometry/energy/forces + the run-status flags, and — for
        NMS steps — the imaginary frequencies + normal-mode tensor (cached for
        :meth:`normal_mode_sample`). Ensemble operations read their sidecar.
        """
        operation = self._effective_operation(ctx)
        text_based = operation.lower().replace("+", "_") in output.TEXT_BASED_OPERATIONS
        prev_by_id = {s.id: s for s in ctx.prev_state.structures}

        # Parse every output first so each input's fan-out is known, then mint
        # child IDs through the shared lineage convention in `ids` instead of
        # re-implementing the ``{parent}-{i}`` format here.
        parsed_per_input: list[tuple[str, list[output.ParsedStructure]]] = []
        for _inp, out_path, sid in inputs.files:
            if text_based:
                text = out_path.read_text(encoding="utf-8", errors="replace")
                parsed = output.parse_text(text, operation, src=str(out_path))
                if ctx.step_cfg.nms and parsed:
                    self._cache_frequencies(sid, text, n_atoms=len(parsed[0].symbols))
            else:
                parsed = output.parse_output(out_path, operation)
            parsed_per_input.append((sid, parsed))

        parents = [sid for sid, _ in parsed_per_input]
        fanouts = [len(parsed) for _, parsed in parsed_per_input]
        child_ids = iter(allocate_child_ids(parents, fanouts))

        out_structures: list[Structure] = []
        for sid, parsed in parsed_per_input:
            input_struct = prev_by_id.get(sid)
            is_fanout = len(parsed) > 1
            for ps in parsed:
                # Fan-out: parent is the input that fanned out.
                # 1:1: child inherits the input's parent lineage unchanged.
                child_parent = (
                    sid
                    if is_fanout
                    else (input_struct.parent_id if input_struct is not None else None)
                )
                atoms = Atoms(symbols=list(ps.symbols), positions=ps.positions)
                out_structures.append(
                    Structure(
                        id=next(child_ids),
                        atoms=atoms,
                        parent_id=child_parent,
                        energy_hartree=ps.energy_hartree,
                        forces_ev_per_a=ps.forces_ev_per_a,
                        converged=ps.converged,
                        terminated=ps.terminated,
                        gibbs_hartree=ps.gibbs_hartree,
                        enthalpy_hartree=ps.enthalpy_hartree,
                        energy_zpe_hartree=ps.energy_zpe_hartree,
                    )
                )
        return StepResults(structures=tuple(out_structures))

    def _cache_frequencies(self, sid: str, text: str, *, n_atoms: int) -> None:
        """Cache the imaginary freqs + normal-mode tensor from this output's text.

        Part of the single parse pass for NMS steps; :meth:`normal_mode_sample`
        reads these instead of re-reading the ``.out``. An output with **no**
        frequency table caches ``None`` — distinct from ``{}`` (a parsed table
        with zero imaginary modes) so a run whose freq module never produced
        results can't be mistaken for a verified minimum. A missing modes block
        leaves ``modes=None`` so the structure simply isn't displaced.
        """
        if "VIBRATIONAL FREQUENCIES" in text:
            self._imag_freqs[sid] = frequencies.parse_imaginary_frequencies_from_text(text)
        else:
            logger.warning(
                "NMS step: %s produced no frequency table — the step's operation / "
                "template must request a frequency calc (e.g. opt+freq), or NMS has "
                "nothing to act on and the structure is left unresolved",
                sid,
            )
            self._imag_freqs[sid] = None
        try:
            self._modes[sid] = frequencies.parse_normal_modes_tensor_from_text(
                text, num_atoms=n_atoms
            )
        except ValueError:
            self._modes[sid] = None

    # -- nms (two-round: displace round-1 survivors, re-optimise, mark resolved) --

    def _resolved_nms_options(self, ctx: StepContext) -> nms.NmsOptions:
        """NMS options with the ``target`` resolved (F1): explicit wins, else inferred.

        When the step doesn't set ``options.target``, derive it from the template:
        an ``OptTS`` run targets a ``ts`` (keep one imaginary mode), anything else a
        ``minimum`` (remove them all). An explicit ``target`` is always honoured.
        """
        raw = ctx.step_cfg.options or {}
        opts = nms.NmsOptions.from_raw(raw)
        if "target" not in raw:
            is_ts = inspect.inspect_template(self._resolve_template(ctx)).is_ts
            opts = opts.model_copy(update={"target": "ts" if is_ts else "minimum"})
        return opts

    def normal_mode_sample(self, results: StepResults, ctx: StepContext) -> StepResults:
        """Displace round-1 survivors per target, re-optimise them, flag resolution.

        For each round-1 structure: if it already matches the target imaginary
        count it passes through (``converged=True``); otherwise it is displaced
        (target-aware, via :mod:`engines.orca.nms`) and the ± children are
        re-optimised (``opt+freq``) in a ``nms/`` subdir (round 2). Each child's
        ``converged`` flag records whether its round-2 imaginary count matches
        the target. :func:`chemrefine.step.run_step` then groups by parent and
        applies the step's ``on_failure`` policy to unresolved parents.

        **Throttling / ordering.** This is the *second* throttled phase: round 1
        (every structure) has already finished under the budget (``run_step``
        blocks on ``submit``'s ``wait_all``) before this runs, then round 2 is a
        **separate** ``submit`` whose ``nms_ctx`` is ``dataclasses.replace(ctx,
        …)`` — so ``max_cores`` / ``max_gpus`` / ``device`` / header all carry
        over and the displacements are throttled too. The two rounds never share
        the budget at once: imag-freq removal starts only *after* round 1 fully
        drains, not the instant an individual structure finishes.
        """
        target = nms.target_imaginary_count(self._resolved_nms_options(ctx))
        already, children = self._nms_displace(results, ctx)
        outputs = list(already)
        if children:
            inputs = self._prepare_round_two(ctx, children)
            round2_ctx = self._round_two_ctx(ctx, children)
            # No separate round-2 manifest: the children are per-id directories
            # nested under their round-1 parent, and the step's round-1 manifest
            # (already written) must not be clobbered. Round-2 is re-derived
            # deterministically on rebuild/reattempt, so it needs no manifest.
            self.wait(self.submit(inputs, round2_ctx))
            outputs.extend(self._flag_resolution(self.parse(inputs, round2_ctx), target))
        return StepResults(structures=tuple(outputs))

    def resolve_nms_from_existing(self, results: StepResults, ctx: StepContext) -> StepResults:
        """Re-resolve NMS from round-2 outputs already on disk — no submission.

        Used by ``rebuild-cache``: re-derives the (deterministic) displaced
        children and parses their existing round-2 ``.out`` files (in each child's
        ``stepN/<parent>/<child>/`` directory) to re-flag resolution. Round-1
        frequencies must already be cached (the caller re-parses round 1 first).
        Children with no output on disk are simply absent, so
        :func:`chemrefine.step_nms.resolve_nms` treats their parent as unresolved.
        """
        target = nms.target_imaginary_count(self._resolved_nms_options(ctx))
        already, children = self._nms_displace(results, ctx)
        outputs = list(already)
        if children:
            round2_ctx = self._round_two_ctx(ctx, children)
            step = ctx.step_cfg.step

            def _triple(c: Structure) -> tuple[Path, Path, str]:
                base = ctx.step_dir / (c.parent_id or c.id)  # nest under the round-1 parent
                return (
                    structure_artifact_path(base, step, c.id, "inp"),
                    structure_artifact_path(base, step, c.id, "out"),
                    c.id,
                )

            present = StepInputs(files=tuple(t for c in children if (t := _triple(c))[1].is_file()))
            if present.files:
                outputs.extend(self._flag_resolution(self.parse(present, round2_ctx), target))
        return StepResults(structures=tuple(outputs))

    def _nms_displace(
        self, results: StepResults, ctx: StepContext
    ) -> tuple[list[Structure], list[Structure]]:
        """Split round-1 survivors into (already-at-target, displaced children)."""
        opts = self._resolved_nms_options(ctx)
        rng = np.random.default_rng(opts.seed)
        target = nms.target_imaginary_count(opts)
        already: list[Structure] = []
        children: list[Structure] = []
        for s in results.structures:
            # ``None`` = no freq table in the output; never "already at target".
            imag = self._imag_freqs.get(s.id)
            if imag is not None and target is not None and len(imag) == target:
                already.append(replace(s, converged=True))  # already at target
                continue
            modes = self._modes.get(s.id)
            if modes is None:
                logger.warning(
                    "NMS %s: no normal-mode tensor; cannot displace (left unresolved)",
                    s.id,
                )
                continue
            for suffix, positions in nms.select_displacements(s, imag or {}, modes, opts, rng):
                child_atoms = s.atoms.copy()
                child_atoms.set_positions(positions)
                children.append(Structure(id=f"{s.id}_{suffix}", atoms=child_atoms, parent_id=s.id))
        return already, children

    def _round_two_ctx(self, ctx: StepContext, children: list[Structure]) -> StepContext:
        """Round-2 context for submit/parse: the displaced children as the prior state.

        ``step_dir`` is unchanged — submit/parse key off the per-file paths and the
        prior state, not the step dir; the per-child *paths* nest under each child's
        round-1 parent (see :meth:`_prepare_round_two`).
        """
        return replace(ctx, prev_state=PipelineState(structures=tuple(children)))

    def _prepare_round_two(self, ctx: StepContext, children: list[Structure]) -> StepInputs:
        """Prepare round-2 inputs, each child nested inside its round-1 parent's dir.

        A displaced re-optimisation of structure ``X`` is just "redo ``X`` with a
        different geometry", so it lives in ``stepN/<X>/<child_id>/`` — a sub-dir of
        the parent's own directory (the same place a future failure-retry would go),
        not a separate ``nms/`` folder. Children are grouped by parent so one
        ``prepare`` call per parent writes into that parent's directory; the combined
        inputs are submitted as a single throttled batch.
        """
        files: list[tuple[Path, Path, str]] = []
        by_parent: dict[str, list[Structure]] = defaultdict(list)
        for c in children:
            by_parent[c.parent_id or c.id].append(c)
        for parent_id, group in by_parent.items():
            parent_ctx = replace(
                ctx,
                step_dir=ctx.step_dir / parent_id,
                prev_state=PipelineState(structures=tuple(group)),
            )
            files.extend(self.prepare(parent_ctx).files)
        return StepInputs(files=tuple(files))

    def _flag_resolution(self, round2: StepResults, target: int | None) -> list[Structure]:
        """Set each round-2 child's ``converged`` to whether it hit the target.

        A child whose output carried no frequency table (``imag is None``) is
        never resolved: without freq evidence, "zero imaginary modes" can't be
        claimed — only counted-and-zero counts as a verified minimum.
        """
        flagged: list[Structure] = []
        for c in round2.structures:
            imag = self._imag_freqs.get(c.id)
            resolved = c.terminated is not False and (
                target is None or (imag is not None and len(imag) == target)
            )
            flagged.append(replace(c, converged=resolved))
        return flagged

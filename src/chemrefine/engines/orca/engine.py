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
from chemrefine.ids import StepTemplate, require_template, step_relative
from chemrefine.state import StepContext


@register("orca")
class OrcaEngine(JobEngine):
    """Standard ORCA DFT engine (also the base for the ExtOpt engines)."""

    name: ClassVar[str] = "orca"
    label: ClassVar[str] = "ORCA"
    template_suffix: ClassVar[str] = "inp"
    output_suffix: ClassVar[str] = "out"
    template_starter: ClassVar[str] = (
        "# ORCA starter — edit the keywords; ChemRefine appends each structure's geometry.\n"
        "! B3LYP D4 def2-SVP Opt\n"
        "%pal nprocs 4 end\n"
        "%maxcore 2000\n"
    )
    """What ``chemrefine scaffold`` writes for a missing ``stepN.inp`` — see
    :attr:`~chemrefine.engines.api.TemplateDriven.template_starter`."""
    operations: ClassVar[tuple[str, ...]] = tuple(sorted(output.known_operations()))
    """The ``operation:`` vocabulary this family interprets — the parser dispatch's own
    set (see :class:`~chemrefine.engines.api.OperationsDeclaring`), inherited by the
    ExtOpt engines, which are ORCA-driven and parse the same outputs."""
    executable_keys: ClassVar[tuple[str, ...]] = ("orca",)
    """Satisfies :class:`~chemrefine.engines.api.ExecutablesDeclaring`: the ORCA binary,
    which the ExtOpt engines run too."""
    frequency_request: ClassVar[str] = (
        "`Freq`, `NumFreq` or `AnFreq` among the keywords, or `NumFreq true` / "
        "`AnFreq true` in a %freq block"
    )
    """Satisfies :class:`~chemrefine.engines.api.NmsCapableEngine`: every spelling
    :func:`~chemrefine.engines.orca.inspect.inspect_template` counts as a frequency job."""
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
    up where it stopped instead of started over. Neither matches any other glob here, so
    without these two entries both stay in ``$WORK_DIR`` and are deleted with it.

    Named exactly rather than as ``*.txt``, which would sweep up whatever a user's template
    happens to write. Note ``*.hess`` also matches ORCA's numbered intermediates
    (``<base>.001.hess``, ``.002.hess``, …), so a long TS search brings back one Hessian per
    recompute; they are large and superseded, and keeping them is a deliberate choice."""

    whitespace_path_reason: ClassVar[str] = orca_input.WHITESPACE_PATH_REASON
    """Satisfies :class:`~chemrefine.engines.api.WhitespacePathIntolerant`.

    Declared here so the ExtOpt engines inherit it: they subclass this one and write the
    same ``* xyzfile`` directive, plus a ``ProgExt`` wrapper path with the same problem.
    The refusal itself lives at the point of use
    (:func:`chemrefine.engines.orca.input.require_whitespace_free`); this ClassVar is what
    lets :mod:`chemrefine.validate` warn about it without importing a concrete engine."""

    # -- input -------------------------------------------------------------

    def template_aux_files(self, template: Path) -> dict[str, Path]:
        """Satisfies :class:`~chemrefine.engines.api.AuxFileConsuming`.

        The enumeration is :func:`chemrefine.engines.orca.input.referenced_aux_files` —
        the same rule ``build_input``'s path rewriter pins references by, so what a job
        reads and what the cache key digests cannot disagree. Inherited by the ExtOpt
        engines, whose templates are the same grammar.
        """
        return orca_input.referenced_aux_files(template)

    def build_input(
        self,
        *,
        xyz_path: Path,
        template: StepTemplate,
        input_path: Path,
        output_path: Path,
        ctx: StepContext,
    ) -> None:
        """Write one ORCA ``.inp`` (the ``.out`` is a stdout redirect, not declared here)."""
        orca_input.build_input(
            xyz_path=xyz_path,
            template=template,
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
        return inspect.inspect_template(require_template(ctx.template, label=self.label).text).pal

    def memory_mb(self, ctx: StepContext) -> int | None:
        """The SLURM total ``%maxcore`` implies: ``ceil(maxcore * pal / 0.75)``.

        ``%maxcore`` is per core and it is a *promise to ORCA*, not a bound ORCA honours —
        it routinely overshoots it per process, which is why qorca holds maxcore to at most
        75% of the granted allocation (and its ``-m`` flag *sets* maxcore to 75% of the
        given amount). Inverting that rule here sizes the request so the declared maxcore
        is exactly 75% of it. The core count is the one the job is actually granted —
        the template's PAL clamped to ``max_cores``, matching :meth:`slurm_layout` and the
        ``.inp`` rewrite in ``build_input``.

        No ``%maxcore`` in the template → ``None``: the header's memory policy stands, as
        it always has. The builder also keeps a header whose own grant already covers this
        — see :func:`chemrefine.slurm.script._apply_memory`.
        """
        info = inspect.inspect_template(require_template(ctx.template, label=self.label).text)
        if info.maxcore is None:
            return None
        cores = min(info.pal, ctx.max_cores)
        return -(-(info.maxcore * cores * 4) // 3)  # ceil(maxcore * cores / 0.75)

    @staticmethod
    def orca_command(ctx: StepContext, inp_name: str, out_name: str) -> str:
        """The quoted ``orca <input> > <output>`` invocation, for every ORCA run block.

        The executable comes from the YAML, and an unquoted path with a space in it
        silently becomes two words while one with a shell metacharacter becomes
        something else entirely. ``$OUTPUT_DIR`` is quoted for the same reason: the
        config validator refuses metacharacters in the directory paths but **not**
        spaces, so ``output_dir: ./my outputs`` would word-split the redirect.

        Spaces are only safe *here*, in bash. The same directory reaches the ``.inp`` as the
        geometry path, where ORCA's own parser truncates it —
        :func:`chemrefine.engines.orca.input.require_whitespace_free` is the refusal, and
        quoting is no help there.

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

    def infer_operation(self, template: StepTemplate) -> str:
        """Satisfies :class:`~chemrefine.engines.api.OperationInferring`: the run kind the
        template's keywords and blocks name (:mod:`engines.orca.inspect`). Inherited by the
        ExtOpt engines, whose templates are the same grammar."""
        return inspect.inspect_template(template.text).operation

    def reads_alike(self, first: str, second: str) -> bool:
        """Satisfies :class:`~chemrefine.engines.api.OperationInferring`: the same operation,
        or two that both read the output as its final structure — every operation but those
        that fan it out into several
        (:func:`~chemrefine.engines.orca.output.multi_structure_operations`)."""
        multi = output.multi_structure_operations()
        return first == second or (first not in multi and second not in multi)

    def parse_one(
        self, output_path: Path, structure_id: str, ctx: StepContext
    ) -> list[ParsedResult]:
        """Parse one ORCA output in a single pass — geometry/energy/forces/thermo + frequencies.

        For a single-structure ``.out`` the frequency values (``imaginary_freqs`` /
        ``normal_modes``) ride on the returned ``ParsedResult`` (and thus on the ``Structure``),
        so NMS never re-parses the file.

        The output is read as the step's operation (:attr:`~chemrefine.state.StepContext.
        operation`), resolved once with its context and keyed by every row, so no output
        re-reads the template. One the parser dispatch has no reader for yet — a reaction
        path's ``irc``, a band's ``neb``, a trajectory's ``md`` — is read as its final
        structure, as the run said when it started
        (:func:`~chemrefine.engines.api.operation_notice`): a run kind runs and yields
        something usable the day it appears in a template, and fans out into its geometries
        the day its reader joins the dispatch, with no edit here. An explicit ``operation:``
        is held to :attr:`operations` at the run's t=0 walk
        (:func:`~chemrefine.engines.api.check_operation`), so only an inferred one is read this
        way. The final-structure read finds what a single-structure ``.out`` carries; a run
        that writes its geometries elsewhere (a trajectory) leaves it nothing, and the
        structure is ledgered unparseable. A context built while its template could not be
        read carries no operation; the output then reads the template itself, through the
        reader that refuses it by name.

        A reason for an output it cannot use names that output relative to the step
        (:func:`~chemrefine.ids.step_relative`), so it still points at the file once the
        tree is copied or moved.
        """
        operation = ctx.operation
        if operation is None:
            operation = self.infer_operation(require_template(ctx.template, label=self.label))
        src = step_relative(output_path, ctx.step_dir)
        if ctx.step_cfg.operation is None and operation not in output.known_operations():
            return output.parse_dft(output_path, src=src)
        return output.parse_output(output_path, operation, src=src)

    # -- nms hook (the only engine-specific half of chemrefine.nms) --------

    def nms_input_info(self, ctx: StepContext) -> NmsInputInfo:
        """Whether the template runs a TS search and computes frequencies (keyword scan)."""
        run = inspect.inspect_template(require_template(ctx.template, label=self.label).text)
        return NmsInputInfo(is_transition_state=run.is_ts, computes_frequencies=run.has_freq)

    def parse_frequency_output(self, output_path: Path) -> list[ParsedResult]:
        """Re-parse one finished ``.out`` ctx-free — the viewer half of NMS capability."""
        return output.parse_dft(output_path)

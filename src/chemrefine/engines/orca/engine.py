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

from chemrefine.config import StepConfig
from chemrefine.engines._job import JobEngine
from chemrefine.engines.api import NmsInputInfo, ParsedResult, RunBlock, register
from chemrefine.engines.orca import input as orca_input
from chemrefine.engines.orca import inspect, output
from chemrefine.errors import ConfigError
from chemrefine.ids import require_template
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
    :class:`~chemrefine.engines.api.StarterProviding`. The ExtOpt engines inherit it: they
    are ORCA-driven and read the same input."""
    preflight_refuses: ClassVar[str] = (
        "an `operation:` outside the parser dispatch's vocabulary, which would otherwise fail "
        "only after every job had run and each output was ledgered UNPARSEABLE"
    )
    operations: ClassVar[tuple[str, ...]] = tuple(sorted(output.known_operations()))
    """The ``operation:`` vocabulary this family interprets — the parser dispatch's own
    set (see :class:`~chemrefine.engines.api.OperationsDeclaring`), inherited by the
    ExtOpt engines, which are ORCA-driven and parse the same outputs."""
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

    # -- preflight ---------------------------------------------------------

    def check_step(self, step_cfg: StepConfig, *, charge: int, multiplicity: int) -> None:
        """Refuse an ``operation`` the parser dispatch does not know, before any job runs.

        ``operation`` never changes the generated input — it only picks the parser — so a
        value outside the dispatch's vocabulary cannot fail until the outputs are read:
        every job runs at full cost first, each output is then ledgered ``UNPARSEABLE``,
        and because the operation is part of every row key, correcting the typo re-keys
        the rows and no recovery command adopts the paid outputs. Decidable from the
        config alone, so it belongs on :class:`~chemrefine.engines.api.PreflightChecking`,
        where the run's t=0 walk and ``chemrefine validate`` both make it.

        The accepted set is the parser's own:
        :func:`~chemrefine.engines.orca.output.known_operations` plus the legacy ``dft``
        spelling the dispatch still reads (deliberately absent from what introspection
        *offers*). Normalised the way the dispatch normalises, so ``GOAT`` and ``OPT+SP``
        stay legal however the config was built — the legacy YAML rewriter lowercases on
        the way in, but a :class:`~chemrefine.config.StepConfig` built directly does not.
        ``None`` is untouched: the template inspection decides, and a template problem
        has refusals of its own. ``charge`` / ``multiplicity`` are unread — the
        signature is the capability's.
        """
        if step_cfg.operation is None:
            return
        normalized = step_cfg.operation.lower().replace("+", "_")
        if normalized not in output.known_operations() | {"dft"}:
            raise ConfigError(
                f"step {step_cfg.step}: unknown ORCA operation {step_cfg.operation!r} — "
                f"this engine parses {sorted(output.known_operations())}. Correct it, or "
                f"omit `operation:` to infer the run type from the template's keywords."
            )

    # -- run ---------------------------------------------------------------

    def pal(self, ctx: StepContext) -> int:
        """PAL is a property of the template (one ``%pal`` for the step), read once."""
        return inspect.inspect_template(require_template(ctx.template, label=self.label)).pal

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
        info = inspect.inspect_template(require_template(ctx.template, label=self.label))
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

    def parse_frequency_output(self, output_path: Path) -> list[ParsedResult]:
        """Re-parse one finished ``.out`` ctx-free — the viewer half of NMS capability."""
        return output.parse_dft(output_path)

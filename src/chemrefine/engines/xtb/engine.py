"""``XtbEngine`` — the xtb program as a per-structure job engine, registered as ``"xtb"``.

A :class:`~chemrefine.engines._job.JobEngine`: it supplies only xtb-specific primitives —
write the xcontrol input (:mod:`chemrefine.engines.xtb.input`), give the run command
(:mod:`chemrefine.engines.xtb.command`), parse the output (:mod:`chemrefine.engines.xtb.output`)
— while the shared base handles ``prepare`` / ``submit`` / ``parse`` and
:mod:`chemrefine.engines._execution` runs the batch. It runs the xtb binary itself, so an
xTB step needs no ORCA install.

What shapes this engine, and how each is answered:

* **xtb's command line decides what it computes.** The run type, method, solvation,
  charge and spin are all arguments, and xtb's xcontrol file has no key for the run type.
  So the step's ``operation:`` *is* the run type (``sp`` / ``opt_sp`` / ``freq`` /
  ``opt_freq``), required on every step, and the options carry the rest. The template is
  an xcontrol file for what only xcontrol can say, refused when it sets what the command
  line sets.
* **The geometry is the structure's own XYZ file**, named on the command line, so it
  stays in Å both ways (xtb writes ``xtbopt.xyz`` in its input's format).
* **Parallelism is threads** (``-P`` and ``OMP_NUM_THREADS``), so the layout is one task
  of ``cores`` CPUs, as for Q-Chem.
* **GFN0's parameters live in a file** xtb finds through ``XTBPATH``. The run block sets
  it from the install the binary sits in (see :meth:`XtbEngine.run_block`).

Not :class:`~chemrefine.engines.api.OperationInferring`: an xcontrol file cannot name a run
type, so the declared ``operation`` is keyed and read as written. NMS-capable: a
``freq`` / ``opt_freq`` step's structures carry the modes xtb's ``g98.out`` lists.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import ClassVar

from chemrefine.config import StepConfig
from chemrefine.engines._job import JobEngine
from chemrefine.engines.api import NmsInputInfo, ParsedResult, RunBlock, register
from chemrefine.engines.xtb import command, output
from chemrefine.engines.xtb import input as xtb_input
from chemrefine.engines.xtb.options import XtbOptions
from chemrefine.errors import ConfigError
from chemrefine.ids import StepTemplate, step_relative
from chemrefine.state import StepContext


@register("xtb")
class XtbEngine(JobEngine):
    """xtb engine — one job per structure, the run type from ``operation:``, threaded."""

    name: ClassVar[str] = "xtb"
    label: ClassVar[str] = "xtb"
    template_suffix: ClassVar[str] = "xcontrol"
    output_suffix: ClassVar[str] = "out"
    template_starter: ClassVar[str] = (
        "# xtb xcontrol input for this step, passed to xtb as --input.\n"
        "# ChemRefine passes the run type (the step's operation), the method, solvation,\n"
        "# charge and multiplicity on xtb's command line, so they do not belong here.\n"
        "# Add what only xcontrol can say: constraints, a wall, a scan, optimiser or SCC\n"
        "# settings. See `man 7 xcontrol`.\n"
        "$end\n"
    )
    """What ``chemrefine scaffold`` writes for a missing ``stepN.xcontrol`` — see
    :attr:`~chemrefine.engines.api.TemplateDriven.template_starter`. It runs as it is."""
    preflight_refuses: ClassVar[str] = (
        "an xtb step that names no `operation:` (its xcontrol template cannot say what to "
        "run), CPCM-X on any operation but `sp`, an `opt_level` on a run that does not "
        "optimise, `device: cuda` or `backend_python`; and a template that sets what the "
        "command line sets — `$chrg`, `$spin`, the `$gfn` method or a solvent"
    )
    operations: ClassVar[tuple[str, ...]] = tuple(sorted(command.RUN_TYPES))
    """The ``operation:`` vocabulary — the run types :mod:`~chemrefine.engines.xtb.command`
    knows (see :class:`~chemrefine.engines.api.OperationsDeclaring`), derived rather than
    restated so the dropdown, the schema document and the t=0 check cannot disagree with
    what the command line can run."""
    executable_keys: ClassVar[tuple[str, ...]] = ("xtb",)
    """Satisfies :class:`~chemrefine.engines.api.ExecutablesDeclaring`: the xtb binary."""
    frequency_request: ClassVar[str] = "`operation: freq` or `operation: opt_freq`"
    """Satisfies :class:`~chemrefine.engines.api.NmsCapableEngine`: the operations
    :meth:`nms_input_info` counts as computing frequencies."""
    output_globs: ClassVar[tuple[str, ...]] = (
        "*.out",
        "*.xtbopt.xyz",
        "*.xtbopt.log",
        "*.gradient",
        "*.hessian",
        "*.vibspectrum",
        "*.charges",
        "*.wbo",
    )
    """Result files copied back out of ``$WORK_DIR``; xtb's restart and topology files stay.

    xtb runs with ``--namespace`` set to the structure's stem, so it names every file
    ``<stem>.<its usual name>`` (``step1_0.xtbopt.xyz``, ``step1_0.gradient``). That is what
    lets an NMS winner's files follow its output when the child is promoted to its parent's
    name (:func:`chemrefine.attempts.promote` carries the files that start with the stem).
    The stdout ``.out`` is written straight to ``$OUTPUT_DIR``; ``*.out`` brings back
    ``<stem>.g98.out``. ``xtbopt.log`` is the optimisation's trajectory, ``hessian`` the
    Hessian (Eh/bohr²) and ``vibspectrum`` every mode's frequency and IR intensity. The
    ``.engrad`` xtb also writes holds what ``gradient`` does, under a name built from the
    geometry file's, and is left in scratch."""

    options_cls: ClassVar[type[XtbOptions]] = XtbOptions

    # -- input -------------------------------------------------------------

    def build_input(
        self,
        *,
        xyz_path: Path,
        template: StepTemplate,
        input_path: Path,
        output_path: Path,
        ctx: StepContext,
    ) -> None:
        """Write one structure's xcontrol input: the template, unchanged.

        ``xyz_path`` is not written into it: the run block names the structure's input
        geometry as xtb's geometry argument (:data:`~chemrefine.engines.xtb.input.
        GEOMETRY_SUFFIX`).
        """
        xtb_input.build_input(template=template, output_path=input_path)

    # -- preflight ---------------------------------------------------------

    def check_step(self, step_cfg: StepConfig, *, charge: int, multiplicity: int) -> None:
        """Refuse what is decidable from the config alone, before any job runs.

        * No ``operation:`` — the template cannot say what to run, and a default would run
          something the step does not name.
        * CPCM-X on a run that needs a gradient — xtb's CPCM-X gives none.
        * An ``opt_level`` on a run that does not optimise — a knob nothing reads.
        * ``device: cuda`` and ``backend_python`` — the shared options model accepts both,
          and nothing here reads either: one would charge a GPU for a CPU job, the other
          names an interpreter for a program that is not a Python backend.

        An explicit operation outside :attr:`operations` is refused before this is asked
        (:func:`~chemrefine.engines.api.check_operation`). ``charge`` / ``multiplicity``
        are unread — the signature is the capability's.
        """
        opts = self.options_cls.from_raw(step_cfg.engine_options())
        operation = self._require_operation(step_cfg)
        if opts.solvation == "cpcmx" and operation != "sp":
            raise ConfigError(
                f"step {step_cfg.step}: `solvation: cpcmx` with `operation: {operation}` — "
                f"xtb's CPCM-X gives no gradient, so it runs only as `operation: sp`."
            )
        if opts.opt_level is not None and operation not in command.OPTIMISING:
            raise ConfigError(
                f"step {step_cfg.step}: `opt_level` on `operation: {operation}`, which does not "
                f"optimise, so nothing reads it. Remove it, or use opt_sp / opt_freq."
            )
        if opts.device != "cpu":
            raise ConfigError(
                f"step {step_cfg.step}: `device: {opts.device}` on an xtb step — xtb runs on "
                f"the CPU, so the knob would only charge a GPU against max_gpus. Leave it unset."
            )
        if opts.backend_python is not None:
            raise ConfigError(
                f"step {step_cfg.step}: `backend_python` on an xtb step — xtb is a program "
                f"named in `executables`, not a Python backend, so nothing reads it. Remove it."
            )

    def check_template(self, ctx: StepContext, template: StepTemplate) -> None:
        """Satisfies :class:`~chemrefine.engines.api.TemplateChecking`: the template sets
        nothing the command line sets (:func:`~chemrefine.engines.xtb.input.
        require_fillable`), refused by the run's t=0 walk and by ``chemrefine validate``.
        ``ctx`` is unread — the signature is the capability's."""
        xtb_input.require_fillable(template)

    @staticmethod
    def _require_operation(step_cfg: StepConfig) -> str:
        """The step's operation, or the refusal of a step that names none."""
        if step_cfg.operation is None:
            raise ConfigError(
                f"step {step_cfg.step}: an xtb step names its `operation:` — one of "
                f"{', '.join(command.RUN_TYPES)} — because the xcontrol template cannot say "
                f"what to run."
            )
        return step_cfg.operation

    # -- run ---------------------------------------------------------------

    def _opts(self, ctx: StepContext) -> XtbOptions:
        """This step's validated options — the single reader of the YAML knobs.

        Strict, over the engine's share of the mapping (:meth:`~chemrefine.config.StepConfig.
        engine_options`, which keeps the NMS knobs out of the read): every knob reaches the
        command line, so a misspelled one has no other reader to fall back on.
        """
        return self.options_cls.from_raw(ctx.step_cfg.engine_options())

    def pal(self, ctx: StepContext) -> int:
        """Threads one job uses: ``options.cores``."""
        return self._opts(ctx).cores

    def slurm_layout(self, ctx: StepContext) -> tuple[int, int]:
        """One threaded process: ``(1, cores)``, clamped to ``max_cores`` like every layout."""
        return (1, min(self.pal(ctx), ctx.max_cores))

    @staticmethod
    def parameter_dir(executable: str | None) -> str | None:
        """The ``share/xtb`` directory of the install ``executable`` sits in, or ``None``.

        For an absolute ``<prefix>/bin/xtb``, ``<prefix>/share/xtb``: where the release
        tarball, a conda environment and a meson install all keep the parameter files GFN0
        reads (GFN1 and GFN2 have theirs built in). ``None`` for a bare ``xtb`` or any other
        layout, which leaves ``XTBPATH`` to the environment.
        """
        if executable is None:
            return None
        path = Path(executable)
        if not path.is_absolute() or path.parent.name != "bin":
            return None
        return str(path.parent.parent / "share" / "xtb")

    def run_block(self, ctx: StepContext, inp_path: Path, out_path: Path) -> RunBlock:
        """The thread environment, ``XTBPATH``, and the xtb invocation.

        The geometry is the structure's ``_inp.xyz`` beside its output, and the structure's
        stem is xtb's ``--namespace`` (see :attr:`output_globs`), the stem derived in bash so
        the job-array ``$INP_NAME`` sentinel still expands. The stack settings are
        xtb's own advice: its Hessian of a large molecule overflows the default stack.
        ``XTBPATH`` is set only when the environment (a module, the header) has not set it
        already.
        """
        opts = self._opts(ctx)
        _ntasks, threads = self.slurm_layout(ctx)
        args = command.arguments(
            opts,
            operation=self._require_operation(ctx.step_cfg),
            charge=ctx.charge,
            multiplicity=ctx.multiplicity,
        )
        executable = ctx.executables.get("xtb")
        lines = [
            f"export OMP_NUM_THREADS={threads},1",
            f"export MKL_NUM_THREADS={threads}",
            'export OMP_STACKSIZE="${OMP_STACKSIZE:-4G}"',
            "ulimit -s unlimited 2>/dev/null || true",
        ]
        parameters = self.parameter_dir(executable)
        if parameters is not None:
            lines.append(
                f'if [ -z "${{XTBPATH:-}}" ]; then export XTBPATH={shlex.quote(parameters)}; fi'
            )
        lines += [
            f'XTB_STEM="{inp_path.name}"; XTB_STEM="${{XTB_STEM%.*}}"',
            f"{shlex.quote(executable or 'xtb')} "
            f'"$OUTPUT_DIR/${{XTB_STEM}}{xtb_input.GEOMETRY_SUFFIX}" --input "{inp_path.name}" '
            f'--namespace "$XTB_STEM" {shlex.join(args)} -P {threads} '
            f'> "$OUTPUT_DIR/{out_path.name}"',
        ]
        return RunBlock(body="\n".join(lines))

    def extra_header_fields(self, ctx: StepContext) -> tuple[tuple[str, object], ...]:
        """Record which xtb binary ran in the runlog header."""
        return (("xtb_executable", ctx.executables.get("xtb", "xtb")),)

    # -- parse -------------------------------------------------------------

    @staticmethod
    def _input_geometry(output_path: Path) -> Path:
        """The structure's input geometry, beside its output and named from the same stem."""
        return output_path.with_name(f"{output_path.stem}{xtb_input.GEOMETRY_SUFFIX}")

    def parse_one(
        self, output_path: Path, structure_id: str, ctx: StepContext
    ) -> list[ParsedResult]:
        """Parse one xtb run — energy, forces, geometry, verdicts, and the NMS feed.

        One reader for every operation (:func:`~chemrefine.engines.xtb.output.parse_xtb`),
        because it reads what stdout says xtb wrote. A reason for an output it cannot use
        names that output relative to the step (:func:`~chemrefine.ids.step_relative`), so
        it still points at the file once the tree is copied or moved.
        """
        return output.parse_xtb(
            output_path,
            input_geometry=self._input_geometry(output_path),
            src=step_relative(output_path, ctx.step_dir),
        )

    # -- nms hook (the only engine-specific half of chemrefine.nms) --------

    def nms_input_info(self, ctx: StepContext) -> NmsInputInfo:
        """Whether the step computes frequencies: its operation does.

        Never a transition-state search: xtb has no TS optimiser, so a ``ts`` NMS target
        is the user's to ask for explicitly.
        """
        return NmsInputInfo(
            is_transition_state=False,
            computes_frequencies=ctx.step_cfg.operation in command.FREQUENCY,
        )

    def parse_frequency_output(self, output_path: Path) -> list[ParsedResult]:
        """Re-parse one finished ``.out`` ctx-free — the viewer half of NMS capability."""
        return output.parse_xtb(output_path, input_geometry=self._input_geometry(output_path))

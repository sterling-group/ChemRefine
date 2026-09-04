"""``QchemEngine`` — Q-Chem as a per-structure job engine, registered as ``"qchem"``.

A :class:`~chemrefine.engines._job.JobEngine`: it supplies only Q-Chem-specific primitives —
write the ``.in`` (:mod:`chemrefine.engines.qchem.input`), give the run command, parse the
``.out`` (:mod:`chemrefine.engines.qchem.output`) — while the shared base handles
``prepare`` / ``submit`` / ``parse`` and :mod:`chemrefine.engines._execution` runs the batch.

Where Q-Chem differs from ORCA, and how each difference is answered:

* **Parallelism is CLI-side** (``-nt`` threads, ``-mpi -np`` ranks), never in the input
  file — so the numbers live in the YAML (``options.cores`` / ``options.nprocs``) and
  :meth:`QchemEngine.slurm_layout` spells them to SLURM as ``(1, threads)`` or
  ``(ranks, threads)``, where ORCA's ranks are ``(pal, 1)``.
* **The install is an environment**, not one binary: the ``qchem`` wrapper needs ``QC`` (and
  ``QCAUX``, which Q-Chem itself defaults to ``$QC/qcaux`` when unset — the manual says so,
  and the run block spells that default visibly). Both come from the config's
  ``executables`` map (machine facts, like the ``orca`` binary); a cluster with
  ``module load qchem`` in its header needs neither.
* **Scratch is ``QCSCRATCH`` + a savename**: the per-job ``$WORK_DIR`` *is* the QCSCRATCH
  role, and the savename (the structure stem, derived in bash so the array sentinel
  survives) is what makes Q-Chem keep its key scratch files instead of deleting them —
  ``options.save`` then copies that directory home via the run block's cleanup.
* **Memory is ``mem_total``** in the ``$rem`` block, ORCA's ``%maxcore`` counterpart —
  declared to the scheduler through :meth:`memory_mb`, no headroom factor (qqchem accepts a
  ``mem_total`` up to the full allocation).

NMS-capable like ORCA: :meth:`nms_input_info` reads the template's ``JOBTYPE`` facts, and
the frequency values ride on each parsed structure (see the output module's mode-index
contract).
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import ClassVar

from chemrefine.config import StepConfig
from chemrefine.engines._job import JobEngine
from chemrefine.engines.api import NmsInputInfo, ParsedResult, RunBlock, register
from chemrefine.engines.qchem import input as qchem_input
from chemrefine.engines.qchem import inspect, output
from chemrefine.engines.qchem.options import QchemOptions
from chemrefine.errors import ConfigError
from chemrefine.ids import require_template
from chemrefine.state import StepContext


@register("qchem")
class QchemEngine(JobEngine):
    """Q-Chem engine — one job per structure, threads by default, MPI opt-in."""

    name: ClassVar[str] = "qchem"
    label: ClassVar[str] = "Q-Chem"
    template_suffix: ClassVar[str] = "in"
    output_suffix: ClassVar[str] = "out"
    # The comment deliberately never spells a section name: the input writer's block regexes
    # are line-anchored, but a starter that does not mention them is one whose rendering can
    # never depend on that anchoring.
    template_starter: ClassVar[str] = (
        "$comment\n"
        "Q-Chem starter — ChemRefine swaps each structure's geometry into the first\n"
        "coordinate block below.\n"
        "$end\n"
        "\n"
        "$molecule\n"
        "0 1\n"
        "H 0.0 0.0 0.0\n"
        "$end\n"
        "\n"
        "$rem\n"
        "  jobtype     opt\n"
        "  method      b3lyp\n"
        "  basis       def2-svp\n"
        "$end\n"
    )
    """What ``chemrefine scaffold`` writes for a missing ``stepN.in`` — see
    :class:`~chemrefine.engines.api.StarterProviding`."""
    operations: ClassVar[tuple[str, ...]] = tuple(sorted(output.known_operations()))
    """The ``operation:`` vocabulary this engine interprets — the parser dispatch's own
    set (see :class:`~chemrefine.engines.api.OperationsDeclaring`), derived rather than
    restated so the dropdown, the schema document and the preflight refusal cannot
    disagree with what :func:`~chemrefine.engines.qchem.output.parse_output`
    accepts."""
    output_globs: ClassVar[tuple[str, ...]] = ("*.out", "*.fchk")
    """Result files copied back out of ``$WORK_DIR``.

    The ``.out`` is written straight to ``$OUTPUT_DIR`` (ORCA's redirect pattern, so it
    live-tails); the glob still covers a template that writes extra ``.out`` artifacts.
    ``.fchk`` is what a ``GUI=2`` job leaves beside the run for visualisation tooling."""

    options_cls: ClassVar[type[QchemOptions]] = QchemOptions

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
        """Write one Q-Chem ``.in`` — the geometry into job 1's ``$molecule`` block."""
        qchem_input.build_input(
            xyz_path=xyz_path,
            template_path=template_path,
            output_path=input_path,
            charge=ctx.charge,
            multiplicity=ctx.multiplicity,
        )

    # -- preflight ---------------------------------------------------------

    def check_step(self, step_cfg: StepConfig, *, charge: int, multiplicity: int) -> None:
        """Refuse an ``operation`` the parser dispatch does not know, before any job runs.

        ``operation`` never changes the generated input — it only picks the parser — so
        a value outside the vocabulary cannot fail until every job has run at full cost
        and each output is ledgered ``UNPARSEABLE``. Decidable from the config alone,
        so it fires at the run's t=0 walk and in ``chemrefine validate``. ``None`` is
        untouched: the template inspection decides. ``charge`` / ``multiplicity`` are
        unread — the signature is the capability's.
        """
        if step_cfg.operation is None:
            return
        normalized = step_cfg.operation.lower().replace("+", "_")
        if normalized not in output.known_operations():
            raise ConfigError(
                f"step {step_cfg.step}: unknown Q-Chem operation {step_cfg.operation!r} — "
                f"this engine parses {sorted(output.known_operations())}. Correct it, or "
                f"omit `operation:` to infer the run type from the template's JOBTYPE."
            )

    # -- run ---------------------------------------------------------------

    def _opts(self, ctx: StepContext) -> QchemOptions:
        """This step's validated options — the single reader of the YAML knobs."""
        return self.options_cls.from_raw_lenient(ctx.step_cfg.options)

    def pal(self, ctx: StepContext) -> int:
        """Total cores one job uses: threads, times MPI ranks when the step opts in."""
        opts = self._opts(ctx)
        return opts.cores * (opts.nprocs or 1)

    def slurm_layout(self, ctx: StepContext) -> tuple[int, int]:
        """Threads are ``(1, cores)``; MPI is ``(nprocs, cores)``.

        The pure-OpenMP case clamps to ``max_cores`` like every default layout — one
        process can use fewer threads than asked. An MPI layout is **not** clamped: its
        factorization is the job's shape, so exceeding the budget is refused upstream
        (:class:`~chemrefine.engines._execution._BatchPlan`) rather than silently reshaped.
        """
        opts = self._opts(ctx)
        if opts.nprocs is None:
            return (1, min(opts.cores, ctx.max_cores))
        return (opts.nprocs, opts.cores)

    def memory_mb(self, ctx: StepContext) -> int | None:
        """The peak ``mem_total`` the template declares — Q-Chem's ``%maxcore`` counterpart.

        No headroom factor, unlike ORCA's 75% rule: ``mem_total`` is the whole job's
        declaration and qqchem accepts one up to the full SLURM allocation. ``None`` when
        the template declares nothing — the header's memory policy stands.
        """
        return inspect.inspect_template(
            require_template(ctx.template, label=self.label)
        ).mem_total_mb

    def _executable(self, ctx: StepContext) -> str:
        """The qchem invocation: explicit ``executables.qchem`` → ``$QC/bin/qchem`` → bare.

        The middle form is spelled with the *variable*, not the configured path: the run
        block exports ``QC`` (quoted) just above, so the command stays readable in the
        script and cannot disagree with the environment Q-Chem's own wrapper reads.
        """
        explicit = ctx.executables.get("qchem")
        if explicit:
            return shlex.quote(explicit)
        if ctx.executables.get("qc"):
            return '"$QC/bin/qchem"'
        return "qchem"

    def _parallel_flags(self, ctx: StepContext) -> str:
        """qqchem's exact emission: ``-nt N`` for threads; ``-mpi -np P [-nt N]`` for MPI."""
        ntasks, threads = self.slurm_layout(ctx)
        if self._opts(ctx).nprocs is None:
            return f"-nt {threads}"
        if threads > 1:
            return f"-mpi -np {ntasks} -nt {threads}"
        return f"-mpi -np {ntasks}"

    def run_block(self, ctx: StepContext, inp_path: Path, out_path: Path) -> RunBlock:
        """The Q-Chem environment, the invocation, and the opt-in scratch copy-back.

        ``QCSCRATCH`` is the job's own ``$WORK_DIR``; the savename (structure stem, derived
        in bash so the job-array ``$INP_NAME`` sentinel still expands) makes Q-Chem keep its
        key scratch files there. ``QCAUX`` is exported only when configured or derivable:
        an explicit ``executables.qcaux`` wins (installs that keep qcaux *beside* the QC
        root), else ``$QC/qcaux`` — Q-Chem's own documented default, spelled visibly. The
        cleanup line runs inside the script's single EXIT trap, so a ``save: true`` step
        keeps its scratch on every exit path.
        """
        opts = self._opts(ctx)
        _ntasks, threads = self.slurm_layout(ctx)
        lines: list[str] = []
        qc = ctx.executables.get("qc")
        qcaux = ctx.executables.get("qcaux")
        if qc:
            lines.append(f"export QC={shlex.quote(qc)}")
            lines.append(
                f"export QCAUX={shlex.quote(qcaux)}" if qcaux else 'export QCAUX="$QC/qcaux"'
            )
            lines.append('export PATH="$PATH:$QC/bin:$QC/bin/perl"')
        elif qcaux:
            lines.append(f"export QCAUX={shlex.quote(qcaux)}")
        lines += [
            'export QCSCRATCH="$WORK_DIR"',
            f"export OMP_NUM_THREADS={threads}",
            f"export QC_THREADS={threads}",
            f'QCSAVE="{inp_path.name}"; QCSAVE="${{QCSAVE%.*}}"',
            f"{self._executable(ctx)} {self._parallel_flags(ctx)} "
            f'{inp_path.name} "$OUTPUT_DIR/{out_path.name}" "$QCSAVE"',
        ]
        # Guarded expansions, because the exit trap is armed *before* the body runs under
        # `set -u`: a TERM landing in that window reaches this cleanup with QCSAVE unset,
        # and a bare `$QCSAVE` is then a nounset abort *inside the handler* — taking the
        # copy-back, the output_dirs copy and the runlog footer with it (`set +e` does not
        # suppress nounset). The `-n` test also keeps an empty QCSAVE from turning the
        # source into a bare scratch root. The rule is held for every engine by
        # `test_every_cleanup_survives_the_armed_trap_window`.
        cleanup = (
            'if [ -n "${QCSAVE:-}" ]; then '
            'cp -r "${QCSCRATCH:-}/${QCSAVE:-}" "$OUTPUT_DIR/" 2>/dev/null || true; fi'
            if opts.save
            else ""
        )
        return RunBlock(body="\n".join(lines), cleanup=cleanup)

    def extra_header_fields(self, ctx: StepContext) -> tuple[tuple[str, object], ...]:
        """Record which qchem binary ran in the runlog header — as raw text, never ``$QC``.

        ORCA's raw-value convention, deliberately not :meth:`_executable`'s quoted
        invocation. This used to record ``"$QC/bin/qchem"`` on the belief that the header
        heredoc would expand ``$QC`` "to the run's real root when set" — but the header
        executes *before* the run block that exports ``QC``, under the script's
        ``set -euo pipefail``, where an unset ``$QC`` in a heredoc is fatal. Every job
        configured the documented way (``executables: {qc: …}``, no explicit ``qchem``)
        died at its own header, before the trap was armed: no footer, no copy-back, the
        whole step ledgered ``output missing``. The recorded value is now the configured
        path text itself, derived by the same precedence :meth:`_executable` uses —
        and :func:`chemrefine.job_log.bash_header` refuses any ``$``-carrying field
        value outright, so this cannot quietly regress.
        """
        explicit = ctx.executables.get("qchem")
        qc = ctx.executables.get("qc")
        recorded = explicit or (f"{qc}/bin/qchem" if qc else "qchem")
        return (("qchem_executable", recorded),)

    # -- parse -------------------------------------------------------------

    def _resolve_operation(self, ctx: StepContext) -> str:
        """Operation for parsing: an explicit ``operation`` wins, else inspect the template.

        The template's ``JOBTYPE`` facts pick the parser when ``operation`` is omitted,
        through the one reader :mod:`~chemrefine.engines.qchem.inspect` already is.
        """
        if ctx.step_cfg.operation is not None:
            return ctx.step_cfg.operation
        return inspect.inspect_template(require_template(ctx.template, label=self.label)).operation

    def parse_one(
        self, output_path: Path, structure_id: str, ctx: StepContext
    ) -> list[ParsedResult]:
        """Parse one Q-Chem output — energy, final geometry, and the NMS frequency feed."""
        return output.parse_output(output_path, self._resolve_operation(ctx))

    # -- nms hook (the only engine-specific half of chemrefine.nms) --------

    def nms_input_info(self, ctx: StepContext) -> NmsInputInfo:
        """Whether the template runs a TS search and computes frequencies (``JOBTYPE`` scan)."""
        info = inspect.inspect_template(require_template(ctx.template, label=self.label))
        return NmsInputInfo(is_transition_state=info.is_ts, computes_frequencies=info.has_freq)

    def parse_frequency_output(self, output_path: Path) -> list[ParsedResult]:
        """Re-parse one finished ``.out`` ctx-free — the viewer half of NMS capability."""
        return output.parse_qchem(output_path)

"""Engine Protocol contract and the ``ENGINES`` registry.

Every engine satisfies :class:`CalculationEngine` structurally — no
inheritance required. Engines register themselves via the :func:`register`
decorator at import time, so importing :mod:`chemrefine.engines`
populates the registry as a side effect.

The orchestrator only ever sees the :class:`CalculationEngine` Protocol
plus the :data:`ENGINES` dict. ORCA-specific imports, MLIP imports, etc.
never reach :mod:`chemrefine.pipeline` — that's how the orchestrator
stays engine-agnostic.

Adding a new engine (e.g. qchem, psi4, cfour)
---------------------------------------------
1. Create a package ``engines/<name>/`` — *everything* engine-specific lives
   there (input writer, output parser, options model, ...). Shared, engine-
   neutral infrastructure stays in ``engines/`` (this module, the SLURM batch
   base, the template renderer, the ``_backend_server`` gradient service).
2. Add ``engines/<name>/engine.py`` with a class that either subclasses
   :class:`SlurmBatchEngine` (implement the ``_pal`` / ``_run_block`` hooks and
   the ``output_globs`` / ``template_suffix`` / ``label`` ClassVars) or
   satisfies :class:`CalculationEngine` directly, decorated with
   ``@register("<name>")``. Import it from ``engines/<name>/__init__.py`` and
   list that package in ``engines/__init__.py`` so registration fires. (Old
   spellings are mapped to the canonical name by the config normalizer, not by
   registration — see :func:`chemrefine.config._normalize_legacy`.)
3. An external binary? Read its path from ``ctx.executables.get("<name>")``
   (set in the YAML ``executables`` map). An importable backend? Ship it as a
   ``pip install chemrefine[<name>]`` extra and import it in-process.
4. To let ORCA optimise using this engine's gradients, add
   ``engines/<name>/extopt_calc.py`` implementing
   :class:`~chemrefine.engines._backend_server.base.ComputeBackend` and one
   line in ``engines/_backend_server/registry.py``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

from chemrefine import slurm, throttle
from chemrefine.errors import EngineNotFoundError
from chemrefine.ids import resolve_step_template
from chemrefine.state import JobBatch, StepContext, StepInputs, StepResults

logger = logging.getLogger(__name__)

# Throttler poll cadence. SLURM ``squeue`` is expensive and jobs are minutes-long,
# so poll slowly; background local processes are right here and often finish in
# well under a second, so poll fast or every local step would stall for a full
# SLURM interval.
_SLURM_POLL_SECONDS = 10.0
_LOCAL_POLL_SECONDS = 0.25


@runtime_checkable
class CalculationEngine(Protocol):
    """Structural contract every engine must satisfy.

    The five lifecycle methods mirror the five stages
    :func:`chemrefine.step.run_step` calls in order. ``supports_nms`` is
    a class-level boolean: when ``True``, :meth:`normal_mode_sample` is
    invoked between :meth:`parse` and filtering for steps that set
    ``nms: true`` in their YAML.
    """

    name: str
    supports_nms: bool

    def prepare(self, ctx: StepContext) -> StepInputs:
        """Write engine-specific input files for this step's seed structures."""
        ...

    def submit(self, inputs: StepInputs, ctx: StepContext) -> JobBatch:
        """Submit a batch of jobs (SLURM or local); return their handles."""
        ...

    def wait(self, batch: JobBatch) -> None:
        """Block until every job in the batch finishes (success or failure)."""
        ...

    def parse(self, inputs: StepInputs, ctx: StepContext) -> StepResults:
        """Parse each output file into a :class:`~chemrefine.state.Structure`."""
        ...

    def normal_mode_sample(self, results: StepResults, ctx: StepContext) -> StepResults:
        """Expand a frequency-step result by displacing along imaginary modes."""
        ...


ENGINES: dict[str, type[CalculationEngine]] = {}
"""Registry mapping the YAML ``engine:`` string to a concrete engine class.

Only **canonical** names live here. Old spellings (``mlff*``, ``dft``) are
rewritten to canonical names by the config normalizer
(:func:`chemrefine.config._normalize_legacy`) — the single place that knows the
legacy vocabulary — before any lookup, so the registry stays alias-free.
"""


def register(name: str) -> Callable[[type], type]:
    """Decorator: register ``cls`` under ``name`` in :data:`ENGINES`."""

    def decorator(cls: type) -> type:
        if name in ENGINES and ENGINES[name] is not cls:
            raise ValueError(f"engine {name!r} is already registered to {ENGINES[name]!r}")
        ENGINES[name] = cls
        return cls

    return decorator


def get_engine(name: str) -> CalculationEngine:
    """Look up an engine by its canonical name and return a fresh instance.

    Raises :class:`~chemrefine.errors.EngineNotFoundError` if ``name`` isn't
    registered. (Legacy spellings are normalized at config-parse time, so the
    name reaching here is already canonical.)
    """
    engine_cls = ENGINES.get(name)
    if engine_cls is None:
        raise EngineNotFoundError(
            f"unknown engine {name!r}; registered: {sorted(ENGINES)}"
        )
    return engine_cls()


class SlurmBatchEngine:
    """Shared base for engines that submit one SLURM job per structure.

    Owns the throttler-bounded submit loop, the no-op :meth:`wait` (submit
    already blocks until every job finishes), and per-step template
    resolution. Subclasses supply only the parts that differ:

    * :meth:`_pal` — the per-job core count (PAL) before clamping.
    * :meth:`_run_block` — the bash that runs inside ``$WORK_DIR``.
    * :meth:`_extra_header_fields` — optional runlog header rows.
    * ``output_globs`` / ``template_suffix`` / ``label`` ClassVars.

    ``prepare`` / ``parse`` / ``normal_mode_sample`` stay engine-specific, so
    a subclass still satisfies :class:`CalculationEngine` structurally.
    """

    output_globs: ClassVar[tuple[str, ...]]
    template_suffix: ClassVar[str]
    label: ClassVar[str]

    def _resolve_template(self, ctx: StepContext) -> Path:
        """Resolve this step's input template (override or ``step{N}.{suffix}``)."""
        return resolve_step_template(
            ctx.template_dir,
            ctx.step_cfg.step,
            template=ctx.step_cfg.template,
            suffix=self.template_suffix,
            label=self.label,
        )

    def _pal(self, ctx: StepContext) -> int:
        """Return the per-job core count (PAL); the base clamps it to ``max_cores``."""
        raise NotImplementedError

    def _run_block(self, ctx: StepContext, inp_path: Path, out_path: Path) -> str:
        """Return the engine-specific bash that runs inside ``$WORK_DIR``."""
        raise NotImplementedError

    def _extra_header_fields(self, ctx: StepContext) -> tuple[tuple[str, object], ...]:
        """Engine-specific ``(key, value)`` rows appended to the runlog header."""
        return ()

    def submit(self, inputs: StepInputs, ctx: StepContext) -> JobBatch:
        """Generate a SLURM script per structure, submit under the PAL budget, block until done."""
        local = not slurm.sbatch_available()
        throttler = throttle.Throttler(
            max_cores=ctx.max_cores,
            poll_interval=_LOCAL_POLL_SECONDS if local else _SLURM_POLL_SECONDS,
        )
        header_path = ctx.template_dir / ctx.slurm_template
        if not header_path.is_file():
            raise FileNotFoundError(f"SLURM header template not found: {header_path}")

        pal = min(self._pal(ctx), ctx.max_cores)
        step_label = ctx.step_cfg.dir_name()
        jobs: dict[Path, str] = {}
        for inp, out, sid in inputs.files:
            throttler.wait_for_room(pal, is_finished=slurm.is_finished)
            script_path = inp.with_suffix(".slurm")
            slurm.build_script(
                job_name=inp.stem,
                pal=pal,
                template_path=header_path,
                script_path=script_path,
                input_path=inp,
                output_dir=out.parent,
                scratch_dir=ctx.scratch_dir,
                run_block=self._run_block(ctx, inp, out),
                engine=ctx.step_cfg.engine,
                operation=ctx.step_cfg.operation,
                step=ctx.step_cfg.step,
                structure_id=sid,
                step_label=step_label,
                output_globs=self.output_globs,
                extra_header_fields=self._extra_header_fields(ctx),
            )
            job_id = slurm.submit(script_path)
            throttler.register(job_id, pal)
            jobs[inp] = job_id
            logger.info("submitted %s as job %s (pal=%d)", inp.name, job_id, pal)

        throttler.wait_all(is_finished=slurm.is_finished)
        return JobBatch(jobs=jobs)

    def wait(self, batch: JobBatch) -> None:
        """No-op: :meth:`submit` already blocked until every job finished."""
        return None

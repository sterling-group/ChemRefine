"""Engine Protocol contract and the ``ENGINES`` registry.

Every engine satisfies :class:`CalculationEngine` structurally — no
inheritance required. Engines register themselves via the :func:`register`
decorator at import time, so importing :mod:`chemrefine.engines`
populates the registry as a side effect.

The orchestrator only ever sees the :class:`CalculationEngine` Protocol
plus the :data:`ENGINES` dict. ORCA-specific imports, MLIP imports, etc.
never reach :mod:`chemrefine.pipeline` — that's how the orchestrator
stays engine-agnostic.

Adding a new engine
---------------------------------------------
Everything engine-specific lives in a new ``engines/<name>/`` package; shared,
engine-neutral infrastructure stays in ``engines/`` (this module, the SLURM
batch base, the template renderer, the ``_backend_server`` gradient service).

**1. Pick a base** for ``engines/<name>/engine.py`` (decorate the class with
``@register("<name>")``):

* **The user supplies a ``step{N}.py``** they want run per structure → subclass
  :class:`~chemrefine.engines._template_engine.TemplateScriptEngine`. You inherit
  ``prepare`` / ``submit`` / ``parse`` / ``_pal`` / ``_run_block``; override only
  ``_template_vars`` to inject ``$VAR`` placeholders from ``step.options``. (See
  ``engines/mlip/engine.py`` — 20 lines.)
* **A real binary / custom SLURM job** → subclass :class:`SlurmBatchEngine`:
  implement ``prepare`` + ``parse``, the required hooks ``_pal`` + ``_run_block``,
  and the ClassVars ``label`` / ``template_suffix`` / ``output_globs``. (See
  ``engines/orca/engine.py``.)
* **ORCA optimises using this engine's gradients** → subclass
  :class:`~chemrefine.engines.orca.extopt.engine.ExtOptOrcaEngine`: set the
  ClassVars ``backend`` / ``wrapper_filename``, implement ``_server_cmd``, and
  add a :class:`~chemrefine.engines._backend_server.base.ComputeBackend` in
  ``engines/<name>/extopt_calc.py`` registered in
  ``engines/_backend_server/registry.py``. (See ``engines/pyscf/extopt_engine.py``.)
* **Local / orchestration-only** (no SLURM compute) → implement the
  :class:`CalculationEngine` Protocol directly (the five lifecycle methods). (See
  ``engines/mlip/train_engine.py`` / ``engines/_fake/engine.py``.)

**2. Metadata** — declare ``name`` and ``supports_nms`` (plus any base-required
ClassVars) as ``ClassVar[...]`` annotations, matching the other engines.

**3. YAML knobs** → a Pydantic model in ``engines/<name>/options.py`` with a
``from_raw`` classmethod (mirror ``engines/mlip/options.py`` /
``engines/pyscf/options.py``); read it in ``prepare`` / ``_template_vars``.

**4. Register** by importing the class in ``engines/<name>/__init__.py`` and
listing the package in ``engines/__init__.py`` (registration is a side effect of
that import). Legacy YAML spellings map to the canonical name in the config
normalizer, not here — see :func:`chemrefine.config._normalize_legacy`.

**5. Resources** — an external binary reads its path from
``ctx.executables.get("<name>")``; an importable backend ships as a
``pip install chemrefine[<name>]`` extra and is imported in-process (lazily).

**6. Tests** go in ``tests/test_engines_<name>*.py``.
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

    def _gpus(self, ctx: StepContext) -> int:
        """GPUs this job needs: 1 for a CUDA/GPU step, else 0.

        Reads the device knobs every GPU-capable engine already exposes —
        ``options.device == "cuda"`` (mlip) or a truthy ``options.gpu`` (pyscf).
        ORCA-only steps have neither (ORCA is CPU/MPI), so they request no GPU.
        Also drives :meth:`_slurm_header_name` (a GPU job → the cuda header).
        """
        options = ctx.step_cfg.options or {}
        if str(options.get("device", "")).lower() == "cuda" or options.get("gpu"):
            return 1
        return 0

    def _slurm_header_name(self, ctx: StepContext) -> str:
        """Pick the SLURM header for this step.

        An explicit per-step ``slurm_template`` wins; otherwise a GPU step
        (:meth:`_gpus` > 0) auto-selects ``cuda.slurm.header`` so the job lands on
        a GPU node, and everything else uses the global ``Config.slurm_template``.
        ORCA-only steps never report a GPU, so they keep the global header.
        """
        if ctx.step_cfg.slurm_template:
            return ctx.step_cfg.slurm_template
        if self._gpus(ctx) > 0:
            return slurm.header_name_for_device("cuda")
        return ctx.slurm_template

    def _run_block(self, ctx: StepContext, inp_path: Path, out_path: Path) -> str:
        """Return the engine-specific bash that runs inside ``$WORK_DIR``."""
        raise NotImplementedError

    def _extra_header_fields(self, ctx: StepContext) -> tuple[tuple[str, object], ...]:
        """Engine-specific ``(key, value)`` rows appended to the runlog header."""
        return ()

    def submit(self, inputs: StepInputs, ctx: StepContext) -> JobBatch:
        """Generate a SLURM script per structure, submit under the CPU+GPU budget, then block.

        Blocks until every job in the batch finishes (success or failure).
        """
        local = not slurm.sbatch_available()
        throttler = throttle.Throttler(
            max_cores=ctx.max_cores,
            max_gpus=slurm.resolve_gpu_budget(ctx.max_gpus),
            poll_interval=_LOCAL_POLL_SECONDS if local else _SLURM_POLL_SECONDS,
        )
        header_path = ctx.template_dir / self._slurm_header_name(ctx)
        if not header_path.is_file():
            raise FileNotFoundError(f"SLURM header template not found: {header_path}")

        pal = min(self._pal(ctx), ctx.max_cores)
        gpus = self._gpus(ctx)
        step_label = ctx.step_cfg.dir_name()
        jobs: dict[Path, str] = {}
        for inp, out, sid in inputs.files:
            throttler.wait_for_room(pal, is_finished=slurm.is_finished, gpus_needed=gpus)
            # Pin a free GPU per local job so concurrent CUDA jobs don't collide on
            # device 0; under SLURM the scheduler sets CUDA_VISIBLE_DEVICES itself.
            device = throttler.assign_device() if (local and gpus) else None
            env = {"CUDA_VISIBLE_DEVICES": str(device)} if device is not None else None
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
            job_id = slurm.submit(script_path, env=env)
            throttler.register(job_id, pal, gpus=gpus, device=device)
            jobs[inp] = job_id
            logger.info(
                "submitted %s as job %s (pal=%d, gpus=%d)", inp.name, job_id, pal, gpus
            )

        throttler.wait_all(is_finished=slurm.is_finished)
        return JobBatch(jobs=jobs)

    def wait(self, batch: JobBatch) -> None:
        """No-op: :meth:`submit` already blocked until every job finished."""
        return None

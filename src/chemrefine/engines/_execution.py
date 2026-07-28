"""Run a batch of per-structure jobs under the CPU + GPU budget.

Engine-independent submission: given an engine's per-structure inputs plus the few
primitives it exposes (``run_block`` / ``pal`` / ``gpus`` / ``output_globs`` /
``output_dirs`` / ``extra_header_fields`` — the :class:`~chemrefine.engines.api.JobExecutable`
surface), build one SLURM script per structure — or a single job array — and run them locally
or via ``sbatch`` under :mod:`chemrefine.throttle`. This is the engine subsystem's scheduler,
composing the flat generic infra (:mod:`chemrefine.slurm` + :mod:`chemrefine.throttle`); a
:class:`~chemrefine.engines._job.JobEngine` provides only the primitives and delegates its
``submit`` here, so submission is not duplicated per engine.
"""

from __future__ import annotations

import logging
from pathlib import Path

from chemrefine import slurm, throttle
from chemrefine.engines.api import JobExecutable
from chemrefine.errors import ConfigError
from chemrefine.state import JobBatch, StepContext, StepInputs

logger = logging.getLogger(__name__)

# Throttler poll cadence. SLURM ``squeue`` is expensive and jobs are minutes-long, so
# poll slowly; background local processes finish in well under a second, so poll fast.
_SLURM_POLL_SECONDS = 10.0
_LOCAL_POLL_SECONDS = 0.25


def _header_name(engine: JobExecutable, ctx: StepContext) -> str:
    """Pick the SLURM header for this step.

    An explicit per-step ``slurm_template`` wins; otherwise a GPU step (``engine.gpus`` > 0)
    auto-selects the cuda header so the job lands on a GPU node, and everything else uses the
    global ``Config.slurm_template``.
    """
    if ctx.step_cfg.slurm_template:
        return ctx.step_cfg.slurm_template
    if engine.gpus(ctx) > 0:
        return slurm.header_name_for_device("cuda")
    return ctx.slurm_template


def _header_path(engine: JobExecutable, ctx: StepContext) -> Path:
    """Resolve + validate the SLURM header template path for this step.

    Raises :class:`~chemrefine.errors.ConfigError` (not a bare ``FileNotFoundError``) so
    a missing header — the other likeliest first-run error, and one a GPU step can hit
    without ever naming ``cuda.slurm.header`` itself — exits with the documented code
    instead of a traceback.
    """
    header_path = ctx.template_dir / _header_name(engine, ctx)
    if not header_path.is_file():
        raise ConfigError(f"SLURM header template not found: {header_path}")
    return header_path


def run_batch(engine: JobExecutable, inputs: StepInputs, ctx: StepContext) -> JobBatch:
    """Submit one SLURM job per structure under the CPU+GPU budget, then block.

    Blocks until every job finishes (success or failure). With ``slurm_array`` set
    (and a real SLURM host) the whole batch goes out as job array(s) — see
    :func:`_run_array`.
    """
    local = slurm.dispatch_locally(ctx.dispatch)
    if ctx.slurm_array and not local:
        return _run_array(engine, inputs, ctx)
    throttler = throttle.Throttler(
        max_cores=ctx.max_cores,
        max_gpus=slurm.resolve_gpu_budget(ctx.max_gpus, dispatch=ctx.dispatch),
        poll_interval=_LOCAL_POLL_SECONDS if local else _SLURM_POLL_SECONDS,
    )
    header_path = _header_path(engine, ctx)
    pal = min(engine.pal(ctx), ctx.max_cores)
    gpus = engine.gpus(ctx)
    if local and gpus > 1:
        # `Throttler.assign_device` hands out a single device index per job and
        # `run_batch` exports it as one `CUDA_VISIBLE_DEVICES` value, so a job
        # asking for several local GPUs would be charged for all of them but
        # pinned to one. No bundled engine requests >1 today; fail loudly rather
        # than silently under-provisioning if one starts to.
        raise ConfigError(
            f"step {ctx.step_cfg.step} requests {gpus} GPUs, but local dispatch pins one "
            f"device per job; run this step under SLURM or set `options.device: cpu`"
        )
    if gpus > throttler.max_gpus:
        # Surface a config mistake (e.g. `max_gpus: 0` with a CUDA step) as a
        # ConfigError with its documented exit code, not the throttler's traceback.
        raise ConfigError(
            f"step {ctx.step_cfg.step} needs {gpus} GPU(s) but the budget is "
            f"{throttler.max_gpus}; raise `max_gpus` or set `options.device: cpu`"
        )
    jobs: dict[Path, str] = {}
    try:
        _submit_all(engine, inputs, ctx, throttler, header_path, pal, gpus, local, jobs)
        throttler.wait_all(finished=slurm.finished_jobs)
    finally:
        # Any job still active here means we are unwinding on an exception — a
        # ThrottleTimeoutError, a mid-batch JobSubmissionError, a KeyboardInterrupt.
        # Local jobs are real background processes owned by this interpreter, so
        # leaving them running would orphan compute that keeps competing for the
        # cores of whatever the user runs next. (SLURM jobs are the scheduler's;
        # cancelling them here would be presumptuous.)
        slurm.terminate_local_jobs(throttler.active_jobs)
    return JobBatch(jobs=jobs)


def _submit_all(
    engine: JobExecutable,
    inputs: StepInputs,
    ctx: StepContext,
    throttler: throttle.Throttler,
    header_path: Path,
    pal: int,
    gpus: int,
    local: bool,
    jobs: dict[Path, str],
) -> None:
    """Submit every prepared input under the budget, recording ids into ``jobs``."""
    step_label = ctx.step_cfg.dir_name()
    operation = ctx.step_cfg.operation or ""
    for inp, out, sid in inputs.files:
        throttler.wait_for_room(pal, finished=slurm.finished_jobs, gpus_needed=gpus)
        # Pin a free GPU per local job so concurrent CUDA jobs don't collide on device
        # 0; under SLURM the scheduler sets CUDA_VISIBLE_DEVICES itself.
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
            run_block=engine.run_block(ctx, inp, out),
            engine=ctx.step_cfg.engine,
            operation=operation,
            step=ctx.step_cfg.step,
            structure_id=sid,
            step_label=step_label,
            output_globs=engine.output_globs,
            output_dirs=engine.output_dirs(ctx),
            extra_header_fields=engine.extra_header_fields(ctx),
        )
        job_id = slurm.submit(script_path, env=env, dispatch=ctx.dispatch)
        throttler.register(job_id, pal, gpus=gpus, device=device)
        jobs[inp] = job_id
        logger.info("submitted %s as job %s (pal=%d, gpus=%d)", inp.name, job_id, pal, gpus)


def _run_array(engine: JobExecutable, inputs: StepInputs, ctx: StepContext) -> JobBatch:
    """Submit the whole batch as SLURM job array(s); block until they drain.

    One script + one manifest per chunk; the scheduler enforces the core budget
    natively via ``--array=...%{max_cores // pal}``, so no Python-side throttling runs.
    The run block is rendered **once** against sentinel paths whose *names* are the bash
    variables the script resolves per task — every engine's ``run_block`` uses only
    ``.name``, so it composes unchanged. GPU placement is the scheduler's (``--gres`` in
    the cuda header), as on the per-job path.
    """
    if not inputs.files:
        return JobBatch(jobs={})
    header_path = _header_path(engine, ctx)
    pal = min(engine.pal(ctx), ctx.max_cores)
    step_label = ctx.step_cfg.dir_name()
    # The shared array script + manifests live at the step dir; each task resolves its
    # own per-structure dir (dirname of its output) at runtime.
    output_dir = ctx.step_dir
    script_path = output_dir / f"{step_label}_array.slurm"
    slurm.build_array_script(
        step_label=step_label,
        pal=pal,
        template_path=header_path,
        script_path=script_path,
        output_dir=output_dir,
        scratch_dir=ctx.scratch_dir,
        run_block=engine.run_block(ctx, Path("$INP_NAME"), Path("$OUT_NAME")),
        engine=ctx.step_cfg.engine,
        operation=ctx.step_cfg.operation or "",
        step=ctx.step_cfg.step,
        output_globs=engine.output_globs,
        output_dirs=engine.output_dirs(ctx),
        extra_header_fields=engine.extra_header_fields(ctx),
    )

    manifests = slurm.write_array_manifests(inputs.files, output_dir, step_label=step_label)
    max_concurrent = max(1, ctx.max_cores // pal)
    jobs: dict[Path, str] = {}
    for manifest, chunk in manifests:
        parent_id = slurm.submit_array(
            script_path, n_tasks=len(chunk), max_concurrent=max_concurrent, manifest=manifest
        )
        for inp, _out, _sid in chunk:
            jobs[inp] = parent_id
        logger.info(
            "submitted %s as array %s (%d tasks, pal=%d, max %d concurrent)",
            step_label,
            parent_id,
            len(chunk),
            pal,
            max_concurrent,
        )

    slurm.wait_for_jobs(
        set(jobs.values()), poll_interval=_SLURM_POLL_SECONDS, finished=slurm.finished_jobs
    )
    return JobBatch(jobs=jobs)

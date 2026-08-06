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
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from chemrefine import slurm, throttle
from chemrefine.engines.api import NULL_SINK, CompletionSink, JobExecutable, sweep
from chemrefine.errors import ChemRefineError, ConfigError
from chemrefine.state import JobBatch, JobTriple, StepContext, StepInputs

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


@dataclass(frozen=True)
class _BatchPlan:
    """What :func:`run_batch` resolves once and every job of the batch then reuses.

    Only the values that cost something to derive or that the budget checks depend on. The
    step label and operation stay off it: they are one-liners off the ``ctx`` that
    :func:`_submit_one` already receives, and copying them here would be a second place for
    them to be read from.
    """

    header_path: Path
    pal: int
    gpus: int
    local: bool
    max_gpus: int

    @classmethod
    def of(cls, engine: JobExecutable, ctx: StepContext, *, local: bool) -> _BatchPlan:
        """Resolve and **validate** this batch's plan, whichever way it will be submitted.

        Built before the array/queue fork rather than inside the queue branch, so a step is
        judged by its configuration and not by the path it happens to take. The GPU checks
        below used to sit after that fork: the same ``max_gpus: 0`` that raised a
        :class:`~chemrefine.errors.ConfigError` per job submitted silently as an array, and
        ``pal`` and the header were re-resolved in the array branch by hand.
        """
        # Header first, so a missing template still reports itself before anything about
        # GPUs — the order these were resolved in before they moved here.
        header_path = _header_path(engine, ctx)
        pal = min(engine.pal(ctx), ctx.max_cores)
        gpus = engine.gpus(ctx)
        max_gpus = slurm.resolve_gpu_budget(ctx.max_gpus, local=local)
        if local and gpus > 1:
            # `Throttler.assign_device` hands out a single device index per job and
            # `_submit_one` exports it as one `CUDA_VISIBLE_DEVICES` value, so a job
            # asking for several local GPUs would be charged for all of them but
            # pinned to one. No bundled engine requests >1 today; fail loudly rather
            # than silently under-provisioning if one starts to.
            raise ConfigError(
                f"step {ctx.step_cfg.step} requests {gpus} GPUs, but local dispatch pins one "
                f"device per job; run this step under SLURM or set `options.device: cpu`"
            )
        if gpus > max_gpus:
            # Surface a config mistake (e.g. `max_gpus: 0` with a CUDA step) as a
            # ConfigError with its documented exit code, not the throttler's traceback.
            raise ConfigError(
                f"step {ctx.step_cfg.step} needs {gpus} GPU(s) but the budget is "
                f"{max_gpus}; raise `max_gpus` or set `options.device: cpu`"
            )
        return cls(
            header_path=header_path,
            pal=pal,
            gpus=gpus,
            local=local,
            max_gpus=max_gpus,
        )


def run_batch(
    engine: JobExecutable,
    inputs: StepInputs,
    ctx: StepContext,
    *,
    sink: CompletionSink = NULL_SINK,
) -> JobBatch:
    """Submit one SLURM job per structure under the CPU+GPU budget, then block.

    Blocks until every job finishes (success or failure). With ``slurm_array`` set
    (and a real SLURM host) the whole batch goes out as job array(s) — see
    :func:`_run_array`.

    ``sink`` hears each job the moment it finishes, and any follow-up work it asks for joins
    the same queue — so a retry lands in the slot its own failed job just freed instead of
    waiting for the whole batch to drain. It defaults to
    :data:`~chemrefine.engines.api.NULL_SINK`, which hears nothing and asks for nothing; that
    is the *only* difference between running a batch and streaming one, so there is one
    submission loop rather than two.

    The array path cannot report completions per task (it maps every input to the parent
    array id), so it drives the sink over the whole batch once it has drained and submits the
    follow-ups as another batch. Correct, but without the overlap.
    """
    local = slurm.dispatch_locally(ctx.dispatch)
    plan = _BatchPlan.of(engine, ctx, local=local)
    if ctx.slurm_array and not local:
        batch = _run_array(engine, plan, inputs, ctx)
        if follow := sweep(inputs, sink):
            more = run_batch(engine, StepInputs(follow), ctx, sink=sink)
            batch = JobBatch(jobs={**batch.jobs, **more.jobs})
        return batch
    throttler = throttle.Throttler(
        max_cores=ctx.max_cores,
        max_gpus=plan.max_gpus,
        poll_interval=_LOCAL_POLL_SECONDS if local else _SLURM_POLL_SECONDS,
    )
    try:
        jobs = _run_queue(engine, plan, ctx, throttler, inputs, sink)
    finally:
        # Any job still active here means the stack is unwinding on an exception — a
        # ThrottleTimeoutError, a mid-batch JobSubmissionError, a KeyboardInterrupt.
        # Local jobs are real background processes owned by this interpreter, so
        # leaving them running would orphan compute that keeps competing for the
        # cores of whatever the user runs next. (SLURM jobs are the scheduler's;
        # cancelling them here would be presumptuous.)
        slurm.terminate_local_jobs(throttler.active_jobs)
    return JobBatch(jobs=jobs)


def _submit_one(
    engine: JobExecutable,
    plan: _BatchPlan,
    ctx: StepContext,
    throttler: throttle.Throttler,
    job: JobTriple,
) -> str:
    """Build one job's script, submit it, charge it to the budget; return its job id.

    :func:`_run_queue` has already established that the budget has room, which is also
    :meth:`~chemrefine.throttle.Throttler.assign_device`'s documented precondition — so this
    does not ask a second time.
    """
    inp, out, sid = job
    # Pin a free GPU per local job so concurrent CUDA jobs don't collide on device
    # 0; under SLURM the scheduler sets CUDA_VISIBLE_DEVICES itself.
    device = throttler.assign_device() if (plan.local and plan.gpus) else None
    env = {"CUDA_VISIBLE_DEVICES": str(device)} if device is not None else None
    script_path = inp.with_suffix(".slurm")
    slurm.build_script(
        job_name=inp.stem,
        pal=plan.pal,
        template_path=plan.header_path,
        script_path=script_path,
        input_path=inp,
        output_dir=out.parent,
        scratch_dir=ctx.scratch_dir,
        run_block=engine.run_block(ctx, inp, out),
        engine=ctx.step_cfg.engine,
        operation=ctx.step_cfg.operation or "",
        step=ctx.step_cfg.step,
        structure_id=sid,
        step_label=ctx.step_cfg.dir_name(),
        output_globs=engine.output_globs,
        output_dirs=engine.output_dirs(ctx),
        extra_header_fields=engine.extra_header_fields(ctx),
    )
    job_id = slurm.submit(script_path, env=env, dispatch=ctx.dispatch)
    throttler.register(job_id, plan.pal, gpus=plan.gpus, device=device)
    logger.info("submitted %s as job %s (pal=%d, gpus=%d)", inp.name, job_id, plan.pal, plan.gpus)
    return job_id


def _run_queue(
    engine: JobExecutable,
    plan: _BatchPlan,
    ctx: StepContext,
    throttler: throttle.Throttler,
    inputs: StepInputs,
    sink: CompletionSink,
) -> dict[Path, str]:
    """Submit under the budget, handing each job to ``sink`` as it finishes.

    **The one submission loop.** Fill every free slot, wait for a completion, tell the sink,
    repeat — which is a plain batch when the sink asks for nothing and a streamed one when it
    does. The queue is a ``deque`` rather than a fixed sequence precisely so the sink can add
    to it: a structure that failed to converge asks for a re-run, and that re-run goes into
    the slot the failed job just freed instead of waiting for the rest of the batch.

    Follow-ups go to the **back** — original work is known to be needed, a follow-up is
    speculative, and the submission log then still reads in manifest order.

    Returns input path → job id. A structure that was re-run appears once, under its latest
    attempt: the mapping is an opaque handle (:class:`~chemrefine.state.JobBatch`) that
    nothing reads back.
    """
    pending: deque[JobTriple] = deque(inputs.files)
    jobs: dict[Path, str] = {}
    index: dict[str, JobTriple] = {}
    while pending or throttler.active_jobs:
        while pending and throttler.has_room(plan.pal, gpus_needed=plan.gpus):
            job = pending.popleft()
            jobs[job[0]] = job_id = _submit_one(engine, plan, ctx, throttler, job)
            index[job_id] = job
        if pending and not throttler.active_jobs:
            # Unreachable: `pal` is clamped to `max_cores` and the GPU budget is checked
            # before the loop, so an empty throttler always has room and the inner loop
            # above will have submitted something. An assertion about this function, not a
            # complaint about the config — those were made above, with their own exit code.
            # Raising rather than breaking because the alternative is silence: the jobs
            # would never be submitted and the failure would surface two layers away, as a
            # missing result for a structure nobody can see was skipped.
            raise ChemRefineError(
                f"step {ctx.step_cfg.step}: {len(pending)} job(s) left unsubmitted with "
                f"nothing running — {plan.pal} cores + {plan.gpus} gpu(s) against a "
                f"budget of {throttler.max_cores} + {throttler.max_gpus}"
            )
        if pending:
            # The only account of *why* nothing is being submitted. Once per wait rather than
            # once per poll: the local cadence is 0.25 s, which would make it noise.
            logger.debug(
                "waiting on budget: cores %d+%d/%d, gpus %d+%d/%d",
                throttler.cores_in_use,
                plan.pal,
                throttler.max_cores,
                throttler.gpus_in_use,
                plan.gpus,
                throttler.max_gpus,
            )
        for job_id in throttler.wait_for_completion(
            finished=slurm.finished_jobs, max_wait_seconds=ctx.job_timeout_seconds
        ):
            pending.extend(sink.on_complete(index[job_id]))
    return jobs


def _run_array(
    engine: JobExecutable, plan: _BatchPlan, inputs: StepInputs, ctx: StepContext
) -> JobBatch:
    """Submit the whole batch as SLURM job array(s); block until they drain.

    One script + one manifest per chunk; the scheduler enforces the core budget natively
    via each array's ``%limit``, so no Python-side throttling runs. That limit is
    ``max_cores // pal`` **divided again by the number of chunks**, because a step past the
    per-array task cap is several arrays and all of them are queued at once — one share
    each is what makes them add up to the budget rather than to a multiple of it.

    The cost of the division, since it is a real one: a chunk's share is not handed back
    when a sibling drains early, so the tail of an N-chunk step runs at 1/N of the budget.
    Chaining the chunks with ``--dependency`` would keep full utilisation inside each, and
    was rejected for it: the slowest task of one chunk would hold up every task of the
    next, a cancelled parent leaves its dependents queued forever, and it would have to be
    ``afterany`` or one failed task would cancel the rest of a batch this pipeline ledgers
    per structure. ``max(1, …)`` is a floor rather than a guarantee — more chunks than
    ``max_cores // pal`` still exceeds the budget, because ``%0`` is not submittable, which
    needs both a >1000-structure step and a budget under one job per chunk.

    The run block is rendered **once** against sentinel paths whose *names* are the bash
    variables the script resolves per task — every engine's ``run_block`` uses only
    ``.name``, so it composes unchanged. GPU placement is the scheduler's (``--gres`` in
    the cuda header), as on the per-job path.

    Takes the same :class:`_BatchPlan` the queue path does — the header and ``pal`` were
    re-derived here once, which is one edit away from an array running under a different
    core count than the budget was checked against.
    """
    if not inputs.files:
        return JobBatch(jobs={})
    step_label = ctx.step_cfg.dir_name()
    # The shared array script + manifests live at the step dir; each task resolves its
    # own per-structure dir (dirname of its output) at runtime.
    output_dir = ctx.step_dir
    script_path = output_dir / f"{step_label}_array.slurm"
    slurm.build_array_script(
        step_label=step_label,
        pal=plan.pal,
        template_path=plan.header_path,
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
    # Divided by the chunk count as well as by `pal`, because every chunk is submitted now
    # rather than after the one before it drained, and each carries its own `%limit`. By
    # `pal` alone each of a 2500-structure step's three arrays got the *whole* budget:
    # 3 x 64 tasks x pal 8 = 1536 cores against a `max_cores: 512`.
    max_concurrent = max(1, ctx.max_cores // (plan.pal * len(manifests)))
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
            plan.pal,
            max_concurrent,
        )

    slurm.wait_for_jobs(
        set(jobs.values()),
        poll_interval=_SLURM_POLL_SECONDS,
        poll=slurm.poll_jobs,
        max_wait_seconds=ctx.job_timeout_seconds,
    )
    return JobBatch(jobs=jobs)

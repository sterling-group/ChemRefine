"""MLIP training engine (registered as ``"mlip-train"``).

A step with ``engine: mlip-train`` fine-tunes or trains an MLIP on the structures the previous
step produced, which must carry energies **and** forces. It is an *artifact* step
(:class:`~chemrefine.engines.api.ArtifactEngine`): one job over the whole ensemble, a model as
the product, and the same structures handed on to the next step unchanged.

Which library trains is ``task_name``, resolved through
:mod:`chemrefine.engines.mlip.registry` — the same registry, and the same word, that selects
the calculator, so a pipeline that fine-tunes ``task_name: mace_off`` and then runs the result
names one library and resolves one environment. Everything library-specific (the dataset
format, the argv, where the model lands) belongs to that library's
:class:`~chemrefine.engines.mlip.train.base.Trainer` and nothing of it is spelled here.

The job runs through the ordinary scheduler
(:func:`chemrefine.engines._execution.run_batch`), so training is throttled against the same
core and GPU budgets as everything else, gets the device-aware SLURM header, the runlog, the
scratch handling and the local-dispatch fallback for free — and honours
``job_timeout_seconds``. It reaches that by satisfying
:class:`~chemrefine.engines.api.JobExecutable` directly rather than by subclassing
:class:`~chemrefine.engines._job.JobEngine`, whose every primitive is per-structure: a
training job has no structure, and routing it through that machinery would mint a synthetic
one that the failure ledger, the attempt directories and ``rerun-errors`` would each then
mishandle in their own way.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from chemrefine import ids
from chemrefine.engines import _provision
from chemrefine.engines.api import BackendRequirement, RunBlock, register
from chemrefine.engines.mlip.backend import MlipBackend
from chemrefine.engines.mlip.options import MlipTrainOptions
from chemrefine.engines.mlip.registry import (
    backend_spec,
    registered_trainers,
    requirement_from_options,
    trainer_for,
)
from chemrefine.engines.mlip.train.base import (
    Trainer,
    TrainingPlan,
    digest_of,
    placeholders_for,
    render_config,
    split_structures,
)
from chemrefine.errors import ConfigError
from chemrefine.state import JobBatch, StepContext, StepInputs, StepResults

logger = logging.getLogger(__name__)

SIDECAR_NAME = "trained_model.json"
"""Record of what a training step produced, written beside the model.

Names the backend, the selection that produced it, the split sizes and the model's own
digest. It exists so a later step — or a person, months on — can tell which run a checkpoint
came from without re-deriving it from the YAML, and so the model can be cited."""

TRAINED_MODEL_FORMAT = 1


@register("mlip-train")
class MlipTrainEngine(MlipBackend):
    """Train an MLIP on the previous step's structures; pass the structures through."""

    name: ClassVar[str] = "mlip-train"
    label: ClassVar[str] = "MLIP training"
    template_suffix: ClassVar[str] = "yaml"
    """The trainer's own config, rendered rather than patched — and the extension says so.

    Every shipped trainer's native config format is YAML (MACE's ``run_train`` config,
    FAIRChem's hydra config), and the template *is* that file with placeholders in it."""

    options_cls: ClassVar[type[MlipTrainOptions]] = MlipTrainOptions

    # -- the backend this step needs ---------------------------------------

    def backend_requirement(self, options: Mapping[str, Any] | None) -> BackendRequirement:
        """The environment this step needs — and it must be one that can *train*.

        Narrows :meth:`~chemrefine.engines.mlip.backend.MlipBackend.backend_requirement` with
        ``require_trainer``: every MLIP library can be run, only some can be trained, and a
        step that named an untrainable one should hear about it from ``preflight_backends``
        before the run submits anything rather than after its upstream steps have computed a
        dataset.

        The other half of provisioning is the launch: ``_plan`` resolves the interpreter
        through :func:`~chemrefine.engines._provision.launcher_for` and the trainer builds its
        command from it. Both halves matter and only together: a preflight without the
        launcher passes a step that then cannot start, and a launcher without the preflight
        finds out at training time — a bare ``mace_run_train`` is on nobody's ``PATH`` once
        MACE lives in an environment of its own.
        """
        return requirement_from_options(options, options_cls=self.options_cls, require_trainer=True)

    # -- resolving the step -------------------------------------------------

    def _opts(self, ctx: StepContext) -> MlipTrainOptions:
        """This step's validated options — strictly, so a typoed knob fails the step."""
        return self.options_cls.from_raw(ctx.step_cfg.options)

    def _trainer(self, ctx: StepContext) -> Trainer:
        """The trainer this step's ``task_name`` dispatches to.

        The same one word :meth:`backend_requirement` resolves the environment from, read off
        the same validated options model — which is what makes it impossible for the env a
        step is provisioned into and the trainer it launches to disagree. Two readers with
        their own rules would fail as a step provisioned for one library asking another to
        train, and only on whichever code path the tests were not watching.
        """
        return trainer_for(self._opts(ctx).task_name)()

    def _run_dir(self, ctx: StepContext) -> Path:
        """Where the run's files live: the training job's own directory under the step.

        The same shape a structure gets, for the same reason — config, script, runlog, logs
        and product in one directory that nothing else writes to.
        """
        return ctx.step_dir / ids.TRAINING_ID

    def artifact(self, ctx: StepContext) -> Path:
        """The model this step produces — its success test, and what the next step loads.

        May sit *inside* the run directory rather than in it: FAIRChem's is at
        ``checkpoints/final/inference_ckpt.pt`` under a directory it names itself. That is why
        it is not also the job's output path — see :meth:`_job_output`.
        """
        return self._trainer(ctx).artifact(self._run_dir(ctx), ids.TRAINING_ID)

    def _job_output(self, ctx: StepContext) -> Path:
        """The job's output path for the scheduler — always directly in the run directory.

        The scheduler reads only its **parent**, as the directory the job writes to: the
        runlog, the ``.err`` and, when no ``scratch_dir`` is configured, ``$WORK_DIR`` itself.
        Handing it :meth:`artifact` works for a trainer whose model lands beside its config
        and fails quietly for one whose model is nested — FAIRChem's parent is
        ``train/train/checkpoints/final``, so a run's own log and scratch were created three
        levels inside the directory it was about to write its final checkpoint into.

        Naming the artifact's basename at the run directory keeps that parent right while
        still saying what the job is for. For MACE the two paths are the same file; for a
        trainer that nests, this one is the anchor and :meth:`artifact` is the product.
        """
        return self._run_dir(ctx) / self.artifact(ctx).name

    def _plan(self, ctx: StepContext) -> TrainingPlan:
        """Where and how this step's job runs — derivable from the config alone.

        Deliberately independent of ``ctx.prev_state``: :meth:`run_block` is asked what bash a
        step generates without any ensemble in hand, and a plan that needed one could not
        answer. The data half is :func:`split_structures`, computed in :meth:`prepare`.
        """
        opts = self._opts(ctx)
        # The granted cores, not the ask: the layout below clamps to max_cores, and the
        # plan's `cores` feeds the thread exports and `$CORES` — a raw pal() here let a
        # `cores:` above the budget thread past what the scheduler grants and charges.
        ntasks, cpus_per_task = self.slurm_layout(ctx)
        return TrainingPlan(
            run_dir=self._run_dir(ctx),
            run_name=ids.TRAINING_ID,
            device=opts.device,
            gpus=self.gpus(ctx),
            cores=ntasks * cpus_per_task,
            seed=opts.seed,
            charge=ctx.charge,
            multiplicity=ctx.multiplicity,
            start_from=opts.model_path or opts.model_name or None,
            launcher=Path(_provision.launcher_for(self, ctx.step_cfg.options)),
        )

    # -- the lifecycle ------------------------------------------------------

    def prepare(self, ctx: StepContext) -> StepInputs:
        """Write the dataset and render the trainer config; return the one job.

        One :class:`~chemrefine.state.JobTriple`, whose "input" is the rendered config and
        whose "output" is the model — so the scheduler runs it in the run directory, the
        manifest records what this step was asked to do, and a resume can prove the product on
        disk belongs to this configuration.
        """
        opts = self._opts(ctx)
        if "task_name" not in opts.model_fields_set:
            raise ConfigError(
                f"step {ctx.step_cfg.step} (mlip-train) must name a `task_name`: it selects "
                f"which library trains, and there is no default worth guessing. Its "
                f"inference default names a foundation model to *run*, which is not the "
                f"same choice. Trainable: {sorted(registered_trainers())}."
            )
        if "device" not in opts.model_fields_set:
            raise ConfigError(
                f"step {ctx.step_cfg.step} (mlip-train) must name a device: training on CPU "
                f"is impractical rather than merely slow, so there is no default. Set "
                f"`options: {{device: cuda}}` (or `cpu` if you mean it)."
            )
        if ctx.step_cfg.on_failure != "stop":
            raise ConfigError(
                f"step {ctx.step_cfg.step} (mlip-train) sets `on_failure: "
                f"{ctx.step_cfg.on_failure}`, which has nothing to act on — a training step "
                f"has no per-structure failures to skip or backfill. Remove it."
            )
        trainer = self._trainer(ctx)
        plan = self._plan(ctx)
        split = split_structures(
            ctx.prev_state.structures,
            valid_fraction=opts.valid_fraction,
            test_fraction=opts.test_fraction,
            seed=opts.seed,
        )
        plan.run_dir.mkdir(parents=True, exist_ok=True)
        data = trainer.write_dataset(plan, split)
        config = render_config(
            ctx.template,
            placeholders_for(trainer, plan, data),
            required=trainer.required_placeholders,
            dest=ids.structure_artifact_path(
                ctx.step_dir, ctx.step_cfg.step, ids.TRAINING_ID, self.template_suffix
            ),
        )
        logger.info(
            "step %d: training %s on %s",
            ctx.step_cfg.step,
            self._opts(ctx).task_name,
            split.counts(),
        )
        return StepInputs(files=((config, self._job_output(ctx), ids.TRAINING_ID),))

    def submit(self, inputs: StepInputs, ctx: StepContext) -> JobBatch:
        """Run the training job through the ordinary scheduler and block until it finishes."""
        from chemrefine.engines import _execution

        return _execution.run_batch(self, inputs, ctx)

    def parse(self, inputs: StepInputs, ctx: StepContext) -> StepResults:
        """Record what was produced, then hand the prior ensemble on unchanged.

        The structures are not this step's product and are not recomputed by it — the model
        is. Passing them through is what lets a training step sit in the middle of a pipeline
        rather than at the end of one.
        """
        self._write_sidecar(ctx)
        return StepResults(structures=ctx.prev_state.structures)

    def _write_sidecar(self, ctx: StepContext) -> None:
        """Write :data:`SIDECAR_NAME` beside the model.

        Written with the standard library rather than through :mod:`chemrefine.cache`'s
        ``write_json``: an engine records what it produced, and caching is not its concern —
        a rule the subsystem is held to by test.
        """
        opts = self._opts(ctx)
        model = self.artifact(ctx)
        record = {
            "trained_model_format": TRAINED_MODEL_FORMAT,
            "task_name": opts.task_name,
            "backend": backend_spec(opts.task_name).extra,
            "model_path": str(model),
            "model_digest": digest_of(model),
            "started_from": opts.model_path or opts.model_name or None,
            "seed": opts.seed,
            "charge": ctx.charge,
            "multiplicity": ctx.multiplicity,
            "n_structures": len(ctx.prev_state.structures),
        }
        (self._run_dir(ctx) / SIDECAR_NAME).write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        logger.info("step %d: trained model at %s", ctx.step_cfg.step, model)

    # -- what the scheduler asks of a job ----------------------------------

    output_globs: ClassVar[tuple[str, ...]] = (
        "*.csv",
        "*.log",
        "*.model",
        "*.pt",
        "*.pth.tar",
        "*.yaml",
        "checkpoint_*.pth",
        "log.sevenn",
    )
    """Loose files to copy back from scratch — the union of every trainer's own.

    A superset rather than this step's trainer, because
    :class:`~chemrefine.engines.api.JobExecutable` declares this a ``ClassVar`` and the
    scheduler reads it with no :class:`~chemrefine.state.StepContext` to resolve one from.
    (``output_dirs`` below *is* asked with a context, which is why that half can be exact.)

    Spelled out rather than computed because the value has to exist when this class body runs
    and the backends are auto-discovered after it — but held to
    :func:`~chemrefine.engines.mlip.registry.trainer_output_globs` **by test**, so a library
    added later cannot leave it stale. Left to hand maintenance the list drifts silently, and
    a missing glob is a model lost to the scratch cleanup for anyone whose template writes
    ``run_dir`` outside the run directory.

    A superset is the safe direction — copying back a pattern nothing wrote costs nothing.
    """

    def output_dirs(self, ctx: StepContext) -> tuple[str, ...]:
        """Whole directories to copy back — the selected trainer's own.

        Exact rather than a union, because the scheduler asks this one with a context.
        """
        return self._trainer(ctx).output_dirs

    def run_block(self, ctx: StepContext, inp_path: Path, out_path: Path) -> RunBlock:
        """The bash that trains, from the trainer, with the thread limits around it.

        No ``cleanup`` and no ``trap``: the job script owns the single exit handler that
        copies results back, and an engine emitting its own would silently replace it.
        """
        plan = self._plan(ctx)
        return RunBlock(
            body=(
                f"export OMP_NUM_THREADS={plan.cores}\n"
                f"export MKL_NUM_THREADS={plan.cores}\n"
                f"export MKL_THREADING_LAYER=GNU\n"
                f"{self._trainer(ctx).command(plan, inp_path)}"
            )
        )

    def slurm_layout(self, ctx: StepContext) -> tuple[int, int]:
        """The ranks spelling, matching what this job has always been granted.

        A training run is one threaded process, so ``(1, cores)`` would be the more honest
        SLURM spelling — but flipping it changes every user's allocation shape and belongs
        to the same follow-up decision as the threaded script engines. Declared explicitly
        because this class satisfies :class:`~chemrefine.engines.api.JobExecutable` directly
        rather than through :class:`~chemrefine.engines._job.JobEngine`.
        """
        return (min(self.pal(ctx), ctx.max_cores), 1)

    def memory_mb(self, ctx: StepContext) -> int | None:
        """No declared memory — a trainer config carries no ``mem_total``-like knob.

        Declared explicitly because this class satisfies
        :class:`~chemrefine.engines.api.JobExecutable` directly; the header's memory
        policy stands untouched, as it always has for training jobs.
        """
        return None

    def pal(self, ctx: StepContext) -> int:
        """Cores for the training job — the step's whole budget unless it names fewer.

        A step that runs one job per structure shares ``max_cores`` between them, so
        ``cores: 1`` is the right default there. This step's single job *is* the step, so an
        unset ``cores`` means all of it. "Unset" is read from the options the YAML actually
        set rather than from the inherited default, which is indistinguishable from someone
        deliberately asking for one core.
        """
        opts = self._opts(ctx)
        return opts.cores if "cores" in opts.model_fields_set else ctx.max_cores

    def gpus(self, ctx: StepContext) -> int:
        """GPUs this job asks for; ``0`` unless the step explicitly said ``cuda``.

        Reads ``device`` through the same model that renders it into the config, so the
        scheduler's GPU demand and the trainer's own device cannot disagree. A step that
        names no device demands none — :meth:`prepare` then refuses it outright, but the
        scheduler is never asked to place a GPU job for a step that will not start.
        """
        opts = self._opts(ctx)
        return opts.gpus if opts.device == "cuda" else 0

    def extra_header_fields(self, ctx: StepContext) -> tuple[tuple[str, object], ...]:
        """Runlog rows naming what the job asked for, next to what SLURM gave it.

        ``gpus`` is here because chemrefine does not write ``--gres`` — the allocation comes
        from the user's own header — so the runlog is where the two can be compared.
        """
        opts = self._opts(ctx)
        return (
            ("trainer", opts.task_name),
            ("gpus", self.gpus(ctx)),
            ("started_from", opts.model_path or opts.model_name or "scratch"),
        )

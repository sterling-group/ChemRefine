"""MLIP training engine (registered as ``"mlip-train"``).

A pipeline step with ``engine: mlip-train`` trains an MLIP on the structures the
previous step produced (which must carry energies + forces), submitting one MACE
training job. The step is a pass-through for the *structures*: downstream steps
keep refining the same ensemble, while the trained model is the side-effect
artifact written under the step dir. The heavy lifting lives in
:mod:`chemrefine.engines.mlip.trainer`; this engine just adapts it to the
:class:`~chemrefine.engines.api.CalculationEngine` lifecycle.
"""

from __future__ import annotations

import logging
from typing import ClassVar

from chemrefine import cache
from chemrefine.engines.api import register
from chemrefine.engines.mlip import trainer
from chemrefine.errors import ConfigError
from chemrefine.state import JobBatch, StepContext, StepInputs, StepResults

logger = logging.getLogger(__name__)


@register("mlip-train")
class MlipTrainEngine:
    """Train an MLIP on the previous step's structures; pass the structures through."""

    name: ClassVar[str] = "mlip-train"

    def prepare(self, ctx: StepContext) -> StepInputs:
        """Training operates on the whole prior ensemble, not per-structure inputs."""
        return StepInputs(files=())

    def submit(self, inputs: StepInputs, ctx: StepContext) -> JobBatch:
        """Write the train/test sets + config + SLURM script and run the MACE job."""
        logger.info(
            "step %d: MLIP training on %d structures",
            ctx.step_cfg.step,
            len(ctx.prev_state.structures),
        )
        trainer.run_training(StepResults(structures=ctx.prev_state.structures), ctx)
        return JobBatch(jobs={})

    def parse(self, inputs: StepInputs, ctx: StepContext) -> StepResults:
        """Pass the prior structures through unchanged (the model is the artifact)."""
        return StepResults(structures=ctx.prev_state.structures)

    def input_digest(self, ctx: StepContext) -> str:
        """SHA-1 (16 hex) of the MACE config template; ``""`` if it's missing.

        The step passes its *structures* through untouched, but it is not
        template-independent: :func:`~chemrefine.engines.mlip.trainer.write_training_config`
        renders a per-step template that **is** the MACE config — epochs, learning rate,
        model width. Returning ``""`` left those out of the cache key, so retuning the
        hyperparameters and running ``resume`` was a cache hit: the step reported "reusing
        N structures", never retrained, and left the previous model on disk for whatever
        loads it downstream.

        A missing template is not a digest failure — the step raises from ``submit`` with
        the actionable message — so it hashes to ``""``, as it does for every job engine.
        """
        try:
            template = trainer.resolve_training_template(ctx)
        except ConfigError:
            return ""
        return cache.template_digest(template)

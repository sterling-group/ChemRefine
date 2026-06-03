"""MLFF training engine (registered as ``"mlff-train"``).

A pipeline step with ``engine: mlff-train`` trains an MLFF on the structures the
previous step produced (which must carry energies + forces), submitting one MACE
training job. The step is a pass-through for the *structures*: downstream steps
keep refining the same ensemble, while the trained model is the side-effect
artifact written under the step dir. The heavy lifting lives in
:mod:`chemrefine.engines.mlff.trainer`; this engine just adapts it to the
:class:`~chemrefine.engines.base.CalculationEngine` lifecycle.
"""

from __future__ import annotations

import logging
from typing import ClassVar

from chemrefine.engines.base import register
from chemrefine.engines.mlff import trainer
from chemrefine.state import JobBatch, StepContext, StepInputs, StepResults

logger = logging.getLogger(__name__)


@register("mlff-train")
class MlffTrainEngine:
    """Train an MLFF on the previous step's structures; pass the structures through."""

    name: ClassVar[str] = "mlff-train"
    supports_nms: ClassVar[bool] = False

    def prepare(self, ctx: StepContext) -> StepInputs:
        """Training operates on the whole prior ensemble, not per-structure inputs."""
        return StepInputs(files=())

    def submit(self, inputs: StepInputs, ctx: StepContext) -> JobBatch:
        """Write the train/test sets + config + SLURM script and run the MACE job."""
        logger.info("step %d: MLFF training on %d structures",
                    ctx.step_cfg.step, len(ctx.prev_state.structures))
        trainer.run_training(
            StepResults(structures=ctx.prev_state.structures), ctx
        )
        return JobBatch(jobs={})

    def wait(self, batch: JobBatch) -> None:
        """No-op: :meth:`submit` blocks until the training job finishes."""
        return None

    def parse(self, inputs: StepInputs, ctx: StepContext) -> StepResults:
        """Pass the prior structures through unchanged (the model is the artifact)."""
        return StepResults(structures=ctx.prev_state.structures)

    def normal_mode_sample(self, results: StepResults, ctx: StepContext) -> StepResults:
        """Not supported — the orchestrator gates this on ``supports_nms``."""
        raise NotImplementedError("mlff-train does not support normal-mode sampling")

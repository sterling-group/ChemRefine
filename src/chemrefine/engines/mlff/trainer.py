"""MLFF training pipeline.

TODO: port the v3 ``MLFFTrainer`` from
:file:`src/chemrefine/mlff.py` on ``main``. The pieces to bring over:

* ``prepare_inputs`` — splits the collected ``(coords, energies,
  forces)`` arrays into train/test ``extxyz`` files (90/10 by default),
  converting Hartree → eV and Hartree/Bohr → eV/Å on the way.
* ``write_training_config`` — fills in a template ``mace_train.yaml``
  with the per-step dataset paths.
* ``write_slurm_script`` — generates the training-job SLURM script
  (GPU-aware, separate from the optimisation header).
* ``submit_training`` — submits and waits.

Verifying these end-to-end needs a real MACE training run, which is
heavy and depends on a working CUDA stack. The placeholder
:func:`run_training` below raises :class:`NotImplementedError` so any
``operation: mlff_train`` step fails loudly until the port lands.
"""

from __future__ import annotations

from chemrefine.state import StepContext, StepResults


def run_training(results: StepResults, ctx: StepContext) -> StepResults:
    """Train an MLFF model on ``results`` and return it unchanged.

    Placeholder — see module docstring for the port plan.
    """
    _ = results, ctx
    raise NotImplementedError(
        "MLFF training pipeline not yet ported — see TODO in engines/mlff/trainer.py"
    )

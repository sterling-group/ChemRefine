"""SevenNet: the calculator chemrefine runs, and the trainer that fine-tunes it.

One module for one library, so the environment is declared once (:data:`SEVENN`) and both
capabilities hang off it — the shape :mod:`chemrefine.engines.mlip.registry` sets and the
MACE and FAIRChem modules already follow.

The trainer drives SevenNet's own CLI (``sevenn train input.yaml``, unified in v0.11.1 —
the ``mlip-sevenn`` extra's floor) against a rendered copy of the user's ``input.yaml``.
Datasets are plain extxyz: SevenNet reads any ASE-readable path listed in
``data.load_trainset_path`` and takes energy and forces off the reconstructed calculator
(``get_potential_energy()`` / ``get_forces()``), which is exactly what ase's extxyz
round-trip provides. Checkpoints land in the working directory —
``checkpoint_{epoch}.pth`` as it goes, ``checkpoint_best.pth`` whenever the tracked
validation metric improves — so the job runs them up in ``$WORK_DIR`` and the declared
``output_globs`` carry them home.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from chemrefine.engines.mlip.registry import CalculatorSpec, MlipLibrary
from chemrefine.engines.mlip.train.base import (
    TrainerBase,
    TrainingPlan,
    write_labelled_extxyz,
)
from chemrefine.state import Structure

SEVENN = MlipLibrary(extra="mlip-sevenn", package="sevenn", import_name="sevenn")
"""The one declaration of what provides this library."""


@SEVENN.calculator("sevenn")
def _build_sevenn(spec: CalculatorSpec) -> Any:
    """SevenNet potential (``task_name: sevenn``); the weights come from name or checked path.

    ``SevenNetCalculator``'s ``model`` argument is typed ``str | Path`` — "Name of
    pretrained models … or path to the checkpoint" — and its resolution checks the
    filesystem before trying release names, so the vetted ``spec.weights`` rides it
    directly.
    """
    from sevenn.calculator import SevenNetCalculator

    if spec.weights is None and not spec.model_name:
        # Neither given: let SevenNet's own default release load — the library owns its
        # default, exactly as FAIRChem's builder owns `uma-s-1p2`.
        return SevenNetCalculator(device=spec.device)
    return SevenNetCalculator(model=spec.weights or spec.model_name, device=spec.device)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@SEVENN.trainer("sevenn")
class SevennTrainer(TrainerBase):
    """Fine-tune a SevenNet model on a step's labelled structures.

    The declarations carry the library facts: a validation set is required because
    ``checkpoint_best.pth`` — the artifact this step adopts — is written when the tracked
    *validation* metric improves, so a run without validation trains to completion and
    leaves no best checkpoint to adopt; and SevenNet has no charge or spin channel, so
    the base warns on a charged or open-shell ensemble rather than silently fitting it
    as neutral-singlet data.
    """

    label = "SevenNet"
    needs_validation = True
    validation_reason = (
        "checkpoint_best.pth, the model this step adopts, is written when the tracked "
        "validation metric improves"
    )

    required_placeholders: ClassVar[frozenset[str]] = frozenset({"TRAIN_SET", "DEVICE"})
    """The dataset, plus ``$DEVICE`` — a plan fact SevenNet reads from this config's own
    ``device:`` key, **falling back to cuda-whenever-torch-sees-one** when the key is
    absent (``sevenn.parse_input``). Unset, a ``device: cpu`` step on a CUDA node would
    train on hardware the scheduler never booked — the silent wrong-hardware class the
    driver-run trainers close over their argv; for a CLI trainer the template gate is
    that closure. No run-dir or run-name knob, unlike MACE and FAIRChem:
    ``checkpoint_best.pth`` is a fixed name wherever the run happened. ``$VALID_SET`` and
    ``$FOUNDATION_MODEL`` (→ ``train.continue.checkpoint``) are supplied too; a template
    is free to use them. No ``$SEED`` requirement — SevenNet's config declares a
    ``random_seed`` key its trainer does not read, so there is no seed fact to lose."""

    output_globs: ClassVar[tuple[str, ...]] = ("checkpoint_*.pth", "log.sevenn", "*.csv")
    """The main path, not a safety net: SevenNet writes checkpoints and its log into the
    *working directory* — ``$WORK_DIR`` under the scheduler — so these are what carry the
    product home. ``checkpoint_best.pth`` matches ``checkpoint_*.pth``; the log and the
    learning-curve CSV ride along (a glob that matches nothing costs nothing)."""

    def write_split(self, plan: TrainingPlan, name: str, structures: tuple[Structure, ...]) -> Path:
        """The shared calculator-labelled extxyz — exactly what SevenNet's loader reads."""
        return write_labelled_extxyz(plan.run_dir / f"{name}.xyz", structures)

    def command(self, plan: TrainingPlan, config: Path) -> str:
        """``sevenn train`` against the rendered config, from the backend env's own bin.

        The console script rather than ``-m``, like FAIRChem's — and by basename against
        the rendered config, for the array-sentinel reason MACE's command spells out.
        ``-s`` streams the log to stdout, where the runlog captures it. Multi-GPU is
        SevenNet's own documented DDP shape: ``torchrun … --no_python sevenn`` with
        ``-d`` in place of ``-s`` (``batch_size`` in the config is then per GPU).
        """
        sevenn_bin = self.console_script(plan, "sevenn")
        if plan.gpus > 1:
            return self.torchrun(plan, "--no_python", sevenn_bin, "train", config.name, "-d")
        return f"{sevenn_bin} train {config.name} -s"

    def artifact(self, run_dir: Path, run_name: str) -> Path:
        """``checkpoint_best.pth`` — the fixed name SevenNet gives its best checkpoint.

        Fixed by the library (``run_name`` plays no part), which is what makes the path
        predictable enough for a later step to name in its own YAML before the run.
        """
        return run_dir / "checkpoint_best.pth"

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

import logging
import shlex
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write as ase_write

from chemrefine.engines.mlip.registry import MlipLibrary
from chemrefine.engines.mlip.training import DatasetFiles, DatasetSplit, TrainingPlan
from chemrefine.errors import ConfigError
from chemrefine.quantities import HARTREE_TO_EV
from chemrefine.state import Structure

logger = logging.getLogger(__name__)

SEVENN = MlipLibrary(extra="mlip-sevenn", package="sevenn", import_name="sevenn")
"""The one declaration of what provides this library."""


@SEVENN.calculator("sevenn")
def _build_sevenn(
    *,
    model_name: str = "",
    device: str = "cuda",
    model_path: str | Path | None = None,
    **_: Any,
) -> Any:
    """SevenNet potential (``task_name: sevenn``); the weights come from name or path.

    ``model_path`` is honoured with SevenNet's own loader, like every builder's
    (:mod:`chemrefine.engines.mlip.registry`'s "three knobs" rule): ``SevenNetCalculator``'s
    ``model`` argument is typed ``str | Path`` — "Name of pretrained models … or path to the
    checkpoint" — and its resolution checks the filesystem before trying release names. The
    existence check runs before the library is imported, so a mistyped checkpoint is reported
    as the configuration mistake it is rather than as a loader traceback that names neither
    the step nor the option.
    """
    weights: str | Path = model_name
    if model_path is not None:
        weights = Path(model_path)
        if not weights.is_file():
            raise FileNotFoundError(f"SevenNet checkpoint not found: {weights}")

    from sevenn.calculator import SevenNetCalculator

    return SevenNetCalculator(model=weights, device=device)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _to_atoms(struct: Structure) -> Atoms:
    """One labelled :class:`~ase.Atoms` in the form SevenNet reads.

    Labels ride on a :class:`~ase.calculators.singlepoint.SinglePointCalculator`: ase's
    extxyz writer emits them as the standard ``energy=`` / ``free_energy=`` comment fields
    plus a ``forces`` column, its reader reconstructs the calculator, and SevenNet's
    loader takes both off it — ``get_potential_energy(force_consistent=True)`` first,
    which is why ``free_energy`` is set alongside ``energy``. Energies are converted
    Hartree → eV; forces are already eV/Å. The copy keeps the pipeline's own structure
    untouched, exactly as the MACE writer's does.
    """
    atoms: Atoms = struct.atoms.copy()
    assert struct.energy_hartree is not None  # noqa: S101 - split_structures rejected these
    assert struct.forces_ev_per_a is not None  # noqa: S101
    energy_ev = struct.energy_hartree * HARTREE_TO_EV
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=energy_ev,
        free_energy=energy_ev,
        forces=np.asarray(struct.forces_ev_per_a, dtype=float),
    )
    return atoms


@SEVENN.trainer("sevenn")
class SevennTrainer:
    """Fine-tune a SevenNet model on a step's labelled structures."""

    required_placeholders: ClassVar[frozenset[str]] = frozenset({"TRAIN_SET"})
    """Only the dataset: SevenNet writes into the working directory by its own rule, so
    unlike MACE and FAIRChem there is no run-dir or run-name knob the template must pin
    for :meth:`artifact` to hold — ``checkpoint_best.pth`` is a fixed name wherever the
    run happened. ``$VALID_SET`` and ``$FOUNDATION_MODEL`` (→ ``train.continue.checkpoint``)
    are supplied too; a template is free to use them."""

    output_globs: ClassVar[tuple[str, ...]] = ("checkpoint_*.pth", "log.sevenn", "*.csv")
    """The main path, not a safety net: SevenNet writes checkpoints and its log into the
    *working directory* — ``$WORK_DIR`` under the scheduler — so these are what carry the
    product home. ``checkpoint_best.pth`` matches ``checkpoint_*.pth``; the log and the
    learning-curve CSV ride along (a glob that matches nothing costs nothing)."""

    output_dirs: ClassVar[tuple[str, ...]] = ()
    """SevenNet writes flat files; there is no directory to copy back wholesale."""

    def write_dataset(self, plan: TrainingPlan, split: DatasetSplit) -> DatasetFiles:
        """Write train / valid / test extxyz files under the run directory.

        A validation set is required for the same reason FAIRChem's trainer requires one,
        one step later: ``checkpoint_best.pth`` — the artifact this step adopts — is
        written when the tracked *validation* metric improves, so a run without
        validation trains to completion and leaves no best checkpoint to adopt.

        SevenNet has no charge or spin channel, so a charged or open-shell ensemble is
        warned about rather than silently fitted as neutral-singlet data — the
        ``_parity_warning`` philosophy: silence is the only wrong answer.
        """
        if not split.valid:
            raise ConfigError(
                "SevenNet training needs a validation set — checkpoint_best.pth, the "
                "model this step adopts, is written when the tracked validation metric "
                "improves. Raise `valid_fraction` above 0."
            )
        if plan.charge != 0 or plan.multiplicity != 1:
            logger.warning(
                "SevenNet has no charge/spin channel: charge %d, multiplicity %d will be "
                "fitted as if neutral singlet — the labels carry no trace of either",
                plan.charge,
                plan.multiplicity,
            )
        plan.run_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path | None] = {"train": None, "valid": None, "test": None}
        for name, structures in (
            ("train", split.train),
            ("valid", split.valid),
            ("test", split.test),
        ):
            if not structures:
                # An empty split gets no file at all — ase refuses a zero-byte extxyz.
                continue
            path = plan.run_dir / f"{name}.xyz"
            ase_write(str(path), [_to_atoms(s) for s in structures], format="extxyz")
            written[name] = path
        assert written["train"] is not None  # noqa: S101 - split_structures guarantees one
        return DatasetFiles(train=written["train"], valid=written["valid"], test=written["test"])

    def placeholders(self, plan: TrainingPlan, data: DatasetFiles) -> dict[str, str]:
        """SevenNet's dataset knobs — ``data.load_trainset_path`` and friends."""
        return {
            "TRAIN_SET": str(data.train),
            "VALID_SET": str(data.valid) if data.valid else "",
            "TEST_SET": str(data.test) if data.test else "",
        }

    def command(self, plan: TrainingPlan, config: Path) -> str:
        """``sevenn train`` against the rendered config, from the backend env's own bin.

        The console script rather than ``-m``, like FAIRChem's — and by basename against
        the rendered config, for the array-sentinel reason MACE's command spells out.
        ``-s`` streams the log to stdout, where the runlog captures it. Multi-GPU is
        SevenNet's own documented DDP shape: ``torchrun … --no_python sevenn`` with
        ``-d`` in place of ``-s`` (``batch_size`` in the config is then per GPU).
        """
        sevenn_bin = shlex.quote(str(plan.bindir / "sevenn"))
        if plan.gpus > 1:
            python = shlex.quote(str(plan.launcher))
            return (
                f"{python} -m torch.distributed.run --standalone --nnodes 1 "
                f"--nproc_per_node {plan.gpus} --no_python {sevenn_bin} "
                f"train {config.name} -d"
            )
        return f"{sevenn_bin} train {config.name} -s"

    def artifact(self, run_dir: Path, run_name: str) -> Path:
        """``checkpoint_best.pth`` — the fixed name SevenNet gives its best checkpoint.

        Fixed by the library (``run_name`` plays no part), which is what makes the path
        predictable enough for a later step to name in its own YAML before the run.
        """
        return run_dir / "checkpoint_best.pth"

"""CHGNet: the calculator chemrefine runs, and the trainer that fine-tunes it.

One module for one library, so the environment is declared once (:data:`CHGNET`) and both
capabilities hang off it — the shape the registry sets and every sibling follows.

CHGNet has no training CLI and no config-file format: fine-tuning is a pure Python API
(``chgnet.trainer.Trainer``). The trainer therefore renders the user's YAML — a
chemrefine-defined schema, since there is no native one to render — and runs the shared
:mod:`chemrefine.engines.mlip.train.driver`, which resolves this same class in the
*backend* environment and calls its :meth:`ChgnetTrainer.run_training`. Both capabilities
and the backend-side hook live in this one module, so the registry docstring's promise —
adding a library is one dropped-in file — holds for API-only libraries too.
"""

from __future__ import annotations

import logging
import shlex
from pathlib import Path
from typing import Any, ClassVar

from chemrefine.engines.mlip.registry import MlipLibrary
from chemrefine.engines.mlip.train.base import (
    DatasetFiles,
    DatasetSplit,
    TrainingPlan,
    write_labelled_extxyz,
)
from chemrefine.errors import ConfigError

logger = logging.getLogger(__name__)

CHGNET = MlipLibrary(extra="mlip-chgnet", package="chgnet", import_name="chgnet")
"""The one declaration of what provides this library."""


@CHGNET.calculator("chgnet")
def _build_chgnet(*, model_path: str | Path | None = None, device: str = "cuda", **_: Any) -> Any:
    """CHGNet universal potential (a local checkpoint via ``model_path``).

    ``model_path`` reaches here whenever a step names ``task_name: chgnet`` with one, which is
    the same rule every library follows: the task names the library, and the path only says
    where its weights come from. The canonical import is
    ``from chgnet.model import CHGNet, CHGNetCalculator``.

    A local checkpoint loads through ``CHGNet.from_file`` — ``CHGNet.load`` is
    keyword-only and accepts *release names* (``"0.3.0"``…), never a path, so the old
    ``CHGNet.load(str(model_path))`` was a ``TypeError`` on every use; the mocked test
    that pinned it accepted any call. ``from_file`` reads the ``{"model": as_dict()}``
    shape the trainer's driver saves, which is what makes train-then-run one round trip.
    The existence check runs before the import, like every sibling's, so a mistyped
    checkpoint names the option rather than a ``torch.load`` traceback.
    """
    weights: Path | None = None
    if model_path is not None:
        weights = Path(model_path)
        if not weights.is_file():
            raise FileNotFoundError(f"CHGNet checkpoint not found: {weights}")

    from chgnet.model import CHGNet, CHGNetCalculator

    model = CHGNet.from_file(str(weights)) if weights is not None else CHGNet.load()
    return CHGNetCalculator(model=model, use_device=device)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@CHGNET.trainer("chgnet")
class ChgnetTrainer:
    """Fine-tune a CHGNet model on a step's labelled structures.

    The template is chemrefine's own schema (CHGNet has none): a YAML the shipped driver
    reads — dataset paths through the standard placeholders, plus the ``Trainer`` knobs
    (``epochs``, ``learning_rate``, ``batch_size``, ``targets``). ``chemrefine scaffold``'s
    pointer and the docs carry a worked example.
    """

    required_placeholders: ClassVar[frozenset[str]] = frozenset(
        {"TRAIN_SET", "VALID_SET", "RUN_NAME"}
    )
    """``VALID_SET`` joins the dataset because ``Trainer.train`` takes a validation loader
    positionally — there is no train-only mode; ``RUN_NAME`` because :meth:`artifact` is
    derived from it: the driver's fixed-name final save is ``{run_name}.pth.tar``, and a
    template that never tells the driver the name trains perfectly and is then reported
    as having produced nothing — MACE's reasoning, one library over."""

    output_globs: ClassVar[tuple[str, ...]] = ("*.pth.tar",)
    """The driver runs in ``$WORK_DIR`` and its fixed-name final save lands there; this is
    what carries it home. CHGNet's own per-epoch ``bestE_…``/``bestF_…`` checkpoints match
    too — small, and the run's history is worth keeping beside the product."""

    output_dirs: ClassVar[tuple[str, ...]] = ()
    """The driver writes flat files; nothing to copy back wholesale."""

    def write_dataset(self, plan: TrainingPlan, split: DatasetSplit) -> DatasetFiles:
        """Write train / valid / test extxyz files under the run directory.

        The same calculator-labelled extxyz SevenNet's trainer writes
        (:func:`~chemrefine.engines.mlip.train.base.write_labelled_extxyz`) — the driver
        converts to pymatgen ``Structure`` + per-atom energies on the other side, where
        chgnet and pymatgen are importable. A validation set is required because
        ``Trainer.train(train_loader, val_loader, …)`` has no train-only mode.

        CHGNet has no charge or spin channel, so a charged or open-shell ensemble is
        warned about rather than silently fitted as neutral data.
        """
        if not split.valid:
            raise ConfigError(
                "CHGNet training needs a validation set — its Trainer.train takes a "
                "validation loader, with no train-only mode. Raise `valid_fraction` "
                "above 0."
            )
        if plan.charge != 0 or plan.multiplicity != 1:
            logger.warning(
                "CHGNet has no charge/spin channel: charge %d, multiplicity %d will be "
                "fitted as if neutral — the labels carry no trace of either",
                plan.charge,
                plan.multiplicity,
            )
        written: dict[str, Path | None] = {"train": None, "valid": None, "test": None}
        for name, structures in (
            ("train", split.train),
            ("valid", split.valid),
            ("test", split.test),
        ):
            if not structures:
                continue
            written[name] = write_labelled_extxyz(plan.run_dir / f"{name}.xyz", structures)
        assert written["train"] is not None  # noqa: S101 - split_structures guarantees one
        return DatasetFiles(train=written["train"], valid=written["valid"], test=written["test"])

    def placeholders(self, plan: TrainingPlan, data: DatasetFiles) -> dict[str, str]:
        """The driver-schema knobs a template names — datasets, like every trainer's."""
        return {
            "TRAIN_SET": str(data.train),
            "VALID_SET": str(data.valid) if data.valid else "",
            "TEST_SET": str(data.test) if data.test else "",
        }

    def command(self, plan: TrainingPlan, config: Path) -> str:
        """The shared train driver, under the backend env's interpreter.

        ``-m chemrefine.engines.mlip.train.driver chgnet`` — the driver is chemrefine's,
        present in the managed env because the env is a ``chemrefine[mlip-chgnet]``
        install, and it resolves this same class over there and calls
        :meth:`run_training`. The config rides by basename for the array-sentinel reason
        MACE's command spells out. Single-process on any device: CHGNet's ``Trainer``
        places itself with ``use_device``, and multi-GPU DDP is not a mode its API
        documents.
        """
        python = shlex.quote(str(plan.launcher))
        return f"{python} -m chemrefine.engines.mlip.train.driver chgnet {config.name}"

    def artifact(self, run_dir: Path, run_name: str) -> Path:
        """:meth:`run_training`'s fixed-name final save: ``{run_name}.pth.tar``.

        Fixed deliberately: CHGNet's own best checkpoints embed the epoch and the error
        in their names (``bestE_epoch{n}_{err}.pth.tar``), which no later step could name
        before the run. The hook re-saves the best model under this name, in the
        ``{"model": as_dict()}`` shape ``CHGNet.from_file`` reads.
        """
        return run_dir / f"{run_name}.pth.tar"

    # -- the backend-side half (runs under the mlip-chgnet env's interpreter) ----------

    _REQUIRED_KEYS = ("train_set", "valid_set", "run_name")

    def run_training(self, config: dict[str, Any]) -> int:
        """Train with CHGNet's Python API — the hook the shared train driver calls.

        Runs in the *backend* environment, inside ``$WORK_DIR``, so the heavy imports
        live here and nowhere the orchestrator reaches. Three CHGNet facts this hook
        owns, so no template has to know them: ``StructureData`` takes **per-atom**
        energies (eV/atom — its own fine-tuning example's convention); each split becomes
        its own ``get_loader`` dataset, because CHGNet's splitting loader would
        re-partition what :func:`~chemrefine.engines.mlip.train.base.split_structures`
        already decided; and the best model is re-saved under :meth:`artifact`'s fixed
        name — ``trainer.model`` standing in when no epoch ever improved the metric.
        """
        missing = [key for key in self._REQUIRED_KEYS if not config.get(key)]
        if missing:
            raise SystemExit(
                f"chgnet training config is missing {missing} — the template must "
                f"reference $TRAIN_SET, $VALID_SET and $RUN_NAME so the render fills them"
            )

        import torch
        from chgnet.model import CHGNet
        from chgnet.trainer import Trainer

        start_from = str(config.get("start_from") or "")
        model = CHGNet.from_file(start_from) if start_from else CHGNet.load()

        batch_size = int(config.get("batch_size", 8))
        seed = int(config.get("seed", 42))
        trainer = Trainer(
            model=model,
            targets=str(config.get("targets", "ef")),
            optimizer=str(config.get("optimizer", "Adam")),
            criterion=str(config.get("criterion", "MSE")),
            epochs=int(config.get("epochs", 50)),
            learning_rate=float(config.get("learning_rate", 1e-3)),
            use_device=str(config.get("device", "cpu")),
            torch_seed=seed,
            data_seed=seed,
        )
        test_set = str(config.get("test_set") or "")
        trainer.train(
            _loader(str(config["train_set"]), batch_size),
            _loader(str(config["valid_set"]), batch_size),
            _loader(test_set, batch_size) if test_set else None,
            save_dir="chgnet_epochs",
        )

        best = getattr(trainer, "best_model", None) or trainer.model
        target = Path(f"{config['run_name']}.pth.tar")
        torch.save({"model": best.as_dict()}, target)
        print(f"chgnet training: saved {target}")
        return 0


def _split_arrays(xyz_path: str) -> tuple[list[Any], list[float], list[Any]]:
    """One split's ``(structures, energies_per_atom, forces)`` from a labelled extxyz.

    The inverse of :func:`~chemrefine.engines.mlip.train.base.write_labelled_extxyz`, read
    on the side where pymatgen lives: ase reconstructs the calculator from each frame,
    the adapter converts the geometry, and the total energy divides down to CHGNet's
    per-atom convention.
    """
    from ase.io import read as ase_read
    from pymatgen.io.ase import AseAtomsAdaptor

    structures: list[Any] = []
    energies_per_atom: list[float] = []
    forces: list[Any] = []
    for atoms in ase_read(xyz_path, index=":"):
        structures.append(AseAtomsAdaptor.get_structure(atoms))
        energies_per_atom.append(float(atoms.get_potential_energy()) / len(atoms))
        forces.append(atoms.get_forces())
    return structures, energies_per_atom, forces


def _loader(xyz_path: str, batch_size: int) -> Any:
    """A DataLoader over one split — ``get_loader``, never a splitting loader."""
    from chgnet.data.dataset import StructureData, get_loader

    structures, energies_per_atom, forces = _split_arrays(xyz_path)
    dataset = StructureData(structures=structures, energies=energies_per_atom, forces=forces)
    return get_loader(dataset, batch_size=batch_size)

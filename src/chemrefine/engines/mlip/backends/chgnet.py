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

from pathlib import Path
from typing import Any, ClassVar

from chemrefine.engines.mlip.registry import CalculatorSpec, MlipLibrary
from chemrefine.engines.mlip.train.base import (
    ApiTrainerBase,
    TrainingPlan,
    write_labelled_extxyz,
)
from chemrefine.state import Structure

CHGNET = MlipLibrary(extra="mlip-chgnet", package="chgnet", import_name="chgnet")
"""The one declaration of what provides this library."""


@CHGNET.calculator("chgnet")
def _build_chgnet(spec: CalculatorSpec) -> Any:
    """CHGNet universal potential; all three weight sources, each through its own door.

    A local checkpoint loads through ``CHGNet.from_file`` — ``CHGNet.load`` is
    keyword-only and accepts *release names* (``"0.3.0"``…), never a path; ``from_file``
    reads the ``{"model": as_dict()}`` shape the trainer saves, which is what makes
    train-then-run one round trip. A ``model_name`` is a release name for ``load`` —
    the old keyword builder silently dropped it into its catch-all, which is the bug
    class the spec exists to end. Neither given, the released default loads.
    """
    from chgnet.model import CHGNet, CHGNetCalculator

    if spec.weights is not None:
        model = CHGNet.from_file(str(spec.weights))
    elif spec.model_name:
        model = CHGNet.load(model_name=spec.model_name)
    else:
        model = CHGNet.load()
    return CHGNetCalculator(model=model, use_device=spec.device)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@CHGNET.trainer("chgnet")
class ChgnetTrainer(ApiTrainerBase):
    """Fine-tune a CHGNet model on a step's labelled structures.

    The template is chemrefine's own schema (CHGNet has none): a YAML the shipped driver
    reads — dataset paths through the standard placeholders, plus the ``Trainer`` knobs
    (``epochs``, ``learning_rate``, ``batch_size``, ``targets``). ``chemrefine scaffold``'s
    pointer and the docs carry a worked example.

    The declarations carry the library facts: a validation set is required because
    ``Trainer.train(train_loader, val_loader, …)`` has no train-only mode, and CHGNet has
    no charge or spin channel, so the base warns on non-neutral data.
    """

    label = "CHGNet"
    needs_validation = True
    validation_reason = "its Trainer.train takes a validation loader, with no train-only mode"
    driver_task = "chgnet"
    required_config_keys = ("train_set", "valid_set", "run_name")
    missing_config_hint = (
        "the template must reference $TRAIN_SET, $VALID_SET and $RUN_NAME so the render fills them"
    )
    artifact_filename = "{run_name}.pth.tar"
    """CHGNet's own best checkpoints embed the epoch and the error in their names
    (``bestE_epoch{n}_{err}.pth.tar``), which no later step could name before the run;
    the hook re-saves the best model under this fixed name, in the
    ``{"model": as_dict()}`` shape ``CHGNet.from_file`` reads."""

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

    def write_split(self, plan: TrainingPlan, name: str, structures: tuple[Structure, ...]) -> Path:
        """The shared calculator-labelled extxyz — the same file SevenNet's trainer writes.

        The driver converts to pymatgen ``Structure`` + per-atom energies on the other
        side, where chgnet and pymatgen are importable.
        """
        return write_labelled_extxyz(plan.run_dir / f"{name}.xyz", structures)

    # -- the backend-side half (runs under the mlip-chgnet env's interpreter) ----------

    def train_with_library(self, config: dict[str, Any]) -> int:
        """Train with CHGNet's Python API — reached through the shared train driver.

        Runs in the *backend* environment, inside ``$WORK_DIR``, so the heavy imports
        live here and nowhere the orchestrator reaches. Three CHGNet facts this hook
        owns, so no template has to know them: ``StructureData`` takes **per-atom**
        energies (eV/atom — its own fine-tuning example's convention); each split becomes
        its own ``get_loader`` dataset, because CHGNet's splitting loader would
        re-partition what :func:`~chemrefine.engines.mlip.train.base.split_structures`
        already decided; and the best model is re-saved under :meth:`artifact`'s fixed
        name — ``trainer.model`` standing in when no epoch ever improved the metric.
        """
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
        target = Path(self.artifact_filename.format(run_name=config["run_name"]))
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

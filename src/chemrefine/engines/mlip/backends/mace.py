"""MACE: the calculators chemrefine runs, and the trainer that fine-tunes them.

One module for one library, so the environment it needs is declared once (:data:`MACE`) and
both capabilities hang off it. A step naming ``task_name: mace_off`` therefore resolves the
same env whether it runs a model or trains one — which it must, since both come out of the
same ``chemrefine[mlip-mace]`` install.

MACE gets an environment of its own because it pins ``e3nn==0.4.4``, which cannot share a
prefix with FAIRChem's ``e3nn>=0.5``. That is also why the trainer's command is built from the
provisioner's launcher rather than a bare ``mace_run_train``, which is on nobody's ``PATH``
once the library lives somewhere the orchestrator does not.

The trainer writes MACE's **own** default label keys, so a template needs no ``energy_key`` /
``forces_key`` line to be correct, and carries the step's charge and multiplicity into the
dataset — MACE silently assumes a neutral singlet when those are absent, which would fit an
ion against the wrong species with nothing said in any log.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from ase import Atoms
from ase.io import write as ase_write

from chemrefine.engines.mlip.registry import LEGACY_MACE_TASK, MlipLibrary
from chemrefine.engines.mlip.train.base import DatasetFiles, DatasetSplit, TrainingPlan
from chemrefine.errors import ConfigError
from chemrefine.quantities import HARTREE_TO_EV
from chemrefine.state import Structure

MACE = MlipLibrary(extra="mlip-mace", package="mace-torch", import_name="mace")
"""The one declaration of what provides MACE — read by every capability below."""


# ---------------------------------------------------------------------------
# Calculators
# ---------------------------------------------------------------------------


FAMILIES = ("mace_off", "mace_mp", "mace_omol")
"""MACE's foundation-model families — organic molecules, Materials Project, charge/spin.

One list, used by both capabilities below, so a family cannot be runnable and not trainable by
an oversight in one decorator."""


@MACE.calculator(*FAMILIES, LEGACY_MACE_TASK)
def _build_mace(
    *,
    task_name: str,
    model_name: str = "",
    device: str = "cuda",
    model_path: str | Path | None = None,
    **_: Any,
) -> Any:
    """A MACE calculator: ``task_name`` is the family, the weights come from name or path.

    One builder for all of them, because MACE's own loaders already are one:
    ``mace_off(model=…)`` is typed ``str | Path`` and documented as "path to the model", and
    its ``model in mace_off_urls or str(model).startswith("https:")`` test falls through to
    loading a local file for anything else. So a fine-tuned checkpoint needs no separate task
    key — which is why there is no ``custom_fairchem`` either, and why
    :data:`~chemrefine.engines.mlip.registry.LEGACY_MACE_TASK` is an alias rather than a rule.

    The selection is resolved **before** the library is imported, so a missing checkpoint or an
    unusable alias is reported as the configuration mistake it is rather than as whichever
    ``ImportError`` or ``torch.load`` traceback the backend would have raised first — neither
    of which names the step or the option the value came from.
    """
    weights: str | Path | None = model_name or None
    if model_path is not None:
        weights = Path(model_path)
        if not weights.is_file():
            raise FileNotFoundError(f"MACE checkpoint not found: {weights}")
    elif task_name == LEGACY_MACE_TASK:
        raise ConfigError(
            f"task_name={LEGACY_MACE_TASK!r} names no MACE family and no `model_path` was "
            f"given, so there is nothing to load. Name the family you mean "
            f"({', '.join(FAMILIES)}) — `model_path` works with any of them."
        )

    from mace.calculators import mace_mp, mace_off, mace_omol

    builders = {"mace_off": mace_off, "mace_mp": mace_mp, "mace_omol": mace_omol}
    # The alias names no family, and none is needed to load a checkpoint: the file carries
    # its own architecture, so MACE-OFF's loader reads a MACE-MP model perfectly well.
    return builders.get(task_name, mace_off)(model=weights, device=device)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


ENERGY_KEY = "REF_energy"
FORCES_KEY = "REF_forces"
"""MACE's own default label keys (``mace.tools.default_keys.DefaultKeys``).

Writing the defaults rather than a name of our own is what makes a template correct with no
``energy_key`` / ``forces_key`` line in it. A template may still set them — it is MACE's
config, not ours — but then it is naming the same strings back."""

CHARGE_KEY = "total_charge"
SPIN_KEY = "total_spin"
"""Where MACE reads a system's charge and spin multiplicity from ``atoms.info``.

``total_spin`` is the multiplicity, not the number of unpaired electrons: MACE forms
``total_charge ± (total_spin - 1)``, so a closed-shell singlet is ``1.0`` — which is also the
value it silently assumes when the key is absent."""


def _to_atoms(struct: Structure, plan: TrainingPlan) -> Atoms:
    """One labelled :class:`~ase.Atoms` in the form MACE reads.

    Energies are converted Hartree → eV; forces are already eV/Å (ASE-native), which is what
    :attr:`~chemrefine.state.Structure.forces_ev_per_a` promises. The copy is what keeps the
    pipeline's own structures untouched — their force arrays are deliberately read-only, and
    ``atoms.arrays`` is not.
    """
    atoms: Atoms = struct.atoms.copy()
    assert struct.energy_hartree is not None  # noqa: S101 - split_structures rejected these
    assert struct.forces_ev_per_a is not None  # noqa: S101
    atoms.info[ENERGY_KEY] = struct.energy_hartree * HARTREE_TO_EV
    atoms.arrays[FORCES_KEY] = np.asarray(struct.forces_ev_per_a, dtype=float)
    atoms.info[CHARGE_KEY] = float(plan.charge)
    atoms.info[SPIN_KEY] = float(plan.multiplicity)
    return atoms


@MACE.trainer(*FAMILIES, LEGACY_MACE_TASK)
class MaceTrainer:
    """Train or fine-tune a MACE model on a step's labelled structures."""

    required_placeholders: ClassVar[frozenset[str]] = frozenset(
        {"TRAIN_SET", "RUN_NAME", "RUN_DIR"}
    )
    """``$RUN_NAME`` and ``$RUN_DIR`` are required alongside the dataset because
    :meth:`artifact` is derived from them: MACE names the model ``{name}.model`` inside
    ``model_dir``, which defaults to ``work_dir``. A template that hardcoded either would
    train perfectly well and then be reported as having produced nothing."""

    output_globs: ClassVar[tuple[str, ...]] = ("*.model", "*.log")
    output_dirs: ClassVar[tuple[str, ...]] = ("logs", "checkpoints", "results")
    """A safety net rather than the main path. The shipped template sets ``work_dir:
    $RUN_DIR``, so MACE writes straight into the run directory and a long training's logs and
    checkpoints are readable *while it runs* rather than only after it copies back. A template
    that leaves ``work_dir`` at MACE's default writes into the job's scratch instead, and
    these are what stop the model disappearing with it."""

    def write_dataset(self, plan: TrainingPlan, split: DatasetSplit) -> DatasetFiles:
        """Write train / valid / test extxyz files under the run directory."""
        plan.run_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path | None] = {"train": None, "valid": None, "test": None}
        for name, structures in (
            ("train", split.train),
            ("valid", split.valid),
            ("test", split.test),
        ):
            if not structures:
                # An empty split gets no file at all. ase refuses to read a zero-byte extxyz
                # ("Empty file"), so naming one would turn "no test set" into a crash.
                continue
            path = plan.run_dir / f"{name}.xyz"
            ase_write(str(path), [_to_atoms(s, plan) for s in structures], format="extxyz")
            written[name] = path
        assert written["train"] is not None  # noqa: S101 - split_structures guarantees one
        return DatasetFiles(train=written["train"], valid=written["valid"], test=written["test"])

    def placeholders(self, plan: TrainingPlan, data: DatasetFiles) -> dict[str, str]:
        """MACE's dataset knobs, plus the label keys for a template that wants them explicit."""
        return {
            "TRAIN_SET": str(data.train),
            "VALID_SET": str(data.valid) if data.valid else "",
            "TEST_SET": str(data.test) if data.test else "",
            "ENERGY_KEY": ENERGY_KEY,
            "FORCES_KEY": FORCES_KEY,
        }

    def command(self, plan: TrainingPlan, config: Path) -> str:
        """``mace_run_train`` against the rendered config, under the backend's interpreter.

        Run as ``-m mace.cli.run_train`` rather than through the ``mace_run_train`` console
        script: the module form works wherever the *interpreter* does, including the
        single-environment case where the provisioner falls back to ``sys.executable`` and no
        MACE console script is on ``PATH``. ``run_train`` carries a ``__main__`` guard, so
        this is its documented entry point rather than a trick.

        The config is named by basename because the job script copies it into ``$WORK_DIR``
        and runs there — the same reason the script engines do it.

        **The basename is interpolated bare, not quoted.** Under ``slurm_array: true`` the run
        block is rendered *once* against sentinel paths whose names are bash variables, and the
        array script assigns them per task; ``shlex.quote`` turns ``$INP_NAME`` into
        ``'$INP_NAME'``, which single quotes protect from expansion, so every task would train
        against a file of that literal name. Every other engine leaves ``.name`` bare for the
        same reason. It is safe to: the value is either that sentinel or
        :data:`~chemrefine.ids.TRAINING_ID`, both minted in :mod:`chemrefine.ids`. The launcher
        is a real filesystem path and stays quoted.
        """
        python = shlex.quote(str(plan.launcher))
        if plan.gpus > 1:
            return (
                f"{python} -m torch.distributed.run --standalone --nnodes=1 "
                f"--nproc_per_node={plan.gpus} -m mace.cli.run_train "
                f"--config {config.name} --distributed"
            )
        return f"{python} -m mace.cli.run_train --config {config.name}"

    def artifact(self, run_dir: Path, run_name: str) -> Path:
        """The trained model MACE leaves in ``model_dir`` — which defaults to ``work_dir``.

        Two names, one meaning: a run that reached stage two (``swa``/``stagetwo``, which the
        recommended fine-tuning recipe enables) writes ``{name}_stagetwo.model``, and one that
        did not writes ``{name}.model``. Neither carries the seed, unlike the tagged copies
        under ``checkpoints/`` — which is what makes this path predictable enough for a later
        step to name in its own YAML before the training has run.
        """
        stage_two = run_dir / f"{run_name}_stagetwo.model"
        return stage_two if stage_two.is_file() else run_dir / f"{run_name}.model"

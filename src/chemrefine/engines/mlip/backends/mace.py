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

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from ase import Atoms
from ase.io import write as ase_write

from chemrefine.engines.mlip.registry import CalculatorSpec, MlipLibrary
from chemrefine.engines.mlip.train.base import DatasetFiles, TrainerBase, TrainingPlan
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

LEGACY_MACE_TASK = "custom_mace"
"""A back-compat alias for MACE, kept only so configs written against v1 still resolve.

It is **not** a mechanism, and there is deliberately no ``custom_fairchem`` beside it. A local
checkpoint is ``model_path``, which every library's builder honours itself — see the registry
module's docstring. Configs naming this should say which MACE family they mean
(:data:`FAMILIES`) instead; it is registered here on the MACE library so that saying nothing
still works. MACE policy, so it lives in MACE's module — the shared registry houses no
library's aliases."""


@MACE.calculator(*FAMILIES, LEGACY_MACE_TASK)
def _build_mace(spec: CalculatorSpec) -> Any:
    """A MACE calculator: ``task_name`` is the family, the weights come from name or path.

    One builder for all of them, because MACE's own loaders already are one:
    ``mace_off(model=…)`` is typed ``str | Path`` and documented as "path to the model", and
    its ``model in mace_off_urls or str(model).startswith("https:")`` test falls through to
    loading a local file for anything else. So a fine-tuned checkpoint needs no separate task
    key — which is why there is no ``custom_fairchem`` either, and why
    :data:`LEGACY_MACE_TASK` is an alias rather than a rule. An unusable alias — the legacy
    task with nothing to load — is reported before the library is imported, naming the
    families to write instead.
    """
    if spec.weights is None and spec.task_name == LEGACY_MACE_TASK:
        raise ConfigError(
            f"task_name={LEGACY_MACE_TASK!r} names no MACE family and no `model_path` was "
            f"given, so there is nothing to load. Name the family you mean "
            f"({', '.join(FAMILIES)}) — `model_path` works with any of them."
        )

    from mace.calculators import mace_mp, mace_off, mace_omol

    builders = {"mace_off": mace_off, "mace_mp": mace_mp, "mace_omol": mace_omol}
    weights: str | Path | None = (
        spec.weights if spec.weights is not None else (spec.model_name or None)
    )
    # The alias names no family, and none is needed to load a checkpoint: the file carries
    # its own architecture, so MACE-OFF's loader reads a MACE-MP model perfectly well.
    return builders.get(spec.task_name, mace_off)(model=weights, device=spec.device)


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
class MaceTrainer(TrainerBase):
    """Train or fine-tune a MACE model on a step's labelled structures."""

    label = "MACE"
    charge_spin_aware = True
    """The dataset speaks ``total_charge`` / ``total_spin`` (:func:`_to_atoms`), so an ion
    or an open-shell species is fitted as itself — no warning to raise."""

    required_placeholders: ClassVar[frozenset[str]] = frozenset(
        {"TRAIN_SET", "RUN_NAME", "RUN_DIR", "DEVICE", "SEED"}
    )
    """``$RUN_NAME`` and ``$RUN_DIR`` are required alongside the dataset because
    :meth:`artifact` is derived from them: MACE names the model ``{name}.model`` inside
    ``model_dir``, which defaults to ``work_dir``. A template that hardcoded either would
    train perfectly well and then be reported as having produced nothing.

    ``$DEVICE`` and ``$SEED`` are required because they are **plan facts** and MACE reads
    both from the config this template becomes (``device:`` falls back to ``cpu``,
    ``seed:`` to MACE's own ``123``). For a CLI trainer the template is the only channel
    the library's program reads, so the gate is the closure the driver-run trainers get
    from their argv: a template that does not reference them would run a ``device: cuda``
    step on CPU with the GPU booked, and seed torch differently from the ``seed`` the
    step's cache fingerprint records."""

    output_globs: ClassVar[tuple[str, ...]] = ("*.model", "*.log")
    output_dirs: ClassVar[tuple[str, ...]] = ("logs", "checkpoints", "results")
    """A safety net rather than the main path. The shipped template sets ``work_dir:
    $RUN_DIR``, so MACE writes straight into the run directory and a long training's logs and
    checkpoints are readable *while it runs* rather than only after it copies back. A template
    that leaves ``work_dir`` at MACE's default writes into the job's scratch instead, and
    these are what stop the model disappearing with it."""

    def write_split(self, plan: TrainingPlan, name: str, structures: tuple[Structure, ...]) -> Path:
        """One split as extxyz with MACE's own ``REF_*`` label keys."""
        path = plan.run_dir / f"{name}.xyz"
        ase_write(str(path), [_to_atoms(s, plan) for s in structures], format="extxyz")
        return path

    def placeholders(self, plan: TrainingPlan, data: DatasetFiles) -> dict[str, str]:
        """The dataset triple, plus the label keys for a template that wants them explicit."""
        return {
            **super().placeholders(plan, data),
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
        if plan.gpus > 1:
            return self.torchrun(
                plan, "-m", "mace.cli.run_train", "--config", config.name, "--distributed"
            )
        return f"{self.quoted_launcher(plan)} -m mace.cli.run_train --config {config.name}"

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

"""FAIRChem backend (UMA / eSEN): one builder, every task head.

FAIRChem is *one family with many checkpoints and a fixed set of task heads*.
``model_name`` selects the checkpoint (any ``uma-…`` or ``esen-…`` id in
``pretrained_mlip.available_models``); ``task_name`` *is* the head and keys the
registry, so it is passed straight through to
:class:`~fairchem.core.FAIRChemCalculator`. The one builder is registered under
each of the 7 heads, so ``task_name: omat`` builds the materials head — not a
hardcoded ``omol``.
"""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from chemrefine.engines.mlip.registry import MlipLibrary
from chemrefine.engines.mlip.train.base import DatasetFiles, DatasetSplit, TrainingPlan
from chemrefine.errors import ConfigError
from chemrefine.quantities import HARTREE_TO_EV
from chemrefine.state import Structure

_METADATA_NAME = "metadata.npz"
"""What FAIRChem's dataset loader looks for beside a database it was given no explicit
``metadata_path`` for. Its batch sampler reads ``natoms`` from it to build atom-budget
batches, so it is required rather than an optimisation."""

_DATA_DIR = "data"
"""Where the split databases go, one sub-directory each, under the run directory.

Namespaced rather than written at the top: FAIRChem's ``timestamp_id`` is pinned to
:data:`~chemrefine.ids.TRAINING_ID` and it creates ``<run_dir>/<timestamp_id>/`` for its own
checkpoints — the same word the training split is called. Without this the training set would
be written at ``train/train/train.db``, inside the tree the model is about to be written to."""

FAIRCHEM = MlipLibrary(extra="mlip-fairchem", package="fairchem-core", import_name="fairchem")
"""The one declaration of what provides FAIRChem — read by every capability below."""

HEADS = {
    "omol": "molecules",
    "omat": "materials",
    "odac": "DAC MOFs",
    "oc20": "catalysis",
    "oc22": "oxides",
    "oc25": "electrolytes",
    "omc": "molecular crystals",
}
"""FAIRChem's task heads. One list, so a capability cannot be registered for a subset of
them by accident — a trainer added later reuses exactly these keys rather than retyping them.
"""

#: Default checkpoint when ``model_name`` is unset — UMA-1.2, the latest small UMA
#: model (fastest while still SOTA on most benchmarks); ships with ``fairchem-core>=2.18``.
_DEFAULT_MODEL = "uma-s-1p2"


def _write_ase_db(
    directory: Path, name: str, structures: Sequence[Structure], *, charge: int, spin: int
) -> Path:
    """Write one split as an ASE database plus its ``metadata.npz``; return the database.

    Labels go on a :class:`~ase.calculators.singlepoint.SinglePointCalculator` because that is
    where ``AseDBDataset`` reads them from — unlike MACE's extxyz, there are no named info or
    array keys to match. Energies convert Hartree → eV; forces are already eV/Å.

    ``charge`` and ``spin`` (the multiplicity) ride in each row's ``data=`` mapping, which is
    the one channel ``AseDBDataset`` copies back into ``atoms.info`` — a plain ``db.write``
    drops ``atoms.info`` entirely. They matter because FAIRChem's ``common_transform``
    silently defaults both to a neutral singlet when absent, so without these an ion would be
    fine-tuned as the wrong species with nothing said in any log. The template's ``a2g_args``
    must also ask for them (``r_data_keys: [charge, spin]``); the shipped example does.

    ``natoms`` is written as an **integer** array: the loader asserts
    ``np.issubdtype(dtype, np.integer)`` and a float column fails there rather than anywhere
    informative.
    """
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.db import connect

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.db"
    path.unlink(missing_ok=True)  # `connect` appends, and a re-run must not stack duplicates
    with connect(str(path)) as db:
        for struct in structures:
            atoms = struct.atoms.copy()
            assert struct.energy_hartree is not None  # noqa: S101 - split_structures rejected these
            assert struct.forces_ev_per_a is not None  # noqa: S101
            atoms.calc = SinglePointCalculator(
                atoms,
                energy=struct.energy_hartree * HARTREE_TO_EV,
                forces=np.asarray(struct.forces_ev_per_a, dtype=float),
            )
            db.write(atoms, data={"charge": charge, "spin": spin})
    np.savez(
        directory / _METADATA_NAME,
        natoms=np.array([len(s.atoms) for s in structures], dtype=np.int64),
    )
    return path


@FAIRCHEM.calculator(*HEADS)
def _build_fairchem(
    *,
    task_name: str,
    model_name: str = "",
    device: str = "cuda",
    model_path: str | Path | None = None,
    **_: Any,
) -> Any:
    """FAIRChem (UMA/eSEN): ``task_name`` is the head; the weights come from name or path.

    Two different loaders, because FAIRChem has two. ``pretrained_mlip.get_predict_unit``
    resolves a **registry name** and raises ``KeyError`` for anything else — there is no path
    branch in it — so a checkpoint a training step just produced cannot be loaded that way.
    ``load_predict_unit`` is the same thing one level down and takes a path.

    Without this branch a fine-tuned FAIRChem model could be trained and never run, which is
    most of the point of being able to train one.
    """
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    if model_path is not None:
        # Imported in the branch that uses it, not beside the others: the two loaders live in
        # different submodules, and a caller running a *named* release should not need the
        # one it will not call to be importable.
        from fairchem.core.units.mlip_unit import load_predict_unit

        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"FAIRChem checkpoint not found: {path}")
        predictor = load_predict_unit(str(path), device=device)
    else:
        predictor = pretrained_mlip.get_predict_unit(
            model_name=model_name or _DEFAULT_MODEL, device=device
        )
    return FAIRChemCalculator(predictor, task_name=task_name)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@FAIRCHEM.trainer(*HEADS)
class FairchemTrainer:
    """Fine-tune a FAIRChem model on a step's labelled structures.

    Registered over the same ``HEADS`` the calculator uses, so the head list is written once
    and a step that fine-tunes ``omol`` and then runs the result names one library.

    FAIRChem's config surface is large and **nothing ships in the wheel to copy**
    (``find … -name '*.yaml'`` → 0 files), so the template is the user's — this class only
    writes the data, names the paths, and launches. Its shape is fixed, though: FAIRChem's
    fine-tuning entry points compose one specific way, published as
    ``configs/uma/finetune/uma_sm_finetune_template.yaml`` in the fairchem repository, and
    the shipped example follows it. The parts a template must keep, and what breaks without
    each:

    * ``transforms: {common_transform: {dataset_name: …}}`` on every dataset stanza — the
      collater dispatches on the name that transform stamps, and data tagged any other way
      (``a2g_args: {task_name: …}``) dies inside ``MTCollater`` with
      ``TypeError: unhashable type: 'list'``.
    * ``tasks_list`` defining fresh ``energy``/``forces`` Tasks whose ``datasets`` name that
      same string, fed to **both** ``mt_collater_adapter`` and the training unit — the
      checkpoint's own task list belongs to its pretraining datasets, not to this one.
    * The model node **literally at** ``runner.train_eval_unit.model``, as
      ``initialize_finetuning_model`` with replacement ``heads`` — nothing else converts the
      final sharded checkpoint into the ``inference_ckpt.pt`` a later step loads.
    * A ``TrainCheckpointCallback`` — a run without one trains to completion and writes
      nothing at all.
    * ``a2g_args: {r_data_keys: [charge, spin]}`` — see :func:`_write_ase_db`.
    """

    required_placeholders: ClassVar[frozenset[str]] = frozenset(
        {"TRAIN_SET", "VAL_SET", "RUN_DIR", "RUN_NAME"}
    )
    """``RUN_DIR`` and ``RUN_NAME`` join the datasets because :meth:`artifact` is derived from
    them: FAIRChem writes under ``run_dir/<timestamp_id>/checkpoints``, and the template is
    what pins ``timestamp_id``. Left to its default it is a fresh timestamp, and the step
    would train correctly and then report having produced nothing."""

    output_globs: ClassVar[tuple[str, ...]] = ("*.pt", "*.yaml", "*.log")
    output_dirs: ClassVar[tuple[str, ...]] = ()
    """Nothing is copied back, and nothing can be: the model must not go through the scratch.

    A template **must** point ``job.run_dir`` at ``$RUN_DIR`` — which is why ``RUN_DIR`` is a
    required placeholder — so FAIRChem writes its checkpoints straight into the run directory
    as it goes, and a long training's logs are readable while it runs.

    The copy-back cannot serve as a fallback for this the way MACE's ``output_dirs`` does. Its
    file half is non-recursive and FAIRChem's product is three directories down; its directory
    half copies ``$WORK_DIR/<name>``, and the name would have to be FAIRChem's ``timestamp_id``
    — a value chosen per run, which a ``ClassVar`` read without a context cannot know. So a
    template that leaves ``run_dir`` at its default trains into the scratch and loses the
    model, and the honest thing is to say so rather than to declare a net that does not catch
    it."""

    def write_dataset(self, plan: TrainingPlan, split: DatasetSplit) -> DatasetFiles:
        """Write one ASE database per split, each with the metadata its sampler demands.

        **A directory per split, not a file per split.** Without an explicit
        ``metadata_path`` FAIRChem looks for ``metadata.npz`` in the database file's *parent*,
        so two databases sharing a directory would both load the first one's metadata and trip
        its length assertion. The template is given explicit paths anyway; this makes the
        fallback correct too.

        All of them under :data:`_DATA_DIR`, which is what keeps the ``train`` split out of the
        ``train`` timestamp directory FAIRChem writes its checkpoints into.

        Sqlite rather than ``.aselmdb``: this runs in the orchestrator's environment, which
        has core ``ase`` but not ``ase_db_backends``. `AseDBDataset` connects with a plain
        ``ase.db.connect``, so any format ase can write is one it can read.
        """
        if not split.valid:
            raise ConfigError(
                "FAIRChem training needs a validation set — its runner takes a train *and* an "
                "eval dataloader, with no train-only mode. Raise `valid_fraction` above 0."
            )
        written: dict[str, Path | None] = {"train": None, "valid": None, "test": None}
        for name, structures in (
            ("train", split.train),
            ("valid", split.valid),
            ("test", split.test),
        ):
            if not structures:
                continue
            written[name] = _write_ase_db(
                plan.run_dir / _DATA_DIR / name,
                name,
                structures,
                charge=plan.charge,
                spin=plan.multiplicity,
            )
        assert written["train"] is not None  # noqa: S101 - split_structures guarantees one
        return DatasetFiles(train=written["train"], valid=written["valid"], test=written["test"])

    def placeholders(self, plan: TrainingPlan, data: DatasetFiles) -> dict[str, str]:
        """FAIRChem's dataset paths, their metadata, and its own spellings of the shared knobs.

        ``$DEVICE`` is overridden rather than shared: FAIRChem's ``device_type`` and
        ``scheduler.mode`` are omegaconf-validated enums that reject their own lowercase
        *values* — ``'cpu'`` raises ``Invalid value 'cpu', expected one of [CPU, CUDA]`` — and
        accept only the member names. A trainer specialising a shared placeholder is what
        :func:`~chemrefine.engines.mlip.train.base.placeholders_for` allows, and this is the
        case it is for: the same word, the spelling this library insists on.

        ``$RANKS_PER_NODE`` exists because ``$NGPUS`` cannot serve FAIRChem's
        ``scheduler.ranks_per_node``: a CPU step demands zero GPUs, and zero *ranks* is a run
        with no workers. The distributed width is ``gpus`` on a GPU step and one otherwise.
        """
        return {
            "TRAIN_SET": str(data.train),
            "TRAIN_METADATA": str(data.train.parent / _METADATA_NAME),
            "VAL_SET": str(data.valid) if data.valid else "",
            "VAL_METADATA": str(data.valid.parent / _METADATA_NAME) if data.valid else "",
            "TEST_SET": str(data.test) if data.test else "",
            "TEST_METADATA": str(data.test.parent / _METADATA_NAME) if data.test else "",
            "DEVICE": plan.device.upper(),
            "RANKS_PER_NODE": str(max(1, plan.gpus)),
        }

    def command(self, plan: TrainingPlan, config: Path) -> str:
        """``fairchem -c <config>`` from the backend environment's own ``bin/``.

        The console script, not ``-m``: ``fairchem.core._cli`` has no ``__main__`` guard, so
        ``python -m fairchem.core._cli`` exits 0 having done nothing — a training step that
        appears to succeed and produces no model.

        No Hydra CLI overrides either. Everything this run needs is already in the rendered
        config, and overrides for keys absent from a YAML have prefix rules that differ by
        Hydra version — a dependency the generated command has no reason to take on.

        The basename is bare rather than quoted, for the reason
        :meth:`~chemrefine.engines.mlip.backends.mace.MaceTrainer.command` spells out: under
        ``slurm_array: true`` it is a ``$INP_NAME`` sentinel the array script expands, and
        single quotes would stop it expanding.
        """
        return f"{shlex.quote(str(plan.bindir / 'fairchem'))} -c {config.name}"

    def artifact(self, run_dir: Path, run_name: str) -> Path:
        """The inference checkpoint FAIRChem writes when training ends.

        ``on_train_end`` saves into ``checkpoints/final``, and ``save_state`` converts the
        sharded training checkpoint there into a single loadable ``inference_ckpt.pt``. That
        conversion is what a later step loads, and it happens only while ``save_inference_ckpt``
        is left on — as is EMA, whose state the conversion reads.

        The ``<run_name>`` component is FAIRChem's ``timestamp_id``, which the template pins to
        this same value; that is what makes the path predictable before the run.
        """
        return run_dir / run_name / "checkpoints" / "final" / "inference_ckpt.pt"

"""Pydantic validator for MLIP engine YAML options.

Mirrors :class:`chemrefine.engines.pyscf.options.PyscfOptions` so a
new engine implementer adds an ``options.py`` per backend by the
same pattern. Both the direct (:class:`~chemrefine.engines.mlip.engine.MlipEngine`)
and ExtOpt (:class:`~chemrefine.engines.mlip.extopt_engine.MlipExtOptEngine`)
engines route ``step.options`` through this validator instead of
reading the raw YAML dict.
"""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, ConfigDict, Field

from chemrefine.engines._options import EngineOptions


class MlipOptions(EngineOptions):
    """Validated knobs for the MLIP backend.

    Two axes, mapped the same way across libraries (the mace↔fairchem
    equivalence): ``task_name`` is the **method/head** that selects the builder
    (see the table in :mod:`chemrefine.engines.mlip.calculator`), and
    ``model_name`` is the **weights** handed to it. Backend-natural YAML aliases
    are accepted (``task`` for ``task_name``; ``model``/``size`` for
    ``model_name``) so each backend reads naturally. ``device`` + ``from_raw`` come
    from :class:`~chemrefine.engines._options.EngineOptions`.
    """

    model_config = ConfigDict(populate_by_name=True)

    model_name: str = Field(
        "uma-s-1p2",
        validation_alias=AliasChoices("model_name", "model", "size"),
    )
    """The model *weights* for the chosen backend.

    A MACE size (``small``/``medium``/``large``), a FAIRChem checkpoint
    (``uma-s-1p2``/``uma-s-1p1``/``esen-…``), a SevenNet id (``7net-0``), an ORB
    loader (``orb_v3_…``), or a local path. YAML aliases: ``model``, ``size``.
    Defaults to ``uma-s-1p2`` (UMA-1.2, ships with ``fairchem-core>=2.18``).
    """

    task_name: str = Field(
        "omol",
        validation_alias=AliasChoices("task_name", "task"),
    )
    """The method / head / family — selects the builder.

    A FAIRChem head (``omol``/``omat``/``odac``/``oc20``/``oc22``/``oc25``/``omc``),
    ``mace_off``/``mace_mp``/``mace_omol``, ``custom_mace``, ``chgnet``,
    ``sevenn``, or ``orb``. YAML alias: ``task``.
    """

    model_path: str | None = None
    """Custom MACE checkpoint path (selects the ``custom_mace`` backend)."""


class MlipTrainOptions(MlipOptions):
    """Validated knobs for the ``mlip-train`` step.

    Exists for one field. Training is GPU work — MACE training on CPU is not a
    slower run, it is an impractical one — so this step keeps requesting a GPU
    when the YAML names no device, where every inference engine now defaults to
    ``cpu`` (the floor that always runs).

    The point of declaring that here rather than leaving a literal in the trainer
    is that a default spelled in two places is a default that drifts: the trainer
    repeated ``"cuda"`` inline, so moving the shared default silently split the
    two apart, with the options model saying ``cpu`` and the SLURM header saying
    ``cuda`` for the very same step.
    """

    device: Literal["cuda", "cpu"] = "cuda"

    valid_fraction: float = Field(0.1, gt=0, lt=1)
    """Share of the structures held out for validation.

    Bounded here rather than checked in the trainer: ``0`` left no validation set and
    ``1`` left nothing to train on, and both only surfaced as a ``ValueError`` from
    :func:`~chemrefine.engines.mlip.trainer.prepare_inputs` after the step had already
    started."""

    seed: int = 42
    """Seed for the train/validation split, so a re-run reproduces the same split."""

    job_name: str = Field("mlip_train", pattern=r"^[A-Za-z0-9._-]+$")
    """SLURM job name for the training job.

    The pattern is the constraint the trainer already enforced by hand, moved to the
    field that owns it: this value lands inside an ``#SBATCH`` directive, where a newline
    would start an arbitrary extra directive and whitespace would split the value."""

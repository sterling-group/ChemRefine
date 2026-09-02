"""Pydantic validator for MLIP engine YAML options.

Mirrors :class:`chemrefine.engines.pyscf.options.PyscfOptions` so a
new engine implementer adds an ``options.py`` per backend by the
same pattern. Both the direct (:class:`~chemrefine.engines.mlip.engine.MlipEngine`)
and ExtOpt (:class:`~chemrefine.engines.mlip.extopt_engine.MlipExtOptEngine`)
engines route ``step.options`` through this validator instead of
reading the raw YAML dict.
"""

from __future__ import annotations

from pydantic import AliasChoices, ConfigDict, Field, field_validator

from chemrefine.config import reject_shell_unsafe
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
        "",
        validation_alias=AliasChoices("model_name", "model", "size"),
    )
    """The model *weights* for the chosen backend, in that library's own spelling.

    A MACE size (``small``/``medium``/``large``), a FAIRChem checkpoint
    (``uma-s-1p2``/``uma-s-1p1``/``esen-…``), a SevenNet id (``7net-0``), an ORB
    loader (``orb_v3_…``). YAML aliases: ``model``, ``size``.

    Unset means **the chosen library's own default** — FAIRChem's builder fills in
    ``uma-s-1p2``, MACE's and SevenNet's loaders theirs — because a default spelled
    *here* would have to be right for every library at once, and the old ``uma-s-1p2``
    was not: it reached every non-FAIRChem step's template too, surviving only because a
    buggy builder happened to drop it. For a training step, unset means training from
    scratch (``started_from: scratch`` in the runlog).
    """

    task_name: str = Field(
        "omol",
        validation_alias=AliasChoices("task_name", "task"),
    )
    """The method / head / family — **the only key that selects a builder**.

    A FAIRChem head (``omol``/``omat``/``odac``/``oc20``/``oc22``/``oc25``/``omc``),
    ``mace_off``/``mace_mp``/``mace_omol``, ``chgnet``, ``sevenn``, or ``orb``. YAML alias:
    ``task``.
    """

    model_path: str | None = None
    """A local checkpoint to load instead of a named release — with the library above.

    It selects nothing. Every builder takes it and loads it with *its own* library, so running
    a model an ``mlip-train`` step produced means naming the same ``task_name`` that trained
    it. Letting a checkpoint imply a library would not survive a second library that can
    produce one: the value travels through channels that carry only its string
    (``model_dump()``, a ``$MODEL_PATH`` placeholder, the ExtOpt CLI), none of which could
    also carry "and which library this means".

    Relative values are made absolute against the **config file's** directory by
    :func:`chemrefine.config.resolve_relative_paths`, with every other path the config
    names. It has to happen there rather than here: a field validator sees only the process
    working directory, which is not where the config sits — and the value reaches a job that
    runs in a scratch directory, so an unresolved relative path is found by nobody."""

    @field_validator("model_name", "model_path")
    @classmethod
    def _shell_safe(cls, v: str | None) -> str | None:
        """Hold both selection strings to the shell rule the other bash-bound values obey.

        ``mlip-train`` records ``model_path or model_name`` as the runlog's
        ``started_from`` row, which lands in the unquoted header heredoc — the same
        channel ``operation`` and ``executables`` are refused metacharacters for, and
        the one through which ``model_name: 'x$(…)'`` executed on the compute node.
        :func:`chemrefine.job_log.bash_header` refuses the value again at emission;
        this is the boundary half of the same rule, so the mistake is named at config
        load with the knob that carries it. Real model names and checkpoint paths never
        carry these characters, so nothing legitimate is narrowed.

        Per knob, deliberately — not a blanket check on
        :class:`~chemrefine.engines._options.EngineOptions`. The rule is keyed to
        *reaching generated bash*, and most option values never do: the ExtOpt server
        CLI is ``shlex.quote``-d, script placeholders land in Python source, and
        ``task_name`` is registry-gated before any script exists. A base-wide check
        would refuse values the property does not cover — PySCF's compound-functional
        ``xc: "b88,lyp"`` carries the comma the rule now rejects, legitimately. The
        general net for future rows is the emission check, which guards the channel
        itself rather than a roster of knobs.
        """
        if v:
            reject_shell_unsafe(v, what="the MLIP model selection", fix="rename it")
        return v


CALCULATOR_KNOBS: tuple[str, ...] = ("model_name", "task_name", "device", "model_path")
"""The four fields that select and place a calculator — the one list consumers iterate.

Spelled once, beside the model that declares the fields, so the direct engine's template
placeholders and the ExtOpt server's CLI table enumerate the same knobs; the model stays
the source of names, defaults and YAML aliases. The sweep that motivated this counted the
tuple hand-enumerated at ten sites."""


class MlipTrainOptions(MlipOptions):
    """Validated knobs for the ``mlip-train`` step.

    Inherits ``task_name`` / ``model_name`` / ``model_path`` from :class:`MlipOptions`, with
    their training meanings: ``task_name`` selects the **trainer** through the same registry
    rule that selects the calculator, so one word names the library whether a step trains a
    model or runs one; ``model_name`` is the foundation model a run starts from, and
    ``model_path`` a local checkpoint to continue.

    **``device`` and ``cores`` are inherited unchanged, and read through
    ``model_fields_set``.** Both want a training-specific meaning for *unset* — a device the
    step must name explicitly, and a core count that defaults to the whole step budget — and
    both could have got it by re-declaring the field with a wider type. That would be a lie
    to anything holding these options as the base type, and the type checker says so. Pydantic
    already records which fields the YAML actually set, so
    :class:`~chemrefine.engines.mlip.train.engine.MlipTrainEngine` asks that instead and the
    inherited types stay honest.
    """

    gpus: int = Field(1, ge=1)
    """GPUs this training job needs — the data-parallel width, not an allocation.

    chemrefine never writes ``--gres``: it charges this against the GPU budget and passes it
    to the trainer as its distributed width (MACE's ``--nproc_per_node``, FAIRChem's
    ``ranks_per_node``). The *allocation* is the ``--gres`` line in your own
    ``cuda.slurm.header``, so raise both together."""

    seed: int = 42
    """Seed for the train/validation/test split, so a re-run reproduces the same partition."""

    valid_fraction: float = Field(0.1, ge=0, lt=1)
    """Share of the structures held out to validate on.

    Bounded here rather than checked in the trainer: ``1`` leaves nothing to train on
    whatever the structure count, which is a fact about the number and belongs on the field.
    ``0`` is allowed and means "no validation set" — a legitimate choice on a tiny dataset,
    and distinct from the count-dependent case (a fraction that rounds to zero structures)
    which only :func:`~chemrefine.engines.mlip.train.base.split_structures` can decide."""

    test_fraction: float = Field(0.0, ge=0, lt=1)
    """Share held out for a final evaluation the training never sees. Defaults to none.

    Distinct from ``valid_fraction`` because the two sets do different jobs: a *validation*
    set steers training (early stopping, model selection), a *test* set only reports. On the
    few-dozen-structure datasets this step is built for, a test set is usually a luxury the
    training set cannot afford — hence the default of zero."""

"""Pydantic validator for MLIP engine YAML options.

Both the direct (:class:`~chemrefine.engines.mlip.engine.MlipEngine`)
and ExtOpt (:class:`~chemrefine.engines.mlip.extopt_engine.MlipExtOptEngine`)
engines route ``step.options`` through this validator instead of
reading the raw YAML dict.
"""

from __future__ import annotations

from typing import Annotated, Any, Self

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from chemrefine.config import reject_shell_unsafe
from chemrefine.engines._options import KEYED_WHEN_SET, EngineOptions, refuse_template_knobs
from chemrefine.engines.orca.extopt.engine import ExtOptOptions

_SOLVATION_KNOBS = ("solvation", "solvation_model", "solvent", "solvation_weights")


class SolvationKnobs(BaseModel):
    """The implicit-solvation knobs an MLIP step's ``options`` may carry.

    Grouped like :class:`~chemrefine.config.NmsKnobs` and inherited by :class:`MlipOptions`,
    so in YAML they sit flat beside the other engine options. All three unset is a
    gas-phase step. The rules between them, and the libraries, live in
    :mod:`chemrefine.engines.mlip.solvation`.

    The registry is imported inside the validators rather than at module top: importing
    :mod:`chemrefine.engines.mlip.solvation` runs the ``chemrefine.engines.mlip`` package,
    which imports this module back while it is still initialising.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    solvation: Annotated[str | None, KEYED_WHEN_SET] = None
    """The library that computes the solvation correction (``anysolv``).

    No default, and required whenever ``solvent`` is set: a default would make a second
    library change what every existing config means."""

    solvation_model: Annotated[str | None, KEYED_WHEN_SET] = None
    """That library's model; unset means the library's ChemRefine default (AnySolv's
    ``compact``). The default is written into the validated options, so the cache key
    names the model whether or not the YAML did.

    ``auto`` asks for the library's best model whose weights the step's environment has
    (AnySolv: ``moe``, else ``compact``). It is chosen once per step before the run, in that
    environment, and the run continues with the model chosen: the cache key, the jobs and
    the step's ``_cache/resolved_options.json`` all name it. A later run that can choose a
    better model recomputes the step with it. One that could only choose a worse model than
    the recorded one (the MoE weights have gone) is refused before anything is submitted,
    rather than replacing the step's results with a lesser model's."""

    solvent: Annotated[str | None, KEYED_WHEN_SET] = None
    """The solvent, by a name in the library's own table (``water``, ``thf``, …). Checked by
    the library, which holds the table, when the step's calculator is built."""

    solvation_weights: Annotated[str | None, KEYED_WHEN_SET] = None
    """A weights file for a model the library downloads separately (AnySolv's ``moe``),
    looked at before the library's own places. Unset, the library looks in those alone.

    A path like ``model_path``: relative values resolve against the config file's directory
    (:data:`chemrefine.config.STEP_OPTION_PATHS`), and the key holds its basename and its
    bytes, so replacing the file recomputes the step."""

    @model_validator(mode="before")
    @classmethod
    def _fill_default_model(cls, data: Any) -> Any:
        """Write the library's default model in when a solvated step names none."""
        from chemrefine.engines.mlip.solvation import registered_solvation, solvation_library

        if not isinstance(data, dict):
            return data
        solvation = data.get("solvation")
        if solvation in registered_solvation() and not data.get("solvation_model"):
            data = {**data, "solvation_model": solvation_library(solvation).default_model}
        return data

    @model_validator(mode="after")
    def _solvation_is_consistent(self) -> Self:
        """Refuse a solvent without a library, a library without a solvent, or an unknown either."""
        from chemrefine.engines.mlip.solvation import solvation_problem

        problem = solvation_problem(
            self.solvation, self.solvation_model, self.solvent, self.solvation_weights
        )
        if problem:
            raise ValueError(problem)
        return self


class MlipOptions(SolvationKnobs, EngineOptions):
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
    """The model *weights* for the chosen backend: a release of the chosen library, in its
    own spelling (:attr:`~chemrefine.engines.mlip.registry.MlipLibrary.releases`; the backend
    table in the docs lists each library's). YAML aliases: ``model``, ``size``.

    Under a ``task_name`` whose library declares its releases
    (:attr:`~chemrefine.engines.mlip.registry.MlipLibrary.releases`), any other name is
    refused before the run starts, whatever the step's template does with it: a template's
    own knobs belong in ``extra``, and a checkpoint in ``model_path``.

    Unset means **the chosen library's own default** — its builder's — because a default
    spelled *here* would have to be right for every library at once, and no release name
    is. For a training step, unset means where the library's own training starts when told
    nothing: the trainer's declared release
    (:attr:`~chemrefine.engines.mlip.train.base.TrainerBase.default_foundation`), named
    as ``started_from`` in the runlog and in the sidecar, or from scratch (``scratch`` in
    the runlog, ``null`` in the sidecar).
    """

    task_name: str = Field(
        "omol",
        validation_alias=AliasChoices("task_name", "task"),
    )
    """The method / head / family — **the only key that selects a builder**.

    A task name a registered library answers to
    (:func:`~chemrefine.engines.mlip.registry.registered_backends`; the backend table in the
    docs lists them). YAML alias: ``task``.
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

        ``mlip-train`` records ``model_path`` or ``model_name`` (unset, its trainer's
        declared default) as the runlog's ``started_from`` row, which lands in the
        unquoted header heredoc — the same channel ``operation`` and ``executables``
        are refused metacharacters for, and the one through which
        ``model_name: 'x$(…)'`` executed on the compute node.
        :func:`chemrefine.job_log.bash_header` refuses the value again at emission;
        this is the boundary half of the same rule, so the mistake is named at config
        load with the knob that carries it. Real model names and checkpoint paths never
        carry these characters, so nothing legitimate is narrowed.

        Per knob, deliberately — not a blanket check on
        :class:`~chemrefine.engines._options.EngineOptions`. The rule is keyed to
        *reaching generated bash*, and most option values never do: the ExtOpt server
        CLI is ``shlex.quote``-d, script placeholders land in Python source, and
        ``task_name`` is registry-gated before any script exists. A base-wide check
        would refuse values the property does not cover — a quoting character in a
        knob that lands in Python source is a legitimate value there. The general net
        for future rows is the emission check, which guards the channel itself rather
        than a roster of knobs.
        """
        if v:
            reject_shell_unsafe(v, what="the MLIP model selection", fix="rename it")
        return v

    extra: dict[str, Any] = Field(default_factory=dict)
    """Free knobs for your own ``step{N}.py``, as a mapping.

    The declared bag for a template's own settings: validated only as a mapping, rendered
    as ``$EXTRA`` (a Python dict literal) and inside ``$OPTIONS_JSON``. Declared rather than
    read off undeclared keys so a key you invent is deliberate and a typo of a real knob
    still warns; the engines that render no template (``mlip-extopt``, ``mlip-train``)
    refuse it."""


CALCULATOR_KNOBS: tuple[str, ...] = (
    "model_name",
    "task_name",
    "device",
    "model_path",
    *_SOLVATION_KNOBS,
)
"""The fields that select, place and solvate a calculator — the one list consumers iterate.

Spelled once, beside the model that declares the fields, so the direct engine's template
placeholders and the ExtOpt server's CLI table enumerate the same knobs; the model stays
the source of names, defaults and YAML aliases. The sweep that motivated this counted the
tuple hand-enumerated at ten sites."""


class MlipExtOptOptions(MlipOptions, ExtOptOptions):
    """The MLIP knobs as the ``mlip-extopt`` engine reads them, plus the ExtOpt family's.

    A subclass rather than the direct model itself because the direct engine and this one
    read ``extra`` differently: ``mlip`` renders it into the user's ``step{N}.py``, and this
    engine renders no template at all — the server is built from the validated knobs alone
    — so a key placed there would be accepted and read by nothing. The refusal lives on the
    model the engine declares, where the preflight, the server command and the docs all
    read it.
    """

    @model_validator(mode="after")
    def _no_template_knobs(self) -> Self:
        """Refuse ``extra``: this engine renders no ``step{N}.py`` for it to reach."""
        refuse_template_knobs(self.extra)
        return self


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

    @model_validator(mode="after")
    def _no_template_knobs(self) -> Self:
        """Refuse ``extra``: this engine renders no ``step{N}.py`` for it to reach."""
        refuse_template_knobs(self.extra)
        return self

    @model_validator(mode="after")
    def _no_solvation(self) -> Self:
        """Refuse the solvation knobs: a training step fits a model and evaluates none.

        Inherited from :class:`SolvationKnobs` with the rest of :class:`MlipOptions`; accepted,
        they would be read by nothing, and a fine-tune on gas-phase labels would look like a
        solvated one in the config.
        """
        if set_knobs := [name for name in _SOLVATION_KNOBS if getattr(self, name)]:
            raise ValueError(
                f"{', '.join(set_knobs)}: mlip-train fits a model on its labels and runs none, "
                f"so a solvation correction has nothing to apply to"
            )
        return self

    gpus: int = Field(1, ge=1)
    """GPUs this training job needs — the data-parallel width, not an allocation.

    chemrefine never writes ``--gres``: it charges this against the GPU budget and passes it
    to the trainer as its distributed width (MACE's ``--nproc_per_node``, FAIRChem's
    ``ranks_per_node``). The *allocation* is the ``--gres`` line in your own
    ``cuda.slurm.header``, so raise both together."""

    seed: int = Field(42, ge=0)
    """Seed for the train/validation/test split, so a re-run reproduces the same partition.

    Non-negative because ``numpy.random.default_rng`` refuses anything else with a bare
    ``ValueError`` — which would be raised in ``prepare``, after every labelling step
    upstream has run."""

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

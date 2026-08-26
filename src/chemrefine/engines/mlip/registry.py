"""Which MLIP library a step selects, and what that library can do.

One registry, keyed by ``task_name``, shared by inference and training. Each entry names a
:class:`MlipLibrary` — the environment that provides the library — plus whichever capabilities
that library offers: an ASE calculator builder, a :class:`~chemrefine.engines.mlip.train.base.
TrainerBase` subclass, or both. :mod:`chemrefine.engines.mlip.calculator` reads the first;
:mod:`chemrefine.engines.mlip.train.engine` reads the second; the provisioner reads the
library.

Why one registry and not two
----------------------------

Because ``task_name: mace_off`` names **one library** whether a step trains a model or runs
one, and the environment it needs is the same environment either way. Two registries said that
in prose and implemented it as a copy: the packaging triple was declared once per registry
rather than once per library, as was the dispatch rule and the task-key list, with nothing to
catch them drifting. The triple is what resolves *which environment a step launches from*, so a
drift is silent at import and surfaces as a step provisioned into one env and launched
expecting another.

Merging them makes two things true by construction rather than by discipline: a trainable task
is a runnable task, and it resolves the same environment in both directions.

This is the rule the rest of ``engines/`` already follows — a concern two siblings both need
gets one home so neither imports from the other (see
:mod:`chemrefine.engines.orca.output.status`, :mod:`chemrefine.engines.pyscf._runtime`,
:mod:`chemrefine.engines.orca.extopt.run_block`).

Three knobs, one dispatch
-------------------------

.. code-block:: text

    task_name   which library (and, for MACE, which family)   mace_off, omol, sevenn
    model_name  a named release of it                         small, uma-s-1p2, 7net-0
    model_path  weights from this file instead                ./outputs/step4/train/train.model

Only the first selects anything. ``model_path`` is passed to the chosen library's builder,
which loads it *with that library* — MACE's own ``mace_off``/``mace_mp``/``mace_omol`` take a
checkpoint path directly, and FAIRChem's builder swaps ``get_predict_unit`` for
``load_predict_unit``. So there is no ``custom_<library>`` key for anyone, and adding a library
never adds one: a fine-tuned model is run by naming the library that trained it, exactly as a
released one is.

Adding a library
----------------

**One dropped-in module** under :mod:`chemrefine.engines.mlip.backends`, auto-discovered, plus
one ``[project.optional-dependencies]`` entry for the extra it names (asserted by
``test_engines_mlip_training.py``, since an extra nothing installs is an env that cannot be
provisioned). It declares its environment once and hangs its capabilities off it::

    from chemrefine.engines.mlip.registry import CalculatorSpec, MlipLibrary

    MY_MLIP = MlipLibrary(
        extra="mlip-my_mlip", package="my-mlip-lib", import_name="my_mlip_library"
    )

    @MY_MLIP.calculator("my_task")
    def _build_my_mlip(spec: CalculatorSpec):
        from my_mlip_library import MyCalculator      # imported lazily, inside the builder

        return MyCalculator(model=spec.weights or spec.model_name, device=spec.device)

    @MY_MLIP.trainer("my_task")                        # optional — omit if it cannot train
    class MyTrainer(TrainerBase): ...                  # or ApiTrainerBase, for API-only libs

Both decorators are variadic, which is what lets one library register a family of heads from a
single list rather than a stack of decorators per capability — FAIRChem's head family is
declared once and used by both.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chemrefine.engines.api import BackendRequirement
from chemrefine.engines.mlip.options import MlipOptions
from chemrefine.errors import ConfigError

if TYPE_CHECKING:
    from chemrefine.engines.mlip.train.base import TrainerBase


@dataclass(frozen=True)
class CalculatorSpec:
    """Everything a builder may read — filled and vetted by the dispatch, never by hand.

    **This frozen value is the whole calculator-builder contract.** The previous contract
    was four keyword arguments plus ``**_``, stated in docstrings and enforced by nothing
    — which is how three shipped builders came to silently swallow a knob the dispatch
    was passing (``model_path`` on sevenn and orb, ``model_name`` on chgnet): a keyword
    that lands in a catch-all vanishes without a trace. A single positional spec has no
    catch-all to vanish into; a builder that ignores a field ignores it *visibly*, in
    code a reviewer can read.

    ``weights`` arrives already existence-checked (the dispatch's job, done once —
    previously five pasted copies of the same ``is_file()`` idiom, one per builder).
    ``charge`` and ``multiplicity`` are ``None`` when the caller did not say — the door
    for charge-aware libraries (AIMNet2's calculator takes a total charge; ``mace_omol``
    reads one), which the old four-knob shape could not express. A builder for a
    charge-blind library simply never reads them.

    The builder's *return* stays ``Any`` deliberately: ASE calculators are an untyped
    third-party surface, and a home-grown protocol over them would be aspirational
    typing with no enforcement value.
    """

    task_name: str
    model_name: str
    device: str
    weights: Path | None
    charge: int | None = None
    multiplicity: int | None = None


CalculatorBuilder = Callable[[CalculatorSpec], Any]
"""The typed shape every registered builder has: one spec in, an ASE calculator out.

The alias is what makes the contract *static* — ``@LIB.calculator`` is typed over it, so
mypy checks each builder at its own decorator site — and :meth:`MlipLibrary.calculator`
verifies the callable's arity at registration, so a malformed drop-in fails at import of
its own module with a message naming the rule."""


@dataclass(frozen=True)
class MlipLibrary:
    """One MLIP library and the environment that provides it.

    Declared **once per module**, and every capability that module registers refers to this
    object — so a library that is both runnable and trainable cannot name two different
    environments for itself.

    ``extra`` is the pip extra that provides it (``chemrefine[<extra>]``, and also the managed
    env's name); ``package`` is the distribution named in an actionable import error;
    ``import_name`` is the top-level module the provisioner probes with
    :func:`importlib.util.find_spec`. Every MLIP library gets an environment of its own —
    their torch/e3nn trees conflict and cannot share a prefix.
    """

    extra: str
    package: str
    import_name: str

    def calculator(self, *task_names: str) -> Callable[[CalculatorBuilder], CalculatorBuilder]:
        """Register a builder for each of ``task_names``; return it unchanged.

        The builder is a :data:`CalculatorBuilder` — one :class:`CalculatorSpec` in, an
        ASE calculator out; the spec's docstring is the contract. Its backend import
        needs no guard — :func:`~chemrefine.engines.mlip.calculator.build_calculator`
        turns an ``ImportError`` into the install hint built from this library's
        metadata.

        The decorator is the gate: a builder that is not a single-parameter callable is
        refused **here**, at import of the module that declares it. The old shape —
        keyword arguments plus a catch-all — was checked by nothing, and three shipped
        builders silently swallowed a knob the dispatch was passing before anyone
        noticed; arity is the property a signature can actually prove, so it is proven.
        """

        def _wrap(fn: CalculatorBuilder) -> CalculatorBuilder:
            if not callable(fn):
                raise TypeError(
                    f"{self.extra}: @calculator({', '.join(map(repr, task_names))}) "
                    f"must decorate a callable, got {fn!r}"
                )
            parameters = [
                p
                for p in inspect.signature(fn).parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
            if len(parameters) != 1 or len(inspect.signature(fn).parameters) != 1:
                raise TypeError(
                    f"{self.extra}: builder {getattr(fn, '__name__', fn)!r} must take "
                    f"exactly one positional parameter — the CalculatorSpec. The spec "
                    f"replaces the old keyword shape so no knob can vanish into a "
                    f"catch-all."
                )
            for name in task_names:
                _put(name, self, builder=fn)
            return fn

        return _wrap

    def trainer(self, *task_names: str) -> Callable[[type[TrainerBase]], type[TrainerBase]]:
        """Register a :class:`~chemrefine.engines.mlip.train.base.TrainerBase` subclass.

        Optional: a library that ships no training entry point simply never calls this, and
        :func:`trainer_for` then reports it as runnable but not trainable — which is a
        different thing from an unknown task, and worth saying differently.

        The decorator is the gate, mirroring :meth:`calculator`: a class that is not a
        concrete ``TrainerBase`` — or that leaves a declaration the machinery reads
        unset — is refused **here**, at import of the module that declares it, naming
        the missing item. The class statement already enforces the abstract hooks at
        instantiation; this check moves the failure to discovery time and covers the
        bare ``ClassVar`` declarations no ``ABCMeta`` machinery watches.
        """

        def _wrap(cls: type[TrainerBase]) -> type[TrainerBase]:
            # Function-level import: the registry cannot import ``train.base`` at module
            # top — ``train/__init__`` pulls in ``train.engine``, which imports this
            # module back while it is still initialising.
            from chemrefine.engines.mlip.train.base import ApiTrainerBase, TrainerBase

            names = ", ".join(map(repr, task_names))
            if not (isinstance(cls, type) and issubclass(cls, TrainerBase)):
                raise TypeError(
                    f"{self.extra}: @trainer({names}) must decorate a TrainerBase "
                    f"subclass (ApiTrainerBase for a library without a CLI), got {cls!r}"
                )
            abstract = sorted(getattr(cls, "__abstractmethods__", ()))
            if abstract:
                raise TypeError(
                    f"{self.extra}: trainer {cls.__name__} leaves {abstract} abstract — "
                    f"every hook must be implemented before registration"
                )
            required = ["label", "output_globs"]
            if issubclass(cls, ApiTrainerBase):
                required += ["driver_task", "missing_config_hint", "artifact_filename"]
            missing = [name for name in required if not hasattr(cls, name)]
            if cls.needs_validation and not cls.validation_reason:
                missing.append("validation_reason")
            if missing:
                raise TypeError(
                    f"{self.extra}: trainer {cls.__name__} is missing declaration(s) "
                    f"{missing} — the machinery reads them; see TrainerBase"
                )
            for name in task_names:
                _put(name, self, trainer=cls)
            return cls

        return _wrap


@dataclass(frozen=True)
class BackendSpec:
    """What one ``task_name`` resolves to: a library, and its capabilities.

    ``builder`` and ``trainer`` are independently optional because the two capabilities are
    independent — every shipped library can be run, only some can be trained. The packaging
    fields read straight through to the library, so a caller asks ``spec.extra`` without
    caring that it is stored once per library rather than once per task.
    """

    library: MlipLibrary
    builder: CalculatorBuilder | None = None
    trainer: type[TrainerBase] | None = None

    @property
    def extra(self) -> str:
        """The pip extra (and managed-env name) that provides this backend."""
        return self.library.extra

    @property
    def package(self) -> str:
        """The pip distribution named in this backend's install hint."""
        return self.library.package

    @property
    def import_name(self) -> str:
        """The top-level module the provisioner probes for this backend."""
        return self.library.import_name


_BACKENDS: dict[str, BackendSpec] = {}


def _put(
    name: str,
    library: MlipLibrary,
    *,
    builder: CalculatorBuilder | None = None,
    trainer: type[TrainerBase] | None = None,
) -> None:
    """Add or extend the entry for ``name`` with one capability.

    Capabilities accumulate: a module registers its builder and its trainer separately, and
    the second call extends the entry the first created rather than replacing it. Registering
    a second library against an existing task is refused — it is the drift this registry
    exists to make impossible, and it can only happen by a genuine mistake in a backend
    module.
    """
    existing = _BACKENDS.get(name)
    if existing is None:
        _BACKENDS[name] = BackendSpec(library, builder=builder, trainer=trainer)
        return
    if existing.library != library:
        raise ValueError(
            f"task_name {name!r} is already registered to {existing.library.extra!r}; "
            f"{library.extra!r} cannot claim it too — one task names one library."
        )
    _BACKENDS[name] = replace(
        existing,
        builder=builder if builder is not None else existing.builder,
        trainer=trainer if trainer is not None else existing.trainer,
    )


def backend_spec(task_name: str) -> BackendSpec:
    """The entry a ``task_name`` resolves to — the whole dispatch rule.

    **``task_name`` alone selects the library.** ``model_path`` does not participate: it says
    where the chosen library takes its weights from, and each library's builder honours it
    itself. Keeping it out of the dispatch is load-bearing — a rule that also read the
    checkpoint would need "stated task / checkpoint only / neither" to survive every channel
    the selection crosses (``model_dump()``, a ``$TASK_NAME`` template placeholder, an
    argparse CLI), and each of those carries exactly one string.

    Raises :class:`~chemrefine.errors.ConfigError` — not ``ValueError`` — because an unknown
    ``task_name`` is a configuration mistake, and :mod:`chemrefine.errors` promises every
    exception carries the exit code the CLI maps. This is reached from ``preflight_backends``,
    where a bare ``ValueError`` would escape that contract as a traceback.
    """
    spec = _BACKENDS.get(task_name)
    if spec is None:
        raise ConfigError(
            f"unsupported MLIP backend: task_name={task_name!r} (known: {sorted(_BACKENDS)})"
        )
    return spec


def calculator_for(task_name: str) -> CalculatorBuilder:
    """The builder a ``task_name`` dispatches to; raises if the library cannot be run."""
    spec = backend_spec(task_name)
    if spec.builder is None:
        raise ConfigError(f"MLIP backend {task_name!r} declares no calculator")
    return spec.builder


def trainer_for(task_name: str) -> type[TrainerBase]:
    """The trainer a ``task_name`` dispatches to; raises if that library cannot be trained.

    "Unknown task" and "known task, no trainer" are different mistakes and get different
    messages: the first is a typo, the second is a library chemrefine can run but not yet
    train, which is a fact about the backend rather than about the config.
    """
    spec = backend_spec(task_name)
    if spec.trainer is None:
        raise ConfigError(
            f"MLIP backend {task_name!r} can be run but not trained. "
            f"Trainable: {sorted(registered_trainers())}."
        )
    return spec.trainer


def registered_backends() -> frozenset[str]:
    """Every ``task_name`` the registry knows."""
    return frozenset(_BACKENDS)


def registered_trainers() -> frozenset[str]:
    """Every ``task_name`` that can be trained — a subset of :func:`registered_backends`."""
    return frozenset(name for name, spec in _BACKENDS.items() if spec.trainer is not None)


def registered_extras() -> frozenset[str]:
    """Every pip extra a registered backend declares (drives CLI listing/validation)."""
    return frozenset(spec.extra for spec in _BACKENDS.values())


def trainer_output_globs() -> tuple[str, ...]:
    """Every registered trainer's ``output_globs``, unioned and sorted.

    The scheduler asks an engine what to copy back out of scratch without a
    :class:`~chemrefine.state.StepContext` in hand, so ``mlip-train`` cannot answer for *this*
    step's trainer and has to answer for all of them. Derived here, from the trainers'
    declarations, because a hand-maintained superset is a list that silently stops being one
    the day a library is added. :class:`~chemrefine.engines.mlip.train.engine.MlipTrainEngine`
    reads this directly — its ``output_globs`` property — so the roster is correct by
    construction, not by a test holding two spellings equal.

    A superset is the safe direction — copying back a pattern nothing wrote costs nothing,
    where *missing* one loses a model to the scratch cleanup.
    """
    return tuple(
        sorted(
            {
                glob
                for spec in _BACKENDS.values()
                if spec.trainer
                for glob in spec.trainer.output_globs
            }
        )
    )


def requirement_from_options(
    options: Mapping[str, Any] | None,
    *,
    options_cls: type[MlipOptions] | None = None,
    require_trainer: bool = False,
) -> BackendRequirement:
    """The environment a step's raw ``options`` imply.

    Reads the task/model selection **through the options model** rather than off the raw dict,
    so the environment the preflight demands is always the one the step would actually load —
    a second reader of the same knobs is free to disagree with the first about aliases and
    defaults, and the disagreement shows up as a preflight that passes a step the engine then
    rejects.

    ``require_trainer`` is what a training step passes. It makes ``preflight_backends`` refuse
    an untrainable ``task_name`` **before any step submits**, rather than at the training step
    itself — which, in a pipeline that spends days computing labels first, is the difference
    between a typo caught in seconds and one caught on Thursday.
    """
    opts = (options_cls or MlipOptions).from_raw_lenient(dict(options or {}))
    spec = backend_spec(opts.task_name)
    if require_trainer:
        trainer_for(opts.task_name)
    return BackendRequirement(extra=spec.extra, import_name=spec.import_name)

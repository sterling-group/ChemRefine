"""Which MLIP library a step selects, and what that library can do.

One registry, keyed by ``task_name``, shared by inference and training. Each entry names a
:class:`MlipLibrary` — the environment that provides the library — plus whichever capabilities
that library offers: an ASE calculator builder, a :class:`~chemrefine.engines.mlip.train.base.
Trainer`, or both. :mod:`chemrefine.engines.mlip.calculator` reads the first;
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

    from chemrefine.engines.mlip.registry import MlipLibrary

    MY_MLIP = MlipLibrary(
        extra="mlip-my_mlip", package="my-mlip-lib", import_name="my_mlip_library"
    )

    @MY_MLIP.calculator("my_task")
    def _build_my_mlip(*, model_name="", device="cuda", model_path=None, **_):
        from my_mlip_library import MyCalculator      # imported lazily, inside the builder

        return MyCalculator(model=model_path or model_name, device=device)

    @MY_MLIP.trainer("my_task")                        # optional — omit if it cannot train
    class MyTrainer: ...

Both decorators are variadic, which is what lets one library register a family of heads from a
single list rather than a stack of decorators per capability — FAIRChem's seven heads are
declared once and used by both.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from chemrefine.engines.api import BackendRequirement
from chemrefine.engines.mlip.options import MlipOptions
from chemrefine.errors import ConfigError

if TYPE_CHECKING:
    from chemrefine.engines.mlip.train.base import Trainer

LEGACY_MACE_TASK = "custom_mace"
"""A back-compat alias for MACE, kept only so configs written against v1 still resolve.

It is **not** a mechanism, and there is deliberately no ``custom_fairchem`` beside it. A local
checkpoint is ``model_path``, which every library's builder honours itself — see the module
docstring. Configs naming this should say which MACE family they mean (``mace_off`` and
friends) instead; it is registered on the MACE library so that saying nothing still works."""


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

    def calculator(self, *task_names: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a builder for each of ``task_names``; return it unchanged.

        The builder takes keyword arguments (``task_name``, ``model_name``, ``device``,
        ``model_path``, plus a catch-all ``**_``) and returns an ASE calculator. Its backend
        import needs no guard — :func:`~chemrefine.engines.mlip.calculator.build_calculator`
        turns an ``ImportError`` into the install hint built from this library's metadata.
        """

        def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
            for name in task_names:
                _put(name, self, builder=fn)
            return fn

        return _wrap

    def trainer(self, *task_names: str) -> Callable[[type[Trainer]], type[Trainer]]:
        """Register a :class:`~chemrefine.engines.mlip.train.base.Trainer` for ``task_names``.

        Optional: a library that ships no training entry point simply never calls this, and
        :func:`trainer_for` then reports it as runnable but not trainable — which is a
        different thing from an unknown task, and worth saying differently.
        """

        def _wrap(cls: type[Trainer]) -> type[Trainer]:
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
    builder: Callable[..., Any] | None = None
    trainer: type[Trainer] | None = None

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
    builder: Callable[..., Any] | None = None,
    trainer: type[Trainer] | None = None,
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


def calculator_for(task_name: str) -> Callable[..., Any]:
    """The builder a ``task_name`` dispatches to; raises if the library cannot be run."""
    spec = backend_spec(task_name)
    if spec.builder is None:
        raise ConfigError(f"MLIP backend {task_name!r} declares no calculator")
    return spec.builder


def trainer_for(task_name: str) -> type[Trainer]:
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
    the day a library is added — nothing else would connect a new trainer's globs to the
    engine's ClassVar. A test holds the two equal.

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

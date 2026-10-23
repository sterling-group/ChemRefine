"""Which solvation library a step names, what it needs, and how its correction is built.

A solvation library adds an implicit-solvent correction on top of a gas-phase MLIP: given a
geometry, its charge and multiplicity and a solvent, it returns ΔE = E(solv) - E(gas) and
the matching ΔF. It never sees the base potential, so the solvated surface is the sum —
``E_MLIP + ΔE``, ``F_MLIP + ΔF`` — whatever library computes ``E_MLIP``, and the one wrapper
in :mod:`chemrefine.engines.mlip.calculator` serves every MLIP route (a direct template, and
ORCA's Opt / OptTS / GOAT / Freq through ExtOpt, whose numerical Hessian differentiates the
corrected gradients).

Shaped like :mod:`chemrefine.engines.mlip.registry`: one module per library under
:mod:`chemrefine.engines.mlip.solvation`, auto-discovered, declaring a
:class:`SolvationLibrary` once and registering its correction builder on it::

    MY_SOLV = SolvationLibrary(
        name="my_solv",
        requirement=PythonRequirement(package="my-solv", import_name="my_solv"),
        models={"small": "my_solv_small"},
        default_model="small",
        pairs_with=frozenset({"mlip-fairchem"}),
    )

    @MY_SOLV.correction
    def _build(spec: SolvationSpec) -> Correction:
        from my_solv import delta                     # imported lazily, inside the builder
        return lambda atoms, charge, multiplicity: delta(atoms, ...)

Keyed by ``solvation`` and never defaulted, so a second library can be added without any
existing config changing what it means.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from chemrefine.errors import ConfigError

if TYPE_CHECKING:
    from ase import Atoms


@dataclass(frozen=True)
class PythonRequirement:
    """A library imported in the step's own MLIP environment, beside the base potential.

    ``package`` is the distribution named in an install hint; ``import_name`` the top-level
    module the correction imports. It shares the MLIP env rather than getting one of its own:
    the correction runs in the same process as the base potential, once per evaluation.
    """

    package: str
    import_name: str


@dataclass(frozen=True)
class ExecutableRequirement:
    """A program the correction runs, at the path the config's ``executables`` names.

    ``key`` is the ``executables`` key (``xtb`` for ``executables: {xtb: …}``) — a machine
    fact, like ORCA's path, rather than anything installed into the MLIP env.
    """

    key: str


SolvationRequirement = PythonRequirement | ExecutableRequirement
"""What a library needs to be runnable: an import in the MLIP env, or a program on disk."""


@dataclass(frozen=True)
class SolvationSpec:
    """Everything a correction builder may read — filled and vetted by :func:`build_correction`.

    ``model`` arrives in the **library's own spelling** (the value side of
    :attr:`SolvationLibrary.models`), already checked, and never ``auto``: that is chosen
    before the spec is made. ``solvent`` is the name the step wrote, which only the library
    can vet, since only it holds its solvent table. ``weights`` is the step's
    ``solvation_weights``, a file of weights the library downloads separately, or ``None``.
    """

    library: str
    model: str
    solvent: str
    device: str
    weights: str | None = None


Correction = Callable[["Atoms", int, int], tuple[float, np.ndarray]]
"""One geometry, its charge and multiplicity in, ``(ΔE eV, ΔF eV/Å)`` out."""

CorrectionBuilder = Callable[[SolvationSpec], Correction]
"""Builds a step's :data:`Correction` once, so its model and solvent are resolved once."""

Locator = Callable[[str, "str | None"], str]
"""A model (the library's spelling) and the step's ``solvation_weights`` in, the path of its
weights file out. Raises :class:`WeightsMissing` when the file is in none of the places the
library looks."""

AUTO = "auto"
"""The ``solvation_model`` that asks for the first model in :attr:`SolvationLibrary.auto`
whose weights the step's environment has."""


class WeightsMissing(ConfigError):
    """A model's weights are in none of the places its library looks.

    The one refusal ``solvation_model: auto`` passes over to try its next model. Anything
    else stays a refusal under ``auto`` too: a ``solvation_weights`` path that is not there is
    a mistake in the config, not a model this machine lacks.
    """


@dataclass(frozen=True)
class SolvationLibrary:
    """One solvation library: what the step names it by, what it needs, which models it has.

    ``models`` maps the name a step writes as ``solvation_model`` to the library's own
    spelling of it, and ``default_model`` is the one an unset ``solvation_model`` means.
    The default is ChemRefine's to choose rather than the library's: a library that picks
    by what is on disk would give one config two answers on two machines, so the options
    model fills it in and it is part of the step's cache key.

    ``pairs_with`` is the MLIP extras the library has been run with. The correction does
    not depend on the base potential, but the library still has to install and run in that
    potential's environment, and this is the list of environments where it has been checked.

    ``auto`` is the preference order ``solvation_model: auto`` walks, best first; empty means
    the library offers no ``auto``. ``weighted`` is the models whose weights a step may name
    as ``solvation_weights``. ``locate`` finds a model's weights; it runs in the step's own
    environment, where the library and its weights are, both when ``auto`` is chosen before
    the run and when the correction is built.
    """

    name: str
    requirement: SolvationRequirement
    models: Mapping[str, str] = field(hash=False)
    default_model: str
    pairs_with: frozenset[str]
    auto: tuple[str, ...] = ()
    weighted: frozenset[str] = frozenset()
    locate: Locator | None = field(default=None, hash=False, compare=False)

    def __post_init__(self) -> None:
        """Refuse a default, an ``auto`` order or a weighted model the library does not have."""
        if self.default_model not in self.models:
            raise TypeError(
                f"solvation library {self.name!r}: default_model {self.default_model!r} is "
                f"not one of its models ({', '.join(self.models)})"
            )
        if stray := sorted((set(self.auto) | self.weighted) - set(self.models)):
            raise TypeError(f"solvation library {self.name!r}: {stray} are not among its models")
        if AUTO in self.models:
            raise TypeError(f"solvation library {self.name!r}: {AUTO!r} cannot name a model")
        if (self.auto or self.weighted) and self.locate is None:
            raise TypeError(
                f"solvation library {self.name!r}: `auto` and `weighted` need a `locate` "
                f"to find the weights"
            )

    def weights_of(self, name: str, weights: str | None) -> str | None:
        """Where model ``name``'s weights are, or ``None`` for a library that finds its own.

        Raises :class:`WeightsMissing` when the library looks and finds none.
        """
        if self.locate is None:
            return None
        return self.locate(self.model(name), weights)

    def choose(self, weights: str | None) -> tuple[str, tuple[str, ...]]:
        """The first :attr:`auto` model whose weights are here, and why each before it was not.

        Called in the step's own environment. A :class:`ConfigError` when none is here.
        """
        skipped: list[str] = []
        for name in self.auto:
            try:
                self.weights_of(name, weights)
            except WeightsMissing as exc:
                skipped.append(f"{name}: {exc}")
                continue
            return name, tuple(skipped)
        raise ConfigError(
            f"solvation: none of {self.name}'s models is installed here ({'; '.join(skipped)})"
        )

    def model(self, name: str | None) -> str:
        """``name`` (unset: :attr:`default_model`) in this library's own spelling."""
        chosen = name or self.default_model
        if chosen == AUTO:
            raise ConfigError(
                f"solvation: `solvation_model: {AUTO}` is chosen before the run, in the step's "
                f"environment; this reader was handed it unchosen"
            )
        if chosen not in self.models:
            raise ConfigError(
                f"solvation: {self.name} has no model {chosen!r} (its models: "
                f"{', '.join(self.models)})"
            )
        return self.models[chosen]

    def require_pairing(self, mlip_extra: str) -> None:
        """Refuse an MLIP environment this library has not been run in."""
        if mlip_extra not in self.pairs_with:
            raise ConfigError(
                f"solvation: {self.name} has not been run with the {mlip_extra} backend "
                f"(it pairs with: {', '.join(sorted(self.pairs_with))})"
            )

    def correction(self, fn: CorrectionBuilder) -> CorrectionBuilder:
        """Register ``fn`` as this library's correction builder; return it unchanged."""
        existing = _LIBRARIES.get(self.name)
        if existing is not None and existing.library != self:
            raise ValueError(f"solvation library {self.name!r} is already registered")
        _LIBRARIES[self.name] = _Registered(self, fn)
        return fn


@dataclass(frozen=True)
class _Registered:
    """A registered library and its correction builder, as :data:`_LIBRARIES` holds them."""

    library: SolvationLibrary
    build: CorrectionBuilder


_LIBRARIES: dict[str, _Registered] = {}


def registered_solvation() -> frozenset[str]:
    """Every name a step's ``solvation`` may take."""
    return frozenset(_LIBRARIES)


def solvation_library(name: str) -> SolvationLibrary:
    """The library ``solvation: <name>`` selects; a :class:`ConfigError` naming the known ones."""
    entry = _LIBRARIES.get(name)
    if entry is None:
        raise ConfigError(
            f"unknown solvation library {name!r} (known: {', '.join(sorted(_LIBRARIES))})"
        )
    return entry.library


def solvation_problem(
    solvation: str | None,
    solvation_model: str | None,
    solvent: str | None,
    solvation_weights: str | None = None,
) -> str | None:
    """Why these knobs do not make a solvated step, or ``None`` when they do.

    One rule for both readers: the options model, which reports it as a validation error,
    and :func:`build_correction`, which a ``step{N}.py`` reaches without the YAML. All of
    them unset is a gas-phase step. ``solvation`` is required whenever any other is set,
    because it is the only knob that says which library the others mean.
    ``solvation_weights`` is accepted for a model that reads it, or under ``auto`` when one
    of the models ``auto`` may choose does.
    """
    if not solvation:
        named = [
            f"`{knob}`"
            for knob, value in (
                ("solvent", solvent),
                ("solvation_model", solvation_model),
                ("solvation_weights", solvation_weights),
            )
            if value
        ]
        if named:
            return (
                f"{', '.join(named)} needs `solvation` to name the library it belongs to "
                f"(one of: {', '.join(sorted(_LIBRARIES))})"
            )
        return None
    try:
        library = solvation_library(solvation)
        if solvation_model == AUTO:
            if not library.auto:
                return f"solvation: {solvation} offers no `solvation_model: {AUTO}`"
            readers = library.weighted & set(library.auto)
        else:
            library.model(solvation_model)
            readers = library.weighted & {solvation_model or library.default_model}
    except ConfigError as exc:
        return str(exc)
    if solvation_weights and not readers:
        return (
            f"`solvation_weights` is a weights file for {solvation}'s "
            f"{', '.join(sorted(library.weighted)) or 'no'} model; "
            f"`solvation_model: {solvation_model or library.default_model}` reads none"
        )
    if not solvent:
        return f"`solvation: {solvation}` needs a `solvent` to solvate in"
    return None


def build_correction(
    *,
    solvation: str | None,
    solvation_model: str | None,
    solvent: str | None,
    device: str,
    solvation_weights: str | None = None,
) -> Correction | None:
    """The step's correction, or ``None`` for a gas-phase step.

    A missing library is reported naming the library, not the MLIP extra: the base
    potential is installed, and what the environment lacks is the solvation library.

    ``solvation_model: auto`` reaching here unchosen — a ``step{N}.py`` run by hand, or a
    calculator built outside a run — is chosen here, in this environment, which is the
    step's own. A run chooses it before any job exists and hands the job the model it
    chose, so a job never chooses for itself.
    """
    problem = solvation_problem(solvation, solvation_model, solvent, solvation_weights)
    if problem:
        raise ConfigError(problem)
    if not solvation:
        return None
    entry = _LIBRARIES[solvation]
    try:
        if solvation_model == AUTO:
            solvation_model, _skipped = entry.library.choose(solvation_weights)
        spec = SolvationSpec(
            library=solvation,
            model=entry.library.model(solvation_model),
            solvent=str(solvent),
            device=device,
            weights=solvation_weights,
        )
        return entry.build(spec)
    except ImportError as exc:
        requirement = entry.library.requirement
        if not isinstance(requirement, PythonRequirement):
            raise
        raise ImportError(
            f"solvation: {solvation} needs '{requirement.package}' in the environment this "
            f"step runs in — install it there, beside the MLIP library."
        ) from exc

"""Engine Protocol contract and the ``ENGINES`` registry.

Every engine satisfies :class:`CalculationEngine` structurally — no
inheritance required. Engines register themselves via the :func:`register`
decorator at import time, so importing :mod:`chemrefine.engines`
populates the registry as a side effect.

The orchestrator only ever sees the :class:`CalculationEngine` Protocol
plus the :data:`ENGINES` dict. ORCA-specific imports, MLFF imports, etc.
never reach :mod:`chemrefine.pipeline` — that's how the orchestrator
stays engine-agnostic.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from chemrefine.errors import EngineNotFoundError
from chemrefine.state import JobBatch, StepContext, StepInputs, StepResults


@runtime_checkable
class CalculationEngine(Protocol):
    """Structural contract every engine must satisfy.

    The five lifecycle methods mirror the five stages
    :func:`chemrefine.step.run_step` calls in order. ``supports_nms`` is
    a class-level boolean: when ``True``, :meth:`normal_mode_sample` is
    invoked between :meth:`parse` and filtering for steps that set
    ``nms: true`` in their YAML.
    """

    name: str
    supports_nms: bool

    def prepare(self, ctx: StepContext) -> StepInputs:
        """Write engine-specific input files for this step's seed structures."""
        ...

    def submit(self, inputs: StepInputs, ctx: StepContext) -> JobBatch:
        """Submit a batch of jobs (SLURM or local); return their handles."""
        ...

    def wait(self, batch: JobBatch) -> None:
        """Block until every job in the batch finishes (success or failure)."""
        ...

    def parse(self, inputs: StepInputs, ctx: StepContext) -> StepResults:
        """Parse each output file into a :class:`~chemrefine.state.Structure`."""
        ...

    def normal_mode_sample(self, results: StepResults, ctx: StepContext) -> StepResults:
        """Expand a frequency-step result by displacing along imaginary modes."""
        ...


ENGINES: dict[str, type[CalculationEngine]] = {}
"""Registry mapping the YAML ``engine:`` string to a concrete engine class."""


def register(name: str) -> Callable[[type], type]:
    """Decorator: register ``cls`` under ``name`` in :data:`ENGINES`."""

    def decorator(cls: type) -> type:
        if name in ENGINES and ENGINES[name] is not cls:
            raise ValueError(f"engine {name!r} is already registered to {ENGINES[name]!r}")
        ENGINES[name] = cls
        return cls

    return decorator


def get_engine(name: str) -> CalculationEngine:
    """Look up an engine by name and return a fresh instance.

    Raises :class:`~chemrefine.errors.EngineNotFoundError` if ``name``
    is not in the registry.
    """
    engine_cls = ENGINES.get(name)
    if engine_cls is None:
        raise EngineNotFoundError(
            f"unknown engine {name!r}; registered: {sorted(ENGINES)}"
        )
    return engine_cls()

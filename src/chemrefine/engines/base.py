"""Engine Protocol contract and the ``ENGINES`` registry.

Every engine satisfies :class:`CalculationEngine` structurally — no
inheritance required. Engines register themselves via the :func:`register`
decorator at import time, so importing :mod:`chemrefine.engines`
populates the registry as a side effect.

The orchestrator only ever sees the :class:`CalculationEngine` Protocol
plus the :data:`ENGINES` dict. ORCA-specific imports, MLIP imports, etc.
never reach :mod:`chemrefine.pipeline` — that's how the orchestrator
stays engine-agnostic.

An engine is a **plugin**: it provides only engine-specific things. The generic
machinery is flat in ``chemrefine/`` — submission + budget
(:mod:`chemrefine.submit` + :mod:`chemrefine.slurm` + :mod:`chemrefine.throttle`),
structure assembly (:mod:`chemrefine.engines._assemble`), caching, and the step
lifecycle. Capability is declared by which Protocol an engine satisfies.

Adding a new engine
---------------------------------------------
**1. Pick a base** for ``engines/<name>/engine.py`` (decorate with ``@register("<name>")``):

* **Runs one job per structure** (a binary, or a user ``step{N}.py``) → subclass
  :class:`~chemrefine.engines._batch.BatchEngine`. You inherit ``prepare`` / ``submit``
  / ``parse``; supply only the primitives ``_build_input``, ``_parse_one``,
  ``_run_block``, ``_pal`` (+ ``_gpus`` if GPU-capable) and the ClassVars
  ``label`` / ``template_suffix`` / ``output_suffix`` / ``output_globs``. (ORCA;
  the template-driven engines subclass
  :class:`~chemrefine.engines._template_engine.TemplateScriptEngine`, itself a thin
  ``BatchEngine``.)
* **ORCA optimises using this engine's gradients** → subclass
  :class:`~chemrefine.engines.orca.extopt.engine.ExtOptOrcaEngine`: set the ClassVars
  ``backend`` / ``wrapper_filename`` and implement ``_server_cmd``, plus a
  :class:`~chemrefine.engines._backend_server.base.ComputeBackend` registered in
  ``engines/_backend_server/registry.py``. (See ``engines/pyscf/extopt_engine.py``.)
* **Not a per-structure batch** (the fake engine, ``mlip-train``) → implement the
  :class:`CalculationEngine` Protocol directly (``prepare`` / ``submit`` / ``parse`` /
  ``input_digest``).

**2. Capabilities** — NMS support is **not** a flag: implement
:class:`NmsCapableEngine`'s two hooks (``nms_input_info`` + ``read_frequencies``) and the
generic coordinator (:mod:`chemrefine.nms`) drives it; capability is detected via
``isinstance``.

**3. YAML knobs** → a Pydantic model in ``engines/<name>/options.py`` with a ``from_raw``
classmethod (mirror ``engines/mlip/options.py``); read it in the primitives.

**4. Register** by importing the class in ``engines/<name>/__init__.py`` and listing the
package in ``engines/__init__.py`` (registration is a side effect of that import). Legacy
YAML spellings map to the canonical name in :func:`chemrefine.config._normalize_legacy`.

**5. Resources** — an external binary reads its path from ``ctx.executables.get("<name>")``;
an importable backend ships as a ``pip install chemrefine[<name>]`` extra (imported lazily).

**6. Tests** go in ``tests/test_engines_<name>*.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from chemrefine.errors import EngineNotFoundError
from chemrefine.state import JobBatch, StepContext, StepInputs, StepResults


@runtime_checkable
class CalculationEngine(Protocol):
    """Structural contract every engine must satisfy.

    The lifecycle methods mirror the stages :func:`chemrefine.step.run_step` calls in
    order: :meth:`prepare` → :meth:`submit` (which blocks until the jobs finish) →
    :meth:`parse`. ``input_digest`` feeds the cache fingerprint. NMS is a *separate
    capability*: an engine that supports it also satisfies :class:`NmsCapableEngine`
    (detected via ``isinstance``), so there is no ``supports_nms`` flag to keep in sync.
    Per-structure batch engines get all of this from
    :class:`chemrefine.engines._batch.BatchEngine` and supply only primitives.
    """

    name: str

    def prepare(self, ctx: StepContext) -> StepInputs:
        """Write engine-specific input files for this step's seed structures."""
        ...

    def submit(self, inputs: StepInputs, ctx: StepContext) -> JobBatch:
        """Run the prepared inputs (locally or via SLURM); block until all finish."""
        ...

    def parse(self, inputs: StepInputs, ctx: StepContext) -> StepResults:
        """Parse each output file into a :class:`~chemrefine.state.Structure`."""
        ...

    def input_digest(self, ctx: StepContext) -> str:
        """Digest of this step's input (e.g. the resolved template), or ``""``.

        Folded into the cache fingerprint so editing the input in place re-runs the
        step — the template *basename* alone never changes the fingerprint.
        """
        ...


@dataclass(frozen=True)
class FrequencyData:
    """Imaginary frequencies + normal-mode tensor parsed from one output.

    The engine-specific half of NMS (see :class:`NmsCapableEngine`): ``imaginary`` maps
    a mode index to its frequency (cm⁻¹); ``None`` means the output had **no** frequency
    table at all — distinct from ``{}`` (a parsed table with zero imaginary modes), so a
    run that never produced frequencies can't be mistaken for a verified minimum.
    ``modes`` is the normal-mode displacement tensor (``None`` if absent).
    """

    imaginary: dict[int, float] | None
    modes: NDArray[np.float64] | None


@dataclass(frozen=True)
class NmsInputInfo:
    """What an engine's configured input does, for the generic NMS coordinator.

    ``is_transition_state`` drives the default NMS target (``ts`` vs ``minimum``);
    ``computes_frequencies`` gates NMS (a step that won't produce frequencies cannot be
    resolved). Both are read from the engine's own input (ORCA: its template keywords).
    """

    is_transition_state: bool
    computes_frequencies: bool


@runtime_checkable
class NmsCapableEngine(CalculationEngine, Protocol):
    """An engine that supports normal-mode sampling, via just two hooks.

    The two-round NMS algorithm — displacement, round-2 submission, resolution, retry —
    is engine-independent and lives in :mod:`chemrefine.nms`, which drives any engine
    through these two hooks plus the standard lifecycle. A new NMS-capable engine just
    implements these two methods; capability is detected with ``isinstance``.
    """

    def nms_input_info(self, ctx: StepContext) -> NmsInputInfo:
        """Introspect this step's configured input (TS search? computes frequencies?)."""
        ...

    def read_frequencies(
        self, structure_id: str, step_dir: Path, ctx: StepContext
    ) -> FrequencyData:
        """Read a structure's imaginary frequencies + normal modes from its output."""
        ...


ENGINES: dict[str, type[CalculationEngine]] = {}
"""Registry mapping the YAML ``engine:`` string to a concrete engine class.

Only **canonical** names live here. Old spellings (``mlff*``, ``dft``) are
rewritten to canonical names by the config normalizer
(:func:`chemrefine.config._normalize_legacy`) — the single place that knows the
legacy vocabulary — before any lookup, so the registry stays alias-free.
"""


def register(name: str) -> Callable[[type], type]:
    """Decorator: register ``cls`` under ``name`` in :data:`ENGINES`."""

    def decorator(cls: type) -> type:
        if name in ENGINES and ENGINES[name] is not cls:
            raise ValueError(f"engine {name!r} is already registered to {ENGINES[name]!r}")
        ENGINES[name] = cls
        return cls

    return decorator


def get_engine(name: str) -> CalculationEngine:
    """Look up an engine by its canonical name and return a fresh instance.

    Raises :class:`~chemrefine.errors.EngineNotFoundError` if ``name`` isn't
    registered. (Legacy spellings are normalized at config-parse time, so the
    name reaching here is already canonical.)
    """
    engine_cls = ENGINES.get(name)
    if engine_cls is None:
        raise EngineNotFoundError(f"unknown engine {name!r}; registered: {sorted(ENGINES)}")
    return engine_cls()

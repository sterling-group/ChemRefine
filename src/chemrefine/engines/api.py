"""The engine plugin contract + the ``ENGINES`` registry — the public face of ``engines/``.

This is the one module the flat pipeline imports from the engine subsystem: the
:class:`CalculationEngine` / :class:`NmsCapableEngine` Protocols, the :class:`JobExecutable`
provision contract, the DTOs (:class:`ParsedResult`, :class:`FrequencyData`,
:class:`NmsInputInfo`), and the registry (:data:`ENGINES` / :func:`register` /
:func:`get_engine`). Importing :mod:`chemrefine.engines` registers every bundled engine, so the
orchestrator looks engines up by name and never imports a concrete engine module.

chemrefine **declares** a small contract; an engine **provides** it. Reusable building blocks
(underscored, never edited to add an engine) live in the subsystem: :class:`._job.JobEngine`
(the per-structure lifecycle), :mod:`._execution` (the scheduler), :class:`._script.ScriptEngine`
(the user-Python-script kind), :mod:`._backend_server` (the ExtOpt server). Plugins are the
bare-named packages (``orca`` / ``mlip`` / ``pyscf``).

Adding a new engine — pick a *kind* and provide the requested pieces
---------------------------------------------------------------------

============================  ==============================  =====================================
Kind                          Base                            The engine provides
============================  ==============================  =====================================
Per-structure program         :class:`._job.JobEngine`        ``build_input`` / ``run_block`` /
(own input format, like ORCA)                                 ``parse_one`` / ``pal`` / ``gpus``
                                                              + the ClassVars
User Python script            :class:`._script.ScriptEngine`  usually only ``_template_vars``
ORCA-driven gradients         :class:`.orca.extopt.engine.    ``backend`` / ``wrapper_filename`` /
                              ExtOptOrcaEngine`               ``options_cls`` / ``calculator_cls``
Custom / non-job              :class:`CalculationEngine`       ``prepare`` / ``submit`` / ``parse``
(mlip-train, fake)            directly                        / ``input_digest``
============================  ==============================  =====================================

* **Capabilities** — NMS is **not** a flag: implement :class:`NmsCapableEngine`'s two hooks
  (``nms_input_info`` + ``read_frequencies``) and the generic coordinator (:mod:`chemrefine.nms`)
  drives it; capability is detected via ``isinstance``.
* **YAML knobs** — a Pydantic model in ``engines/<name>/options.py`` subclassing
  :class:`~chemrefine.engines._options.EngineOptions`; read it in the primitives.
* **Register** — import the class in ``engines/<name>/__init__.py`` and add the package to
  :mod:`chemrefine.engines`'s import list (registration is a side effect). Legacy YAML spellings
  map to the canonical name in :func:`chemrefine.config._normalize_legacy`.
* **Resources** — an external binary reads its path from ``ctx.executables.get("<name>")``; an
  importable backend ships as a ``pip install chemrefine[<name>]`` extra (imported lazily).
* **Tests** go in ``tests/test_engines_<name>*.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

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
    Per-structure *job* engines get all of this from
    :class:`chemrefine.engines._job.JobEngine` and supply only primitives.
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


@dataclass(frozen=True)
class ParsedResult:
    """One structure extracted from an engine's output, before lineage is assigned.

    The DTO a :class:`JobEngine`'s ``parse_one`` returns; the assembler
    (:func:`chemrefine.engines._job.build_structures`) turns a step's
    ``ParsedResult``s into :class:`~chemrefine.state.Structure` objects with IDs +
    parents. ``terminated`` / ``converged`` are run-status flags (``None`` when the
    engine doesn't report them, e.g. sidecar ensemble frames); a structure is a
    *failure* only when one is explicitly ``False``. The thermochemistry fields are
    populated only by a frequency run.
    """

    symbols: tuple[str, ...]
    positions: NDArray[np.float64]
    energy_hartree: float
    forces_ev_per_a: NDArray[np.float64] | None
    converged: bool | None = None
    terminated: bool | None = None
    gibbs_hartree: float | None = None
    enthalpy_hartree: float | None = None
    energy_zpe_hartree: float | None = None


@runtime_checkable
class JobExecutable(Protocol):
    """The primitives :func:`chemrefine.engines._execution.run_batch` needs from an engine.

    The narrow provision surface a :class:`~chemrefine.engines._job.JobEngine` exposes so
    the flat scheduler can run one job per structure without knowing the engine's type —
    Interface Segregation: ``run_batch`` depends on these six members, not the whole
    engine. ``JobEngine`` satisfies it structurally.
    """

    output_globs: ClassVar[tuple[str, ...]]

    def run_block(self, ctx: StepContext, inp_path: Path, out_path: Path) -> str:
        """The engine-specific bash that runs one job inside ``$WORK_DIR``."""
        ...

    def pal(self, ctx: StepContext) -> int:
        """Per-job core count (PAL) before the scheduler clamps it to ``max_cores``."""
        ...

    def gpus(self, ctx: StepContext) -> int:
        """GPUs one job needs (``0`` = CPU)."""
        ...

    def output_dirs(self, ctx: StepContext) -> tuple[str, ...]:
        """Scratch sub-directories to copy back wholesale."""
        ...

    def extra_header_fields(self, ctx: StepContext) -> tuple[tuple[str, object], ...]:
        """Engine-specific ``(key, value)`` rows for the runlog header."""
        ...


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

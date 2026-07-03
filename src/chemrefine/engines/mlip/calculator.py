"""Backend-agnostic factory for ASE calculators across MLIP libraries.

A *registry of small builders* pattern: each supported MLIP library
contributes a tiny ``_build_<name>`` function decorated with
:func:`register_backend`, living in its own module under
:mod:`chemrefine.engines.mlip.backends`. Adding a new MLIP backend is
**one small module plus one import line** — no edits to a god-class.

The public entry point is :func:`build_calculator` (used by users'
templates and by the ExtOpt server adapter). The :class:`MlipCalculator`
class is a thin wrapper that calls :func:`build_calculator` under the
hood, so the direct engine and the ExtOpt server share one selection path.

Two axes map the same way across libraries (the mace↔fairchem equivalence):
``task_name`` is the **method/head** that keys the registry; ``model_name`` is
the **weights** handed to that builder.

==================  ===============  =========================================
``task_name`` (key) backend          ``model_name`` (weights)
==================  ===============  =========================================
``omol`` … ``omc``  ``_build_fairchem`` FAIRChem checkpoint (``uma-s-1``/``esen-…``)
``mace_off``        ``mace_off``     MACE-OFF size (``small``/``medium``/``large``)
``mace_mp``         ``mace_mp``      MACE-MP size / named model
``mace_omol``       ``mace_omol``    MACE-OMOL size (``extra_large``)
``sevenn``          ``sevenn``       SevenNet id (``7net-0``)
``orb``             ``orb``          ORB loader (``orb_v3_…``)
``chgnet``          ``chgnet``       — (single model)
*(model_path set)*  ``custom_mace``  local MACE checkpoint
==================  ===============  =========================================

The 7 FAIRChem heads (``omol``/``omat``/``odac``/``oc20``/``oc22``/``oc25``/
``omc``) all register the one ``_build_fairchem`` and pass the head straight
through to ``FAIRChemCalculator``. Backend imports happen inside each builder so
this module imports cleanly even when the optional MLIP deps aren't installed;
:func:`build_calculator` converts a missing library into an error naming the
``chemrefine[mlip-<backend>]`` extra to install (every backend is a dedicated-env
extra — their torch/e3nn trees conflict; ``[mlip]`` = FAIRChem/UMA).

Adding a new backend = a new ``backends/<name>.py`` (decorated builder) listed in
``backends/__init__.py``. The decorator carries the backend's packaging metadata — the pip
extra that provides it, the pip distribution named in the actionable import error, and the
top-level module the provisioner probes — so the backend module is the **single** declaration
point (no central table anywhere) and a missing library is reported with the extra to
install::

    from chemrefine.engines.mlip.calculator import register_backend

    @register_backend(
        "my_mlip", extra="mlip-my_mlip", package="my-mlip-lib", import_name="my_mlip_library"
    )
    def _build_my_mlip(*, model_name="", device="cuda", **_):
        from my_mlip_library import MyCalculator

        return MyCalculator(model=model_name, device=device)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ase import Atoms

from chemrefine.engines.api import BackendRequirement
from chemrefine.engines.mlip.options import MlipOptions

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendSpec:
    """One registered MLIP backend: its builder plus the packaging metadata.

    ``extra`` is the pip extra that provides the backend (``chemrefine[<extra>]`` — also the
    managed-env name the provisioner uses); ``package`` is the pip distribution named in the
    actionable import error; ``import_name`` is the top-level module the provisioner probes
    with :func:`importlib.util.find_spec`. Declared once, at ``@register_backend`` in the
    backend's own module — there is no central backend table.
    """

    builder: Callable[..., Any]
    extra: str
    package: str
    import_name: str


_BACKENDS: dict[str, BackendSpec] = {}


def register_backend(
    name: str, *, extra: str, package: str, import_name: str
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a builder + its packaging metadata under ``name``; return the function.

    The builder takes keyword arguments (``task_name``, ``model_name``, ``device``,
    ``model_path``, plus a catch-all ``**_``) and returns an ASE calculator; its lazy backend
    import needs no guard — :func:`build_calculator` converts an ``ImportError`` into the
    actionable install hint from this metadata. Stackable: applying the decorator twice
    registers one function under two keys (the FAIRChem builder uses this — one function,
    every head).
    """

    def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        _BACKENDS[name] = BackendSpec(fn, extra=extra, package=package, import_name=import_name)
        return fn

    return _wrap


def backend_spec(task_name: str, model_path: str | Path | None = None) -> BackendSpec:
    """The :class:`BackendSpec` a task/model selection dispatches to.

    The one dispatch rule, shared by :func:`build_calculator` and the provisioner: a
    ``model_path`` selects ``custom_mace`` (a local checkpoint), else ``task_name`` keys the
    registry. Raises :class:`ValueError` listing known keys if nothing is registered.
    """
    key = "custom_mace" if model_path is not None else task_name
    spec = _BACKENDS.get(key)
    if spec is None:
        raise ValueError(
            f"unsupported MLIP backend: task_name={task_name!r} (known: {sorted(_BACKENDS)})"
        )
    return spec


def requirement_from_options(options: dict[str, Any] | None) -> BackendRequirement:
    """The :class:`BackendRequirement` a step's raw ``options`` imply.

    Reads the task/model selection tolerantly (same aliases the direct engine's template
    vars accept, defaults from :class:`~chemrefine.engines.mlip.options.MlipOptions`) and maps
    it through :func:`backend_spec` — so the env requirement always matches what
    :func:`build_calculator` would actually load.
    """
    raw = options or {}
    task = str(raw.get("task_name") or raw.get("task") or MlipOptions().task_name)
    spec = backend_spec(task, raw.get("model_path"))
    return BackendRequirement(extra=spec.extra, import_name=spec.import_name)


def build_calculator(
    *,
    task_name: str,
    model_name: str = "",
    device: str = "cuda",
    model_path: str | Path | None = None,
    **extra: Any,
) -> Any:
    """Dispatch to the builder registered for ``task_name``; return the calculator.

    ``task_name`` is the registry key (the method/head — a FAIRChem head,
    ``mace_off``…, ``chgnet``, ``sevenn``, ``orb``). ``model_name`` is the weights
    handed to it. A ``model_path`` selects ``custom_mace`` (a local checkpoint).
    Raises :class:`ValueError` listing known keys if nothing is registered; a missing
    backend library surfaces as an ``ImportError`` naming the extra to install.
    """
    spec = backend_spec(task_name, model_path)
    try:
        return spec.builder(
            task_name=task_name,
            model_name=model_name,
            device=device,
            model_path=model_path,
            **extra,
        )
    except ImportError as exc:
        raise ImportError(
            f"this MLIP backend needs '{spec.package}' — install it with "
            f"`pip install chemrefine[{spec.extra}]` (in its own environment)."
        ) from exc


# ---------------------------------------------------------------------------
# Thin wrapper class (shared by the direct engine and the ExtOpt adapter)
# ---------------------------------------------------------------------------


class MlipCalculator:
    """Thin wrapper around :func:`build_calculator` for callers that want a class.

    The ExtOpt server adapter
    (:class:`~chemrefine.engines.mlip.extopt_calc.MlipExtOptCalculator`) and a
    direct ``step{N}.py`` template can both construct a calculator via this one
    object API, so both paths share the same backend selection. New code may
    call :func:`build_calculator` directly.
    """

    def __init__(
        self,
        *,
        model_name: str,
        task_name: str = "omol",
        device: str = "cuda",
        model_path: str | Path | None = None,
    ):
        self.model_name = model_name
        self.task_name = task_name
        self.device = device
        self.model_path = Path(model_path) if model_path else None
        self.calculator = build_calculator(
            task_name=task_name,
            model_name=model_name,
            device=device,
            model_path=self.model_path,
        )

    # -- inference ---------------------------------------------------------

    def single_point(self, atoms: Atoms) -> tuple[float, list[list[float]]]:
        """Return ``(energy_eV, gradient_eV_per_A)`` for one geometry."""
        atoms.calc = self.calculator
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        gradient = (-forces).tolist()
        return energy, gradient

    def optimize(self, atoms: Atoms, *, fmax: float = 0.03, steps: int = 200) -> Atoms:
        """In-process LBFGS optimisation; returns the relaxed ``atoms``."""
        from ase.optimize import LBFGS

        atoms.calc = self.calculator
        # ase accepts logfile=None (no log) at runtime; its annotation says IO|str.
        LBFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)  # type: ignore[arg-type]
        return atoms

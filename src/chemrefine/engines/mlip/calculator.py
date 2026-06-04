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
this module imports cleanly even when the optional MLIP deps aren't installed.

Adding a new backend = a new ``backends/<name>.py`` (decorated builder) listed in
``backends/__init__.py``::

    from chemrefine.engines.mlip.calculator import register_backend

    @register_backend("my_mlip")
    def _build_my_mlip(*, model_name="", device="cuda", **_):
        from my_mlip_library import MyCalculator
        return MyCalculator(model=model_name, device=device)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ase import Atoms

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

_BACKEND_BUILDERS: dict[str, Callable[..., Any]] = {}


def register_backend(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a builder under ``name``; return the unchanged function.

    The builder takes keyword arguments (``task_name``, ``model_name``,
    ``device``, ``model_path``, plus a catch-all ``**_``) and returns an ASE
    calculator. Stackable: applying the decorator twice registers one function
    under two keys (the FAIRChem builder uses this — one function, every head).
    """
    def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        _BACKEND_BUILDERS[name] = fn
        return fn
    return _wrap


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
    Raises :class:`ValueError` listing known keys if nothing is registered.
    """
    key = "custom_mace" if model_path is not None else task_name
    builder = _BACKEND_BUILDERS.get(key)
    if builder is None:
        raise ValueError(
            f"unsupported MLIP backend: task_name={task_name!r} "
            f"model_name={model_name!r} (known: {sorted(_BACKEND_BUILDERS)})"
        )
    return builder(
        task_name=task_name,
        model_name=model_name,
        device=device,
        model_path=model_path,
        **extra,
    )


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

    def single_point(
        self, atoms: Atoms
    ) -> tuple[float, list[list[float]]]:
        """Return ``(energy_eV, gradient_eV_per_A)`` for one geometry."""
        atoms.calc = self.calculator
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        gradient = (-forces).tolist()
        return energy, gradient

    def optimize(
        self, atoms: Atoms, *, fmax: float = 0.03, steps: int = 200
    ) -> Atoms:
        """In-process LBFGS optimisation; returns the relaxed ``atoms``."""
        from ase.optimize import LBFGS

        atoms.calc = self.calculator
        LBFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)
        return atoms

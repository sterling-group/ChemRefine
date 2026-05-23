"""Backend-agnostic factory for ASE calculators across MLFF libraries.

A *registry of small builders* pattern: each supported MLFF library
contributes a tiny ``_build_<name>`` function decorated with
:func:`register_backend`. Adding a new MLFF backend is **one small
function plus one decorator line** — no edits to a god-class
required.

The public entry point is :func:`build_calculator` (used by users'
templates and by the ExtOpt server adapter). The :class:`MlffCalculator`
class is a thin compatibility wrapper that calls
:func:`build_calculator` under the hood.

Built-in backends:

* ``mace_off``, ``mace_mp``, ``mace_omol`` — MACE foundation models.
* ``custom_mace`` — caller-supplied MACE checkpoint via ``model_path``.
* ``omol``, ``omat``, ``odac``, ``oc20``, ``omc`` — FAIRChem UMA tasks.
* ``chgnet`` — CHGNet.
* ``sevenn`` — Seven-Net.
* ``orb`` — ORB / Orbital Materials.

Adding a new backend looks like:

.. code-block:: python

   from chemrefine.engines.mlff.calculator import register_backend

   @register_backend("my_mlff")
   def _build_my_mlff(*, model_name, device="cuda", **_):
       from my_mlff_library import MyCalculator
       return MyCalculator(model=model_name, device=device)

Backend imports happen inside each builder so this module imports
cleanly even when the optional MLFF dependencies aren't installed.
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

    The builder must accept keyword arguments and return an ASE
    calculator. The accepted kwargs are at least ``model_name`` and
    ``device`` (catch-all ``**_`` lets the caller pass extra knobs
    without breaking the dispatcher).
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

    Raises :class:`ValueError` if no backend is registered. Pattern
    matches the YAML-driven path: ``step.options['task_name']`` keys
    into the registry.
    """
    # Custom-MACE shortcut: the user supplied an explicit checkpoint.
    if model_path is not None:
        builder = _BACKEND_BUILDERS.get("custom_mace")
        if builder is None:
            raise ValueError("custom_mace backend not registered")
        return builder(
            model_name=model_name, device=device, model_path=model_path, **extra
        )

    # Resolve the backend by exact task name, then by a small set of
    # prefix aliases that match the legacy MLFFCalculator dispatch
    # rules (FAIRChem tasks all map to the "omol" / "uma" family).
    resolved = _resolve_alias(task_name=task_name, model_name=model_name)
    builder = _BACKEND_BUILDERS.get(resolved)
    if builder is None:
        raise ValueError(
            f"unsupported MLFF backend: task_name={task_name!r} "
            f"model_name={model_name!r} (known: {sorted(_BACKEND_BUILDERS)})"
        )
    return builder(model_name=model_name, device=device, **extra)


def _resolve_alias(*, task_name: str, model_name: str) -> str:
    """Return the registry key for a given ``task_name`` / ``model_name`` pair.

    Returns ``task_name`` unchanged if it's already a registered key
    or matches one of the FAIRChem task families; otherwise falls
    back to a model-name prefix match for SevenNet / ORB.
    """
    if task_name in _BACKEND_BUILDERS:
        return task_name
    if task_name.startswith(("omol", "omat", "odac", "uma", "fairchem", "oc20", "omc")):
        return "omol"  # FAIRChem family share one builder
    if model_name.startswith("sevenn"):
        return "sevenn"
    if model_name.startswith("orb"):
        return "orb"
    return task_name


# ---------------------------------------------------------------------------
# Built-in backends
# ---------------------------------------------------------------------------


@register_backend("mace_off")
def _build_mace_off(*, model_name: str, device: str = "cuda", **_: Any) -> Any:
    """MACE-OFF foundation model."""
    from mace.calculators import mace_off

    return mace_off(model=model_name, device=device)


@register_backend("mace_mp")
def _build_mace_mp(*, model_name: str, device: str = "cuda", **_: Any) -> Any:
    """MACE-MP foundation model."""
    from mace.calculators import mace_mp

    return mace_mp(model=model_name, device=device)


@register_backend("mace_omol")
def _build_mace_omol(*, device: str = "cuda", **_: Any) -> Any:
    """MACE-OMOL foundation model."""
    from mace.calculators import mace_omol

    return mace_omol(device=device)


@register_backend("custom_mace")
def _build_custom_mace(
    *,
    model_name: str = "",
    device: str = "cuda",
    model_path: str | Path,
    **_: Any,
) -> Any:
    """User-supplied MACE checkpoint at ``model_path``."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"custom MACE model not found: {path}")
    from mace.calculators import MACECalculator

    return MACECalculator(model_path=str(path), device=device)


@register_backend("omol")
def _build_fairchem(*, model_name: str, device: str = "cuda", **_: Any) -> Any:
    """FAIRChem / UMA family (omol, omat, odac, oc20, omc all route here)."""
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    predictor = pretrained_mlip.get_predict_unit(
        model_name=model_name, device=device
    )
    return FAIRChemCalculator(predictor, task_name="omol")


@register_backend("chgnet")
def _build_chgnet(
    *,
    model_name: str = "",
    device: str = "cuda",
    model_path: str | Path | None = None,
    **_: Any,
) -> Any:
    """CHGNet."""
    from chgnet.calculators import CHGNetCalculator
    from chgnet.model import CHGNet

    model = CHGNet.load(str(model_path)) if model_path else CHGNet.load()
    return CHGNetCalculator(model=model)


@register_backend("sevenn")
def _build_sevenn(*, model_name: str, device: str = "cuda", **_: Any) -> Any:
    """Seven-Net."""
    from sevenn.calculator import SevenNetCalculator

    return SevenNetCalculator(model=model_name, device=device)


@register_backend("orb")
def _build_orb(*, model_name: str, device: str = "cuda", **_: Any) -> Any:
    """ORB / Orbital Materials — currently routes through Seven-Net's CLI."""
    return _build_sevenn(model_name=model_name, device=device)


# ---------------------------------------------------------------------------
# Backwards-compatible wrapper class
# ---------------------------------------------------------------------------


class MlffCalculator:
    """Thin wrapper around :func:`build_calculator` for callers that want a class.

    Kept so existing in-process callers (the direct ``MlffEngine``) and
    the ExtOpt server adapter can construct calculators via the same
    object API. New code should call :func:`build_calculator` directly.
    """

    def __init__(
        self,
        *,
        model_name: str,
        task_name: str = "mace_off",
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

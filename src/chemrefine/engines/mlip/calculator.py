"""Backend-agnostic factory for ASE calculators across MLIP libraries.

The public face of the MLIP backends: :func:`build_calculator` dispatches a task/model
selection to the right library's builder, and :class:`MlipCalculator` wraps it for callers
that want an object. Users' ``step{N}.py`` templates and the ExtOpt server adapter both come
through here, so the direct engine and the gradient server share one selection path.

*Which* library a selection resolves to, and what it can do, belongs to
:mod:`chemrefine.engines.mlip.registry` — one registry shared with training, so ``task_name``
names one library whether a step runs a model or trains one. This module only builds.

The knob rules, not a roster (the live roster is the registry's, rendered on the docs'
generated backends table): ``task_name`` is the **only** knob that selects a library;
``model_name`` names a released set of its weights, in whatever spelling that library
uses; ``model_path`` says to take the weights from a local file instead — it selects
nothing, so a model fine-tuned by an ``mlip-train`` step is run by naming the library
that trained it, the same word in the same place as for a released one.

The dispatch below is where a selection becomes a vetted :class:`~chemrefine.engines.
mlip.registry.CalculatorSpec`: the checkpoint's existence is checked **here, once** —
a missing file names the ``task_name`` the user wrote, before any library imports —
and the defaults come off :class:`~chemrefine.engines.mlip.options.MlipOptions`'s own
fields, so a template call and a YAML step cannot disagree about what "unspecified"
means (the ``device`` default is the model's ``cpu``: the floor that always runs).

Backend imports happen inside each builder so this module imports cleanly even when the
optional MLIP deps aren't installed; :func:`build_calculator` converts a missing library into
an error naming the ``chemrefine[mlip-<backend>]`` extra to install (every backend is a
dedicated-env extra — their torch/e3nn trees conflict; ``[mlip]`` = FAIRChem/UMA).

Adding a backend is one dropped-in module under
:mod:`chemrefine.engines.mlip.backends` — see :mod:`chemrefine.engines.mlip.registry` for the
shape.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from ase import Atoms

from chemrefine.engines.mlip.options import MlipOptions
from chemrefine.engines.mlip.registry import CalculatorSpec, backend_spec, calculator_for

logger = logging.getLogger(__name__)

DEFAULT_TASK: str = MlipOptions.model_fields["task_name"].default
"""The library a caller who names none gets — read off the options model, not restated.

This function is public API a user's ``step{N}.py`` may call without going through the YAML at
all, so it needs a default of its own; taking it from the field means the template and the
config cannot come to disagree about what "unspecified" runs. The other defaults below are
read the same way, for the same reason — the ``device`` fallback in particular was once a
``"cuda"`` literal here while the model said ``"cpu"``, the exact two-readers drift the
model-fields technique exists to prevent."""

_DEFAULT_MODEL_NAME: str = MlipOptions.model_fields["model_name"].default
_DEFAULT_DEVICE: str = MlipOptions.model_fields["device"].default


def build_calculator(
    *,
    task_name: str = DEFAULT_TASK,
    model_name: str = _DEFAULT_MODEL_NAME,
    device: str = _DEFAULT_DEVICE,
    model_path: str | Path | None = None,
    charge: int | None = None,
    multiplicity: int | None = None,
) -> Any:
    """Vet the selection into a :class:`CalculatorSpec` and hand it to the one builder.

    ``task_name`` is the **only** thing that selects a library. ``model_name`` is the
    weights handed to it, and ``model_path`` says to take those weights from a local file
    instead — checked for existence *here*, once, so a mistyped checkpoint is refused
    naming the ``task_name`` the user wrote, before any library imports. Inferring the
    library from the checkpoint is deliberately impossible: a ``.pt`` file does not say
    which loader it belongs to, and handing it to the wrong one fails as a tensor-shape
    error deep inside that library rather than as anything naming the actual mistake.

    ``charge`` / ``multiplicity`` are optional and reach the spec untouched — the door for
    charge-aware libraries; a charge-blind builder never reads them. There is deliberately
    no ``**extra``: a knob no field names has nowhere to hide, which is the property the
    spec exists for (a backend-specific knob starts life as a declared option instead).

    Raises :class:`~chemrefine.errors.ConfigError` listing known keys if nothing is
    registered; a missing backend *library* surfaces as an ``ImportError`` naming the
    extra to install.
    """
    registered = backend_spec(task_name)
    builder = calculator_for(task_name)
    weights: Path | None = None
    if model_path:
        weights = Path(model_path)
        if not weights.is_file():
            raise FileNotFoundError(f"{task_name} checkpoint not found: {weights}")
    spec = CalculatorSpec(
        task_name=task_name,
        model_name=model_name,
        device=device,
        weights=weights,
        charge=charge,
        multiplicity=multiplicity,
    )
    try:
        return builder(spec)
    except ImportError as exc:
        raise ImportError(
            f"this MLIP backend needs '{registered.package}' — install it with "
            f"`pip install chemrefine[{registered.extra}]` (in its own environment)."
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
        model_name: str = _DEFAULT_MODEL_NAME,
        task_name: str = DEFAULT_TASK,
        device: str = _DEFAULT_DEVICE,
        model_path: str | Path | None = None,
        charge: int | None = None,
        multiplicity: int | None = None,
    ):
        """``task_name`` selects the library; ``model_path`` only says where its weights are.

        ``charge`` / ``multiplicity`` are optional and additive — the charge-aware door,
        ignored by libraries without a charge channel. Everything else is the shape the
        shipped templates have always called.
        """
        self.model_name = model_name
        self.task_name = task_name
        self.device = device
        self.model_path = Path(model_path) if model_path else None
        self.calculator = build_calculator(
            task_name=task_name,
            model_name=model_name,
            device=device,
            model_path=self.model_path,
            charge=charge,
            multiplicity=multiplicity,
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
        # Named rather than passed as `None`: ase's own `IOContext.openfile` turns `None`
        # into `open(os.devnull)`, so this is the same file by the shorter route — and it
        # is the `str` the signature asks for.
        LBFGS(atoms, logfile=os.devnull).run(fmax=fmax, steps=steps)
        return atoms

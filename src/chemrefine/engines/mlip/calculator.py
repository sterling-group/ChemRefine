"""Backend-agnostic factory for ASE calculators across MLIP libraries.

The public face of the MLIP backends: :func:`build_calculator` dispatches a task/model
selection to the right library's builder, and :class:`MlipCalculator` wraps it for callers
that want an object. Users' ``step{N}.py`` templates and the ExtOpt server adapter both come
through here, so the direct engine and the gradient server share one selection path.

*Which* library a selection resolves to, and what it can do, belongs to
:mod:`chemrefine.engines.mlip.registry` — one registry shared with training, so ``task_name``
names one library whether a step runs a model or trains one. This module only builds.

Two axes map the same way across libraries (the mace↔fairchem equivalence):
``task_name`` is the **method/head** that keys the registry; ``model_name`` is
the **weights** handed to that builder.

==================  ===================  =========================================
``task_name`` (key) backend              ``model_name`` (weights)
==================  ===================  =========================================
``omol`` … ``omc``  ``_build_fairchem``  FAIRChem checkpoint (``uma-s-1p2``/``esen-…``)
``mace_off``        ``_build_mace``      MACE-OFF size (``small``/``medium``/``large``)
``mace_mp``         ``_build_mace``      MACE-MP size / named model
``mace_omol``       ``_build_mace``      MACE-OMOL size (``extra_large``)
``sevenn``          ``sevenn``           SevenNet id (``7net-0``)
``orb``             ``orb``              ORB loader (``orb_v3_…``)
``chgnet``          ``chgnet``           — (single model)
==================  ===================  =========================================

``model_path`` is the third knob and appears in no row, because it selects nothing: it is
handed to whichever builder ``task_name`` chose, and that library loads the file itself. A
model fine-tuned by an ``mlip-train`` step is therefore run by naming the library that trained
it — the same word, in the same place, as for a released one.

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
from chemrefine.engines.mlip.registry import backend_spec, calculator_for

logger = logging.getLogger(__name__)

DEFAULT_TASK: str = MlipOptions.model_fields["task_name"].default
"""The library a caller who names none gets — read off the options model, not restated.

This function is public API a user's ``step{N}.py`` may call without going through the YAML at
all, so it needs a default of its own; taking it from the field means the template and the
config cannot come to disagree about what "unspecified" runs."""


def build_calculator(
    *,
    task_name: str = DEFAULT_TASK,
    model_name: str = "",
    device: str = "cuda",
    model_path: str | Path | None = None,
    **extra: Any,
) -> Any:
    """Dispatch to the builder registered for ``task_name``; return the calculator.

    ``task_name`` names the library (a FAIRChem head, ``mace_off``…, ``chgnet``, ``sevenn``,
    ``orb``) and is the **only** thing that selects one. ``model_name`` is the weights handed
    to it, and ``model_path`` says to take those weights from a local file instead — both are
    passed to the builder, which loads whichever it was given with *its own* library.

    So running a fine-tuned model means naming the library that produced it — explicitly.
    Inferring the library from the checkpoint is not possible: a ``.pt`` file does not say
    which loader it belongs to, and handing it to the wrong one fails as a tensor-shape error
    deep inside that library rather than as anything naming the actual mistake.

    Raises :class:`~chemrefine.errors.ConfigError` listing known keys if nothing is
    registered; a missing backend *library* surfaces as an ``ImportError`` naming the extra
    to install.
    """
    spec = backend_spec(task_name)
    builder = calculator_for(task_name)
    try:
        return builder(
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
        model_name: str = "",
        task_name: str = DEFAULT_TASK,
        device: str = "cuda",
        model_path: str | Path | None = None,
    ):
        """``task_name`` selects the library; ``model_path`` only says where its weights are."""
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
        # Named rather than passed as `None`: ase's own `IOContext.openfile` turns `None`
        # into `open(os.devnull)`, so this is the same file by the shorter route — and it
        # is the `str` the signature asks for.
        LBFGS(atoms, logfile=os.devnull).run(fmax=fmax, steps=steps)
        return atoms

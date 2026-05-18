"""Backend-agnostic wrapper that produces ASE calculators for MLFF backends.

Supported backends — selected by the ``task_name`` (or model name)
field in the YAML ``options:`` block:

* MACE      — ``task_name`` starts with ``mace``
* FAIRChem  — ``task_name`` starts with ``omol`` / ``omat`` / ``odac`` /
  ``uma`` / ``fairchem``
* CHGNet    — ``task_name`` starts with ``chgnet``
* SevenN    — ``model_name`` starts with ``sevenn``
* ORB       — ``model_name`` starts with ``orb``
* Custom MACE — caller passes an explicit ``model_path``

Backend imports happen lazily inside the corresponding setup methods so
ChemRefine still imports cleanly when an optional MLFF dependency
isn't installed.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ase import Atoms

logger = logging.getLogger(__name__)


class MlffCalculator:
    """Build and hold the right ASE calculator for the selected MLFF backend."""

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
        self.calculator = self._build()

    # -- backend dispatch --------------------------------------------------

    def _build(self):
        """Return the configured ASE calculator instance."""
        if self.model_path is not None:
            return self._build_custom_mace()
        if self.task_name.startswith("mace"):
            return self._build_mace()
        if self.task_name.startswith(("omol", "omat", "odac", "uma", "fairchem")):
            return self._build_fairchem()
        if self.task_name.startswith("chgnet"):
            return self._build_chgnet()
        if self.model_name.startswith("sevenn"):
            return self._build_sevenn()
        if self.model_name.startswith("orb"):
            return self._build_orb()
        raise ValueError(
            f"unsupported MLFF backend: task_name={self.task_name!r} "
            f"model_name={self.model_name!r}"
        )

    def _build_mace(self):
        """Load a pre-trained MACE model (mace_off / mace_mp / mace_omol)."""
        if self.task_name == "mace_off":
            from mace.calculators import mace_off

            return mace_off(model=self.model_name, device=self.device)
        if self.task_name == "mace_mp":
            from mace.calculators import mace_mp

            return mace_mp(model=self.model_name, device=self.device)
        if self.task_name == "mace_omol":
            from mace.calculators import mace_omol

            return mace_omol(device=self.device)
        raise ValueError(f"unsupported MACE task: {self.task_name!r}")

    def _build_custom_mace(self):
        """Load a user-supplied MACE model file."""
        if self.model_path is None or not self.model_path.is_file():
            raise FileNotFoundError(f"custom MACE model not found: {self.model_path}")
        from mace.calculators import MACECalculator

        return MACECalculator(model_path=str(self.model_path), device=self.device)

    def _build_fairchem(self):
        """Load a FAIRChem / UMA pretrained predictor."""
        from fairchem.core import FAIRChemCalculator, pretrained_mlip

        predictor = pretrained_mlip.get_predict_unit(
            model_name=self.model_name, device=self.device
        )
        return FAIRChemCalculator(predictor, task_name=self.task_name)

    def _build_chgnet(self):
        """Load a CHGNet model."""
        from chgnet.calculators import CHGNetCalculator
        from chgnet.model import CHGNet

        model = CHGNet.load(str(self.model_path)) if self.model_path else CHGNet.load()
        return CHGNetCalculator(model=model)

    def _build_sevenn(self):
        """Load a SevenN model."""
        from sevenn.calculator import SevenNetCalculator

        return SevenNetCalculator(model=self.task_name, device=self.device)

    def _build_orb(self):
        """Load an ORB model — uses the SevenN-style entry point in v3."""
        return self._build_sevenn()

    # -- inference ---------------------------------------------------------

    def single_point(self, atoms: Atoms) -> tuple[float, list[list[float]]]:
        """Return ``(energy_eV, gradient_eV_per_A)`` for one geometry.

        Energy is taken from ``atoms.get_potential_energy()`` and forces
        from ``atoms.get_forces()`` (gradient is the negative of forces).
        """
        atoms.calc = self.calculator
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        gradient = (-forces).tolist()
        return energy, gradient

    def optimize(self, atoms: Atoms, *, fmax: float = 0.03, steps: int = 200) -> Atoms:
        """In-process LBFGS optimisation; returns the relaxed ``atoms``."""
        from ase.optimize import LBFGS

        atoms.calc = self.calculator
        LBFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)
        return atoms

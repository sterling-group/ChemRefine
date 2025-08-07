import logging
from typing import List, Tuple, Optional
from ase import Atoms
from .utils import Utility
from pathlib import Path
import time
import socket
import pickle
import torch
from multiprocessing import Process

class MLFFCalculator:
    """Flexible MLFF calculator supporting MACE, UMA (FairChem), CHGNet, etc."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        model_path: Optional[str] = None,
        task_name: str = "mace_off"
    ):
        self.model_name = model_name
        self.device = device
        self.task_name = task_name
        self.model_path = model_path
        self.calculator = self._setup_calculator()

    def _setup_calculator(self):
        """Initialize and return the correct ASE calculator based on task_name."""
        if self.model_path:
            logging.info(f"Using custom MACE model from path: {self.model_path} on device: {self.device}")
            return self._setup_custom_mace()
        if self.task_name.startswith("mace"):
            logging.info(f"Using MACE model: {self.task_name} on device: {self.device}")
            return self._setup_mace()
        elif self.task_name.startswith(("omol", "omat", "odac", "uma", "fairchem")):
            logging.info(f"Using FAIRChem model: {self.task_name} on device: {self.device}")
            return self._setup_fairchem()
        elif self.task_name.startswith("chgnet"):
            logging.info(f"Using CHGNet model on device: {self.device}")
            return self._setup_chgnet()
        else:
            raise ValueError(f"Unsupported task name: {self.task_name}")

    def _setup_mace(self):
        from mace.calculators import mace_off, mace_mp
        if self.task_name == "mace_off":
            logging.info(f"Using MACE OFF model: {self.model_name} on device: {self.device}")   
            return mace_off(model=self.model_name, device=self.device)
        elif self.task_name == "mace_mp":
            logging.info(f"Using MACE MP model: {self.model_name} on device: {self.device}")
            return mace_mp(model=self.model_name, device=self.device)
        else:
            raise ValueError(f"Unsupported MACE task name: {self.task_name}")

    def _setup_fairchem(self):
        from fairchem.core import pretrained_mlip, FAIRChemCalculator
        predictor = pretrained_mlip.get_predict_unit(
            model_name=self.model_name,
            device=self.device,
        )
        logging.info(f"Using FAIRChem model: {self.model_name} on device: {self.device}")
        return FAIRChemCalculator(predictor, task_name=self.task_name)

    def _setup_chgnet(self):
        from chgnet.model import CHGNet
        from chgnet.calculators import CHGNetCalculator
        model = CHGNet.load(self.model_path) if self.model_path else CHGNet.load()
        return CHGNetCalculator(model=model)

    def get_single_point(self, atoms: Atoms) -> Tuple[float, List[float]]:
        """Compute energy and gradient using the configured MLFF."""
        atoms.calc = self.calculator
        from chemrefine.utils_extopt import process_output
        return process_output(atoms)

    def _setup_custom_mace(self):
        from pathlib import Path
        from mace.calculators import MACECalculator

        model_file = Path(self.model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Custom MACE model file does not exist: {self.model_path}")

        logging.info(f"Using custom MACE model from path: {self.model_path} on device: {self.device}")
        return MACECalculator(model_path=self.model_path, device=self.device)

    def calculate(self, atoms: Atoms, fmax: float = 0.03, steps: int = 200) -> Atoms:
        """Optional: geometry optimization using LBFGS."""
        atoms.calc = self.calculator
        from ase.optimize import LBFGS
        optimizer = LBFGS(atoms, logfile=None)
        optimizer.run(fmax=fmax, steps=steps)
        return atoms


    
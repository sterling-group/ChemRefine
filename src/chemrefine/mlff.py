import logging
from typing import List, Tuple, Optional
from ase import Atoms
from .utils import Utility
from pathlib import Path
import nump as np

#GLOBALS
HARTREE_TO_EV = 27.211386245988
HARTREE_BOHR_TO_EV_A = 51.422067  # if you decide to convert here

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

class MLFFTrainer:
    """
    Handle training of MLFF models inside ChemRefine.

    Parameters
    ----------
    step_id : int
        Current workflow step number.
    step_dir : str
        Directory where training outputs (logs, checkpoints) are stored.
    template_dir : str
        Path to training templates (e.g., mace_train.yaml).
    trainer_cfg : dict
        Training parameters from workflow YAML (e.g., model_name, task_name).
    coordinates : list
        Coordinates passed from previous step.
    structure_ids : list
        IDs corresponding to the structures.
    """

    def __init__(self, step_id, step_dir, template_dir, trainer_cfg,
                 coordinates=None, structure_ids=None):
        self.step_id = step_id
        self.step_dir = step_dir
        self.template_dir = template_dir
        self.trainer_cfg = trainer_cfg
        self.coordinates = coordinates
        self.structure_ids = structure_ids

    
    def prepare_inputs(self):
        """
        Convert coordinates, energies, and forces into extended XYZ train/test files.
        """
        if not self.coordinates or not self.structure_ids:
            raise ValueError("No structures available for MLFF training.")

        os.makedirs(self.step_dir, exist_ok=True)

        atoms_list = []
        for coords, energy, forces in zip(self.coordinates, self.trainer_cfg.get("energies", []), self.trainer_cfg.get("forces", [])):
            atoms = _to_atoms(coords, energy, forces)
            atoms_list.append(atoms)

        # Split train/test
        n_total = len(atoms_list)
        n_test = max(1, int(0.1 * n_total))
        train_structs = atoms_list[:-n_test]
        test_structs = atoms_list[-n_test:]

        # Optionally convert energies/forces here to eV/eVÅ before writing
        for atoms in atoms_list:
            if "DFT_energy" in atoms.info:
                atoms.info["DFT_energy"] *= HARTREE_TO_EV
            if "DFT_Forces" in atoms.arrays:
                atoms.arrays["DFT_Forces"] *= HARTREE_BOHR_TO_EV_A

        train_path = os.path.join(self.step_dir, "mace_train.xyz")
        test_path = os.path.join(self.step_dir, "mace_test.xyz")

        write(train_path, train_structs, format="extxyz")
        write(test_path, test_structs, format="extxyz")

        logging.info(f"Wrote {len(train_structs)} training structures to {train_path}")
        logging.info(f"Wrote {len(test_structs)} test structures to {test_path}")

        return train_path, test_path

    def write_training_config(self):
        """Generate training config from template and trainer_cfg."""
        # TODO: read mace_train.yaml (or chgnet_train.yaml) from template_dir,
        # merge with trainer_cfg, and write into step_dir.
        pass

    def submit(self):
        """Submit training job (SLURM/local)."""
        # TODO: call Filesubmit or direct subprocess
        pass

    def monitor(self):
        """Monitor until training completes."""
        # TODO: implement job monitoring
        pass

    def run(self):
        """Main entry point: orchestrates full training step."""
        self.prepare_inputs()
        self.write_training_config()
        self.submit()
        self.monitor()
        # Model path is defined inside the template; return as hint
        return self.trainer_cfg.get("save_dir", self.step_dir)

def _to_atoms(coords, energy, forces):
        """
        Convert parsed ORCA outputs into an ASE Atoms object.

        Parameters
        ----------
        coords : list[list[str]]
            Each row = [element, x, y, z] in Å.
        energy : float
            Total energy in Hartree.
        forces : np.ndarray
            Forces in Hartree/Bohr, shape (N_atoms, 3).

        Returns
        -------
        Atoms
            ASE Atoms object with energy and forces attached.
        """
        symbols = [row[0] for row in coords]
        positions = np.array([[float(x), float(y), float(z)] for _, x, y, z in coords], dtype=float)

        atoms = Atoms(symbols=symbols, positions=positions)

        if energy is not None:
            atoms.info["DFT_energy"] = energy
        if isinstance(forces, np.ndarray) and forces.size > 0:
            atoms.arrays["DFT_Forces"] = forces

        return atoms    
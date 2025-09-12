import logging
from typing import List, Tuple, Optional
from ase import Atoms
from ase.io import write
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import os
import yaml
import time

# GLOBALS
HARTREE_TO_EV = 27.211386245988
HARTREE_BOHR_TO_EV_A = 51.422067  # if you decide to convert here


class MLFFCalculator:
    """Flexible MLFF calculator supporting MACE, UMA (FairChem), CHGNet, etc."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        model_path: Optional[str] = None,
        task_name: str = "mace_off",
    ):
        self.model_name = model_name
        self.device = device
        self.task_name = task_name
        self.model_path = model_path
        self.calculator = self._setup_calculator()

    def _setup_calculator(self):
        """Initialize and return the correct ASE calculator based on task_name."""
        if self.model_path:
            logging.info(
                f"Using custom MACE model from path: {self.model_path} on device: {self.device}"
            )
            return self._setup_custom_mace()
        if self.task_name.startswith("mace"):
            logging.info(f"Using MACE model: {self.task_name} on device: {self.device}")
            return self._setup_mace()
        elif self.task_name.startswith(("omol", "omat", "odac", "uma", "fairchem")):
            logging.info(
                f"Using FAIRChem model: {self.task_name} on device: {self.device}"
            )
            return self._setup_fairchem()
        elif self.task_name.startswith("chgnet"):
            logging.info(f"Using CHGNet model on device: {self.device}")
            return self._setup_chgnet()
        else:
            raise ValueError(f"Unsupported task name: {self.task_name}")

    def _setup_mace(self):
        from mace.calculators import mace_off, mace_mp

        if self.task_name == "mace_off":
            logging.info(
                f"Using MACE OFF model: {self.model_name} on device: {self.device}"
            )
            return mace_off(model=self.model_name, device=self.device)
        elif self.task_name == "mace_mp":
            logging.info(
                f"Using MACE MP model: {self.model_name} on device: {self.device}"
            )
            return mace_mp(model=self.model_name, device=self.device)
        else:
            raise ValueError(f"Unsupported MACE task name: {self.task_name}")

    def _setup_fairchem(self):
        from fairchem.core import pretrained_mlip, FAIRChemCalculator

        predictor = pretrained_mlip.get_predict_unit(
            model_name=self.model_name,
            device=self.device,
        )
        logging.info(
            f"Using FAIRChem model: {self.model_name} on device: {self.device}"
        )
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
        from mace.calculators import MACECalculator

        model_file = Path(self.model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"Custom MACE model file does not exist: {self.model_path}"
            )

        logging.info(
            f"Using custom MACE model from path: {self.model_path} on device: {self.device}"
        )
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
        Parsed coordinates from ORCA (list of lists).
    energies : list
        Energies in Hartree.
    forces : list
        Forces in Hartree/Bohr (arrays, shape N×3).
    structure_ids : list
        IDs corresponding to the structures.
    """

    def __init__(
        self,
        step_id,
        step_dir,
        template_dir,
        trainer_cfg,
        coordinates=None,
        energies=None,
        forces=None,
        structure_ids=None,
        utils=None,
    ):
        self.step_id = step_id
        self.step_dir = step_dir
        self.template_dir = template_dir
        self.trainer_cfg = trainer_cfg
        self.coordinates = coordinates or []
        self.energies = energies or []
        self.forces = forces or []
        self.structure_ids = structure_ids or []
        self.utils = utils

    def prepare_inputs(self):
        """
        Convert coordinates, energies, and forces into extended XYZ train/test files.
        """
        n_coords = len(self.coordinates) if self.coordinates else 0
        n_energies = len(self.energies) if self.energies else 0
        n_forces = len(self.forces) if self.forces else 0
        n_ids = len(self.structure_ids) if self.structure_ids else 0

        logging.info(
            f"[MLFFTrainer] Received {n_coords} coordinates, {n_energies} energies, "
            f"{n_forces} forces, {n_ids} IDs"
        )

        if n_coords == 0 or n_forces == 0:
            raise ValueError(
                f"No usable structures for MLFF training "
                f"(coords={n_coords}, forces={n_forces})"
            )

        if not (n_coords == n_energies == n_forces == n_ids):
            raise ValueError(
                f"Inconsistent dataset lengths: coords={n_coords}, "
                f"energies={n_energies}, forces={n_forces}, ids={n_ids}"
            )

        os.makedirs(self.step_dir, exist_ok=True)

        # Build Atoms list
        atoms_list = []
        for coords, energy, forces in zip(self.coordinates, self.energies, self.forces):
            atoms = _to_atoms(coords, energy, forces)
            atoms_list.append(atoms)

        # Use sklearn for random split (default 90/10, can override in trainer_cfg)
        valid_fraction = self.trainer_cfg.get("valid_fraction", 0.1)
        train_structs, test_structs = train_test_split(
            atoms_list,
            test_size=valid_fraction,
            random_state=self.trainer_cfg.get("seed", 42),
            shuffle=True,
        )

        # Convert to eV/eVÅ just before writing
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

    def write_training_config(self, train_path, test_path):
        """
        Generate training config for MACE from step template.
        Only overrides dataset paths and directories.
        """

        # Load the step-specific training template
        template_file = os.path.join(self.template_dir, f"step{self.step_id}.inp")
        with open(template_file, "r") as f:
            config = yaml.safe_load(f)

        # Override dataset paths
        config["train_file"] = train_path
        config["test_file"] = test_path

        # Override directories relative to step_dir
        for key in ["log_dir", "checkpoints_dir", "results_dir"]:
            config[key] = os.path.join(self.step_dir, key)

        # Write out to input.yaml
        config_path = os.path.join(self.step_dir, "input.yaml")
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        logging.info(f"[MLFFTrainer] Wrote training config to {config_path}")
        return config_path

    def write_slurm_script(
        self, step_dir: str, device: str, job_name: str = "mlff_train"
    ):
        """
        Create a SLURM submission script for MACE training.

        Parameters
        ----------
        step_dir : str
            Path to the step directory where training files are stored.
        device : str
            Either "cuda" or "cpu" to select the appropriate header.
        job_name : str, optional
            Name for the SLURM job. Default "mlff_train".

        Returns
        -------
        str
            Path to the generated SLURM script.
        """
        # pick header based on device
        header_file = "cuda.slurm.header" if device == "cuda" else "cpu.slurm.header"
        header_path = os.path.join(self.template_dir, header_file)
        if not os.path.exists(header_path):
            raise FileNotFoundError(f"SLURM header file not found: {header_path}")

        # paths for logs and config
        slurm_out = os.path.join(step_dir, "slurm-%j.out")
        slurm_err = os.path.join(step_dir, "slurm-%j.err")
        config_path = os.path.join(step_dir, "input.yaml")

        # read header
        with open(header_path, "r") as f:
            header_text = f.read().rstrip()  # removes extra blank lines at end

        # build script
        script_path = os.path.join(step_dir, "train.slurm")
        with open(script_path, "w") as f:
            f.write("#!/bin/bash\n")
            # header first
            f.writelines(header_text)
            # job info
            f.write(f"#SBATCH --job-name={job_name}\n")
            f.write(f"#SBATCH --output={slurm_out}\n")
            f.write(f"#SBATCH --error={slurm_err}\n")
            f.write("\n")
            # training command
            f.write("export MKL_THREADING_LAYER=GNU\n")
            f.write(f"mace_run_train --config {config_path}\n")

        return script_path

    def submit_training(self, slurm_script, max_cores=32):
        """
        Submit the MLFF training SLURM script, track the job until completion.

        Parameters
        ----------
        slurm_script : str
            Path to the SLURM script generated for training.
        max_cores : int
            Maximum cores allowed (for consistency with ORCA submitter).
        """
        active_jobs = {}
        total_cores_used = 0

        # For MLFF we don't parse PAL; just assume a full node allocation
        pal_value = max_cores
        logging.info(f"[MLFFTrainer] Submitting training with up to {pal_value} cores.")

        # Submit job
        job_id = self.utils.submit_job(slurm_script)
        logging.info(f"[MLFFTrainer] Submitted MLFF training job with ID: {job_id}")

        if job_id and str(job_id).isdigit():
            active_jobs[job_id] = pal_value
            total_cores_used += pal_value
        else:
            logging.warning(
                f"[MLFFTrainer] Skipping job tracking for invalid job ID '{job_id}'"
            )
            return

        # Wait for completion
        logging.info("[MLFFTrainer] Waiting for training job to finish...")
        while active_jobs:
            completed_jobs = []
            for job_id, cores in list(active_jobs.items()):
                if self.utils.is_job_finished(job_id):
                    completed_jobs.append(job_id)
                    total_cores_used -= cores
                    logging.info(
                        f"[MLFFTrainer] Training job {job_id} completed. Freed {cores} cores."
                    )

            for job_id in completed_jobs:
                del active_jobs[job_id]

            time.sleep(30)

        logging.info("[MLFFTrainer] Training finished successfully.")

    def run(self):
        """Simplified run: just prepare Atoms objects and write XYZ files."""

        train_path, test_path = self.prepare_inputs()
        self.write_training_config(train_path, test_path)
        self.write_slurm_script(self.step_dir, self.trainer_cfg.get("device", "cuda"))
        self.submit_training(os.path.join(self.step_dir, "train.slurm"))


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
    positions = np.array(
        [[float(x), float(y), float(z)] for _, x, y, z in coords], dtype=float
    )

    atoms = Atoms(symbols=symbols, positions=positions)

    if energy is not None:
        atoms.info["DFT_energy"] = energy
    if isinstance(forces, np.ndarray) and forces.size > 0:
        atoms.arrays["DFT_Forces"] = forces

    return atoms

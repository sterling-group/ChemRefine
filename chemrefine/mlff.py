import logging
import os
import logging
from typing import List, Tuple, Optional
from ase.optimize import LBFGS
from ase.io import read
from ase import Atoms
from .utils import Utility

class MLFFCalculator:
    """Flexible MLFF calculator supporting multiple backend models."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        model_path: Optional[str] = None,
        task_name: str = "mace_off"  # Default task for MACE,
        ):
        """
        Initialize the MLFF calculator.

        Parameters
        ----------
        model_name : str
            Name of the MLFF model (e.g. "mace", "uma-s-1").
        device : str, optional
            Device for model evaluation ("cpu" or "cuda").
        model_path : str, optional
            Path to a local model checkpoint.
        """
        self.model_name = model_name
        self.device = device

        if model_name.startswith("mace"):
            self._setup_mace(model_name)
        elif model_name.startswith("uma") or model_name.startswith("fairchem"):
            self._setup_fairchem(model_name, model_path)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def _setup_mace(self):
        """Setup the MACE calculator."""
        if self.task_name == "mace_off":
            from mace.calculators import mace_off
            self.calc = mace_off.MACE(device=self.device)
        else:
            from mace.calculators import mace_mp
            self.calc = mace_mp(model_name=self.task_name, device=self.device)

    def _setup_fairchem(self, model_name="uma-s-1"):
        """Setup the FairChem calculator."""
        from fairchem.core import pretrained_mlip, FAIRChemCalculator
        predictor = pretrained_mlip.get_predict_unit(
            model_name=model_name,
            device=self.device,
            #TODO currently offline models don't work on new FAIRCHEM checkpoint_path=None  # Use default or specify a path
        )

        self.calc = FAIRChemCalculator(predictor, self.task_name)

    def calculate(self, atoms: Atoms, fmax: float = 0.03, steps: int = 200) -> Tuple[List[List], float]:
        """
        Optimize geometry using the selected MLFF.

        Parameters
        ----------
        atoms : Atoms
            ASE Atoms object with initial geometry.
        fmax : float, optional
            Force convergence criterion.
        steps : int, optional
            Maximum number of optimization steps.

        Returns
        -------
        tuple
            Optimized coordinates and energy in Hartree.
        """
        atoms.calc = self.calc
        optimizer = LBFGS(atoms, logfile=None)
        optimizer.run(fmax=fmax, steps=steps)

        energy_ev = atoms.get_potential_energy()
        energy_hartree = energy_ev / 27.211386245988
        coords = [
            [atom.symbol, atom.position[0], atom.position[1], atom.position[2]]
            for atom in atoms
        ]
        return coords, energy_hartree

class MLFFJobSubmitter:
    """
    Generate and submit SLURM jobs for MLFF calculations.
    Handles job submission, active job tracking, and waiting for job completion.
    """

    def __init__(self, scratch_dir: str | None = None, max_jobs: int = 32):
        """
        Initialize the MLFF job submitter.

        Parameters
        ----------
        scratch_dir : str, optional
            Directory for scratch files.
        max_jobs : int, optional
            Maximum number of concurrent jobs.
        """
        self.scratch_dir = scratch_dir or "./tmp/mlff_scratch"
        self.max_jobs = max_jobs
        self.utility = Utility()

    def generate_slurm_script(
        self,
        xyz_file: str,
        template_dir: str,
        output_dir: str = ".",
        job_name: str | None = None,
        model_name: str = "uma-s-1",
        device: str | None = None,
        fmax: float = 0.03,
        steps: int = 200,
        task_name: str = "mace_off"
    ) -> str:
        """
        Create a SLURM script for an MLFF optimization.

        Parameters
        ----------
        xyz_file : str
            Path to the XYZ file.
        template_dir : str
            Directory containing SLURM header template.
        output_dir : str, optional
            Directory to write the SLURM script to.
        job_name : str, optional
            Name of the SLURM job.
        model_name : str, optional
            MLFF model name.
        device : str, optional
            Device for model evaluation.
        fmax : float, optional
            Force convergence criterion.
        steps : int, optional
            Maximum optimization steps.
        task_name : str, optional
            MLFF task name.

        Returns
        -------
        str
            Path to the generated SLURM script.
        """
        header = Path(template_dir) / "mlff.slurm.header"
        if not header.is_file():
            raise FileNotFoundError(f"SLURM header template {header} not found")

        if job_name is None:
            job_name = Path(xyz_file).stem

        with open(header) as f:
            lines = f.readlines()

        sbatch = []
        non = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#SBATCH"):
                sbatch.append(line.rstrip())
            else:
                non.append(line.rstrip())

        sbatch.append(f"#SBATCH --job-name={job_name}")
        sbatch.append(f"#SBATCH --output={job_name}.out")
        sbatch.append(f"#SBATCH --error={job_name}.err")

        slurm_path = Path(output_dir) / f"{job_name}.slurm"
        with open(slurm_path, "w") as f:
            f.write("#!/bin/bash\n\n")
            f.write("\n".join(sbatch) + "\n\n")
            f.write("\n".join(non) + "\n\n")
            f.write(
                f"python -m chemrefine.mlff_runner {xyz_file} --model {model_name} --task-name {task_name}"
            )
            if device:
                f.write(f" --device {device}")
            f.write(f" --fmax {fmax} --steps {steps}\n")

        logging.info(f"Generated SLURM script: {slurm_path}")
        return str(slurm_path)

    def submit_jobs(
        self,
        xyz_files: list,
        template_dir: str,
        output_dir: str = ".",
        model_name: str = "uma-s-1",
        device: str | None = None,
        fmax: float = 0.03,
        steps: int = 200,
        task_name: str = "mace_off"
    ):
        """
        Submit multiple MLFF XYZ files to SLURM and monitor their progress.

        Parameters
        ----------
        xyz_files : list of str
            Paths to XYZ files to submit.
        template_dir : str
            Directory containing the SLURM header template.
        output_dir : str, optional
            Directory for output.
        model_name : str, optional
            MLFF model name.
        device : str, optional
            Device for model evaluation.
        fmax : float, optional
            Force convergence criterion.
        steps : int, optional
            Maximum optimization steps.
        task_name : str, optional
            MLFF task name.
        """
        active_jobs = {}

        for xyz_file in xyz_files:
            xyz_path = Path(xyz_file).resolve()
            job_name = xyz_path.stem

            # Check concurrency
            while len(active_jobs) >= self.max_jobs:
                logging.info("Maximum concurrent jobs reached. Waiting...")
                completed_jobs = []
                for job_id in list(active_jobs.keys()):
                    if self.utility.is_job_finished(job_id):
                        completed_jobs.append(job_id)
                        logging.info(f"Job {job_id} completed.")
                for job_id in completed_jobs:
                    del active_jobs[job_id]
                time.sleep(10)

            slurm_script = self.generate_slurm_script(
                xyz_file=xyz_path,
                template_dir=template_dir,
                output_dir=output_dir,
                job_name=job_name,
                model_name=model_name,
                device=device,
                fmax=fmax,
                steps=steps,
                task_name=task_name
            )

            job_id = self.utility.submit_job(Path(slurm_script))
            logging.info(f"Submitted MLFF job with ID: {job_id} for XYZ: {xyz_path.name}")

            if job_id.isdigit():
                active_jobs[job_id] = xyz_path
            else:
                logging.warning(f"Skipping job tracking for invalid job ID '{job_id}'")

        logging.info("All MLFF jobs submitted. Waiting for remaining jobs to complete...")
        while active_jobs:
            completed_jobs = []
            for job_id in list(active_jobs.keys()):
                if self.utility.is_job_finished(job_id):
                    completed_jobs.append(job_id)
                    logging.info(f"Job {job_id} completed.")
            for job_id in completed_jobs:
                del active_jobs[job_id]
            time.sleep(30)

        logging.info("All MLFF calculations finished.")

    
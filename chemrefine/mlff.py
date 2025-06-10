import logging
import os
import logging
from typing import List, Tuple
from ase.optimize import LBFGS
from ase.io import read


class MLFFCalculator:
    """Flexible MLFF calculator supporting multiple backend models."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        model_path: Optional[str] = None
    ):
        """
        Initialize the MLFF calculator.

        Parameters
        ----------
        model_name : str
            Name of the MLFF model (e.g. "mace_off", "mace_omat-0", "uma-s-1").
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

    def _setup_mace(self, task_name="mace_off"):
        """Setup the MACE calculator."""
        if task_name == "mace_off":
            from mace.calculators import mace_off
            self.calc = mace_off.MACE(device=self.device)
        else:
            from mace.calculators import mace_mp
            self.calc = mace_mp(model_name=task_name, device=self.device)

    def _setup_fairchem(self, model_name="uma-s-1",task_name="oc20"):
        """Setup the FairChem calculator."""
        from fairchem.core import pretrained_mlip, FAIRChemCalculator
        predictor = pretrained_mlip.get_predict_unit(
            model_name=model_name,
            device=self.device,
            #TODO currently offline models don't work on new FAIRCHEM checkpoint_path=None  # Use default or specify a path
        )

        self.calc = FAIRChemCalculator(predictor, task_name="omol")

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
    """Generate and submit SLURM jobs for MLFF calculations."""

    def __init__(self, scratch_dir: str | None = None):
        self.scratch_dir = scratch_dir or "./tmp/mlff_scratch"

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
    ) -> str:
        """Create a SLURM script for an MLFF optimisation."""
        from pathlib import Path

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
                f"python -m chemrefine.mlff_runner {xyz_file} --model {model_name}"
            )
            if device:
                f.write(f" --device {device}")
            f.write(f" --fmax {fmax} --steps {steps}\n")

        return str(slurm_path)

    def submit_job(self, slurm_script: str) -> str:
        """Submit a SLURM script if ``sbatch`` is available."""
        import subprocess
        import shutil

        if shutil.which("sbatch"):
            try:
                result = subprocess.run(
                    ["sbatch", slurm_script],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                return result.stdout.strip()
            except subprocess.CalledProcessError as exc:
                logging.error(f"sbatch failed: {exc.stderr}")
                return "ERROR"
        else:
            logging.info("sbatch not found; running script locally")
            subprocess.run(["bash", slurm_script], check=True)
            return "LOCAL"

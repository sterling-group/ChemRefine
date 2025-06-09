import logging
import os
from typing import List, Tuple


def get_available_device() -> str:
    """Return ``"cuda"`` if a GPU is available, otherwise ``"cpu"``."""
    try:  # pragma: no cover - optional dependency
        import torch

        if torch.cuda.is_available():
            logging.info("GPU detected for MLFF calculations.")
            return "cuda"
    except Exception:
        pass

    logging.info("No GPU detected; falling back to CPU for MLFF calculations.")
    return "cpu"


def run_mlff_calculation(
    xyz_path: str,
    model_name: str = "mol",
    device: str | None = None,
    model_path: str | None = None,
    fmax: float = 0.03,
    steps: int = 200,
) -> Tuple[List[List], float]:
    """Optimize geometry using farichem-core.

    Parameters
    ----------
    xyz_path : str
        Path to an XYZ file containing the starting geometry.
    model_name : str, optional
        Name of the MLFF model to load. Defaults to ``"mol"``.
    device : str or None, optional
        Device for model evaluation ("cpu" or "cuda"). If ``None``,
        ``get_available_device()`` is used to automatically select the
        device.
    model_path : str, optional
        Path to a local model checkpoint. If provided, the model is
        loaded from this path instead of downloading from Hugging Face.
    fmax : float, optional
        Force convergence criterion for the optimiser. Defaults to ``0.03``.
    steps : int, optional
        Maximum optimisation steps. Defaults to ``200``.

    Returns
    -------
    tuple
        Optimised coordinates ``[[symbol, x, y, z], ...]`` and the energy in Hartree.
    """
    try:
        from ase.io import read
        from ase.optimize import LBFGS
        from fairchem.core import pretrained_mlip, FAIRChemCalculator
    except ImportError as exc:  # pragma: no cover - dependency missing at runtime
        raise ImportError(
            "MLFF calculations require the 'fairchem-core' and 'ase' packages"
        ) from exc

    if device is None:
        device = get_available_device()

    if model_path is None:
        model_path = os.environ.get("CHEMREFINE_MLFF_CHECKPOINT")
        if model_path is None:
            from pathlib import Path

            pkg_dir = Path(__file__).resolve().parent
            candidate = pkg_dir / "models" / f"{model_name}.pt"
            if candidate.is_file():
                model_path = str(candidate)

    logging.info(f"Loading MLFF model '{model_name}' on {device}.")
    atoms = read(xyz_path)
    if model_path:
        predictor = pretrained_mlip.get_predict_unit(
            model_name=model_name, device=device, checkpoint_path=model_path
        )
    else:
        predictor = pretrained_mlip.get_predict_unit(model_name=model_name, device=device)
    calc = FAIRChemCalculator(predictor, task_name="oc20")
    atoms.calc = calc

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
        model_name: str = "mol",
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

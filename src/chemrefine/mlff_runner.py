import argparse
from ase.io import read
from chemrefine.mlff import run_mlff_calculation
from chemrefine.utils import Utility

def main():
    parser = argparse.ArgumentParser(description="Run MLFF optimization on an XYZ file.")
    parser.add_argument("xyz", help="Path to the XYZ file.")
    parser.add_argument("--model", default="mace-off", help="Foundation model backend (mace-off, mace-mp, fairchem).")
    parser.add_argument("--device", default=None, help="Computation device.")
    parser.add_argument("--model-name", default=None, help="Path to a local model checkpoint.")
    parser.add_argument("--task-name", default=None, help="Task name for MACE or FairChem models.")
    parser.add_argument("--model-path", default=None, help="Path to a local model checkpoint.")
    parser.add_argument("--fmax", type=float, default=0.03, help="LBFGS force convergence.")
    parser.add_argument("--steps", type=int, default=200, help="Maximum optimization steps.")
    args = parser.parse_args()

    atoms= run_mlff_calculation(
    xyz_path=args.xyz,
    model_name=args.model,
    task_name=args.task_name,  # required by MLFFCalculator
    device=args.device,
    model_path=args.model_path,
    fmax=args.fmax,
    steps=args.steps
)

    
if __name__ == "__main__":
    main()

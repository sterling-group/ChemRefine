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

    coords, energy,forces= run_mlff_calculation(
    xyz_path=args.xyz,
    model_name=args.model,
    task_name=args.task_name,  # required by MLFFCalculator
    device=args.device,
    model_path=args.model_path,
    fmax=args.fmax,
    steps=args.steps
)


    base = args.xyz.rsplit(".", 1)[0]
    # Read original Atoms object
    atoms = read(args.xyz)
    # Update positions with optimized coordinates
    for atom, coord in zip(atoms, coords):
        atom.position = coord[1:]

    # Write extended XYZ file
    atoms.info["mlff_energy"] = energy
    atoms.arrays["mlff_forces"] = forces  # if forces are available from optimization

    utility = Utility()
    utility.write_single_xyz(atoms, f"{base}.opt.extxyz")

    # Write separate energy file
    with open(f"{base}.energy", "w") as f:
        f.write(str(energy))

if __name__ == "__main__":
    main()

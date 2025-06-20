import argparse
from ase import Atoms
from chemrefine.mlff import MLFFCalculator
from chemrefine.utils_extopt import read_input, read_xyzfile, write_engrad, process_output
import logging
import torch
logging.basicConfig(filename="mlff_debug.log", level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="ORCA external wrapper using MLFF.")
    parser.add_argument("inputfile", help="ORCA .extinp.tmp input file")
    parser.add_argument("--model_name", default="uma-s-1", help="Name of the MLFF model (e.g., uma-s-1)")
    parser.add_argument("--task_name", default="omol", help="Task name (e.g., omol, mace_off)")
    parser.add_argument("--device", default="cuda", help="Device to run on (cpu or cuda)")
    parser.add_argument("--model-path", default=None, help="Optional path to a local model checkpoint")
    args = parser.parse_args()

    if args.device == "cuda":
        print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        logging.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logging.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    xyzname, charge, mult, ncores, dograd = read_input(args.inputfile)
    atom_types, coordinates = read_xyzfile(xyzname)
    atoms = Atoms(symbols=atom_types, positions=coordinates)
    atoms = Atoms(symbols=atom_types, positions=coordinates)


    logging.info(f"Running MLFF with model {args.model_name} on {args.device} for task {args.task_name}")
    # Only set charge/mult if backend supports it
    if args.task_name.startswith(("uma", "omol", "odac", "omat", "fairchem")):
        atoms.info = {"charge": charge, "spin": mult}


    calc = MLFFCalculator(
        model_name=args.model_name,
        task_name=args.task_name,
        device=args.device,
        model_path=args.model_path
    )
    atoms.calc = calc.calculator

    energy, gradient = process_output(atoms)
    write_engrad(xyzname.replace(".xyz", ".engrad"), len(atoms), energy, dograd, gradient)


if __name__ == "__main__":
    main()

import argparse
from ase import Atoms
from chemrefine.mlff import MLFFCalculator
from chemrefine.utils_extopt import read_input, read_xyzfile, write_engrad, process_output


def main():
    parser = argparse.ArgumentParser(description="ORCA external wrapper using MLFF.")
    parser.add_argument("inputfile", help="ORCA .extinp.tmp input file")
    parser.add_argument("--model", default="uma-s-1", help="Name of the MLFF model (e.g., uma-s-1)")
    parser.add_argument("--task-name", default="omol", help="Task name (e.g., omol, mace_off)")
    parser.add_argument("--device", default="cpu", help="Device to run on (cpu or cuda)")
    parser.add_argument("--model-path", default=None, help="Optional path to a local model checkpoint")
    args = parser.parse_args()

    xyzname, charge, mult, ncores, dograd = read_input(args.inputfile)
    atom_types, coordinates = read_xyzfile(xyzname)
    atoms = Atoms(symbols=atom_types, positions=coordinates)
    atoms = Atoms(symbols=atom_types, positions=coordinates)

    # Only set charge/mult if backend supports it
    if args.task_name.startswith(("uma", "omol", "odac", "omat", "fairchem")):
        atoms.info = {"charge": charge, "spin": mult}


    calc = MLFFCalculator(
        model_name=args.model,
        task_name=args.task_name,
        device=args.device,
        model_path=args.model_path
    )
    atoms.calc = calc.calculator

    energy, gradient = process_output(atoms)
    write_engrad(xyzname.replace(".xyz", ".engrad"), len(atoms), energy, dograd, gradient)


if __name__ == "__main__":
    main()

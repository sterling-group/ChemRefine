import argparse
import logging
from ase import Atoms
from chemrefine.mlff import MLFFCalculator, query_mlff_server
from chemrefine.utils_extopt import read_input, read_xyzfile, write_engrad, process_output

def main():
    parser = argparse.ArgumentParser(description="ORCA external wrapper using MLFF.")
    parser.add_argument("inputfile", help="ORCA .extinp.tmp input file")
    parser.add_argument("--model_name", default="uma-s-1", help="Name of the MLFF model (e.g., uma-s-1)")
    parser.add_argument("--task_name", default="omol", help="Task name (e.g., omol, mace_off)")
    parser.add_argument("--device", default="cuda", help="Device to run on (cpu or cuda)")
    parser.add_argument("--model-path", default=None, help="Optional path to a local model checkpoint")
    parser.add_argument("--use-server", action="store_true", help="Use MLFF socket server instead of local call")
    parser.add_argument("--port", type=int, default=50051, help="Port for MLFF socket server (default: 50051)")
    args = parser.parse_args()

    xyzname, charge, mult, ncores, dograd = read_input(args.inputfile)
    atom_types, coordinates = read_xyzfile(xyzname)
    atoms = Atoms(symbols=atom_types, positions=coordinates)

    if args.task_name.startswith(("uma", "omol", "odac", "omat", "fairchem")):
        atoms.info = {"charge": charge, "spin": mult}

    logging.info(f"Running MLFF with model {args.model_name} on {args.device} for task {args.task_name} (use_server={args.use_server})")

    if args.use_server:
        energy, forces = query_mlff_server(atoms, port=args.port)
    else:
        calc = MLFFCalculator(
            model_name=args.model_name,
            task_name=args.task_name,
            device=args.device,
            model_path=args.model_path
        )
        atoms.calc = calc.calculator
        energy, forces = process_output(atoms)

    write_engrad(xyzname.replace(".xyz", ".engrad"), len(atoms), energy, dograd, forces)

if __name__ == "__main__":
    main()

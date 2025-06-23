import argparse
import logging
from ase import Atoms
from chemrefine.mlff import MLFFCalculator, query_mlff_server
from chemrefine.utils_extopt import read_input, read_xyzfile, write_engrad, process_output
import time
import os
def main():
    parser = argparse.ArgumentParser(description="ORCA external wrapper using MLFF.")
    parser.add_argument("inputfile", help="ORCA .extinp.tmp input file")
    parser.add_argument("--model_name", default="uma-s-1")
    parser.add_argument("--task_name", default="omol")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--use-server", action="store_true", default=True)  # Always use server
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args()
     # Wait for ORCA input file to be written
    timeout = 30
    start_time = time.time()
    while not os.path.exists(args.inputfile):
        if time.time() - start_time > timeout:
            raise FileNotFoundError(f"Input file {args.inputfile} not found after waiting {timeout} seconds.")
        time.sleep(1)
    xyzname, charge, mult, ncores, dograd = read_input(args.inputfile)
    atom_types, coordinates = read_xyzfile(xyzname)
    atoms = Atoms(symbols=atom_types, positions=coordinates)

    if args.task_name.startswith(("uma", "omol", "odac", "omat", "fairchem")):
        atoms.info = {"charge": charge, "spin": mult}

    logging.info(f"Running MLFF with model {args.model_name} on {args.device} for task {args.task_name} (use_server=True)")

    energy, forces = query_mlff_server(atoms, port=args.port)
    write_engrad(xyzname.replace(".xyz", ".engrad"), len(atoms), energy, dograd, forces)

if __name__ == "__main__":
    main()

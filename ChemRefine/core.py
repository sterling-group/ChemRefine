import os
import yaml
import logging
from .parse import parse_arguments
from .file_submission import submit_files
from .refine import filter_structures
from .utils import move_step_files, save_step_csv
from .mlip import run_mlip_calculation

def main():
    args, qorca_flags = parse_arguments()
    cores, yaml_input, skip = args.maxcores, args.input_file, args.skip

    with open(yaml_input, 'r') as file:
        config = yaml.safe_load(file)

    steps = config['steps']
    charge = config['charge']
    multiplicity = config['multiplicity']

    filtered_coordinates, filtered_ids = None, None

    for step in steps:
        step_number = step['step']
        ctype = step['calculation_type']
        sample_method = step['sample_type']['method']
        parameters = step['sample_type']['parameters']

        if ctype == 'MLFF':
            run_mlip_calculation()

        if skip and os.path.exists(f"step{step_number}"):
            # parse previously completed output here
            continue

        # Assuming xyz generation and .inp creation was done
        input_files = [f"step{step_number}_structure_{i}.inp" for i in filtered_ids or [0]]
        submit_files(input_files, cores, qorca_flags=qorca_flags)
        # Assume energies fetched
        energies = [0.0] * len(input_files)
        ids = list(range(len(energies)))
        save_step_csv(energies, ids, step_number)
        filtered_coordinates, filtered_ids = filter_structures([], energies, ids, sample_method, parameters=parameters)
        move_step_files(step_number)

if __name__ == "__main__":
    main()

import os
import yaml
import logging
from .parse import ArgumentParser
from .file_submission import FileSubmitter
from .refine import StructureRefiner
from .utils import Utility
from .orca_interface import OrcaInterface  # Keep this if parse_output/create_input are here

class ChemRefiner:
    def __init__(self):
        self.arg_parser = ArgumentParser()
        self.submitter = FileSubmitter()
        self.refiner = StructureRefiner()
        self.utils = Utility()
        self.orca = OrcaInterface()

        # Parse args
        self.args, self.qorca_flags = self.arg_parser.parse()

        # Load YAML config
        with open(self.args.input_file, 'r') as file:
            self.config = yaml.safe_load(file)

        # Pull top-level config
        self.charge = self.config.get('charge', 0)
        self.multiplicity = self.config.get('multiplicity', 1)
        self.template_dir = self.config.get('template_dir', './templates')
        self.scratch_dir = self.config.get('scratch_dir', os.getenv("SCRATCH", "/tmp/orca_scratch"))


    def run(self):
        cores = self.args.maxcores
        skip = self.args.skip
        steps = self.config['steps']

        filtered_coordinates, filtered_ids = None, None
        for step in steps:
            step_number = step['step']
            ctype = step['calculation_type']
            sample_method = step['sample_type']['method']
            parameters = step['sample_type']['parameters']

            if ctype.upper() == 'MLFF':
                logging.warning("MLIP support is under construction.")
                continue

            if skip and os.path.exists(f"step{step_number}"):
                logging.info(f"Skipping step {step_number} because its directory already exists.")
                output_files = [
                    os.path.join(f"step{step_number}", f)
                    for f in os.listdir(f"step{step_number}")
                    if f.endswith('.out')
                ]
                coordinates, energies = self.orca.parse_output(output_files, ctype, dir=f"./step{step_number}")
                continue

            if step_number == 1:
                xyz_file = "step1.xyz"
                inp_file = os.path.join(self.template_dir, "step1.inp")
                if not os.path.exists(xyz_file):
                    logging.error(f"XYZ file {xyz_file} does not exist. Please provide a valid file.")
                    return
                xyz_filenames = [xyz_file]
                input_files, output_files = self.orca.create_input(
                    xyz_filenames, inp_file, self.charge, self.multiplicity
                )
                self.submitter.submit_files(input_files, cores, self.qorca_flags)
                coordinates, energies = self.orca.parse_output(output_files, ctype)
                filtered_ids = list(range(len(energies)))
                self.utils.save_step_csv(energies, filtered_ids, step_number)
                filtered_coordinates, filtered_ids = self.refiner.filter(
                    coordinates, energies, filtered_ids, sample_method, parameters
                )
                self.utils.move_step_files(step_number)
                continue

            xyz_filenames = self.utils.write_xyz(filtered_coordinates, step_number, filtered_ids)
            input_template = os.path.join(self.template_dir, f"step{step_number}.inp")
            input_files, output_files = self.orca.create_input(
                xyz_filenames, input_template, self.charge, self.multiplicity
            )
            self.submitter.submit_files(input_files, cores, self.qorca_flags)
            coordinates, energies = self.orca.parse_output(output_files, ctype)
            self.utils.save_step_csv(energies, filtered_ids, step_number)
            filtered_coordinates, filtered_ids = self.refiner.filter(
                coordinates, energies, filtered_ids, sample_method, parameters
            )
            self.utils.move_step_files(step_number)

def main():
    ChemRefiner().run()

if __name__ == "__main__":
    ChemRefiner().run()

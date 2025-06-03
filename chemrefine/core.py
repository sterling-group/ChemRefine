import os
import yaml
import logging
from .parse import ArgumentParser
from .refine import StructureRefiner
from .utils import Utility
from .orca_interface import OrcaInterface, OrcaJobSubmitter

class ChemRefiner:
    def __init__(self):
        self.arg_parser = ArgumentParser()
        self.orca_submitter = OrcaJobSubmitter()
        self.refiner = StructureRefiner()
        self.utils = Utility()
        self.orca = OrcaInterface()
        self.output_dir = os.path.abspath(self.config.get('outputs', '.'))
        os.makedirs(self.output_dir, exist_ok=True)

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
        steps = self.config.get('steps', [])
        calculation_functions = ["GOAT", "DFT", "XTB", "MLFF"]

        filtered_coordinates, filtered_ids = None, None

        for step in steps:
            step_number = step['step']
            calculation_type = step['calculation_type']
            sample_method = step['sample_type']['method']
            parameters = step['sample_type']['parameters']

            # Validate calculation type
            if calculation_type not in calculation_functions:
                raise ValueError(f"Invalid calculation type '{calculation_type}' in step {step_number}. Exiting...")

            # Skip logic
            if skip and os.path.exists(f"step{step_number}"):
                logging.info(f"Skipping step {step_number} because its directory already exists.")
                output_files = [
                    os.path.join(f"step{step_number}", f)
                    for f in os.listdir(f"step{step_number}")
                    if f.endswith('.out')
                ]
                coordinates, energies = self.orca.parse_output(output_files, calculation_type, dir=f"./step{step_number}")
                filtered_coordinates, filtered_ids = self.refiner.filter(
                    coordinates, energies, list(range(len(energies))), sample_method, parameters
                )
                continue

            logging.info(f"Running step {step_number}: {calculation_type} with sampling method '{sample_method}'")

            # === STEP 1 ===
            if step_number == 1:
                xyz_file = os.path.join(self.template_dir,"step1.xyz")
                inp_file = os.path.join(self.template_dir, "step1.inp")
                if not os.path.exists(xyz_file):
                    raise FileNotFoundError(f"XYZ file '{xyz_file}' not found for step {step_number}. Exiting...")

                xyz_filenames = [xyz_file]
                input_files, output_files = self.orca.create_input(
                    xyz_filenames, inp_file, self.charge, self.multiplicity
                )

                self.orca_submitter = OrcaJobSubmitter()

                for input_file in input_files:
                    input_path = Path(input_file)
                    pal_value = self.orca_submitter.parse_pal_from_input(input_path)
                    pal_value = min(pal_value, cores)
                    self.orca_submitter.adjust_pal_in_input(input_path, pal_value)
                    slurm_script = self.orca_submitter.generate_slurm_script(
                        input_path, pal_value, self.template_dir
                    )
                    job_id = self.orca_submitter.submit_job(slurm_script)
                    logging.info(f"Job submitted with ID: {job_id}")

                # Parse output after job completion (simplified sequential logic)
                coordinates, energies = self.orca.parse_output(output_files, calculation_type)
                filtered_ids = list(range(len(energies)))
                self.utils.save_step_csv(energies, filtered_ids, step_number,output_dir=self.output_dir)
                filtered_coordinates, filtered_ids = self.refiner.filter(
                    coordinates, energies, filtered_ids, sample_method, parameters
                )
                self.utils.move_step_files(step_number)
                continue

            # === Subsequent Steps ===
            if calculation_type != "MLFF":
                xyz_filenames = self.utils.write_xyz(filtered_coordinates, step_number, filtered_ids)
                input_template = os.path.join(self.template_dir, f"step{step_number}.inp")
                input_files, output_files = self.orca.create_input(
                    xyz_filenames, input_template, self.charge, self.multiplicity
                )

                self.orca_submitter = OrcaJobSubmitter()

                for input_file in input_files:
                    input_path = Path(input_file)
                    pal_value = self.orca_submitter.parse_pal_from_input(input_path)
                    pal_value = min(pal_value, cores)
                    self.orca_submitter.adjust_pal_in_input(input_path, pal_value)
                    slurm_script = self.orca_submitter.generate_slurm_script(
                        input_path, pal_value, self.template_dir
                    )
                    job_id = self.orca_submitter.submit_job(slurm_script)
                    logging.info(f"Job submitted with ID: {job_id}")

                coordinates, energies = self.orca.parse_output(output_files, calculation_type)
                self.utils.save_step_csv(energies, filtered_ids, step_number)
                filtered_coordinates, filtered_ids = self.refiner.filter(
                    coordinates, energies, filtered_ids, sample_method, parameters
                )
                self.utils.move_step_files(step_number)
            else:
                raise ValueError("MLFF support is still under construction. Exiting...")



def main():
    ChemRefiner().run()

if __name__ == "__main__":
    ChemRefiner().run()

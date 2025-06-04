import os
import yaml
import logging
from .parse import ArgumentParser
from .refine import StructureRefiner
from .utils import Utility
from .orca_interface import OrcaInterface, OrcaJobSubmitter
from pathlib import Path
import shutil

class ChemRefiner:
    """
    ChemRefiner class orchestrates the ChemRefine workflow, handling input parsing,
    job submission, output parsing, and structure refinement based on a YAML configuration.
    It supports multiple steps with different calculation types and sampling methods.
    """
    def __init__(self):
        self.arg_parser = ArgumentParser()
        self.orca_submitter = OrcaJobSubmitter()
        self.refiner = StructureRefiner()
        self.utils = Utility()
        self.orca = OrcaInterface()

        # === Parse command-line arguments first ===
        self.args, self.qorca_flags = self.arg_parser.parse()

        # === Load the YAML configuration ===
        with open(self.args.input_file, 'r') as file:
            self.config = yaml.safe_load(file)

        # === Pull top-level config ===
        self.charge = self.config.get('charge', 0)
        self.multiplicity = self.config.get('multiplicity', 1)
        self.template_dir = os.path.abspath(self.config.get('template_dir', './templates'))
        self.scratch_dir = self.config.get('scratch_dir', os.getenv("SCRATCH", "/tmp/orca_scratch"))

        # === Setup output directory AFTER config is loaded ===
        output_dir_raw = self.config.get('outputs', './outputs')  # Default to './outputs'
        self.output_dir = os.path.abspath(output_dir_raw)
        os.makedirs(self.output_dir, exist_ok=True)
        logging.info(f"Output directory set to: {self.output_dir}")

        os.makedirs(self.output_dir, exist_ok=True)

    def prepare_step1_directory(self, step_number):
        step_dir = os.path.join(self.output_dir, f"step{step_number}")
        os.makedirs(step_dir, exist_ok=True)        
        logging.info(f"step_dir BEFORE: {step_dir}")


        # Copy input files from template_dir to step_dir
        src_xyz = os.path.join(self.template_dir, "step1.xyz")
        dst_xyz = os.path.join(step_dir, "step1.xyz")
        if not os.path.exists(src_xyz):
            raise FileNotFoundError(
                f"XYZ file '{src_xyz}' not found. Please ensure that 'step1.xyz' exists in the template directory."
            )
        shutil.copyfile(src_xyz, dst_xyz)

        # Use template from template_dir directly
        template_inp = os.path.join(self.template_dir, "step1.inp")
        if not os.path.exists(template_inp):
            raise FileNotFoundError(
                f"Input file '{template_inp}' not found. Please ensure that 'step1.inp' exists in the template directory."
            )

        xyz_filenames = [dst_xyz]

        input_files, output_files = self.orca.create_input(
            xyz_filenames, template_inp, self.charge, self.multiplicity, output_dir=step_dir
        )

        return step_dir, input_files, output_files

    def prepare_subsequent_step_directory(self, step_number, filtered_coordinates, filtered_ids):
        """
        Prepares the directory for subsequent steps by writing XYZ files, copying the template input,
        and generating ORCA input files.

        Args:
            step_number (int): The current step number.
            filtered_coordinates (list): Filtered coordinates from the previous step.
            filtered_ids (list): Filtered IDs from the previous step.

        Returns:
            step_dir (str): Path to the step directory.
            input_files (list): List of generated ORCA input files.
            output_files (list): List of expected ORCA output files.
        """
        step_dir = os.path.join(self.output_dir, f"step{step_number}")
        os.makedirs(step_dir, exist_ok=True)

        # Write XYZ files in step_dir
        xyz_filenames = self.utils.write_xyz(filtered_coordinates, step_number, filtered_ids, output_dir=step_dir)


        # Copy the template input file from template_dir to step_dir
        input_template_src = os.path.join(self.template_dir, f"step{step_number}.inp")
        input_template_dst = os.path.join(step_dir, f"step{step_number}.inp")
        shutil.copyfile(input_template_src, input_template_dst)

        # Create ORCA input files in step_dir
        input_files, output_files = self.orca.create_input(
            xyz_filenames, input_template_dst, self.charge, self.multiplicity, output_dir=step_dir
        )

        return step_dir, input_files, output_files

    def handle_skip_step(self, step_number, calculation_type, sample_method, parameters):
        """
        Handles skip logic for a step if its directory already exists.

        Args:
            step_number (int): The current step number.
            calculation_type (str): The type of calculation (DFT, XTB, etc.).
            sample_method (str): The sampling method.
            parameters (dict): Additional parameters for filtering.

        Returns:
            tuple: (filtered_coordinates, filtered_ids) if step is skipped, else (None, None).
        """
        step_dir = os.path.join(self.output_dir, f"step{step_number}")
        if os.path.exists(step_dir):
            logging.info(f"Skipping step {step_number} because its directory already exists.")
            output_files = [
                os.path.join(step_dir, f)
                for f in os.listdir(step_dir)
                if f.endswith('.out')
            ]
            coordinates, energies = self.orca.parse_output(output_files, calculation_type, dir=step_dir)
            filtered_coordinates, filtered_ids = self.refiner.filter(
                coordinates, energies, list(range(len(energies))), sample_method, parameters
            )
            return filtered_coordinates, filtered_ids
        else:
            return None, None

    def submit_orca_jobs(self, input_files, cores, step_dir, partition="sterling"):
        """
        Submits ORCA jobs for each input file in the step directory using the OrcaJobSubmitter.

        Args:
            input_files (list): List of ORCA input files.
            cores (int): Maximum total cores allowed for all jobs.
            step_dir (str): Path to the step directory.
            partition (str): SLURM partition to monitor jobs.
        """
        logging.info(f"Switching to working directory: {step_dir}")
        original_dir = os.getcwd()
        os.chdir(step_dir)
        try:
            self.orca_submitter = OrcaJobSubmitter(scratch_dir=self.scratch_dir)
            self.orca_submitter.submit_files(
                input_files=input_files,
                max_cores=cores,
                partition=partition,
                template_dir=self.template_dir,
                output_dir=step_dir
            )
        except Exception as e:
            logging.error(f"Error while submitting ORCA jobs in {step_dir}: {str(e)}")
            raise
        finally:
            os.chdir(original_dir)
            logging.info(f"Returned to original directory: {original_dir}")

    def parse_and_filter_outputs(self, output_files, calculation_type, step_number, sample_method, parameters, step_dir):
        """
        Parses ORCA outputs, saves CSV, filters structures, and moves step files.

        Args:
            output_files (list): List of ORCA output files.
            calculation_type (str): Calculation type.
            step_number (int): Current step number.
            sample_method (str): Sampling method.
            parameters (dict): Filtering parameters.
            step_dir (str): Path to the step directory.

        Returns:
            tuple: Filtered coordinates and IDs.
        """
        coordinates, energies = self.orca.parse_output(output_files, calculation_type, dir=step_dir)
        filtered_ids = list(range(len(energies)))
        self.utils.save_step_csv(energies, filtered_ids, step_number, output_dir=step_dir)
        filtered_coordinates, filtered_ids = self.refiner.filter(
            coordinates, energies, filtered_ids, sample_method, parameters
        )
        
        return filtered_coordinates, filtered_ids

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

            if calculation_type not in calculation_functions:
                raise ValueError(f"Invalid calculation type '{calculation_type}' in step {step_number}. Exiting...")

            if skip:
                filtered_coordinates, filtered_ids = self.handle_skip_step(
                    step_number, calculation_type, sample_method, parameters
                )
                if filtered_coordinates is not None and filtered_ids is not None:
                    continue

            logging.info(f"Running step {step_number}: {calculation_type} with sampling method '{sample_method}'")

            if step_number == 1:
                step_dir, input_files, output_files = self.prepare_step1_directory(step_number)
            else:
                if calculation_type != "MLFF":
                    step_dir, input_files, output_files = self.prepare_subsequent_step_directory(
                        step_number, filtered_coordinates, filtered_ids
                    )
                else:
                    raise ValueError("MLFF support is still under construction. Exiting...")

            logging.info(f"Step directory prepared: {step_dir}")

            self.submit_orca_jobs(input_files, cores, step_dir)
            filtered_coordinates, filtered_ids = self.parse_and_filter_outputs(
                output_files, calculation_type, step_number, sample_method, parameters, step_dir
            )

def main():
    ChemRefiner().run()

if __name__ == "__main__":
        ChemRefiner().run()

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
        self.args, self.qorca_flags = self.arg_parser.parse()
        self.input_yaml = self.args.input_yaml
        self.max_cores = self.args.maxcores
        self.skip_steps = self.args.skip
        self.parameters = self.load_yaml_parameters(self.input_yaml)
        

        # === Load the YAML configuration ===
        with open(self.input_yaml, 'r') as file:
            self.config = yaml.safe_load(file)

        # === Pull top-level config ===
        self.charge = self.config.get('charge', 0)
        self.multiplicity = self.config.get('multiplicity', 1)
        self.template_dir = os.path.abspath(self.config.get('template_dir', './templates'))
        self.scratch_dir = self.config.get('scratch_dir')
        logging.info(f"Using template directory: {self.template_dir}")
        logging.info(f"Using scratch directory: {self.scratch_dir}")

        # === Setup output directory AFTER config is loaded ===
        output_dir_raw = self.config.get('outputs', './outputs')  # Default to './outputs'
        self.scratch_dir = os.path.abspath(self.scratch_dir)
        self.output_dir = os.path.abspath(output_dir_raw)
        os.makedirs(self.output_dir, exist_ok=True)
        logging.info(f"Output directory set to: {self.output_dir}")

        os.makedirs(self.output_dir, exist_ok=True)

    def load_yaml_parameters(self, yaml_path):
        """
        Loads the YAML configuration from file.

        Args:
            yaml_path (str): Path to the YAML file.

        Returns:
            dict: Parsed YAML configuration.
        """
        import yaml
        with open(yaml_path, 'r') as file:
            return yaml.safe_load(file)

    def prepare_step1_directory(self, step_number):
        step_dir = os.path.join(self.output_dir, f"step{step_number}")
        os.makedirs(step_dir, exist_ok=True)        
        logging.debug(f"step_dir BEFORE: {step_dir}")


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
        Handles skip logic for a step if its directory already exists and required outputs are present.

        Args:
            step_number (int): The current step number.
            calculation_type (str): The type of calculation ('dft', 'goat', etc.).
            sample_method (str): The sampling method.
            parameters (dict): Additional parameters for filtering.

        Returns:
            tuple: (filtered_coordinates, filtered_ids) if step is skipped, else (None, None).
        """
        step_dir = os.path.join(self.output_dir, f"step{step_number}")
        if os.path.exists(step_dir):
            logging.info(f"Checking skip condition for step {step_number} at: {step_dir}")

            if calculation_type.lower() == 'goat':
                output_files = [
                    os.path.join(step_dir, f)
                    for f in os.listdir(step_dir)
                    if f.endswith('.finalensemble.xyz')
                ]
                if not output_files:
                    logging.warning(f"No GOAT ensemble files found in {step_dir}. Will rerun this step.")
                    return None, None
                logging.info(f"Found {len(output_files)} GOAT ensemble file(s) in {step_dir}. Skipping this step.")
            else:
                output_files = [
                    os.path.join(step_dir, f)
                    for f in os.listdir(step_dir)
                    if f.endswith('.out')
                ]
                if not output_files:
                    logging.warning(f"No .out files found in {step_dir}. Will rerun this step.")
                    return None, None
                logging.info(f"Found {len(output_files)} .out file(s) in {step_dir}. Skipping this step.")

            coordinates, energies = self.orca.parse_output(output_files, calculation_type, dir=step_dir)
            filtered_coordinates, filtered_ids = self.refiner.filter(
                coordinates, energies, list(range(len(energies))), sample_method, parameters
            )
            return filtered_coordinates, filtered_ids
        else:
            logging.info(f"Step directory {step_dir} does not exist. Will run this step.")
            return None, None

    def submit_orca_jobs(self, input_files, cores, step_dir):
        """
        Submits ORCA jobs for each input file in the step directory using the OrcaJobSubmitter.

        Args:
            input_files (list): List of ORCA input files.
            cores (int): Maximum total cores allowed for all jobs.
            step_dir (str): Path to the step directory.
        """
        logging.info(f"Switching to working directory: {step_dir}")
        original_dir = os.getcwd()
        os.chdir(step_dir)
        logging.info(f"Current working directory: {os.getcwd()}")
        logging.info(f"Running in {self.scratch_dir} from submit_orca_jobs helper function.")
        try:
            self.orca_submitter = OrcaJobSubmitter(scratch_dir=self.scratch_dir)
            self.orca_submitter.submit_files(
                input_files=input_files,
                max_cores=cores,
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
        self.utils.save_step_csv(energies, filtered_ids, step_number, output_dir=self.output_dir)
        filtered_coordinates, filtered_ids = self.refiner.filter(
            coordinates, energies, filtered_ids, sample_method, parameters
        )
        
        return filtered_coordinates, filtered_ids

    def run(self):
        """
        Main pipeline execution function for ChemRefine.
        Dynamically loops over all steps defined in the YAML input.
        Handles skip-step logic and standard pipeline logic.
        """
        logging.info("Starting ChemRefine pipeline.")

        previous_coordinates, previous_ids = None, None

        steps = self.config.get('steps', [])
        charge = self.config.get('charge', 0)
        multiplicity = self.config.get('multiplicity', 1)
        calculation_functions = ["GOAT", "DFT", "XTB", "MLFF"]

        for step in steps:
            step_number = step['step']
            calculation_type = step['calculation_type'].lower()
            sample_method = step['sample_type']['method']
            parameters = step['sample_type'].get('parameters', {})

            if calculation_type.upper() not in calculation_functions:
                raise ValueError(f"Invalid calculation type '{calculation_type}' in step {step_number}. Exiting...")

            filtered_coordinates, filtered_ids = (None, None)
            if self.skip_steps:
                filtered_coordinates, filtered_ids = self.handle_skip_step(
                    step_number, calculation_type, sample_method, parameters
                )

            if filtered_coordinates is None or filtered_ids is None:
                logging.info(f"No valid skip outputs for step {step_number}. Proceeding with normal execution.")
                if step_number == 1:
                    step_dir, input_files, output_files = self.prepare_step1_directory(step_number)
                else:
                    step_dir, input_files, output_files = self.prepare_subsequent_step_directory(
                        step_number,
                        previous_coordinates,
                        previous_ids
                    )

                self.submit_orca_jobs(input_files, self.max_cores, step_dir)


                coordinates, energies = self.orca.parse_output(
                    output_files,
                    calculation_type,
                    dir=step_dir
                )

                ids = list(range(len(energies)))
                filtered_coordinates, filtered_ids = self.refiner.filter(
                    coordinates,
                    energies,
                    ids,
                    sample_method,
                    parameters
                )
            else:
                step_dir = os.path.join(self.output_dir, f"step{step_number}")
                logging.info(f"Skipping step {step_number} using existing outputs.")

            self.utils.save_step_csv(step_number, filtered_coordinates, filtered_ids,output_dir=self.output_dir)


            previous_coordinates, previous_ids = filtered_coordinates, filtered_ids


        logging.info("ChemRefine pipeline completed.")

def main():
    ChemRefiner().run()

if __name__ == "__main__":
        ChemRefiner().run()

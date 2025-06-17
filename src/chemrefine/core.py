import os
import yaml
import logging
from .parse import ArgumentParser
from .refine import StructureRefiner
from .utils import Utility
from .orca_interface import OrcaInterface, OrcaJobSubmitter
from .mlff import MLFFCalculator, MLFFJobSubmitter
from pathlib import Path
import shutil
import sys

class ChemRefiner:

    """
    ChemRefiner class orchestrates the ChemRefine workflow, handling input parsing,
    job submission, output parsing, and structure refinement based on a YAML configuration.
    It supports multiple steps with different calculation types and sampling methods.
    """
    def __init__(self):
        self.arg_parser = ArgumentParser()
        self.orca_submitter = OrcaJobSubmitter()
        self.mlff_submitter = MLFFJobSubmitter()
        self.refiner = StructureRefiner()
        self.utils = Utility()
        self.orca = OrcaInterface()
        self.args, self.qorca_flags = self.arg_parser.parse()
        self.input_yaml = self.args.input_yaml
        self.max_cores = self.args.maxcores
        self.skip_steps = self.args.skip
        

        # === Load the YAML configuration ===
        with open(self.input_yaml, 'r') as file:
            self.config = yaml.safe_load(file)

        # === Pull top-level config ===
        self.charge = self.config.get('charge', 0)
        self.multiplicity = self.config.get('multiplicity', 1)
        self.template_dir = os.path.abspath(self.config.get('template_dir', './templates'))
        self.scratch_dir = self.config.get('scratch_dir')
        if self.scratch_dir is None:
            self.scratch_dir = "./scratch"
        logging.info(f"Using template directory: {self.template_dir}")
        logging.info(f"Using scratch directory: {self.scratch_dir}")

        # === Setup output directory AFTER config is loaded ===
        output_dir_raw = self.config.get('outputs', './outputs')  # Default to './outputs'
        self.scratch_dir = os.path.abspath(self.scratch_dir)
        self.output_dir = os.path.abspath(output_dir_raw)
        os.makedirs(self.output_dir, exist_ok=True)
        logging.info(f"Output directory set to: {self.output_dir}")

        os.makedirs(self.output_dir, exist_ok=True)

    def prepare_step1_directory(self, step_number, initial_xyz=None,charge=None, multiplicity=None):
        """ Prepares the directory for the first step by copying the initial XYZ file,"""
        if charge is None:
            charge = self.charge
        if multiplicity is None:
            multiplicity = self.multiplicity

        step_dir = os.path.join(self.output_dir, f"step{step_number}")
        os.makedirs(step_dir, exist_ok=True)        
        logging.debug(f"step_dir BEFORE: {step_dir}")

        # Determine source xyz: use override if provided
        if initial_xyz is None:
            src_xyz = os.path.join(self.template_dir, "step1.xyz")
        else:
            src_xyz = initial_xyz

        dst_xyz = os.path.join(step_dir, "step1.xyz")

        if not os.path.exists(src_xyz):
            raise FileNotFoundError(
                f"Initial XYZ file '{src_xyz}' not found. Please ensure the path is correct."
            )

        shutil.copyfile(src_xyz, dst_xyz)

        # Use input template from template_dir
        template_inp = os.path.join(self.template_dir, "step1.inp")
        if not os.path.exists(template_inp):
            raise FileNotFoundError(
                f"Input file '{template_inp}' not found. Please ensure that 'step1.inp' exists in the template directory."
            )

        xyz_filenames = [dst_xyz]

        input_files, output_files = self.orca.create_input(
            xyz_filenames, template_inp, charge, multiplicity, output_dir=step_dir
        )

        return step_dir, input_files, output_files

    def prepare_subsequent_step_directory(self, step_number, filtered_coordinates, filtered_ids,charge=None, multiplicity=None):
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
        if charge is None:
            charge = self.charge
        if multiplicity is None:
            multiplicity = self.multiplicity

        step_dir = os.path.join(self.output_dir, f"step{step_number}")
        os.makedirs(step_dir, exist_ok=True)

        # Write XYZ files in step_dir
        xyz_filenames = self.utils.write_xyz(filtered_coordinates, step_number, filtered_ids, output_dir=step_dir)


        # Copy the template input file from template_dir to step_dir
        input_template_src = os.path.join(self.template_dir, f"step{step_number}.inp")
        input_template_dst = os.path.join(step_dir, f"step{step_number}.inp")
        if not os.path.exists(input_template_src):
            logging.warning(f"Input file '{input_template_src}' not found. Exiting pipeline.")
            sys.exit(1)
            raise FileNotFoundError(
                f"Input file '{input_template_src}' not found. Please ensure that 'step{step_number}.inp' exists in the template directory."
            )        
        shutil.copyfile(input_template_src, input_template_dst)

        # Create ORCA input files in step_dir
        input_files, output_files = self.orca.create_input(
            xyz_filenames, input_template_dst, charge, multiplicity, output_dir=step_dir
        )

        return step_dir, input_files, output_files

    def prepare_mlff_step1_directory(self, step_number, initial_xyz=None):
        step_dir = os.path.join(self.output_dir, f"step{step_number}")
        os.makedirs(step_dir, exist_ok=True)

        if initial_xyz is None:
            src_xyz = os.path.join(self.template_dir, "step1.xyz")
        else:
            src_xyz = initial_xyz

        dst_xyz = os.path.join(step_dir, "step1.xyz")

        if not os.path.exists(src_xyz):
            raise FileNotFoundError(
                f"XYZ file '{src_xyz}' not found. Please ensure the path is correct."
            )

        shutil.copyfile(src_xyz, dst_xyz)
        return step_dir, [dst_xyz]

    def prepare_mlff_directory(self, step_number, coordinates, ids):
        step_dir = os.path.join(self.output_dir, f"step{step_number}")
        os.makedirs(step_dir, exist_ok=True)
        xyz_files = self.utils.write_xyz(coordinates, step_number, ids, output_dir=step_dir)
        return step_dir, xyz_files

    def handle_skip_step(self, step_number, calculation_type, sample_method, parameters):
        """
        Handles skip logic for a step if its directory already exists and required outputs are present.

        Args:
            step_number (int): The current step number.
            calculation_type (str): The type of calculation ('dft', 'goat', etc.).
            sample_method (str): The sampling method.
            parameters (dict): Additional parameters for filtering.

        Returns:
            tuple: (filtered_coordinates, filtered_ids, energies) if step is skipped, else (None, None, None).
        """
        step_dir = os.path.join(self.output_dir, f"step{step_number}")
        if os.path.exists(step_dir):
            logging.info(f"Checking skip condition for step {step_number} at: {step_dir}")

            if calculation_type.lower() == 'mlff':
                return None, None,None
            elif calculation_type.lower() == 'goat':
                output_files = [
                    os.path.join(step_dir, f)
                    for f in os.listdir(step_dir)
                    if f.endswith('.finalensemble.xyz')
                ]
                if not output_files:
                    logging.warning(f"No GOAT ensemble files found in {step_dir}. Will rerun this step.")
                    return None, None,None
                logging.info(f"Found {len(output_files)} GOAT ensemble file(s) in {step_dir}. Skipping this step.")
            else:
                output_files = [
                    os.path.join(step_dir, f)
                    for f in os.listdir(step_dir)
                    if f.endswith('.out') and not f.endswith('.smd.out')
                ]
                if not output_files:
                    logging.warning(f"No .out files found in {step_dir}. Will rerun this step.")
                    return None, None,None
                logging.info(f"Found {len(output_files)} .out file(s) in {step_dir}. Skipping this step.")

            coordinates, energies = self.orca.parse_output(output_files, calculation_type, dir=step_dir)
            logging.info(f"Parsed {len(coordinates)} coordinates and {len(energies)} energies from {output_files}.")
            filtered_coordinates, filtered_ids = self.refiner.filter(
                coordinates, energies, list(range(len(energies))), sample_method, parameters
            )
            logging.info(f"Filtered {len(filtered_coordinates)} coordinates and {len(filtered_ids)} IDs after filtering.")
            return filtered_coordinates, filtered_ids, energies
        else:
            logging.info(f"Step directory {step_dir} does not exist. Will run this step.")
            return None, None, None

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

        #self.utils.move_step_files(step_number, output_dir=step_dir)

        return filtered_coordinates, filtered_ids

    def run_mlff_step(
        self,
        step_number: int,
        model_name: str,
        task_name: str,
        sample_method: str,
        parameters: dict,
        previous_coordinates,
        previous_ids
    ):
        """
        Handles MLFF job preparation, submission, parsing, and result filtering for a given step.

        Returns
        -------
        tuple: (filtered_coordinates, filtered_ids, step_dir, xyz_files)
        """
        from .mlff import parse_mlff_output
        from ase.io import read

        if step_number == 1:
            step_dir, xyz_files = self.prepare_mlff_step1_directory(step_number)
            # Generate initial IDs for step 1
            structures = read(xyz_files[0], index=":")
            previous_ids = list(range(len(structures)))
        else:
            step_dir, xyz_files = self.prepare_mlff_directory(
                step_number,
                previous_coordinates,
                previous_ids,
            )

        original_dir = os.getcwd()
        logging.info(f"Switching to working directory: {step_dir}")
        os.chdir(step_dir)
        try:
            self.mlff_submitter.submit_jobs(
                xyz_files=xyz_files,
                template_dir=self.template_dir,
                output_dir=step_dir,
                model_name=model_name,
                fmax=0.03,
                steps=200,
                task_name=task_name
            )
        except Exception as e:
            logging.error(f"Failed to submit MLFF jobs in {step_dir}: {e}")
            raise
        finally:
            os.chdir(original_dir)
            logging.info(f"Returned to original directory: {original_dir}")

        # === Parse MLFF output ===
        coords, energy, forces = parse_mlff_output(xyz_files)  # Assuming one XYZ per step

        # === Filter ===
        filtered_coordinates, filtered_ids = self.refiner.filter(
            coords,
            energy,
            previous_ids,
            sample_method,
            parameters,
        )

        return filtered_coordinates, filtered_ids, step_dir, xyz_files

    def run(self):
        """
        Main pipeline execution function for ChemRefine.
        Dynamically loops over all steps defined in the YAML input.
        Handles skip-step logic and standard pipeline logic.
        """
        logging.info("Starting ChemRefine pipeline.")
        previous_coordinates, previous_ids = None, None
        steps = self.config.get('steps', [])
        calculation_functions = ["GOAT", "DFT", "XTB", "MLFF"]

        for step in steps:
            logging.info(f"Processing step {step['step']} with calculation type '{step['calculation_type']}'.")
            step_number = step['step']
            calculation_type = step['calculation_type'].lower()
            
            # Handle MLFF-specific nested structure
            if calculation_type == 'mlff':
                mlff_config = step.get('mlff', {})
                model_name = mlff_config.get('model_name', 'mace')
                task_name = mlff_config.get('task_name', 'mace_off')
            else:
                model_name = step.get('model_name', 'mace')
                task_name = step.get('task_type', 'mace_off')

            sample_method = step['sample_type']['method']
            parameters = step['sample_type'].get('parameters', {})

            if calculation_type.upper() not in calculation_functions:
                raise ValueError(f"Invalid calculation type '{calculation_type}' in step {step_number}. Exiting...")

            filtered_coordinates, filtered_ids, energies = (None, None, None)
            if self.skip_steps:
                filtered_coordinates, filtered_ids, energies = self.handle_skip_step(
                    step_number, calculation_type, sample_method, parameters
                )

            if filtered_coordinates is None or filtered_ids is None:
                logging.info(f"No valid skip outputs for step {step_number}. Proceeding with normal execution.")

                if calculation_type == 'mlff':
                    logging.info(f"Running MLFF step {step_number} with model '{model_name}' and task '{task_name}'.")
                    filtered_coordinates, filtered_ids, step_dir, xyz_files = self.run_mlff_step(
                        step_number=step_number,
                        model_name=model_name,
                        task_name=task_name,
                        sample_method=sample_method,
                        parameters=parameters,
                        previous_coordinates=previous_coordinates,
                        previous_ids=previous_ids,
                    )

                else:
                    if step_number == 1:
                        initial_xyz = self.config.get("initial_xyz", None)
                        charge = step.get('charge', self.charge)
                        multiplicity = step.get('multiplicity', self.multiplicity)
                        logging.info(f"Step {step_number} using charge={charge}, multiplicity={multiplicity}")
                        step_dir, input_files, output_files = self.prepare_step1_directory(
                            step_number,
                            initial_xyz=initial_xyz,
                            charge=charge,
                            multiplicity=multiplicity
                            )
                    else:
                        charge = step.get('charge', self.charge)
                        multiplicity = step.get('multiplicity', self.multiplicity)
                        logging.info(f"Step {step_number} using charge={charge}, multiplicity={multiplicity}")
                        step_dir, input_files, output_files = self.prepare_subsequent_step_directory(
                            step_number,
                            previous_coordinates,
                            previous_ids,
                            charge=charge,
                            multiplicity=multiplicity
    )


                    self.submit_orca_jobs(input_files, self.max_cores, step_dir)

                    coordinates, energies = self.orca.parse_output(
                        output_files,
                        calculation_type,
                        dir=step_dir,  # Correct output directory used here
                    )

                    ids = list(range(len(energies)))
                    filtered_coordinates, filtered_ids = self.refiner.filter(
                        coordinates,
                        energies,
                        ids,
                        sample_method,
                        parameters,
                    )
            else:
                step_dir = os.path.join(self.output_dir, f"step{step_number}")
                logging.info(f"Skipping step {step_number} using existing outputs.")


            previous_coordinates, previous_ids = filtered_coordinates, filtered_ids

        logging.info("ChemRefine pipeline completed.")

def main():
    ChemRefiner().run()

if __name__ == "__main__":
    main()

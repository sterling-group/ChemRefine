import os
import yaml
import logging
from .parse import ArgumentParser
from .refine import StructureRefiner
from .utils import Utility
from .orca_interface import OrcaInterface, OrcaJobSubmitter
import shutil
import sys

class ChemRefiner:

    """
    ChemRefiner class orchestrates the ChemRefine workflow, handling input parsing,
    job submission, output parsing, and structure refinement based on a YAML configuration.
    It supports multiple steps with different calculation types and sampling methods.
    """
    def __init__(self,):
        self.arg_parser = ArgumentParser()
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
        self.scratch_dir = self.config.get('scratch_dir', "./scratch")
        self.orca_executable = self.config.get('orca_executable', 'orca')

        # === Setup output directory ===
        output_dir_raw = self.config.get('output_dir', './outputs')
        self.output_dir = os.path.abspath(output_dir_raw)
        os.makedirs(self.output_dir, exist_ok=True)
        self.scratch_dir = os.path.abspath(self.scratch_dir)

        logging.info(f"Using template directory: {self.template_dir}")
        logging.info(f"Using scratch directory: {self.scratch_dir}")
        logging.info(f"Output directory set to: {self.output_dir}")

        # === Instantiate components AFTER config ===
        self.refiner = StructureRefiner()
        self.utils = Utility()
        self.orca = OrcaInterface()

    def prepare_step1_directory(self, 
                                step_number, 
                                initial_xyz=None,
                                charge=None, 
                                multiplicity=None,
                                operation='OPT+SP',
                                engine='dft',
                                model_name=None, 
                                task_name=None,
                                device='cpu',
                                bind='127.0.0.1:8888'):
        
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
            xyz_filenames, 
            template_inp, 
            charge, 
            multiplicity, 
            output_dir=step_dir,
            operation=operation,
            engine=engine,
            model_name=model_name,
            task_name=task_name,
            device=device,
            bind=bind
        )

        return step_dir, input_files, output_files

    def prepare_subsequent_step_directory(self, 
                                          step_number, 
                                          filtered_coordinates, 
                                          filtered_ids,charge=None, 
                                          multiplicity=None,
                                          operation='OPT+SP',
                                          engine='dft',
                                          model_name=None, 
                                          task_name=None,
                                          device='cuda',
                                          bind='127.0.0.1:8888'
                                          ):
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
            xyz_filenames, 
            input_template_dst, 
            charge, 
            multiplicity, 
            output_dir=step_dir,
            operation=operation,
            engine=engine,
            model_name=model_name, 
            task_name=task_name,
            device=device,
            bind=bind
        )

        return step_dir, input_files, output_files

    def handle_skip_step(self, step_number, operation,engine, sample_method, parameters):
        """
        Handles skip logic for a step if its directory already exists and required outputs are present.

        Args:
            step_number (int): The current step number.
            operation (str): Algorithm used in ORCA (i.e GOAT, DOCKER, PES, OPT+SP).
            engine (str): The calculation engine (e.g., 'dft', 'mlff
            sample_method (str): The sampling method.
            parameters (dict): Additional parameters for filtering.

        Returns:
            tuple: (filtered_coordinates, filtered_ids, energies) if step is skipped, else (None, None, None).
        """
        step_dir = os.path.join(self.output_dir, f"step{step_number}")
        if os.path.exists(step_dir):
            logging.info(f"Checking skip condition for step {step_number} at: {step_dir}")

            
            if operation.lower() == 'goat':
                output_files = [
                    os.path.join(step_dir, f)
                    for f in os.listdir(step_dir)
                    if f.endswith('.finalensemble.xyz')
                ]
                if not output_files:
                    logging.warning(f"No GOAT ensemble files found in {step_dir}. Will rerun this step.")
                    return None, None,None
                logging.info(f"Found {len(output_files)} GOAT ensemble file(s) in {step_dir}. Skipping this step.")

            elif operation.lower() == 'docker':
                output_files = [
                    os.path.join(step_dir, f)
                    for f in os.listdir(step_dir)
                    if f.endswith('struc1.allopt.xyz')
                ]
                if not output_files:
                    logging.warning(f"No GOAT ensemble files found in {step_dir}. Will rerun this step.")
                    return None, None,None
                logging.info(f"Found {len(output_files)} GOAT ensemble file(s) in {step_dir}. Skipping this step.")
            else:
                output_files = [
                    os.path.join(step_dir, f)
                    for f in os.listdir(step_dir)
                    if f.endswith('.out') and not f.endswith('.smd.out') and not f.startswith('slurm')
                ]
                if not output_files:
                    logging.warning(f"No .out files found in {step_dir}. Will rerun this step.")
                    return None, None,None
                logging.info(f"Found {len(output_files)} .out file(s) in {step_dir}. Skipping this step.")

            coordinates, energies = self.orca.parse_output(output_files, operation, dir=step_dir)
            logging.info(f"Parsed {len(coordinates)} coordinates and {len(energies)} energies from {output_files}.")
            filtered_coordinates, filtered_ids = self.refiner.filter(
                coordinates, energies, list(range(len(energies))), sample_method, parameters
            )
            logging.info(f"Filtered {len(filtered_coordinates)} coordinates and {len(filtered_ids)} IDs after filtering.")
            return filtered_coordinates, filtered_ids, energies
        else:
            logging.info(f"Step directory {step_dir} does not exist. Will run this step.")
            return None, None, None

    def submit_orca_jobs(self, input_files, cores, step_dir,device='cpu',operation='OPT+SP',engine='DFT', model_name=None, task_name=None):
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
            logging.info(f"Submitting ORCA jobs in {step_dir} with {len(input_files)} input files using {device}.")
            self.orca_submitter = OrcaJobSubmitter(scratch_dir=self.scratch_dir,orca_executable=self.orca_executable,device=device)
            self.orca_submitter.submit_files(
                input_files=input_files,
                max_cores=cores,
                template_dir=self.template_dir,
                output_dir=step_dir,
                engine=engine,
                operation=operation,
                model_name=model_name,
                task_name=task_name
                
               
            )
        except Exception as e:
            logging.error(f"Error while submitting ORCA jobs in {step_dir}: {str(e)}")
            raise
        finally:
            os.chdir(original_dir)
            logging.info(f"Returned to original directory: {original_dir}")

    def parse_and_filter_outputs(self, output_files, operation,engine, step_number, sample_method, parameters, step_dir,previous_ids=None):
        """
        Parses ORCA outputs, saves CSV, filters structures, and moves step files.

        Args:
            output_files (list): List of ORCA output files.
            step_number (int): Current step number.
            sample_method (str): Sampling method.
            parameters (dict): Filtering parameters.
            step_dir (str): Path to the step directory.

        Returns:
            tuple: Filtered coordinates and IDs.
        """
        coordinates, energies = self.orca.parse_output(output_files, operation, dir=step_dir)
        if not coordinates or not energies:
            logging.error(f"No valid coordinates or energies found in outputs for step {step_number}. Exiting pipeline.")
            logging.error(f"Error in your output file, please check reason for failure")
            sys.exit(1)
        if previous_ids is None:
            previous_ids = list(range(len(energies)))  # only for step 1

        self.utils.save_step_csv(
                    energies=energies,
                    ids=previous_ids,
                    step=step_number,
                    output_dir=self.output_dir
                    )
        filtered_coordinates, selected_ids = self.refiner.filter(
                                            coordinates, 
                                            energies, 
                                            previous_ids, 
                                            sample_method, 
                                            parameters
                                            )

        return filtered_coordinates, selected_ids

    def run(self):
        """
        Main pipeline execution function for ChemRefine.
        Dynamically loops over all steps defined in the YAML input.
        Handles skip-step logic and standard pipeline logic.
        """
        logging.info("Starting ChemRefine pipeline.")
        previous_coordinates, previous_ids = None, None

        valid_operations = {"OPT+SP","GOAT", "PES", "DOCKER", "SOLVATOR"}
        valid_engines = {"dft", "mlff"}

        steps = self.config.get('steps', [])

        for step in steps:
            step_id = step['step']
            operation = step['operation'].upper()
            engine = step.get('engine', 'dft').lower()

            logging.info(f"Processing step {step_id}: operation '{operation}', engine '{engine}'.")

            if operation not in valid_operations:
                raise ValueError(f"Invalid operation '{operation}' at step {step_id}. Must be one of {valid_operations}.")

            if engine not in valid_engines:
                raise ValueError(f"Invalid engine '{engine}' at step {step_id}. Must be one of {valid_engines}.")

            if engine == 'mlff':
                mlff_config = step.get('mlff', {})
                mlff_model = mlff_config.get('model_name', 'medium')
                mlff_task = mlff_config.get('task_name', 'mace_off')
                bind_address = mlff_config.get('bind', '127.0.0.1:8888')
                device = mlff_config.get('device', 'cuda')

                logging.info(f"Using MLFF model '{mlff_model}' with task '{mlff_task}' for step {step_id}.")
            else:
                mlff_model = step.get('model_name', 'medium')
                mlff_task = step.get('task_name', 'mace_off')
                bind_address = '127.0.0.1:8888'
                device = 'cpu'

            sample_method = step['sample_type']['method']
            parameters = step['sample_type'].get('parameters', {})

            filtered_coordinates, filtered_ids, energies = (None, None, None)
            if self.skip_steps:
                filtered_coordinates, filtered_ids, energies = self.handle_skip_step(
                    step_id, operation,engine, sample_method, parameters
                )

            if filtered_coordinates is None or filtered_ids is None:
                logging.info(f"No valid skip outputs for step {step_id}. Proceeding with normal execution.")
                charge = step.get('charge', self.charge)
                multiplicity = step.get('multiplicity', self.multiplicity)

                if step_id == 1:
                    initial_xyz = self.config.get("initial_xyz", None)
                    step_dir, input_files, output_files = self.prepare_step1_directory(
                        step_id,
                        initial_xyz=initial_xyz,
                        charge=charge,
                        multiplicity=multiplicity,
                        operation=operation,
                        engine=engine,
                        model_name=mlff_model,
                        task_name=mlff_task,
                        device=device,
                        bind=bind_address
                    )
                else:
                    step_dir, input_files, output_files = self.prepare_subsequent_step_directory(
                        step_id,
                        previous_coordinates,
                        previous_ids,
                        charge=charge,
                        multiplicity=multiplicity,
                        operation=operation,
                        engine=engine,
                        model_name=mlff_model,
                        task_name=mlff_task,
                        device=device,
                        bind=bind_address,
                    )

                self.submit_orca_jobs(
                    input_files,
                    self.max_cores,
                    step_dir,
                    operation=operation,
                    engine=engine,
                    model_name=mlff_model,
                    task_name=mlff_task,
                    device=device
                )

                filtered_coordinates, filtered_ids = self.parse_and_filter_outputs(
                    output_files,
                    operation,
                    engine,
                    step_id,
                    sample_method,
                    parameters,
                    step_dir
                )

                if filtered_coordinates is None or filtered_ids is None:
                    logging.error(f"Filtering failed at step {step_id}. Exiting pipeline.")
                    return
            else:
                step_dir = os.path.join(self.output_dir, f"step{step_id}")
                logging.info(f"Skipping step {step_id} using existing outputs.")

            previous_coordinates, previous_ids = filtered_coordinates, filtered_ids

        logging.info("ChemRefine pipeline completed.")


def main():
    ChemRefiner().run()

if __name__ == "__main__":
    main()

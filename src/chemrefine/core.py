import os
import yaml
import logging
from .parse import ArgumentParser
from .refine import StructureRefiner
from .utils import Utility
from .orca_interface import OrcaInterface, OrcaJobSubmitter
import shutil
import re
import sys
from chemrefine.utils import (
    write_step_manifest,
    update_step_manifest_outputs,
    map_outputs_to_ids,
    extract_structure_id,
    write_step_manifest,
    write_synthetic_manifest_for_ensemble,
    get_ensemble_ids,
    validate_structure_ids_or_raise
)

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
                                bind='127.0.0.1:8888',
                                ):
        
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
            bind=bind,
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
                                          bind='127.0.0.1:8888',
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
            bind=bind,
            
        )

        return step_dir, input_files, output_files
    
    def handle_skip_step(self, step_number, operation, engine, sample_method, parameters):
        """
        Decide whether to skip a step by validating that expected outputs already exist.
        Preserves persistent structure IDs via the per-step manifest when available.
        If a manifest is missing (legacy runs), reconstruct IDs from filenames or synthesize
        them for ensemble-like outputs (GOAT, SOLVATOR). Also persists a manifest so
        subsequent runs can skip cleanly.

        Parameters
        ----------
        step_number : int
            Current step index.
        operation : str
            Operation ("OPT+SP", "GOAT", "PES", "DOCKER", "SOLVATOR"), case-insensitive.
        engine : str
            Calculation engine ("dft" or "mlff").
        sample_method : str
            Refiner filtering method.
        parameters : dict
            Parameters for the filtering method.

        Returns
        -------
        tuple[list|None, list|None, list|None]
            (filtered_coordinates, filtered_ids, energies) when outputs are reusable;
            otherwise (None, None, None) to signal re-run.
        """
        step_dir = os.path.join(self.output_dir, f"step{step_number}")
        if not os.path.exists(step_dir):
            logging.info(f"Step directory {step_dir} does not exist. Will run this step.")
            return None, None, None

        op = operation.strip().upper()
        def _ensure_solvator_ids(step_dir, step_number, engine, output_files, energies, structure_ids):
            """
            Normalize SOLVATOR IDs to per-structure cardinality and persist a synthetic manifest.

            Parameters
            ----------
            step_dir : str
                Step directory path.
            step_number : int
                Current step index.
            engine : str
                Calculation engine ("dft" or "mlff").
            output_files : list[str]
                SOLVATOR outputs (we use the first as basename anchor).
            energies : list[float]
                Energies parsed (one per structure).
            structure_ids : list[int]
                Current IDs (often length 1 for SOLVATOR).

            Returns
            -------
            list[int]
                Expanded per-structure IDs [0..N-1] of length len(energies).

            Notes
            -----
            Keeps it simple: child IDs are 0..N-1. This avoids the length-mismatch
            error and keeps downstream typing unchanged (ints). A synthetic manifest
            is written so subsequent runs skip cleanly.
            """
            n_structs = len(energies)
            if n_structs == 0:
                return structure_ids
            if len(structure_ids) == n_structs:
                return structure_ids  # already aligned

            solv_base = os.path.basename(output_files[0])
            write_synthetic_manifest_for_ensemble(
                step_number=step_number,
                step_dir=step_dir,
                n_structures=n_structs,
                operation="SOLVATOR",
                engine=engine,
                output_basename=solv_base,
            )
            return list(range(n_structs))
        # ---------- Discover outputs by operation ----------
        if op == "GOAT":
            output_files = [
                os.path.join(step_dir, f)
                for f in os.listdir(step_dir)
                if f.endswith(".finalensemble.xyz")
            ]
            missing_msg = "No GOAT ensemble (.finalensemble.xyz) files found"
            found_msg = f"Found {len(output_files)} GOAT ensemble file(s)"

        elif op == "DOCKER":
            output_files = [
                os.path.join(step_dir, f)
                for f in os.listdir(step_dir)
                if f.endswith("struc1.allopt.xyz")
            ]
            missing_msg = "No DOCKER output (struc1.allopt.xyz) files found"
            found_msg = f"Found {len(output_files)} DOCKER output file(s)"

        elif op == "SOLVATOR":
            # Prefer *.solvator.xyz (canonical). Fall back to a single .out if present.
            xyz_candidates = [
                os.path.join(step_dir, f)
                for f in os.listdir(step_dir)
                if f.endswith("solvator.solventbuild.xyz")
            ]
            if xyz_candidates:
                output_files = xyz_candidates[:1]
            else:
                output_files = [
                    os.path.join(step_dir, f)
                    for f in os.listdir(step_dir)
                    if f.endswith(".out") and not f.startswith("slurm") and not f.startswith("atom")
                ][:1]
            missing_msg = "No SOLVATOR outputs (.solvator.xyz or .out) found"
            found_msg = f"Found {len(output_files)} SOLVATOR output file(s)"

        else:
            # OPT+SP / PES: ORCA .out files (exclude SLURM and atom-specific files).
            # Be strict: only 'stepN.out' or 'stepN_structure_{id}.out'
            out_pat = re.compile(rf"^step{step_number}(?:_structure_-?\d+)?\.out$", re.IGNORECASE)
            output_files = [
                os.path.join(step_dir, f)
                for f in os.listdir(step_dir)
                if f.endswith(".out")
                and not f.startswith("slurm")
                and not f.startswith("atom")
                and out_pat.match(f) is not None
            ]
            missing_msg = "No ORCA .out files found"
            found_msg = f"Found {len(output_files)} .out file(s)"

        if not output_files:
            logging.warning(f"{missing_msg} in {step_dir}. Will rerun this step.")
            return None, None, None

        logging.info(f"{found_msg} in {step_dir}. Reusing existing outputs.")

        # Update manifest (noop if missing)
        update_step_manifest_outputs(step_dir, step_number, output_files)

        # ---------- Parse outputs ----------
        coordinates, energies = self.orca.parse_output(output_files, op, dir=step_dir)
        if not coordinates or not energies or len(coordinates) != len(energies):
            logging.warning(
                f"Parsed outputs are incomplete or inconsistent for step {step_number} "
                f"(coords={len(coordinates)}, energies={len(energies)}). Will rerun this step."
            )
            return None, None, None

        # ---------- Resolve IDs ----------
        # 1) Try manifest mapping first.
        structure_ids = map_outputs_to_ids(step_dir, step_number, output_files)

        # 2) If unresolved, recover/synthesize depending on operation.
        if all(i < 0 for i in structure_ids):
            if op == "GOAT":
                # One ensemble file → N structures; synthesize 0..N-1 and persist.
                logging.info(f"No usable manifest for GOAT step {step_number}; synthesizing ensemble IDs.")
                ensemble_base = os.path.basename(output_files[0])
                n_structs = len(energies)
                write_synthetic_manifest_for_ensemble(
                    step_number=step_number,
                    step_dir=step_dir,
                    n_structures=n_structs,
                    operation=op,
                    engine=engine,
                    output_basename=ensemble_base,
                )
                structure_ids = list(range(n_structs))

            elif op == "SOLVATOR":
                # SOLVATOR: treat as ensemble of length len(energies), usually 1.
                logging.info(f"No usable manifest for SOLVATOR step {step_number}; synthesizing IDs.")
                solv_base = os.path.basename(output_files[0])
                n_structs = len(energies)
                if n_structs <= 0:
                    logging.warning(f"SOLVATOR parsed no structures for step {step_number}; rerunning.")
                    return None, None, None
                write_synthetic_manifest_for_ensemble(
                    step_number=step_number,
                    step_dir=step_dir,
                    n_structures=n_structs,
                    operation=op,
                    engine=engine,
                    output_basename=solv_base,
                )
                structure_ids = list(range(n_structs))

            else:
                # Non-GOAT/SOLVATOR: per-structure outputs or scalar case.
                logging.info(f"No manifest for step {step_number}; reconstructing IDs from filenames.")

                # Scalar OPT/SP: single .out and single structure → assign ID 0 and persist manifest.
                if len(output_files) == 1 and len(energies) == 1:
                    logging.info(
                        f"Step {step_number}: single-output scalar case; assigning ID 0 and writing manifest."
                    )
                    structure_ids = [0]
                    # If a real input exists, use it; else synthesize the filename in the record.
                    candidate_inp = os.path.join(step_dir, f"step{step_number}_structure_0.inp")
                    input_list = [candidate_inp] if os.path.exists(candidate_inp) else [f"step{step_number}_structure_0.inp"]
                    write_step_manifest(step_number, step_dir, input_list, op, engine)
                    update_step_manifest_outputs(step_dir, step_number, output_files)

                else:
                    # General multi-file/ID recovery (handles suffixes like _atom46).
                    pat = re.compile(rf"^step{step_number}_structure_(\d+)(?:_|$)", re.IGNORECASE)
                    recovered_ids = []
                    recovered_inps = []

                    for outp in output_files:
                        stem = os.path.splitext(os.path.basename(outp))[0]
                        m = pat.match(stem)
                        if m:
                            sid = int(m.group(1))
                        else:
                            # Try a corresponding .inp with truncated suffix
                            truncated = stem.rsplit("_", 1)[0] if "_" in stem else stem
                            candidate_inp = os.path.join(step_dir, truncated + ".inp")
                            sid = extract_structure_id(candidate_inp)

                        if sid is None:
                            logging.warning(f"Could not resolve ID for {outp}; rerunning step.")
                            return None, None, None

                        recovered_ids.append(sid)
                        candidate_inp2 = os.path.join(step_dir, f"step{step_number}_structure_{sid}.inp")
                        if os.path.exists(candidate_inp2):
                            recovered_inps.append(candidate_inp2)

                    structure_ids = recovered_ids
                    # Persist recovered mapping if we found any corresponding inputs
                    if recovered_inps:
                        write_step_manifest(step_number, step_dir, recovered_inps, op, engine)
                        update_step_manifest_outputs(step_dir, step_number, output_files)

        # ---------- Minimal change: expand SOLVATOR IDs if counts mismatch ----------
        if op == "SOLVATOR" and len(structure_ids) != len(energies):
            structure_ids = _ensure_solvator_ids(
                step_dir=step_dir,
                step_number=step_number,
                engine=engine,
                output_files=output_files,
                energies=energies,
                structure_ids=structure_ids,
            )

        # Sanity: lengths must agree before filtering
        if len(structure_ids) != len(energies) or len(coordinates) != len(energies):
            logging.error(
                f"Step {step_number} ({op}): length mismatch before filtering: "
                f"coords={len(coordinates)}, energies={len(energies)}, ids={len(structure_ids)}."
            )
            return None, None, None

        # ---------- Filter ----------
        filtered_coordinates, filtered_ids = self.refiner.filter(
            coordinates, energies, structure_ids, sample_method, parameters
        )
        logging.info(f"After filtering step {step_number}: kept {len(filtered_coordinates)} structures.")
        return filtered_coordinates, filtered_ids, energies
 
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
                task_name=task_name,
                model_path=model_path
               
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
        Handles skip-step logic, ID persistence, and normal pipeline execution.
        """
        logging.info("Starting ChemRefine pipeline.")
        previous_coordinates, previous_ids = None, None

        valid_operations = {"OPT+SP", "GOAT", "PES", "DOCKER", "SOLVATOR"}
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
                mlff_model_path = mlff_config.get('model_path', None)
                bind_address = mlff_config.get('bind', '127.0.0.1:8888')
                device = mlff_config.get('device', 'cuda')
                logging.info(f"Using MLFF model '{mlff_model}' with task '{mlff_task}' for step {step_id}.")
                if mlff_model_path:
                    logging.info(f"Custom MLFF model path specified {mlff_model_path}.")
            else:
               
                mlff_model = None
                mlff_task = None
                mlff_model_path = None
                bind_address = None
                device = None

            sample_method = step['sample_type']['method']
            parameters = step['sample_type'].get('parameters', {})

            filtered_coordinates, filtered_ids, energies = (None, None, None)
            if self.skip_steps:
                filtered_coordinates, filtered_ids, energies = self.handle_skip_step(
                    step_id, operation, engine, sample_method, parameters
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
                        bind=bind_address,
                    )
                else:
                    validate_structure_ids_or_raise(previous_ids, step_id)
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

                # Save manifest with input structure IDs
                write_step_manifest(step_id, step_dir, input_files, operation, engine)

                # Submit jobs
                self.submit_orca_jobs(
                    input_files,
                    self.max_cores,
                    step_dir,
                    operation=operation,
                    engine=engine,
                    model_name=mlff_model,
                    task_name=mlff_task,
                    device=device,
                    model_path = mlff_model_path
                )

                # Parse outputs
                filtered_coordinates, energies = self.orca.parse_output(output_files, operation, dir=step_dir)

                # Update manifest with output file names
                update_step_manifest_outputs(step_dir, step_id, output_files)

                # Get persistent IDs for outputs
                filtered_ids = map_outputs_to_ids(step_dir, step_id, output_files)

                # Apply filtering
                filtered_coordinates, filtered_ids = self.refiner.filter(
                    filtered_coordinates,
                    energies,
                    filtered_ids,
                    sample_method,
                    parameters
                )

                if filtered_coordinates is None or filtered_ids is None:
                    logging.error(f"Filtering failed at step {step_id}. Exiting pipeline.")
                    return
            else:
                step_dir = os.path.join(self.output_dir, f"step{step_id}")
                logging.info(f"Skipping step {step_id} using existing outputs.")

            # Optional: normal mode sampling
            if step.get("normal_mode_sampling", False):
                nms_params = step.get("normal_mode_sampling_parameters", {})
                calc_type = nms_params.get("calc_type", "rm_imag")
                displacement_vector = nms_params.get("displacement_vector", 1.0)
                nms_random_displacements = nms_params.get("num_random_displacements", 1)
                if 'output_files' not in locals() or not output_files:
                    output_files = [
                        os.path.join(step_dir, f)
                        for f in os.listdir(step_dir)
                        if f.endswith(".out") and not f.startswith("slurm")
                    ]

                if not output_files:
                    logging.warning(f"No valid .out files found for normal mode sampling in step {step_id}. Skipping NMS.")
                else:
                    logging.info(f"Normal mode sampling requested for step {step_id}.")
                    input_template_path = os.path.join(self.template_dir, f"step{step_id}.inp")
                    filtered_coordinates, filtered_ids = self.orca.normal_mode_sampling(
                        file_paths=output_files,
                        calc_type=calc_type,
                        input_template=input_template_path,
                        slurm_template=self.template_dir,
                        charge=step.get('charge', self.charge),
                        multiplicity=step.get('multiplicity', self.multiplicity),
                        output_dir=self.output_dir,
                        operation=operation,
                        engine=engine,
                        model_name=mlff_model,
                        step_number=step_id,
                        structure_ids=filtered_ids,
                        max_cores=self.max_cores,
                        task_name=mlff_task,
                        mlff_model=mlff_model,
                        displacement_value=displacement_vector,
                        device=device,
                        bind=bind_address,
                        orca_executable=self.orca_executable,
                        scratch_dir=self.scratch_dir,
                        num_random_modes=nms_random_displacements,
                    )

            previous_coordinates, previous_ids = filtered_coordinates, filtered_ids

        logging.info("ChemRefine pipeline completed.")

def main():
    ChemRefiner().run()

if __name__ == "__main__":
    main()

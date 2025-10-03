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
import glob
from .mlff import MLFFTrainer
from chemrefine.utils import (
    update_step_manifest_outputs,
    map_outputs_to_ids,
    extract_structure_id,
    write_step_manifest,
    write_synthetic_manifest_for_ensemble,
    validate_structure_ids_or_raise,
    resolve_persistent_ids,
    smiles_to_xyz,  # your planned utility
)


class ChemRefiner:
    """
    ChemRefiner class orchestrates the ChemRefine workflow, handling input parsing,
    job submission, output parsing, and structure refinement based on a YAML configuration.
    It supports multiple steps with different calculation types and sampling methods.
    """

    def __init__(
        self,
    ):
        self.arg_parser = ArgumentParser()
        self.args, self.qorca_flags = self.arg_parser.parse()
        self.input_yaml = self.args.input_yaml
        self.max_cores = self.args.maxcores
        self.skip_steps = self.args.skip

        # === Load the YAML configuration ===
        with open(self.input_yaml, "r") as file:
            self.config = yaml.safe_load(file)

        # === Pull top-level config ===
        self.charge = self.config.get("charge", 0)
        self.multiplicity = self.config.get("multiplicity", 1)
        self.template_dir = os.path.abspath(
            self.config.get("template_dir", "./templates")
        )
        self.scratch_dir = self.config.get("scratch_dir", "./scratch")
        self.orca_executable = self.config.get("orca_executable", "orca")

        # === Setup output directory ===
        output_dir_raw = self.config.get("output_dir", "./outputs")
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
        self.next_id = 1  # 0 will be the initial seed; next fresh ID starts at 1

    def prepare_step1_directory(
        self,
        step_number,
        initial_xyz=None,
        charge=None,
        multiplicity=None,
        operation="OPT+SP",
        engine="dft",
        model_name=None,
        task_name=None,
        device="cpu",
        bind="127.0.0.1:8888",
    ):
        """
        Prepare the directory for the first step by copying one or more initial XYZ files,
        or generating XYZ files from a CSV of SMILES strings. Produces input/output files
        and assigns seed IDs (one per XYZ).
        """
        if charge is None:
            charge = self.charge
        if multiplicity is None:
            multiplicity = self.multiplicity

        step_dir = os.path.join(self.output_dir, f"step{step_number}")
        os.makedirs(step_dir, exist_ok=True)

        # --- Discover/generate initial xyz files ---
        if initial_xyz is None:
            # Default: look for "step1.xyz" in template_dir
            src_xyz_files = [os.path.join(self.template_dir, "step1.xyz")]

        elif os.path.isdir(initial_xyz):
            # User provided a directory → take all *.xyz inside
            src_xyz_files = sorted(
                f
                for f in glob.glob(os.path.join(initial_xyz, "*.xyz"))
                if os.path.isfile(f)
            )
            if not src_xyz_files:
                raise FileNotFoundError(
                    f"No .xyz files found in directory '{initial_xyz}'."
                )

        elif initial_xyz.endswith(".csv"):
            # User provided a CSV with SMILES strings → convert them into XYZs
            src_xyz_files = smiles_to_xyz(initial_xyz, step_dir)
            if not src_xyz_files:
                raise ValueError(f"No SMILES could be converted from '{initial_xyz}'.")

        else:
            # User provided a single XYZ file
            src_xyz_files = [initial_xyz]

        # --- Copy/normalize names into step_dir ---
        xyz_filenames = []
        for idx, src in enumerate(src_xyz_files):
            if not os.path.exists(src):
                raise FileNotFoundError(f"Initial XYZ file '{src}' not found.")
            dst = os.path.join(step_dir, f"step{step_number}_structure_{idx}.xyz")
            if src != dst:  # avoid redundant copy
                shutil.copyfile(src, dst)
            xyz_filenames.append(dst)

        # --- Input template ---
        template_inp = os.path.join(self.template_dir, "step1.inp")
        if not os.path.exists(template_inp):
            raise FileNotFoundError(
                f"Input file '{template_inp}' not found. Please ensure it exists."
            )

        # --- Generate inputs/outputs ---
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

        # --- Assign seed IDs (one per input structure) ---
        seed_ids = list(range(len(input_files)))

        return step_dir, input_files, output_files, seed_ids

    def prepare_subsequent_step_directory(
        self,
        step_number,
        filtered_coordinates,
        filtered_ids,
        charge=None,
        multiplicity=None,
        operation="OPT+SP",
        engine="dft",
        model_name=None,
        task_name=None,
        device="cuda",
        bind="127.0.0.1:8888",
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
        xyz_filenames = self.utils.write_xyz(
            filtered_coordinates, step_number, filtered_ids, output_dir=step_dir
        )

        # Copy the template input file from template_dir to step_dir
        input_template_src = os.path.join(self.template_dir, f"step{step_number}.inp")
        input_template_dst = os.path.join(step_dir, f"step{step_number}.inp")
        if not os.path.exists(input_template_src):
            logging.warning(
                f"Input file '{input_template_src}' not found. Exiting pipeline."
            )
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

    def handle_skip_step(
        self, step_number, operation, engine, sample_method, parameters
    ):
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
            Operation ("OPT+SP", "GOAT", "PES", "DOCKER", "SOLVATOR", "MLFF_TRAIN").
        engine : str
            Calculation engine ("dft" or "mlff").
        sample_method : str
            Refiner filtering method.
        parameters : dict
            Parameters for the filtering method.

        Returns
        -------
        tuple[list|None, list|None, list|None, list|None]
            (filtered_coordinates, filtered_ids, energies, forces)
            when outputs are reusable; otherwise (None, None, None, None).
        """
        step_dir = os.path.join(self.output_dir, f"step{step_number}")
        if not os.path.exists(step_dir):
            logging.info(
                f"Step directory {step_dir} does not exist. Will run this step."
            )
            return None, None, None, None

        op = operation.strip().upper()

        # === Special case: MLFF_TRAIN ===
        if op == "MLFF_TRAIN":
            prev_step = step_number - 1
            logging.info(
                f"MLFF_TRAIN step {step_number}: reusing results from step {prev_step}."
            )

            prev_cfg = self.config["steps"][prev_step - 1]  # YAML steps are 1-indexed
            return self.handle_skip_step(
                prev_step,
                prev_cfg["operation"],
                prev_cfg.get("engine", "dft"),
                sample_method,
                parameters,
            )

        def _ensure_solvator_ids(
            step_dir, step_number, engine, output_files, energies, structure_ids
        ):
            n_structs = len(energies)
            if n_structs == 0:
                return structure_ids
            if len(structure_ids) == n_structs:
                return structure_ids
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

        def _skip_pes(step_dir, step_number, engine, sample_method, parameters):
            """
            Reuse PES outputs in a skip-run. Builds persistent IDs from the number
            of parsed frames (not from filenames), writes a synthetic manifest, and
            applies filtering.
            """
            op = "PES"
            logging.info(f"Attempting to skip PES step {step_number} in {step_dir}.")
            # discover the single PES .out; adjust pattern if you also allow plain 'stepN.out'
            candidates = [
                os.path.join(step_dir, f)
                for f in os.listdir(step_dir)
                if f.endswith(".out") and f.startswith(f"step{step_number}")
            ]
            if not candidates:
                logging.warning(
                    f"No PES outputs found in {step_dir}. Will rerun this step."
                )
                return None, None, None, None

            outpath = candidates[0]
            coordinates, energies, forces = self.orca.parse_output(
                [outpath], op, dir=step_dir
            )
            if not coordinates or not energies or len(coordinates) != len(energies):
                logging.warning(
                    f"PES parse failed for {outpath}. Will rerun step {step_number}."
                )
                return None, None, None, None

            # IDs must come from the number of frames (NOT from output files)
            n = len(energies)
            structure_ids = list(range(n))

            # Persist a manifest for future skips
            write_synthetic_manifest_for_ensemble(
                step_number=step_number,
                step_dir=step_dir,
                n_structures=n,
                operation=op,
                engine=engine,
                output_basename=os.path.basename(outpath),
            )
            update_step_manifest_outputs(step_dir, step_number, [outpath])

            # Apply filter
            filtered_coordinates, filtered_ids = self.refiner.filter(
                coordinates, energies, structure_ids, sample_method, parameters
            )
            logging.info(
                f"After filtering PES step {step_number}: kept {len(filtered_coordinates)} structures."
            )

            # Forces are typically unavailable for PES; keep as list of None
            if not forces or len(forces) != len(coordinates):
                forces = [None] * len(coordinates)

            return filtered_coordinates, filtered_ids, energies, forces

        # ---------- GOAT special case ----------
        if op == "GOAT":
            output_files = [
                os.path.join(step_dir, f)
                for f in os.listdir(step_dir)
                if f.endswith("finalensemble.xyz")
            ]
            if not output_files:
                logging.warning(
                    f"No GOAT ensemble files in {step_dir}. Will rerun this step."
                )
                return None, None, None, None

            goat_file = output_files[0]
            logging.info(f"Found GOAT ensemble file: {goat_file}")

            coords, ens = self.orca.parse_goat_finalensemble(goat_file)
            if not coords or not ens:
                logging.warning(
                    f"GOAT parse failed at {goat_file}. Will rerun step {step_number}."
                )
                return None, None, None, None

            # Placeholder IDs just to run filter
            tmp_ids = list(range(len(ens)))
            forces = [None] * len(coords)

            filtered_coordinates, _ = self.refiner.filter(
                coords, ens, tmp_ids, sample_method, parameters
            )

            # Rebuild IDs after filtering
            final_n = len(filtered_coordinates)
            ensemble_base = os.path.basename(goat_file)
            write_synthetic_manifest_for_ensemble(
                step_number=step_number,
                step_dir=step_dir,
                n_structures=final_n,
                operation=op,
                engine=engine,
                output_basename=ensemble_base,
            )
            structure_ids = list(range(final_n))

            logging.info(
                f"After filtering GOAT step {step_number}: kept {final_n} structures."
            )
            return filtered_coordinates, structure_ids, ens, forces

        elif op == "PES":
            return _skip_pes(step_dir, step_number, engine, sample_method, parameters)

        elif op == "DOCKER":
            output_files = [
                os.path.join(step_dir, f)
                for f in os.listdir(step_dir)
                if f.endswith("struc1.allopt.xyz")
            ]
            if not output_files:
                logging.warning(
                    f"No DOCKER outputs found in {step_dir}. Will rerun this step."
                )
                return None, None, None, None

            logging.info(
                f"Found {len(output_files)} DOCKER output file(s) in {step_dir}. Reusing existing outputs."
            )

            coordinates, energies, forces = self.orca.parse_output(
                output_files, op, dir=step_dir
            )
            if not coordinates or not energies or len(coordinates) != len(energies):
                logging.warning(
                    f"DOCKER parse failed for {output_files[0]}. Will rerun step {step_number}."
                )
                return None, None, None, None

            # --- DOCKER structures have no IDs; synthesize sequential ones ---
            structure_ids = list(range(len(energies)))
            docker_base = os.path.basename(output_files[0])
            write_synthetic_manifest_for_ensemble(
                step_number=step_number,
                step_dir=step_dir,
                n_structures=len(energies),
                operation=op,
                engine=engine,
                output_basename=docker_base,
            )
            update_step_manifest_outputs(step_dir, step_number, output_files)

            # Run filter
            filtered_coordinates, filtered_ids = self.refiner.filter(
                coordinates, energies, structure_ids, sample_method, parameters
            )
            logging.info(
                f"After filtering DOCKER step {step_number}: kept {len(filtered_coordinates)} structures."
            )

            return filtered_coordinates, filtered_ids, energies, forces

        elif op == "SOLVATOR":
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
                    if f.endswith(".out")
                    and not f.startswith("slurm")
                    and not f.startswith("atom")
                ][:1]
            missing_msg = "No SOLVATOR outputs (.solvator.xyz or .out) found"
            found_msg = f"Found {len(output_files)} SOLVATOR output file(s)"

        else:
            # --- Replace the generic discovery block with this ---
            struct_pat = re.compile(
                rf"^step{step_number}_structure_-?\d+\.out$", re.IGNORECASE
            )
            plain_pat = re.compile(rf"^step{step_number}\.out$", re.IGNORECASE)

            all_outs = [
                os.path.join(step_dir, f)
                for f in os.listdir(step_dir)
                if f.endswith(".out")
                and not f.startswith("slurm")
                and not f.startswith("atom")
            ]

            # Prefer per-structure outputs; ignore the plain aggregator if structure files exist.
            struct_outs = [p for p in all_outs if struct_pat.match(os.path.basename(p))]
            if struct_outs:
                output_files = sorted(struct_outs)
                missing_msg = "No per-structure ORCA .out files found"
                found_msg = f"Found {len(output_files)} per-structure .out file(s)"
            else:
                # fallback to single aggregator only if no per-structure files exist
                output_files = [
                    p for p in all_outs if plain_pat.match(os.path.basename(p))
                ]
                missing_msg = "No ORCA .out files found"
                found_msg = f"Found {len(output_files)} .out file(s)"

        if not output_files:
            logging.warning(f"{missing_msg} in {step_dir}. Will rerun this step.")
            return None, None, None, None

        logging.info(f"{found_msg} in {step_dir}. Reusing existing outputs.")

        # Update manifest
        update_step_manifest_outputs(step_dir, step_number, output_files)

        # ---------- Parse outputs ----------
        coordinates, energies, forces = self.orca.parse_output(
            output_files, op, dir=step_dir
        )
        if not coordinates or not energies or len(coordinates) != len(energies):
            logging.warning(
                f"Parsed outputs are incomplete or inconsistent for step {step_number} "
                f"(coords={len(coordinates)}, energies={len(energies)}). Will rerun this step."
            )
            return None, None, None, None

        # ---------- Resolve IDs ----------
        structure_ids = map_outputs_to_ids(
            step_dir, step_number, output_files, operation
        )

        if all(i < 0 for i in structure_ids):

            if op == "SOLVATOR":
                logging.info(
                    f"No usable manifest for SOLVATOR step {step_number}; synthesizing IDs."
                )
                solv_base = os.path.basename(output_files[0])
                n_structs = len(energies)
                if n_structs <= 0:
                    logging.warning(
                        f"SOLVATOR parsed no structures for step {step_number}; rerunning."
                    )
                    return None, None, None, None
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
                logging.info(
                    f"No manifest for step {step_number}; reconstructing IDs from filenames."
                )
                if len(output_files) == 1 and len(energies) == 1:
                    logging.info(
                        f"Step {step_number}: single-output scalar case; assigning ID 0 and writing manifest."
                    )
                    structure_ids = [0]
                    candidate_inp = os.path.join(
                        step_dir, f"step{step_number}_structure_0.inp"
                    )
                    input_list = (
                        [candidate_inp]
                        if os.path.exists(candidate_inp)
                        else [f"step{step_number}_structure_0.inp"]
                    )
                    write_step_manifest(step_number, step_dir, input_list, op, engine)
                    update_step_manifest_outputs(step_dir, step_number, output_files)
                else:
                    pat = re.compile(
                        rf"^step{step_number}_structure_(\d+)(?:_|$)", re.IGNORECASE
                    )
                    recovered_ids = []
                    recovered_inps = []
                    for outp in output_files:
                        stem = os.path.splitext(os.path.basename(outp))[0]
                        m = pat.match(stem)
                        if m:
                            sid = int(m.group(1))
                        else:
                            truncated = stem.rsplit("_", 1)[0] if "_" in stem else stem
                            candidate_inp = os.path.join(step_dir, truncated + ".inp")
                            sid = extract_structure_id(candidate_inp)
                        if sid is None:
                            logging.warning(
                                f"Could not resolve ID for {outp}; rerunning step."
                            )
                            return None, None, None, None
                        recovered_ids.append(sid)
                        candidate_inp2 = os.path.join(
                            step_dir, f"step{step_number}_structure_{sid}.inp"
                        )
                        if os.path.exists(candidate_inp2):
                            recovered_inps.append(candidate_inp2)
                    structure_ids = recovered_ids
                    if recovered_inps:
                        write_step_manifest(
                            step_number, step_dir, recovered_inps, op, engine
                        )
                        update_step_manifest_outputs(
                            step_dir, step_number, output_files
                        )

        # ---------- Expand SOLVATOR IDs if needed ----------
        if op == "SOLVATOR" and len(structure_ids) != len(energies):
            structure_ids = _ensure_solvator_ids(
                step_dir=step_dir,
                step_number=step_number,
                engine=engine,
                output_files=output_files,
                energies=energies,
                structure_ids=structure_ids,
            )

        if len(structure_ids) != len(energies) or len(coordinates) != len(energies):
            logging.error(
                f"Step {step_number} ({op}): length mismatch before filtering: "
                f"coords={len(coordinates)}, energies={len(energies)}, ids={len(structure_ids)}."
            )
            return None, None, None, None

        # ---------- Filter ----------
        filtered_coordinates, filtered_ids = self.refiner.filter(
            coordinates, energies, structure_ids, sample_method, parameters
        )
        logging.info(
            f"After filtering step {step_number}: kept {len(filtered_coordinates)} structures."
        )

        # If any unresolved IDs slipped through, rebuild them
        if not structure_ids or any(i < 0 for i in structure_ids):
            logging.info(f"Step {step_number}: repairing invalid IDs.")
            structure_ids = list(range(len(energies)))
            write_synthetic_manifest_for_ensemble(
                step_number=step_number,
                step_dir=step_dir,
                n_structures=len(energies),
                operation=op,
                engine=engine,
                output_basename=os.path.basename(output_files[0]),
            )

        return filtered_coordinates, filtered_ids, energies, forces

    def parse_and_filter_outputs(
        self,
        output_files,
        operation,
        engine,
        step_number,
        sample_method,
        parameters,
        step_dir,
        previous_ids=None,
    ):
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
        coordinates, energies, forces = self.orca.parse_output(
            output_files, operation, dir=step_dir
        )
        if not coordinates or not energies:
            logging.error(
                f"No valid coordinates or energies found in outputs for step {step_number}. Exiting pipeline."
            )
            logging.error("Error in your output file, please check reason for failure")
            sys.exit(1)
        if previous_ids is None:
            previous_ids = list(range(len(energies)))  # only for step 1

        self.utils.save_step_csv(
            energies=energies,
            ids=previous_ids,
            step=step_number,
            output_dir=self.output_dir,
        )
        filtered_coordinates, selected_ids = self.refiner.filter(
            coordinates, energies, previous_ids, sample_method, parameters
        )

        return filtered_coordinates, selected_ids

    def submit_orca_jobs(
        self,
        input_files,
        max_cores,
        step_dir,
        device="cpu",
        operation="OPT+SP",
        engine="DFT",
        model_name=None,
        task_name=None,
    ):
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
        logging.info(
            f"Running in {self.scratch_dir} from submit_orca_jobs helper function."
        )
        try:
            logging.info(
                f"Submitting ORCA jobs in {step_dir} with {len(input_files)} input files using {device}."
            )
            self.orca_submitter = OrcaJobSubmitter(
                scratch_dir=self.scratch_dir,
                orca_executable=self.orca_executable,
                device=device,
            )
            self.orca_submitter.submit_files(
                input_files=input_files,
                max_cores=max_cores,
                template_dir=self.template_dir,
                output_dir=step_dir,
                engine=engine,
                operation=operation,
                model_name=model_name,
                task_name=task_name,
            )
        except Exception as e:
            logging.error(f"Error while submitting ORCA jobs in {step_dir}: {str(e)}")
            raise
        finally:
            os.chdir(original_dir)
            logging.info(f"Returned to original directory: {original_dir}")

    def run_mlff_train(
        self, step_number, step, last_coords, last_ids, last_energies, last_forces
    ):
        """
        Handle MLFF_TRAIN steps: prepare training dataset or skip if already completed.

        Parameters
        ----------
        step_number : int
            Current step index.
        step : dict
            Step configuration from YAML.
        last_coords, last_ids, last_energies, last_forces
            Results from the previous step, required for training.
        """
        if step_number == 1:
            raise ValueError("Invalid workflow: MLFF_TRAIN cannot be used at step 1.")

        step_dir = os.path.join(self.output_dir, f"step{step_number}")
        manifest_path = os.path.join(step_dir, f"step{step_number}_manifest.json")

        # --- Skip handling ---
        if self.skip_steps and os.path.exists(manifest_path):
            logging.info(
                f"Skipping MLFF_TRAIN at step {step_number}; training already completed."
            )
            return  # nothing new is produced

        # --- Normal execution ---
        if not (last_coords and last_ids and last_energies and last_forces):
            raise ValueError(
                f"MLFF_TRAIN at step {step_number} requires a prior step with "
                f"coordinates, energies, and forces. None found."
            )

        os.makedirs(step_dir, exist_ok=True)
        logging.info(f"Preparing MLFF dataset at step {step_number}.")

        trainer_cfg = step.get("trainer", {})
        trainer = MLFFTrainer(
            step_number=step_number,
            step_dir=step_dir,
            template_dir=self.template_dir,
            trainer_cfg=trainer_cfg,
            coordinates=last_coords,
            energies=last_energies,
            forces=last_forces,
            structure_ids=last_ids,
            utils=self.utils,
        )
        trainer.run()

        # Write manifest so skip can detect completion later
        write_step_manifest(step_number, step_dir, [], "MLFF_TRAIN", "mlff_train")

    def run(self):
        """
        Main pipeline execution function for ChemRefine.

        Dynamically loops over all steps defined in the YAML input. Supports skip-step
        logic with ID/energy/force persistence. When encountering an MLFF_TRAIN step,
        it consumes the results (coordinates, energies, forces, IDs) from the
        immediately preceding step and only prepares training datasets (XYZ).
        """
        logging.info("Starting ChemRefine pipeline.")

        # Results from the last completed (or skipped) step; used by MLFF_TRAIN.
        last_coords = None
        last_ids = None
        last_energies = None
        last_forces = None

        valid_operations = {
            "OPT+SP",
            "GOAT",
            "PES",
            "DOCKER",
            "SOLVATOR",
            "MLFF_TRAIN",
            "MLIP_TRAIN",
        }
        valid_engines = {"dft", "mlff", "mlip"}

        steps = self.config.get("steps", [])

        for step in steps:
            step_number = step["step"]
            operation = step["operation"].upper()
            engine = step.get("engine", "dft").lower()

            logging.info(
                f"Processing step {step_number}: operation '{operation}', engine '{engine}'."
            )

            if operation not in valid_operations:
                raise ValueError(
                    f"Invalid operation '{operation}' at step {step_number}. "
                    f"Must be one of {valid_operations}."
                )
            if engine not in valid_engines:
                raise ValueError(
                    f"Invalid engine '{engine}' at step {step_number}. "
                    f"Must be one of {valid_engines}."
                )

            # MLFF config (only relevant when engine == 'mlff' in compute steps)
            # MLFF/MLIP config (engine-dependent)
            if engine in {"mlff", "mlip"}:
                ml_config = step.get(
                    engine, {}
                )  # pulls "mlff:" or "mlip:" block from YAML
                model = ml_config.get("model_name", "medium")
                task = ml_config.get("task_name", "mace_off")
                bind_address = ml_config.get("bind", "127.0.0.1:8888")
                device = ml_config.get("device", "cuda")

                logging.info(
                    f"Using {engine.upper()} model '{model}' with task '{task}' for step {step_number}."
                )
            else:
                model = None
                task = None
                bind_address = None
                device = None

            # Sampling (optional)
            st = step.get("sample_type")
            sample_method = st.get("method") if st else None
            parameters = st.get("parameters", {}) if st else {}

            # === Special: MLFF_TRAIN consumes the previous step results and writes datasets ===
            if operation == "MLFF_TRAIN" or operation == "MLIP_TRAIN":
                self.run_mlff_train(
                    step_number, step, last_coords, last_ids, last_energies, last_forces
                )
                continue

            # === Non-training steps (compute / parsing / filtering) ===
            charge = step.get("charge", self.charge)
            multiplicity = step.get("multiplicity", self.multiplicity)

            filtered_coordinates = None
            filtered_ids = None
            energies = None
            forces = None

            # Try to reuse outputs via skip
            if self.skip_steps:
                filtered_coordinates, filtered_ids, energies, forces = (
                    self.handle_skip_step(
                        step_number, operation, engine, sample_method, parameters
                    )
                )
                if filtered_coordinates is not None and filtered_ids is not None:
                    last_coords, last_ids = filtered_coordinates, filtered_ids
                    last_energies, last_forces = energies, forces

            if filtered_coordinates is None or filtered_ids is None:
                logging.info(
                    f"No valid skip outputs for step {step_number}. Proceeding with normal execution."
                )

                if step_number == 1:
                    initial_xyz = self.config.get("initial_xyz", None)
                    step_dir, input_files, output_files, seeds_ids = (
                        self.prepare_step1_directory(
                            step_number=step_number,
                            initial_xyz=initial_xyz,
                            charge=charge,
                            multiplicity=multiplicity,
                            operation=operation,
                            engine=engine,
                            model_name=model,
                            task_name=task,
                            device=device,
                            bind=bind_address,
                        )
                    )
                    last_ids = seeds_ids
                else:
                    # Validate IDs for compute steps (not for training)
                    validate_structure_ids_or_raise(last_ids, step_number)
                    step_dir, input_files, output_files = (
                        self.prepare_subsequent_step_directory(
                            step_number=step_number,
                            filtered_coordinates=last_coords,
                            filtered_ids=last_ids,
                            charge=charge,
                            multiplicity=multiplicity,
                            operation=operation,
                            engine=engine,
                            model_name=model,
                            task_name=task,
                            device=device,
                            bind=bind_address,
                        )
                    )

                # Save manifest with input structure IDs
                write_step_manifest(
                    step_number, step_dir, input_files, operation, engine
                )

                # Submit jobs
                self.submit_orca_jobs(
                    input_files=input_files,
                    max_cores=self.max_cores,
                    step_dir=step_dir,
                    operation=operation,
                    engine=engine,
                    model_name=model,
                    task_name=task,
                    device=device,
                )

                # Parse outputs (must return coords, energies, forces)
                filtered_coordinates, energies, forces = self.orca.parse_output(
                    output_files, operation, dir=step_dir
                )

                # Update manifest with output files
                update_step_manifest_outputs(step_dir, step_number, output_files)

                # CORE CHANGE: persistent ID resolution
                num_out = len(filtered_coordinates)
                filtered_ids, self.next_id = resolve_persistent_ids(
                    step_number=step_number,
                    last_ids=last_ids,  # parents for this step
                    coords_count=num_out,  # children produced this step
                    output_files=output_files,
                    operation=operation,
                    next_id=self.next_id,  # fresh-ID counter
                    file_map_fn=map_outputs_to_ids,  # used only when 1:1 cases
                    step_dir=step_dir,
                )
                # Apply filtering if configured
                if operation in {"GOAT", "PES", "DOCKER", "SOLVATOR"}:
                    filtered_coordinates, filtered_ids = self.refiner.filter(
                        filtered_coordinates,
                        energies,
                        filtered_ids,
                        sample_method,
                        parameters,
                        by_parent=True,  # NEW: refine within each ensemble group
                    )
                else:
                    filtered_coordinates, filtered_ids = self.refiner.filter(
                        filtered_coordinates,
                        energies,
                        filtered_ids,
                        sample_method,
                        parameters,
                        by_parent=False,  # default global filtering
                    )

                if filtered_coordinates is None or filtered_ids is None:
                    logging.error(
                        f"Filtering failed at step {step_number}. Exiting pipeline."
                    )
                    return

            else:
                step_dir = os.path.join(self.output_dir, f"step{step_number}")
                logging.info(
                    f"Skipping step {step_number} using existing outputs."
                )  # Even for skipped steps, carry forward filtered results

            # Optional: normal mode sampling (may mutate coordinates/ids)
            if step.get("normal_mode_sampling", False):
                nms_params = step.get("normal_mode_sampling_parameters", {})
                calc_type = nms_params.get("calc_type", "rm_imag")
                displacement_vector = nms_params.get("displacement_vector", 1.0)
                nms_random_displacements = nms_params.get("num_random_displacements", 1)

                if "output_files" not in locals() or not output_files:
                    output_files = [
                        os.path.join(step_dir, f)
                        for f in os.listdir(step_dir)
                        if f.endswith(".out") and not f.startswith("slurm")
                    ]

                if not output_files:
                    logging.warning(
                        f"No valid .out files found for normal mode sampling in step {step_number}. Skipping NMS."
                    )
                else:
                    logging.info(
                        f"Normal mode sampling requested for step {step_number}."
                    )
                    input_template_path = os.path.join(
                        self.template_dir, f"step{step_number}.inp"
                    )
                    filtered_coordinates, filtered_ids = self.orca.normal_mode_sampling(
                        file_paths=output_files,
                        calc_type=calc_type,
                        input_template=input_template_path,
                        slurm_template=self.template_dir,
                        charge=step.get("charge", self.charge),
                        multiplicity=step.get("multiplicity", self.multiplicity),
                        output_dir=self.output_dir,
                        operation=operation,
                        engine=engine,
                        model_name=model,
                        step_number=step_number,
                        structure_ids=filtered_ids,
                        max_cores=self.max_cores,
                        task_name=task,
                        mlff_model=model,
                        displacement_value=displacement_vector,
                        device=device,
                        bind=bind_address,
                        orca_executable=self.orca_executable,
                        scratch_dir=self.scratch_dir,
                        num_random_modes=nms_random_displacements,
                    )

            # Commit this step's results for potential consumption by next step (e.g., MLFF_TRAIN)
            last_coords, last_ids = filtered_coordinates, filtered_ids
            last_energies, last_forces = energies, forces

        logging.info("ChemRefine pipeline completed.")


def main():
    ChemRefiner().run()


if __name__ == "__main__":
    main()

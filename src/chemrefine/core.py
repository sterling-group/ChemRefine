import os
from attr import dataclass,field
import yaml
import logging
from .parse import ArgumentParser
from .refine import StructureRefiner
from .utils import Utility
from .orca_interface import OrcaInterface, OrcaJobSubmitter
import shutil
import sys
import glob
from typing import Any, Dict, List, Optional
from .mlff import MLFFTrainer
from chemrefine.utils import (
    update_step_manifest_outputs,
    map_outputs_to_ids,
    write_step_manifest,
    validate_structure_ids_or_raise,
    resolve_persistent_ids,
    smiles_to_xyz,
)
from chemrefine.cache_utils import (
    CACHE_VERSION,
    StepCache,
    save_step_cache,
    load_step_cache,
    build_step_fingerprint,
)

@dataclass
class PipelineState:
    """
    Holds the running "memory" of the pipeline between steps.
    """
    last_coords: Any = None
    last_ids: Optional[List[int]] = None
    last_energies: Any = None
    last_forces: Any = None
    next_id: int = 1

    # Optional: keep lightweight records for debugging / future agent behavior
    history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class StepResult:
    """
    The output of a completed step (either computed or restored from cache).
    """
    step_number: int
    operation: str
    engine: str
    step_dir: str

    coords: Any
    ids: List[int]
    energies: Any
    forces: Any

    # Used for NMS fallback discovery and debugging
    output_files: List[str] = field(default_factory=list)

    # Flags
    from_cache: bool = False
    nms_ran: bool = False

class Controller:
    """
    Minimal deterministic controller.
    Later this can be replaced by an LLM-based controller.
    """

    def decide(self, chemrefine, step, state, fp_now, step_dir):
        # If skip_steps is enabled and a cache exists, restore it
        if chemrefine.skip_steps:
                return {"action": "restore_cache"}

        # Otherwise run the step normally
        return {"action": "run_step"}


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
        self.rebuild_cache = self.args.rebuild_cache
        self.rebuild_nms = self.args.rebuild_nms
        self.rerrun_errors = self.args.rerun_errors
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

    def process_step_with_parent_allocation(
        self,
        step_number: int,
        operation: str,
        step_dir: str,
        output_files: list[str],
        last_ids: list[str],
        sample_method: str,
        parameters: dict,
    ):
        """
        Handle per-parent child allocation and filtering for ensemble steps
        (GOAT, PES, DOCKER, SOLVATOR) or multi-XYZ step1 cases.
        Preserves hierarchical IDs using allocate_child_ids().
        """
        from chemrefine.utils import allocate_child_ids

        logging.info(
            f"Per-parent allocation enabled for {operation} at step {step_number}."
        )

        all_coords, all_ids, all_energies, all_forces = [], [], [], []
        fanouts = []

        for parent_id, out_file in zip(last_ids, output_files):
            coords_i, energies_i, forces_i = self.orca.parse_output(
                [out_file], operation, dir=step_dir
            )
            if not coords_i or not energies_i:
                logging.warning(f"Skipping {out_file}: no coords/energies parsed.")
                fanouts.append(0)
                continue

            # Temporarily assign flat children to filter
            tmp_ids = [f"{parent_id}-{k}" for k in range(len(coords_i))]

            f_coords_i, f_ids_i = self.refiner.filter(
                coords_i,
                energies_i,
                tmp_ids,
                sample_method,
                parameters,
                by_parent=False,
            )

            # Only keep structures that survived filtering
            keep_mask = [cid in set(f_ids_i) for cid in tmp_ids]
            kept_coords = [c for c, keep in zip(coords_i, keep_mask) if keep]
            kept_energies = [e for e, keep in zip(energies_i, keep_mask) if keep]
            kept_forces = [
                f
                for f, keep in zip(forces_i or [None] * len(coords_i), keep_mask)
                if keep
            ]

            fanouts.append(len(kept_coords))
            all_coords.extend(kept_coords)
            all_energies.extend(kept_energies)
            all_forces.extend(kept_forces)

        # --- Allocate hierarchical child IDs using true fanouts ---
        child_ids, _ = allocate_child_ids(parents=last_ids, fanouts=fanouts, next_id=0)
        all_ids = child_ids[: len(all_coords)]

        logging.info(
            f"[parent allocation] step{step_number}: built {len(all_coords)} children "
            f"from {len(last_ids)} parents ({sum(fanouts)} total fanouts)."
        )

        return all_coords, all_ids, all_energies, all_forces

    def rerun_errors(self, target_step: int | None = None):
        """
        Rerun failed ORCA/MLFF jobs recorded in _cache/failed_jobs.json
        for the given step, using the same settings stored in its manifest.
        Successfully re-submitted entries are removed from the cache.
        """
        import json
        import logging
        from pathlib import Path

        base_dir = Path(self.output_dir)
        step_dirs = sorted(
            [d for d in base_dir.glob("step*") if d.is_dir()],
            key=lambda p: int(p.name.replace("step", "")),
        )
        if not step_dirs:
            logging.error("No step directories found under outputs/.")
            return

        # --- Resolve target step ---
        if target_step is None:
            step_dir = step_dirs[-1]
            step_number = int(step_dir.name.replace("step", ""))
            logging.info(f"No target provided. Using latest step: {step_number}")
        else:
            step_dir = base_dir / f"step{target_step}"
            step_number = target_step
            if not step_dir.exists():
                logging.error(f"Step directory {step_dir} does not exist.")
                return

        cache_file = step_dir / "_cache" / "failed_jobs.json"
        manifest_file = step_dir / f"step{step_number}_manifest.json"

        if not cache_file.exists():
            logging.info(
                f"No failed_jobs.json found for step {step_number}. Nothing to rerun."
            )
            return

        try:
            with open(cache_file) as f:
                failed_jobs = json.load(f)
        except json.JSONDecodeError:
            logging.error(f"Corrupted failed_jobs.json at {cache_file}.")
            return

        if not failed_jobs:
            logging.info(f"No failed entries found in {cache_file}.")
            return

        # --- Read manifest for metadata ---
        if manifest_file.exists():
            with open(manifest_file) as f:
                manifest = json.load(f)
            operation = manifest.get("operation", "OPT+SP")
            engine = manifest.get("engine", "dft")
        else:
            operation, engine = "OPT+SP", "dft"

        logging.info(
            f"[rerun_errors] Found {len(failed_jobs)} failed jobs to rerun in step {step_number}."
        )

        # --- Retrieve model/task/device from config (if present) ---
        step_cfg = next(
            (s for s in self.config.get("steps", []) if s["step"] == step_number), None
        )
        if step_cfg:
            engine_cfg = step_cfg.get(engine, {})
            model = engine_cfg.get("model_name", "medium")
            task = engine_cfg.get("task_name", "mace_off")
            device = engine_cfg.get("device", "cpu")
        else:
            model, task, device = None, None, "cpu"

        updated_entries = []
        corrected_entries = []

        # --- Build list of input files and resubmit ---
        for job in failed_jobs:
            structure_id = job.get("structure_id")
            reason = job.get("reason", "Unknown")
            input_file = step_dir / f"{structure_id.replace('.out', '.inp')}"

            if not input_file.exists():
                logging.warning(f"Input file not found for {structure_id}. Skipping.")
                job["status"] = "missing_input"
                updated_entries.append(job)
                continue

            logging.info(f"Resubmitting {structure_id} (Reason: {reason})")

            try:
                self.submit_orca_jobs(
                    input_files=[str(input_file)],
                    max_cores=self.max_cores,
                    step_dir=str(step_dir),
                    operation=operation,
                    engine=engine,
                    model_name=model,
                    task_name=task,
                    device=device,
                )
                job["status"] = "corrected"
                corrected_entries.append(job)
                logging.info(
                    f"[rerun_errors] Re-submitted {structure_id} successfully."
                )
            except Exception as e:
                job["status"] = f"resubmit_failed: {e}"
                updated_entries.append(job)
                logging.error(f"[rerun_errors] Failed to resubmit {structure_id}: {e}")

        # --- Keep only unresolved entries ---
        remaining = [
            j
            for j in updated_entries
            if not j.get("status", "").startswith("corrected")
        ]

        if remaining:
            with open(cache_file, "w") as f:
                json.dump(remaining, f, indent=2)
            logging.info(
                f"[rerun_errors] Updated {cache_file}: {len(remaining)} unresolved entries remain."
            )
        else:
            cache_file.unlink(missing_ok=True)
            logging.info(
                f"[rerun_errors] All failed jobs for step {step_number} were corrected. Cache cleared."
            )

        logging.info(f"[rerun_errors] Step {step_number} rerun procedure completed.")

    def rebuild_step_cache_and_exit(self):
        """
        Rebuild the cache for a target step (settings.rebuild_target_step) or, if not set,
        the last step folder under outputs/. No job submission; we only parse outputs,
        re-run the same ID allocation + filtering used during a normal run,
        then write the StepCache and exit.
        """
        logging.info("[rebuild_cache] Starting cache rebuild process.")
        # --- Pick target step directory ---
        if not os.path.isdir(self.output_dir):
            logging.error("[rebuild_cache] Output directory does not exist.")
            return

        step_dirs = sorted(
            [
                (int(name.replace("step", "")), os.path.join(self.output_dir, name))
                for name in os.listdir(self.output_dir)
                if name.startswith("step")
                and os.path.isdir(os.path.join(self.output_dir, name))
            ],
            key=lambda x: x[0],
        )
        if not step_dirs:
            logging.error("[rebuild_cache] No step directories under outputs/.")
            return

        target = getattr(self, "rebuild_target_step", None)
        if target is not None:
            step_number = int(target)
            step_dir = os.path.join(self.output_dir, f"step{step_number}")
            if not os.path.isdir(step_dir):
                logging.error(f"[rebuild_cache] step{step_number} not found.")
                return
        else:
            step_number, step_dir = step_dirs[-1]

        # --- Load this step's YAML config ---
        step_cfg = next(
            (
                s
                for s in self.config.get("steps", [])
                if int(s.get("step")) == step_number
            ),
            None,
        )
        if not step_cfg:
            logging.error(f"[rebuild_cache] No YAML config for step {step_number}.")
            return

        operation = step_cfg["operation"].upper()
        engine = step_cfg.get("engine", "dft").lower()

        # --- Determine parent IDs (None for step 1) ---
        last_ids = None
        if step_number > 1:
            prev_cache_path = os.path.join(self.output_dir, f"step{step_number-1}")
            prev = load_step_cache(prev_cache_path)
            if not prev or not prev.ids:
                logging.error(
                    f"[rebuild_cache] Previous step cache (step{step_number-1}) not found or empty."
                )
                return
            last_ids = list(prev.ids)
        else:
            # Step 1 → synthetic roots (no parents)
            logging.debug(
                "[rebuild_cache] Step 1 detected — initializing synthetic root IDs."
            )
            last_ids = []

        # --- Gather OUTPUT files to parse ---
        try:
            output_files = sorted(
                [
                    os.path.join(step_dir, f)
                    for f in os.listdir(step_dir)
                    if f.endswith(".out") and not f.startswith("slurm")
                ]
            )
        except FileNotFoundError:
            logging.error(f"[rebuild_cache] Step directory missing: {step_dir}")
            return

        if not output_files:
            logging.error(f"[rebuild_cache] No .out files in {step_dir}.")
            return

        # --- Build fingerprint consistent with normal skip logic ---
        st = step_cfg.get("sample_type")
        sample_method = (st or {}).get("method")
        parameters = (st or {}).get("parameters", {})
        fp_now = build_step_fingerprint(
            step_cfg,
            (last_ids if step_number > 1 else None),
            parameters,
            step_number,
        )

        # --- Parse outputs ---
        logging.info(
            f"[rebuild_cache] Parsing {len(output_files)} outputs for step {step_number} ({operation})."
        )

        # === STEP 1 special case ===
        if step_number == 1:
            logging.info(
                f"[rebuild_cache] Step 1 rebuild detected — parsing {operation} outputs directly."
            )
            ensemble_ops = {"GOAT", "PES", "DOCKER", "SOLVATOR"}
            is_ensemble = operation in ensemble_ops

            if is_ensemble:
                # --- Parse per output file to preserve parent grouping ---
                all_coords, all_energies, all_forces, tmp_ids = [], [], [], []
                # parents = [str(i) for i in range(len(output_files))]

                for pidx, out_file in enumerate(sorted(output_files)):
                    ci, ei, fi = self.orca.parse_output(
                        [out_file], operation, dir=step_dir
                    )
                    if not ci or not ei:
                        logging.warning(
                            f"[rebuild_cache] No data parsed from {out_file}; skipping."
                        )
                        continue

                    n = len(ci)
                    all_coords.extend(ci)
                    all_energies.extend(ei)
                    all_forces.extend(fi if fi else [None] * n)
                    tmp_ids.extend([f"{pidx}-{j}" for j in range(n)])

                if not all_coords:
                    logging.error("[rebuild_cache] Step 1 parsing produced no data.")
                    return

                # Map tmp_id -> index so we can realign energies/forces after filtering
                id_to_idx = {sid: i for i, sid in enumerate(tmp_ids)}

                # --- Filter by parent groups using the temporary hierarchical IDs ---
                filtered_coords, filtered_ids = self.refiner.filter(
                    all_coords,
                    all_energies,
                    tmp_ids,
                    sample_method,
                    parameters,
                    by_parent=True,
                )
                if not filtered_coords or not filtered_ids:
                    logging.error(
                        "[rebuild_cache] Step 1 filtering returned no survivors."
                    )
                    return

                # Realign energies/forces to surviving structures
                keep_idx = [id_to_idx[sid] for sid in filtered_ids]
                energies = [all_energies[i] for i in keep_idx]
                forces = (
                    [all_forces[i] for i in keep_idx]
                    if any(all_forces)
                    else [None] * len(keep_idx)
                )

                # --- Renumber children per parent: parent-0, parent-1, ...
                from collections import defaultdict

                buckets = defaultdict(
                    list
                )  # parent -> list of positions in filtered_ids
                for pos, fid in enumerate(filtered_ids):
                    parent = fid.rsplit("-", 1)[0] if "-" in fid else fid
                    buckets[parent].append(pos)

                new_fids = list(filtered_ids)
                for parent, positions in buckets.items():
                    for k, pos in enumerate(positions):
                        new_fids[pos] = f"{parent}-{k}"

                filtered_ids = new_fids

            else:
                # Non-ensemble step1: flat filtering
                coords, energies_all, forces_all = self.orca.parse_output(
                    output_files, operation, dir=step_dir
                )
                if not coords or not energies_all:
                    logging.error("[rebuild_cache] Step 1 parsing produced no data.")
                    return

                filtered_coords, filtered_ids = self.refiner.filter(
                    coords,
                    energies_all,
                    [str(i) for i in range(len(coords))],
                    sample_method,
                    parameters,
                    by_parent=False,
                )
                if not filtered_coords or not filtered_ids:
                    logging.error(
                        "[rebuild_cache] Step 1 filtering returned no survivors."
                    )
                    return

                energies = energies_all[: len(filtered_coords)]
                forces = (forces_all or [None] * len(filtered_coords))[
                    : len(filtered_coords)
                ]

        else:
            # === Normal path for step >= 2 ===
            filtered_coordinates, energies, forces = self.orca.parse_output(
                output_files, operation, dir=step_dir
            )

            # --- Special case: Normal Mode Sampling (stepN/normal_mode_sampling/*.xyz) ---
            nms_dir = os.path.join(step_dir, "normal_mode_sampling")
            if os.path.isdir(nms_dir):
                xyz_files = sorted(
                    [
                        os.path.join(nms_dir, f)
                        for f in os.listdir(nms_dir)
                        if f.endswith(".xyz")
                    ]
                )
                if xyz_files:
                    from ase.io import read

                    coords, ids = [], []
                    for fpath in xyz_files:
                        try:
                            atoms = read(fpath)
                            coords.append(atoms)
                            sid = os.path.splitext(os.path.basename(fpath))[0]
                            # remove "stepN_structure_" prefix for clarity
                            sid = sid.replace(f"step{step_number}_structure_", "")
                            ids.append(sid)
                        except Exception as e:
                            logging.warning(
                                f"[rebuild_cache] Failed to parse {fpath}: {e}"
                            )

                    if coords:
                        logging.info(
                            f"[rebuild_cache] Detected Normal Mode Sampling step ({len(coords)} structures)."
                        )
                        step_cache = StepCache(
                            version=CACHE_VERSION,
                            step=step_number,
                            operation="NMS",
                            engine=engine,
                            fingerprint=None,
                            parent_ids=(last_ids if step_number > 1 else None),
                            ids=ids,
                            n_outputs=len(coords),
                            by_parent=None,
                            coords=coords,
                            energies=[None] * len(coords),
                            forces=[None] * len(coords),
                            extras={"nms_generation": True, "rebuild": True},
                        )
                        save_step_cache(step_dir, step_cache)
                        logging.info(
                            f"[rebuild_cache] Rebuilt NMS cache with {len(coords)} structures."
                        )
                        print(
                            f"[rebuild_cache] ✅ step {step_number} NMS cache rebuilt. Now run with --skip to continue."
                        )
                        return

            ensemble_ops = {"GOAT", "PES", "DOCKER", "SOLVATOR"}
            needs_per_parent = operation in ensemble_ops or (
                step_number == 1 and last_ids is not None and len(last_ids) > 1
            )

            if needs_per_parent:
                # Ensemble: allocate children by parent using your function
                result = self.process_step_with_parent_allocation(
                    step_number,
                    operation,
                    step_dir,
                    output_files,
                    last_ids,
                    sample_method,
                    parameters,
                )
                # Back-compat if function returns 5-tuple (includes by_parent)
                if isinstance(result, tuple) and len(result) == 5:
                    filtered_coordinates, filtered_ids, energies, forces, _by_parent = (
                        result
                    )
                else:
                    filtered_coordinates, filtered_ids, energies, forces = result
            else:
                # 1:1 (OPT+SP) default path
                if (
                    (step_number != 1)
                    and (last_ids is not None)
                    and (len(last_ids) == len(output_files))
                    and operation in {"OPT+SP"}
                ):
                    logging.info(
                        "[rebuild_cache] 1:1 propagation detected (rebuild), reusing parent IDs."
                    )
                    filtered_ids = last_ids[:]
                else:
                    num_out = len(filtered_coordinates)
                    filtered_ids, self.next_id = resolve_persistent_ids(
                        step_number=step_number,
                        last_ids=last_ids,
                        coords_count=num_out,
                        output_files=output_files,
                        operation=operation,
                        next_id=self.next_id,
                        file_map_fn=map_outputs_to_ids,
                        step_dir=step_dir,
                    )

                # Filtering for non-ensemble ops
                filtered_coordinates, filtered_ids = self.refiner.filter(
                    filtered_coordinates,
                    energies,
                    filtered_ids,
                    sample_method,
                    parameters,
                    by_parent=False,
                )

                if step_number > 1 and last_ids:
                    from chemrefine.utils import allocate_child_ids

                    # --- Determine true fanouts from filtering ---
                    # Prefer an explicit by_parent dict if available
                    by_parent = getattr(self.refiner, "by_parent", None)
                    if by_parent and isinstance(by_parent, dict):
                        fanouts = [len(by_parent.get(pid, [])) for pid in last_ids]
                    else:
                        # Fallback: infer fanouts from the filtered IDs
                        fanouts = []
                        for pid in last_ids:
                            count = sum(
                                1
                                for fid in filtered_ids
                                if str(fid).startswith(f"{pid}-") or fid == pid
                            )
                            fanouts.append(count)

                    # --- Use your allocate_child_ids() as-is ---
                    filtered_ids, _ = allocate_child_ids(
                        last_ids, fanouts, self.next_id
                    )
                    filtered_ids = filtered_ids[: len(filtered_coordinates)]

                    logging.info(
                        f"[rebuild_cache] Hierarchical IDs rebuilt: {sum(fanouts)} children "
                        f"from {len(last_ids)} parents."
                    )
                elif step_number == 1:
                    # Root step: assign flat IDs
                    filtered_ids = [str(i) for i in range(len(filtered_coordinates))]

            if filtered_coordinates is None or filtered_ids is None:
                logging.error(
                    f"[rebuild_cache] Filtering/ID step failed while rebuilding step {step_number}."
                )
                return

        # --- Write StepCache (same as normal end-of-step) ---
        try:
            step_cache = StepCache(
                version=CACHE_VERSION,
                step=step_number,
                operation=operation,
                engine=engine,
                fingerprint=fp_now,
                parent_ids=(last_ids if step_number > 1 else None),
                ids=filtered_ids,
                n_outputs=len(filtered_ids),
                by_parent=None,
                coords=(filtered_coords if step_number == 1 else filtered_coordinates),
                energies=energies,
                forces=forces,
                extras={"rebuild": True},
            )
            save_step_cache(step_dir, step_cache)
            logging.info(
                f"[rebuild_cache] Wrote step{step_number} cache with {len(filtered_ids)} items."
            )
            print(
                f"[rebuild_cache] ✅ step {step_number} cache rebuilt. Now run with --skip to continue."
            )
        except Exception as e:
            logging.error(f"[rebuild_cache] Failed to write cache: {e}")

    def rebuild_nms_cache_and_exit(self):
        """
        Rebuild the Normal Mode Sampling (NMS) displacements and write a new StepCache.
        Allows skipping directly to the next step without rerunning the parent step.
        """
        logging.info("[rebuild_nms] Starting NMS rebuild process.")
        # --- Fallback defaults ---
        device = getattr(self, "device", "cpu")
        bind_address = getattr(self, "bind_address", "127.0.0.1:8888")

        # --- Determine which step to rebuild ---
        step_number = int(getattr(self, "rebuild_target_step", 2))
        step_dir = os.path.join(self.output_dir, f"step{step_number}")
        if not os.path.isdir(step_dir):
            logging.error(f"[rebuild_nms] Step directory not found: {step_dir}")
            return

        # --- Load YAML config ---
        step_cfg = next(
            (
                s
                for s in self.config.get("steps", [])
                if int(s.get("step")) == step_number
            ),
            None,
        )
        if not step_cfg:
            logging.error(f"[rebuild_nms] No YAML config found for step {step_number}.")
            return

        operation = step_cfg["operation"].upper()
        engine = step_cfg.get("engine", "dft").lower()
        nms_params = step_cfg.get("normal_mode_sampling_parameters", {})
        calc_type = nms_params.get("calc_type", "random")
        displacement_value = nms_params.get("displacement_vector", 1.0)
        num_random_modes = nms_params.get("num_random_displacements", 1)

        # --- Gather .out files ---
        output_files = sorted(
            [
                os.path.join(step_dir, f)
                for f in os.listdir(step_dir)
                if f.endswith(".out") and not f.startswith("slurm")
            ]
        )
        if not output_files:
            logging.error(f"[rebuild_nms] No .out files found in {step_dir}.")
            return

        logging.info(
            f"[rebuild_nms] Rebuilding normal mode sampling from {len(output_files)} .out files."
        )

        # --- Load structure IDs (from cache if exists) ---
        prev_cache = load_step_cache(step_dir)
        structure_ids = getattr(
            prev_cache, "ids", [str(i) for i in range(len(output_files))]
        )

        input_template = os.path.join(self.template_dir, f"step{step_number}.inp")
        slurm_template = self.template_dir

        # --- Run the normal mode sampling ---
        logging.info(f"[rebuild_nms] Starting NMS ({calc_type}) for step {step_number}")
        displaced_coords, displaced_ids = self.orca.normal_mode_sampling(
            file_paths=output_files,
            calc_type=calc_type,
            input_template=input_template,
            slurm_template=slurm_template,
            charge=step_cfg.get("charge", self.charge),
            multiplicity=step_cfg.get("multiplicity", self.multiplicity),
            output_dir=self.output_dir,
            operation=operation,
            engine=engine,
            model_name=step_cfg.get("model_name", None),
            step_number=step_number,
            structure_ids=structure_ids,
            num_random_modes=num_random_modes,
            displacement_value=displacement_value,
            device=device,
            bind=bind_address,
            orca_executable=self.orca_executable,
            scratch_dir=self.scratch_dir,
        )

        fp_now = build_step_fingerprint(step_cfg, structure_ids, {}, step_number)
        step_cache = StepCache(
            version="1.0",
            step=step_number,
            operation=operation,
            engine=engine,
            fingerprint=fp_now,
            parent_ids=structure_ids,
            ids=displaced_ids,
            n_outputs=len(displaced_ids),
            by_parent=None,
            coords=displaced_coords,
            energies=None,
            forces=None,
            extras={"rebuild_nms": True},
        )

        save_step_cache(step_dir, step_cache)

        logging.info(
            f"[rebuild_nms] ✅ Wrote rebuilt NMS cache with {len(displaced_ids)} items for step{step_number}."
        )
        print(
            f"[rebuild_nms] Done. You can now run with --skip to continue from step {step_number + 1}."
        )

    def _handle_rebuild_modes_or_exit(self) -> bool:
        """
        Handle CLI flags that should execute a special mode and exit early.

        Returns:
            True  -> a rebuild/rerun mode was executed and the caller should stop.
            False -> normal pipeline should continue.
        """
        args = getattr(self, "args", None)
        if args is None:
            return False

        # ---- --rebuild_cache [step] ----
        rebuild_cache = getattr(args, "rebuild_cache", False)
        if rebuild_cache:
            target = rebuild_cache if isinstance(rebuild_cache, int) else None
            if target is not None:
                self.rebuild_target_step = target
            self.rebuild_step_cache_and_exit()
            return True

        # ---- --rebuild_nms [step] ----
        rebuild_nms = getattr(args, "rebuild_nms", False)
        if rebuild_nms:
            target = rebuild_nms if isinstance(rebuild_nms, int) else None
            if target is not None:
                self.rebuild_target_step = target
            self.rebuild_nms_cache_and_exit()
            return True

        # ---- --rerun_errors [step] ----
        rerun_errors = getattr(args, "rerun_errors", False)
        if rerun_errors:
            target = rerun_errors if isinstance(rerun_errors, int) else None
            if target is not None:
                self.rebuild_target_step = target
            self.rerun_errors(target_step=target)
            return True

        return False

    def _restore_from_step_cache(
    self,
    step_number: int,
    operation: str,
    engine: str,
    step_dir: str,
) -> Optional[StepResult]:
        cached = load_step_cache(step_dir)
        if not cached or cached.coords is None or cached.ids is None:
            return None

        logging.info(f"[step {step_number}] Step-level cache hit: {len(cached.ids)} items restored.")
        return StepResult(
            step_number=step_number,
            operation=operation,
            engine=engine,
            step_dir=step_dir,
            coords=cached.coords,
            ids=cached.ids,
            energies=cached.energies,
            forces=cached.forces,
            output_files=[],
            from_cache=True,
        )
          
    def _execute_non_training_step(
        self,
        step: dict,
        state: PipelineState,
        fp_now: str,
    ) -> StepResult:
        step_number = step["step"]
        operation = step["operation"].upper()
        engine = step.get("engine", "dft").lower()

        # MLFF/MLIP config
        if engine in {"mlff", "mlip"}:
            ml_config = step.get(engine, {})
            model = ml_config.get("model_name", "medium")
            task = ml_config.get("task_name", "mace_off")
            bind_address = ml_config.get("bind", "127.0.0.1:8888")
            device = ml_config.get("device", "cuda")
            logging.info(
                f"Using {engine.upper()} model '{model}' with task '{task}' for step {step_number}."
            )
        else:
            model = task = bind_address = device = None

        # Sampling
        st = step.get("sample_type")
        sample_method = st.get("method") if st else None
        parameters = st.get("parameters", {}) if st else {}

        charge = step.get("charge", self.charge)
        multiplicity = step.get("multiplicity", self.multiplicity)

        step_dir = os.path.join(self.output_dir, f"step{step_number}")
        os.makedirs(os.path.join(step_dir, "_cache"), exist_ok=True)

        output_files: List[str] = []
        ids: Optional[List[int]] = None  # ensure defined

        # --- Prepare inputs ---
        if step_number == 1:
            initial_xyz = self.config.get("initial_xyz", None)
            step_dir, input_files, output_files, seeds_ids = self.prepare_step1_directory(
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
            parent_ids_for_step = seeds_ids  # use as "parents" for step-1 outputs
        else:
            validate_structure_ids_or_raise(state.last_ids, step_number)
            step_dir, input_files, output_files = self.prepare_subsequent_step_directory(
                step_number=step_number,
                filtered_coordinates=state.last_coords,
                filtered_ids=state.last_ids,
                charge=charge,
                multiplicity=multiplicity,
                operation=operation,
                engine=engine,
                model_name=model,
                task_name=task,
                device=device,
                bind=bind_address,
            )
            parent_ids_for_step = state.last_ids

        write_step_manifest(step_number, step_dir, input_files, operation, engine)

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

        # Parse outputs
        coords, energies, forces = self.orca.parse_output(output_files, operation, dir=step_dir)
        update_step_manifest_outputs(step_dir, step_number, output_files)

        # === ID resolution + filtering ===
        ensemble_ops = {"GOAT", "PES", "DOCKER", "SOLVATOR"}
        needs_per_parent = operation in ensemble_ops or (
            step_number == 1 and parent_ids_for_step is not None and len(parent_ids_for_step) > 1
        )

        if needs_per_parent:
            result = self.process_step_with_parent_allocation(
                step_number,
                operation,
                step_dir,
                output_files,
                parent_ids_for_step,
                sample_method,
                parameters,
            )
            if isinstance(result, tuple) and len(result) == 5:
                coords, ids, energies, forces, _by_parent = result
            else:
                coords, ids, energies, forces = result
        else:
            # 1:1 propagation for OPT+SP
            if (
                step_number != 1
                and parent_ids_for_step is not None
                and len(parent_ids_for_step) == len(output_files)
                and operation in {"OPT+SP"}
            ):
                logging.info(f"[step {step_number}] 1:1 propagation detected, reusing parent IDs.")
                ids = parent_ids_for_step[:]
            else:
                num_out = len(coords)
                ids, state.next_id = resolve_persistent_ids(
                    step_number=step_number,
                    last_ids=parent_ids_for_step,
                    coords_count=num_out,
                    output_files=output_files,
                    operation=operation,
                    next_id=state.next_id,
                    file_map_fn=map_outputs_to_ids,
                    step_dir=step_dir,
                )

            coords, ids = self.refiner.filter(
                coords,
                energies,
                ids,
                sample_method,
                parameters,
                by_parent=False,
            )

        if coords is None or ids is None:
            raise RuntimeError(f"Filtering failed at step {step_number}.")

        # --- Write step cache ---
        try:
            step_cache = StepCache(
                version=CACHE_VERSION,
                step=step_number,
                operation=operation,
                engine=engine,
                fingerprint=fp_now,
                parent_ids=(parent_ids_for_step if step_number > 1 else None),
                ids=ids,
                n_outputs=len(ids),
                by_parent=None,
                coords=coords,
                energies=energies,
                forces=forces,
                extras=None,
            )
            save_step_cache(step_dir, step_cache)
            logging.info(f"[step {step_number}] Wrote step cache ({len(ids)} items).")
        except Exception as e:
            logging.warning(f"[step {step_number}] Cache write failed: {e}")

        # Optional NMS
        nms_ran = False
        if step.get("normal_mode_sampling", False):
            nms_params = step.get("normal_mode_sampling_parameters", {})
            calc_type = nms_params.get("calc_type", "rm_imag")
            displacement_vector = nms_params.get("displacement_vector", 1.0)
            nms_random_displacements = nms_params.get("num_random_displacements", 1)

            if not output_files:
                output_files = [
                    os.path.join(step_dir, f)
                    for f in os.listdir(step_dir)
                    if f.endswith(".out") and not f.startswith("slurm")
                ]

            if output_files:
                input_template_path = os.path.join(self.template_dir, f"step{step_number}.inp")
                coords, ids = self.orca.normal_mode_sampling(
                    file_paths=output_files,
                    calc_type=calc_type,
                    input_template=input_template_path,
                    slurm_template=self.template_dir,
                    charge=charge,
                    multiplicity=multiplicity,
                    output_dir=self.output_dir,
                    operation=operation,
                    engine=engine,
                    model_name=model,
                    step_number=step_number,
                    structure_ids=ids,
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
                nms_ran = True

                # energies/forces no longer aligned after NMS; keep it simple
                energies = None
                forces = None

                try:
                    step_cache = StepCache(
                        version=CACHE_VERSION,
                        step=step_number,
                        operation=operation,
                        engine=engine,
                        fingerprint=None,
                        parent_ids=(parent_ids_for_step if step_number > 1 else None),
                        ids=ids,
                        n_outputs=len(ids),
                        by_parent=None,
                        coords=coords,
                        energies=[None] * len(ids),
                        forces=[None] * len(ids),
                        extras={"nms_generation": True},
                    )
                    save_step_cache(step_dir, step_cache)
                    logging.info(f"[step {step_number}] Updated step cache after NMS ({len(ids)} items).")
                except Exception as e:
                    logging.warning(f"[step {step_number}] Cache write (post-NMS) failed: {e}")
            else:
                logging.warning(f"No valid .out files found for NMS in step {step_number}. Skipping NMS.")

        return StepResult(
            step_number=step_number,
            operation=operation,
            engine=engine,
            step_dir=step_dir,
            coords=coords,
            ids=ids,
            energies=energies,
            forces=forces,
            output_files=output_files,
            from_cache=False,
            nms_ran=nms_ran,
        )

    def _restore_from_step_cache(
    self,
    step_number: int,
    step_dir: str,
    operation: str,
    engine: str,
) -> Optional[StepResult]:
        cached = load_step_cache(step_dir)
        if not cached:
            return None
        if cached.coords is None or cached.ids is None:
            return None

        logging.info(
            f"[step {step_number}] Step-level cache hit: {len(cached.ids)} items restored."
        )

        return StepResult(
            step_number=step_number,
            operation=operation,
            engine=engine,
            step_dir=step_dir,
            coords=cached.coords,
            ids=cached.ids,
            energies=cached.energies,
            forces=cached.forces,
            output_files=[],
            from_cache=True,
        )

    def _commit_step_result(self, state: PipelineState, result: StepResult) -> None:
        state.last_coords = result.coords
        state.last_ids = result.ids
        state.last_energies = result.energies
        state.last_forces = result.forces

        print(f"Step {result.step_number} completed: {len(state.last_coords)} structures ready.")

        if state.last_energies is not None and len(state.last_energies) == len(state.last_ids):
            self.utils.save_step_csv(
                energies=state.last_energies,
                ids=state.last_ids,
                step=result.step_number,
                output_dir=self.output_dir,
            )
        else:
            logging.info(f"[step {result.step_number}] Skipping CSV export (energy/ID mismatch or missing energies).")

        print(f"Your ID's for this step are: {state.last_ids}")
   
    ###NEW____RUN____###

    def run(self):
        if self._handle_rebuild_modes_or_exit():
            return

        logging.info("Starting ChemRefine pipeline.")

        controller = Controller()
        state = PipelineState(next_id=getattr(self, "next_id", 1))

        for step in self.config.get("steps", []):
            op = step["operation"].upper()

            if op in {"MLFF_TRAIN", "MLIP_TRAIN"}:
                self.run_mlff_train(
                    step["step"], step, state.last_coords, state.last_ids, state.last_energies, state.last_forces
                )
                continue

            step_number = step["step"]
            step_dir = os.path.join(self.output_dir, f"step{step_number}")
            os.makedirs(os.path.join(step_dir, "_cache"), exist_ok=True)

            # fingerprint computed as before (even if you don't enforce it yet)
            st = step.get("sample_type")
            parameters = st.get("parameters", {}) if st else {}
            fp_now = build_step_fingerprint(
                step,
                state.last_ids if step_number > 1 else None,
                parameters,
                step_number,
            )

            decision = controller.decide(self, step, state, fp_now, step_dir)

            if decision["action"] == "restore_cache":
                result = self._restore_from_step_cache(
                    step_number=step_number,
                    step_dir=step_dir,
                    operation=op,
                    engine=step.get("engine", "dft").lower(),
                )
                if result is None:
                    logging.warning(
                        f"No valid step cache for step {step_number}. Proceeding with normal execution."
                    )
                    result = self._execute_non_training_step(step, state, fp_now)
            else:
                result = self._execute_non_training_step(step, state, fp_now)

            self._commit_step_result(state, result)
            self.next_id = state.next_id

        logging.info("ChemRefine pipeline completed.")


def main():
    ChemRefiner().run()


if __name__ == "__main__":
    main()

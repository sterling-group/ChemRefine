from __future__ import annotations

import os
import sys
import glob
import shutil
import logging
from typing import Any, Optional, Tuple
import json
import socket
from pathlib import Path
import yaml
import fcntl
from .parse import ArgumentParser
from .refine import StructureRefiner
from .utils import Utility

from .orca_interface import OrcaInterface, OrcaJobSubmitter
from .mlff import MLFFTrainer

# Types (NEW)
from .types import (
    EngineConfig,
    SamplingConfig,
    StepContext,
    StepResult,
    PipelineState,
    StepIO,
)

# Utility functions (make these relative, not chemrefine.utils)
from .utils import (
    update_step_manifest_outputs,
    map_outputs_to_ids,
    write_step_manifest,
    validate_structure_ids_or_raise,
    resolve_persistent_ids,
    smiles_to_xyz,
)

# Cache utilities (relative import)
from .cache_utils import (
    CACHE_VERSION,
    StepCache,
    save_step_cache,
    load_step_cache,
    build_step_fingerprint,
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
        self._bind_counter = 0
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
    basis=None,
    functional=None,
    engine_extras=None,
):
        """
        Prepare step1 directory. For ML engines, assign a unique bind per initial structure.
        Returns (step_dir, input_files, output_files, seed_ids, binds).
        """
        if charge is None:
            charge = self.charge
        if multiplicity is None:
            multiplicity = self.multiplicity

        step_dir = os.path.join(self.output_dir, f"step{step_number}")
        os.makedirs(step_dir, exist_ok=True)

        eng = (engine or "dft").lower()
        extras = engine_extras or {}

        # discover initial xyz files (your existing logic)
        if initial_xyz is None:
            src_xyz_files = [os.path.join(self.template_dir, "step1.xyz")]
        elif os.path.isdir(initial_xyz):
            src_xyz_files = sorted(
                f for f in glob.glob(os.path.join(initial_xyz, "*.xyz")) if os.path.isfile(f)
            )
            if not src_xyz_files:
                raise FileNotFoundError(f"No .xyz files found in directory '{initial_xyz}'.")
        elif initial_xyz.endswith(".csv"):
            src_xyz_files = smiles_to_xyz(initial_xyz, step_dir)
            if not src_xyz_files:
                raise ValueError(f"No SMILES could be converted from '{initial_xyz}'.")
        else:
            src_xyz_files = [initial_xyz]

        xyz_filenames = []
        for idx, src in enumerate(src_xyz_files):
            if not os.path.exists(src):
                raise FileNotFoundError(f"Initial XYZ file '{src}' not found.")
            dst = os.path.join(step_dir, f"step{step_number}_structure_{idx}.xyz")
            if src != dst:
                shutil.copyfile(src, dst)
            xyz_filenames.append(dst)

        template_inp = os.path.join(self.template_dir, "step1.inp")
        if not os.path.exists(template_inp):
            raise FileNotFoundError(f"Input file '{template_inp}' not found.")

        input_files: list[str] = []
        output_files: list[str] = []
        binds: dict[str, str] = {}

        needs_bind = eng in {"mlff", "mlip", "pyscf"}

        for i, xyz in enumerate(xyz_filenames):
            bind_i = self._increment_bind(bind, i) if needs_bind else bind

            inp_i, out_i = self.orca.create_input(
                [xyz],
                template_inp,
                charge,
                multiplicity,
                output_dir=step_dir,
                operation=operation,
                engine=eng,
                model_name=model_name,
                task_name=task_name,
                device=device,
                bind=bind_i,
                basis=basis,
                xc=(functional or extras.get("xc") or extras.get("functional")),
                df=bool(extras.get("df", False)),
                gpu=extras.get("gpu", None),
                pyscf_method=extras.get("method", "dft"),
                pyscf_prog=extras.get("prog", None),
            )

            input_files.extend(inp_i)
            output_files.extend(out_i)
            binds[str(Path(inp_i[0]).resolve())] = bind_i

        seed_ids = list(range(len(input_files)))
        return step_dir, input_files, output_files, seed_ids, binds

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
    basis=None,
    functional=None,
    engine_extras=None,
):
        """
        Prepare a subsequent step directory and generate ORCA inputs/outputs.

        For MLFF/MLIP/PySCF engines, assigns a unique bind per input structure by
        incrementing the base port (bind) by the structure index, and returns a
        mapping from resolved input file path -> bind.
        """
        if charge is None:
            charge = self.charge
        if multiplicity is None:
            multiplicity = self.multiplicity

        step_dir = os.path.join(self.output_dir, f"step{step_number}")
        os.makedirs(step_dir, exist_ok=True)

        eng = (engine or "dft").lower()
        extras = engine_extras or {}

        xyz_filenames = self.utils.write_xyz(
            filtered_coordinates, step_number, filtered_ids, output_dir=step_dir
        )

        input_template_src = os.path.join(self.template_dir, f"step{step_number}.inp")
        input_template_dst = os.path.join(step_dir, f"step{step_number}.inp")
        if not os.path.exists(input_template_src):
            raise FileNotFoundError(
                f"Input file '{input_template_src}' not found. Please ensure it exists."
            )
        shutil.copyfile(input_template_src, input_template_dst)

        input_files: list[str] = []
        output_files: list[str] = []
        binds: dict[str, str] = {}

        # Only engines that use server/client need per-input binds
        needs_bind = eng in {"mlff", "mlip", "pyscf"}

        for i, xyz in enumerate(xyz_filenames):
            bind_i = self._increment_bind(bind, i) if needs_bind else bind

            inp_i, out_i = self.orca.create_input(
                [xyz],
                input_template_dst,
                charge,
                multiplicity,
                output_dir=step_dir,
                operation=operation,
                engine=eng,
                model_name=model_name,
                task_name=task_name,
                device=device,
                bind=bind_i,
                basis=basis,
                xc=(functional or extras.get("xc") or extras.get("functional")),
                df=bool(extras.get("df", False)),
                gpu=extras.get("gpu", None),
                pyscf_method=extras.get("method", "dft"),
                pyscf_prog=extras.get("prog", None),
            )

            input_files.extend(inp_i)
            output_files.extend(out_i)

            # Key by resolved input path so submitter can look it up reliably
            binds[str(Path(inp_i[0]).resolve())] = bind_i

        return step_dir, input_files, output_files, binds

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
    *,
    input_files,
    max_cores,
    step_dir,
    operation="OPT+SP",
    engine="dft",
    engine_cfg=None,
    binds: dict[str, str] | None = None,
):
        """
        Submit ORCA jobs for a step.

        - DFT: batch submit as usual.
        - MLFF/MLIP/PySCF: batch submit, but per-input binds are used when writing SLURM scripts.
        """
        engine = (engine or "dft").lower()

        device = getattr(engine_cfg, "device", None) or "cpu"
        model_name = getattr(engine_cfg, "model_name", None)
        task_name = getattr(engine_cfg, "task_name", None)
        default_bind = getattr(engine_cfg, "bind", None) or "127.0.0.1:8888"
        basis = getattr(engine_cfg, "basis", None)
        functional = getattr(engine_cfg, "functional", None)
        model_path = getattr(engine_cfg, "model_path", None)

        binds = binds or {}

        logging.info(f"Switching to working directory: {step_dir}")
        original_dir = os.getcwd()
        os.chdir(step_dir)

        try:
            self.orca_submitter = OrcaJobSubmitter(
                scratch_dir=self.scratch_dir,
                orca_executable=self.orca_executable,
                device=device,
                bind=default_bind,  # fallback only
                basis=basis,
                functional=functional,
            )

            # IMPORTANT: submit in one batch and DO NOT WAIT
            self.orca_submitter.submit_files(
                input_files=input_files,
                max_cores=max_cores,
                template_dir=self.template_dir,
                output_dir=step_dir,
                engine=engine,
                operation=operation,
                model_name=model_name,
                task_name=task_name,
                model_path=model_path,
                binds=binds,     # <-- NEW
            )

        finally:
            os.chdir(original_dir)

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

    def _handle_rebuild_modes(self, args) -> bool:
        """Return True if we handled a rebuild mode and exited early."""
        if getattr(args, "rebuild_cache", False):
            target = args.rebuild_cache if isinstance(args.rebuild_cache, int) else None
            if target is not None:
                self.rebuild_target_step = target
            self.rebuild_step_cache_and_exit()
            return True

        if getattr(args, "rebuild_nms", False):
            target = args.rebuild_nms if isinstance(args.rebuild_nms, int) else None
            if target is not None:
                self.rebuild_target_step = target
            self.rebuild_nms_cache_and_exit()
            return True

        if getattr(args, "rerun_errors", False):
            target = args.rerun_errors if isinstance(args.rerun_errors, int) else None
            if target is not None:
                self.rebuild_target_step = target
            self.rerun_errors(target_step=target)
            return True

        return False
   
    def _build_step_context(self, step: dict) -> StepContext:
        step_number = step["step"]
        operation = step["operation"].upper()
        engine = step.get("engine", "dft").lower()

        engine_cfg = self._extract_engine_config(step, engine)
        sampling = self._extract_sampling(step)

        charge = step.get("charge", self.charge)
        multiplicity = step.get("multiplicity", self.multiplicity)

        return StepContext(
            step_number=step_number,
            operation=operation,
            engine=engine,
            charge=charge,
            multiplicity=multiplicity,
            engine_cfg=engine_cfg,
            sampling=sampling,
            step=step,
        )

    def _validate_step_context(self, ctx: StepContext) -> None:
        valid_operations = {
            "OPT+SP", "GOAT", "PES", "DOCKER", "SOLVATOR", "MLFF_TRAIN", "MLIP_TRAIN"
        }
        valid_engines = {"dft", "mlff", "mlip", "pyscf"}

        if ctx.operation not in valid_operations:
            raise ValueError(
                f"Invalid operation '{ctx.operation}' at step {ctx.step_number}. "
                f"Must be one of {valid_operations}."
            )
        if ctx.engine not in valid_engines:
            raise ValueError(
                f"Invalid engine '{ctx.engine}' at step {ctx.step_number}. "
                f"Must be one of {valid_engines}."
            )
        if ctx.engine == "pyscf":
            if not ctx.engine_cfg.basis:
                raise ValueError(f"[step {ctx.step_number}] pyscf engine requires 'pyscf: basis'.")
            if not ctx.engine_cfg.functional:
                raise ValueError(f"[step {ctx.step_number}] pyscf engine requires 'pyscf: functional' (or 'xc').")
            if ctx.engine_cfg.device not in {"cpu", "cuda"}:
                raise ValueError(f"[step {ctx.step_number}] pyscf device must be 'cpu' or 'cuda'.")
            extras = ctx.engine_cfg.extras or {}
            if not isinstance(extras.get("save_tensors", False), bool):
                raise ValueError(f"[step {ctx.step_number}] pyscf.save_tensors must be true/false.")
            if not isinstance(extras.get("localized", False), bool):
                raise ValueError(f"[step {ctx.step_number}] pyscf.localized must be true/false.")
            tensor_folder = extras.get("tensor_folder", "tensors")
            if not isinstance(tensor_folder, str) or not tensor_folder.strip():
                raise ValueError(f"[step {ctx.step_number}] pyscf.tensor_folder must be a non-empty string.")
            ###____RUN____###

    def _extract_engine_config(self, step: dict, engine: str) -> EngineConfig:
        engine = engine.lower()

        # --- defaults shared across engines ---
        default_bind = "127.0.0.1:8888"

        # ----------------
        # MLFF / MLIP
        # ----------------
        if engine in {"mlff", "mlip"}:
            cfg = step.get(engine, {}) or {}
            return EngineConfig(
                engine=engine,
                model_name=cfg.get("model_name", "medium"),
                task_name=cfg.get("task_name", "mace_off"),
                bind=cfg.get("bind", default_bind),
                device=cfg.get("device", "cuda"),
                extras=dict(cfg),
            )

        # ----------------
        # PySCF (ORCA optimizer + PySCF backend)
        # ----------------
        if engine == "pyscf":
            cfg = step.get("pyscf", {}) or {}

            # allow either gpu: true/false or device: cuda/cpu
            device = cfg.get("device")
            if device is None:
                device = "cuda" if cfg.get("gpu", False) else "cpu"

            extras = dict(cfg)
            extras["save_tensors"] = bool(cfg.get("save_tensors", False))
            extras["localized"] = bool(cfg.get("localized", False))
            extras["tensor_folder"] = cfg.get("tensor_folder", "tensors")

            return EngineConfig(
                engine="pyscf",
                device=device,
                bind=cfg.get("bind", default_bind),
                basis=cfg.get("basis", step.get("basis")),               # allow top-level fallback if you want
                functional=cfg.get("functional", cfg.get("xc", "pbe")),   # support xc alias; default pbe
                extras=extras,
            )

        # ----------------
        # Plain DFT (ORCA)
        # ----------------
        return EngineConfig(engine=engine, bind=default_bind, device="cpu")

    def _extract_sampling(self, step: dict) -> SamplingConfig:
        st = step.get("sample_type")
        method = st.get("method") if st else None
        params = st.get("parameters", {}) if st else {}
        return SamplingConfig(method=method, parameters=params)

    def _try_restore_from_step_cache(self, step_dir: str, fp_now: str, ctx: StepContext) -> Optional[StepResult]:
        if not self.skip_steps:
            return None

        cached = load_step_cache(step_dir)
        if not cached:
            logging.info(f"[step {ctx.step_number}] No step-level cache; computing.")
            return None

        # NOTE: your original code has a weird nested `if cached: else: if cached:` branch.
        # Here we assume load_step_cache already returns None on mismatch OR you can check here.
        # If you want strict checking, do it here using cached.fingerprint, cached.operation, cached.engine, etc.

        logging.info(f"[step {ctx.step_number}] Step-level cache hit: {len(cached.ids)} items restored.")
        return StepResult(
            coords=cached.coords,
            ids=cached.ids,
            energies=cached.energies,
            forces=cached.forces,
            output_files=[],  # not stored in your StepCache (unless you add it)
        )

    def _execute_step_heavy_work(
    self,
    ctx: StepContext,
    state: PipelineState,
    step_dir: str,
    fp_now: str,
) -> StepResult:
        io = self._prepare_step_io(ctx, state)
        write_step_manifest(ctx.step_number, io.step_dir, io.input_files, ctx.operation, ctx.engine)

        self.submit_orca_jobs(
            input_files=io.input_files,
            max_cores=self.max_cores,
            step_dir=io.step_dir,
            operation=ctx.operation,
            engine=ctx.engine,
            engine_cfg=ctx.engine_cfg,  
            binds=io.binds, 
        )


        coords, energies, forces = self.orca.parse_output(io.output_files, ctx.operation, dir=io.step_dir)
        update_step_manifest_outputs(io.step_dir, ctx.step_number, io.output_files)

        coords, ids, energies, forces = self._resolve_ids_and_filter(
            ctx=ctx,
            state=state,
            step_dir=io.step_dir,
            output_files=io.output_files,
            coords=coords,
            energies=energies,
            forces=forces,
        )

        if coords is None or ids is None:
            logging.error(f"Filtering failed at step {ctx.step_number}. Exiting pipeline.")
            raise RuntimeError(f"Filtering failed at step {ctx.step_number}")

        self._write_step_cache(ctx, step_dir, fp_now, state.last_ids, coords, ids, energies, forces)

        return StepResult(coords=coords, ids=ids, energies=energies, forces=forces, output_files=io.output_files)

    def _prepare_step_io(self, ctx: StepContext, state: PipelineState) -> StepIO:
        cfg = ctx.engine_cfg or EngineConfig()

        base_bind = getattr(cfg, "bind", None) or "127.0.0.1:8888"
        cfg.bind = self._increment_bind(base_bind)
        ctx.engine_cfg = cfg

        if ctx.step_number == 1:
            initial_xyz = self.config.get("initial_xyz", None)

            step_dir, input_files, output_files, seeds_ids,binds = self.prepare_step1_directory(
                step_number=ctx.step_number,
                initial_xyz=initial_xyz,
                charge=ctx.charge,
                multiplicity=ctx.multiplicity,
                operation=ctx.operation,
                engine=ctx.engine,

                # unified engine config threading
                model_name=cfg.model_name,
                task_name=cfg.task_name,
                device=cfg.device,
                bind=cfg.bind,

                # pyscf-specific (safe for others; they can ignore)
                basis=cfg.basis,
                functional=cfg.functional,
                engine_extras=cfg.extras,
            )

            state.last_ids = seeds_ids
            return StepIO(
                step_dir=step_dir,
                input_files=input_files,
                output_files=output_files,
                seeds_ids=seeds_ids,
                binds=binds,
            )

        validate_structure_ids_or_raise(state.last_ids, ctx.step_number)

        step_dir, input_files, output_files,binds = self.prepare_subsequent_step_directory(
            step_number=ctx.step_number,
            filtered_coordinates=state.last_coords,
            filtered_ids=state.last_ids,
            charge=ctx.charge,
            multiplicity=ctx.multiplicity,
            operation=ctx.operation,
            engine=ctx.engine,

            # unified engine config threading
            model_name=cfg.model_name,
            task_name=cfg.task_name,
            device=cfg.device,
            bind=cfg.bind,

            # pyscf-specific
            basis=cfg.basis,
            functional=cfg.functional,
            engine_extras=cfg.extras,
        )

        return StepIO(step_dir=step_dir, input_files=input_files, output_files=output_files, binds=binds)

    def _resolve_ids_and_filter(
    self,
    *,
    ctx: StepContext,
    state: PipelineState,
    step_dir: str,
    output_files: list[str],
    coords,
    energies,
    forces,
):
        last_ids = state.last_ids

        ensemble_ops = {"GOAT", "PES", "DOCKER", "SOLVATOR"}
        needs_per_parent = (
            ctx.operation in ensemble_ops
            or (ctx.step_number == 1 and last_ids is not None and len(last_ids) > 1)
        )

        if needs_per_parent:
            result = self.process_step_with_parent_allocation(
                ctx.step_number,
                ctx.operation,
                step_dir,
                output_files,
                last_ids,
                ctx.sampling.method,
                ctx.sampling.parameters,
            )
            if isinstance(result, tuple) and len(result) == 5:
                coords, ids, energies, forces, _by_parent = result
            else:
                coords, ids, energies, forces = result
            return coords, ids, energies, forces

        # 1:1 OPT+SP propagation
        if (ctx.step_number != 1) and (len(last_ids) == len(output_files)) and (ctx.operation in {"OPT+SP"}):
            logging.info(f"[step {ctx.step_number}] 1:1 propagation detected, reusing parent IDs.")
            ids = last_ids[:]
        else:
            num_out = len(coords)
            ids, self.next_id = resolve_persistent_ids(
                step_number=ctx.step_number,
                last_ids=last_ids,
                coords_count=num_out,
                output_files=output_files,
                operation=ctx.operation,
                next_id=self.next_id,
                file_map_fn=map_outputs_to_ids,
                step_dir=step_dir,
            )

        coords, ids = self.refiner.filter(
            coords,
            energies,
            ids,
            ctx.sampling.method,
            ctx.sampling.parameters,
            by_parent=False,
        )
        return coords, ids, energies, forces

    def _write_step_cache(self, ctx: StepContext, step_dir: str, fp_now, parent_ids, coords, ids, energies, forces) -> None:
        try:
            step_cache = StepCache(
                version=CACHE_VERSION,
                step=ctx.step_number,
                operation=ctx.operation,
                engine=ctx.engine,
                fingerprint=fp_now,
                parent_ids=(parent_ids if ctx.step_number > 1 else None),
                ids=ids,
                n_outputs=len(ids),
                by_parent=None,
                coords=coords,
                energies=energies,
                forces=forces,
                extras=None,
            )
            save_step_cache(step_dir, step_cache)
            logging.info(f"[step {ctx.step_number}] Wrote step cache ({len(ids)} items).")
        except Exception as e:
            logging.warning(f"[step {ctx.step_number}] Cache write failed: {e}")

    def _maybe_run_normal_mode_sampling(self, ctx: StepContext, step_dir: str, result: StepResult) -> StepResult:
        if not ctx.step.get("normal_mode_sampling", False):
            return result

        nms_params = ctx.step.get("normal_mode_sampling_parameters", {})
        calc_type = nms_params.get("calc_type", "rm_imag")
        displacement_vector = nms_params.get("displacement_vector", 1.0)
        nms_random_displacements = nms_params.get("num_random_displacements", 1)

        output_files = result.output_files or [
            os.path.join(step_dir, f)
            for f in os.listdir(step_dir)
            if f.endswith(".out") and not f.startswith("slurm")
        ]
        if not output_files:
            logging.warning(f"No valid .out files found for normal mode sampling in step {ctx.step_number}. Skipping NMS.")
            return result

        logging.info(f"Normal mode sampling requested for step {ctx.step_number}.")
        input_template_path = os.path.join(self.template_dir, f"step{ctx.step_number}.inp")

        cfg = ctx.engine_cfg  # EngineConfig

        coords, ids = self.orca.normal_mode_sampling(
            file_paths=output_files,
            calc_type=calc_type,
            input_template=input_template_path,
            slurm_template=self.template_dir,
            charge=ctx.charge,
            multiplicity=ctx.multiplicity,
            output_dir=self.output_dir,
            operation=ctx.operation,
            engine=ctx.engine,

            # --- MLFF/MLIP/PySCF config comes from engine_cfg ---
            model_name=cfg.model_name,
            step_number=ctx.step_number,
            structure_ids=result.ids,
            max_cores=self.max_cores,
            task_name=cfg.task_name,
            mlff_model=cfg.model_name,          # keep if your API expects a separate field; otherwise drop
            displacement_value=displacement_vector,
            device=cfg.device,
            bind=cfg.bind,

            # --- ORCA execution ---
            orca_executable=self.orca_executable,
            scratch_dir=self.scratch_dir,
            num_random_modes=nms_random_displacements,
        )

        # (Optional) refresh cache after NMS like you do now
        try:
            step_cache = StepCache(
                version=CACHE_VERSION,
                step=ctx.step_number,
                operation=ctx.operation,
                engine=ctx.engine,
                fingerprint=None,  # disables matching
                parent_ids=None,
                ids=ids,
                n_outputs=len(ids),
                by_parent=None,
                coords=coords,
                energies=[None] * len(ids),
                forces=[None] * len(ids),
                extras={"nms_generation": True},
            )
            save_step_cache(step_dir, step_cache)
            logging.info(f"[step {ctx.step_number}] Updated step cache after NMS ({len(ids)} items).")
        except Exception as e:
            logging.warning(f"[step {ctx.step_number}] Cache write (post-NMS) failed: {e}")

        return StepResult(coords=coords, ids=ids, energies=result.energies, forces=result.forces, output_files=output_files)

    def _commit_step(self, ctx: StepContext, result: StepResult, state: PipelineState) -> None:
        state.last_coords, state.last_ids = result.coords, result.ids
        state.last_energies, state.last_forces = result.energies, result.forces

        print(f"Step {ctx.step_number} completed: {len(state.last_coords)} structures ready.")
        print(f"Your ID's for this step are: {state.last_ids}")

    def _maybe_export_step_csv(self, ctx: StepContext, state: PipelineState) -> None:
        if state.last_energies is not None and len(state.last_energies) == len(state.last_ids):
            self.utils.save_step_csv(
                energies=state.last_energies,
                ids=state.last_ids,
                step=ctx.step_number,
                output_dir=self.output_dir,
            )
        else:
            logging.info(f"[step {ctx.step_number}] Skipping CSV export (energy/ID mismatch or missing energies).")

    def _parse_bind(self, bind: str) -> tuple[str, int]:
        """Parse 'host:port' into (host, port)."""
        host, port_s = bind.split(":")
        return host.strip(), int(port_s)

    def _format_bind(self, host: str, port: int) -> str:
        """Format (host, port) into 'host:port'."""
        return f"{host}:{port}"

    def _increment_bind(self, bind: str, delta=1) -> str:
        """Return bind with port increased by delta."""
        host, port = self._parse_bind(bind)
        return self._format_bind(host, port + int(delta))

    def run(self) -> None:
        """Main pipeline execution function for ChemRefine."""
        args = getattr(self, "args", None)
        if args is not None and self._handle_rebuild_modes(args):
            return

        logging.info("Starting ChemRefine pipeline.")
        state = PipelineState()

        for step in self.config.get("steps", []):
            ctx = self._build_step_context(step)
            self._validate_step_context(ctx)
            logging.info(
                f"Processing step {ctx.step_number}: operation '{ctx.operation}', engine '{ctx.engine}'."
            )

            # Training-only steps
            if ctx.operation in {"MLFF_TRAIN", "MLIP_TRAIN"}:
                self._run_training_step(ctx, state)
                continue

            # Paths & fingerprint
            step_dir = os.path.join(self.output_dir, f"step{ctx.step_number}")
            os.makedirs(os.path.join(step_dir, "_cache"), exist_ok=True)

            parent_ids_for_fp = state.last_ids if ctx.step_number > 1 else None
            fp_now = build_step_fingerprint(
                ctx.step, parent_ids_for_fp, ctx.sampling.parameters, ctx.step_number
            )

            # Try cache, else do heavy work
            restored = self._try_restore_from_step_cache(step_dir, fp_now, ctx)
            if restored is not None:
                result = restored
                logging.info(f"Skipping heavy work for step {ctx.step_number} (cache restored).")
            else:
                result = self._execute_step_heavy_work(ctx, state, step_dir, fp_now)

            # Optional normal mode sampling
            result = self._maybe_run_normal_mode_sampling(ctx, step_dir, result)

            # Commit results to pipeline state + export
            self._commit_step(ctx, result, state)
            self._maybe_export_step_csv(ctx, state)

        logging.info("ChemRefine pipeline completed.")



def main():
    ChemRefiner().run()


if __name__ == "__main__":
    main()

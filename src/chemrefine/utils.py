import re
import glob
import shutil
import logging
from .constants import HARTREE_TO_KCAL_MOL, R_KCAL_MOL_K, CSV_PRECISION
from pathlib import Path
import subprocess
import getpass
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class Utility:
    def extract_pal_from_qorca_output(self, output):
        patterns = [
            r"PAL.*?(\d+)", r"nprocs\s+(\d+)", r"--pal\s+(\d+)", r"pal(\d+)", r"Using (\d+) cores"
        ]
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    def save_step_csv(self, energies, ids, step, filename="steps.csv", T=298.15, output_dir="."):
        """
        Appends filtered structures for a step to a cumulative CSV, sorted by energy.

        Parameters:
            energies (list): Filtered energies in Hartrees.
            ids (list): Persistent IDs of the filtered structures.
            step (int): Step number.
            filename (str): Cumulative CSV file.
            T (float): Temperature in Kelvin for Boltzmann weighting.
            output_dir (str): Output directory.
        """
        import os
        import pandas as pd
        import numpy as np

        df = pd.DataFrame({
            'Conformer': ids,
            'Energy (Hartrees)': energies
        })
        df['Energy (kcal/mol)'] = df['Energy (Hartrees)'] * HARTREE_TO_KCAL_MOL

        # ✅ Sort table by energy (kcal/mol)
        df = df.sort_values(by='Energy (kcal/mol)', ascending=True).reset_index(drop=True)

        # Boltzmann statistics
        df['dE (kcal/mol)'] = df['Energy (kcal/mol)'] - df['Energy (kcal/mol)'].min()
        dE_RT = df['dE (kcal/mol)'] / (R_KCAL_MOL_K * T)
        df['Boltzmann Weight'] = np.exp(-dE_RT)
        df['Boltzmann Weight'] /= df['Boltzmann Weight'].sum()
        df['% Total'] = df['Boltzmann Weight'] * 100
        df['% Cumulative'] = df['% Total'].cumsum()

        df.insert(0, 'Step', step)
        df = df.round({
            'Energy (kcal/mol)': CSV_PRECISION,
            'dE (kcal/mol)': CSV_PRECISION,
            'Boltzmann Weight': CSV_PRECISION,
            '% Total': CSV_PRECISION,
            '% Cumulative': CSV_PRECISION
        })

        output_path = os.path.join(output_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        mode = 'w' if step == 1 else 'a'
        header = step == 1
        df.to_csv(output_path, mode=mode, index=False, header=header)
        logging.info(f"Saved filtered structures for step {step} to {output_path}")


    def write_xyz(self, structures, step_number, structure_ids, output_dir='.'):
        """
        Writes XYZ files for each structure in the step directory.

        Parameters:
        - structures (list of lists or arrays): Coordinates and element data.
        - step_number (int): The step number.
        - structure_ids (list): List of structure IDs.
        - output_dir (str): Output directory path.

        Returns:
        - List of XYZ filenames written.
        """
        import os
        import logging

        logging.info(f"Writing Ensemble XYZ files to {output_dir} for step {step_number}")
        base_name = f"step{step_number}"
        xyz_filenames = []

        os.makedirs(output_dir, exist_ok=True)

        for structure, structure_id in zip(structures, structure_ids):
            output_file = os.path.join(output_dir, f"{base_name}_structure_{structure_id}.xyz")
            xyz_filenames.append(output_file)
            with open(output_file, 'w') as file:
                file.write(f"{len(structure)}\n\n")
                for atom in structure:
                    element, x, y, z = atom  # Unpack atom data
                    file.write(f"{element} {x} {y} {z}\n")

        return xyz_filenames
    
    def submit_job(self, slurm_script: Path) -> str:
        """
        Submit a SLURM job and extract the job ID.

        Parameters
        ----------
        slurm_script : Path
            Path to the SLURM script.

        Returns
        -------
        str
            Job ID or 'ERROR' if submission failed.
        """
        try:
            result = subprocess.run(
                ["sbatch", str(slurm_script)],
                capture_output=True,
                text=True,
                check=True
            )
            logging.info(f"sbatch output: {result.stdout.strip()}")
            job_id = self._extract_job_id(result.stdout)
            if job_id:
                return job_id
            else:
                logging.warning("Failed to extract job ID from sbatch output.")
                return "UNKNOWN"
        except subprocess.CalledProcessError as e:
            logging.error(f"Job submission failed: {e.stderr.strip()}")
            return "ERROR"

    def is_job_finished(self, job_id: str) -> bool:
        """
        Check if a SLURM job with a given job ID has finished.

        Parameters
        ----------
        job_id : str
            SLURM job ID.

        Returns
        -------
        bool
            True if job has finished; False otherwise.
        """
        try:
            username = getpass.getuser()
            command = f"squeue -u {username} -o %i"
            output = subprocess.check_output(command, shell=True, text=True)
            job_ids = output.strip().splitlines()
            return job_id not in job_ids[1:]  # skip header
        except subprocess.CalledProcessError as e:
            logging.info(f"Error running squeue: {e}")
            return False

    def _extract_job_id(self, sbatch_output: str) -> str | None:
        """
        Extract the job ID from sbatch output.

        Parameters
        ----------
        sbatch_output : str
            Output from sbatch command.

        Returns
        -------
        str or None
            Extracted job ID or None if not found.
        """
        match = re.search(r"Submitted batch job (\d+)", sbatch_output)
        return match.group(1) if match else None
    
    def write_single_xyz(self, atoms, output_file):
        """
        Writes a single ASE Atoms object to an XYZ file.

        Parameters:
        - atoms (ase.Atoms): The atoms object to write.
        - output_file (str): Path to output XYZ file.
        """
        from ase.io import write
        write(output_file, atoms)

    # --- Append below your existing imports in chemrefine/utils.py ---
import os
import json
from typing import List, Dict, Optional

_ID_PATTERN = re.compile(r"^step(?P<step>\d+)_structure_(?P<id>\d+)\.inp$", re.IGNORECASE)


def extract_structure_id(inp_filename: str) -> Optional[int]:
    """
    Extract the integer structure ID from an input filename of the form
    'step{N}_structure_{ID}.inp'.

    Parameters
    ----------
    inp_filename : str
        Basename or path to the .inp file.

    Returns
    -------
    int | None
        Parsed ID if pattern matches; otherwise None.
    """
    name = os.path.basename(inp_filename)
    m = _ID_PATTERN.match(name)
    return int(m.group("id")) if m else None


def step_manifest_path(step_dir: str, step_number: int) -> str:
    """
    Compute the per-step manifest path.

    Parameters
    ----------
    step_dir : str
        Step folder path (e.g., outputs/step1).
    step_number : int
        Step index.

    Returns
    -------
    str
        JSON manifest path: 'step{n}_manifest.json' inside step_dir.
    """
    return os.path.join(step_dir, f"step{step_number}_manifest.json")


def write_step_manifest(step_number: int, step_dir: str, input_files: List[str],
                        operation: str, engine: str) -> None:
    """
    Create/update a per-step manifest mapping structure IDs to generated inputs.

    Parameters
    ----------
    step_number : int
        Step index.
    step_dir : str
        Step directory.
    input_files : list[str]
        Absolute (or relative) paths to the generated '.inp' files.
    operation : str
        Operation label (e.g., 'OPT+SP', 'GOAT').
    engine : str
        Engine label ('dft' or 'mlff').
    """
    manifest_file = step_manifest_path(step_dir, step_number)
    records = []
    for inp in input_files:
        sid = extract_structure_id(inp)
        records.append({
            "structure_id": sid,
            "input_file": os.path.basename(inp),
            "output_file": None,
            "operation": operation.upper(),
            "engine": engine.lower(),
        })
    with open(manifest_file, "w") as f:
        json.dump({"step": step_number, "records": records}, f, indent=2)


def read_step_manifest(step_dir: str, step_number: int) -> Optional[Dict]:
    """
    Load a per-step manifest if it exists.

    Parameters
    ----------
    step_dir : str
        Step directory.
    step_number : int
        Step index.

    Returns
    -------
    dict | None
        Parsed manifest or None if missing.
    """
    p = step_manifest_path(step_dir, step_number)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def update_step_manifest_outputs(step_dir: str, step_number: int,
                                 output_files: List[str]) -> None:
    """
    Update the per-step manifest with resolved output filenames by matching stems.

    Parameters
    ----------
    step_dir : str
        Step directory.
    step_number : int
        Step index.
    output_files : list[str]
        Output files discovered after the run.

    Notes
    -----
    Inputs and outputs typically share a stem:
    'stepX_structure_ID' -> 'stepX_structure_ID.out' or similar.
    """
    manifest = read_step_manifest(step_dir, step_number)
    if not manifest or "records" not in manifest:
        return

    # Build lookup from input stem -> record
    by_input_stem = {}
    for r in manifest["records"]:
        stem = os.path.splitext(r["input_file"])[0]
        by_input_stem[stem] = r

    for outp in output_files:
        out_stem = os.path.splitext(os.path.basename(outp))[0]
        rec = by_input_stem.get(out_stem)
        if rec is not None:
            rec["output_file"] = os.path.basename(outp)

    with open(step_manifest_path(step_dir, step_number), "w") as f:
        json.dump(manifest, f, indent=2)


def map_outputs_to_ids(step_dir: str, step_number: int, output_files: List[str]) -> List[int]:
    """
    Resolve structure IDs for outputs using the manifest; fall back to robust parsing
    from output filenames (handles suffixes like '_atom46', '_trj', etc.) and prefix matching.

    Returns
    -------
    list[int]
        Structure IDs aligned with output_files order. Unresolved IDs are -1.
    """
    manifest = read_step_manifest(step_dir, step_number)
    ids: List[int] = []

    # Build indices from manifest
    by_input_stem: Dict[str, Optional[int]] = {}
    if manifest and "records" in manifest:
        for r in manifest["records"]:
            stem = os.path.splitext(r["input_file"])[0]  # e.g., 'step3_structure_0'
            by_input_stem[stem] = r.get("structure_id")

    input_stems = list(by_input_stem.keys())

    for outp in output_files:
        out_base = os.path.basename(outp)
        out_stem, _ = os.path.splitext(out_base)

        # 1) Exact stem match via manifest (when outputs share stem with inputs)
        if out_stem in by_input_stem and by_input_stem[out_stem] is not None:
            ids.append(int(by_input_stem[out_stem]))
            continue

        # 2) Prefix match: output stem starts with input stem (handles '_atom46', '_trj', etc.)
        pref = [s for s in input_stems if out_stem.startswith(s)]
        if len(pref) == 1 and by_input_stem[pref[0]] is not None:
            ids.append(int(by_input_stem[pref[0]]))
            continue

        # 3) Regex extraction from the output name itself
        sid = extract_structure_id_from_any_name(out_base)
        if sid is not None:
            ids.append(sid)
            continue

        # 4) Try a corresponding .inp with truncated suffix (common case)
        #    e.g., 'step3_structure_0_atom46' -> try 'step3_structure_0.inp'
        if "_" in out_stem:
            truncated = out_stem.rsplit("_", 1)[0]
            candidate_inp = os.path.join(step_dir, truncated + ".inp")
            sid2 = extract_structure_id(candidate_inp)
            if sid2 is not None:
                ids.append(sid2)
                continue

        # 5) Last resort: -1 (caller should fail fast before creating inputs)
        ids.append(-1)

    return ids

from typing import Sequence

def validate_structure_ids_or_raise(structure_ids: Sequence[int], step_number: int) -> None:
    """
    Ensure all structure IDs are resolved (>= 0) and non-empty.
    Raises ValueError if invalid, to prevent creating '..._-1.inp' files.

    Parameters
    ----------
    structure_ids : Sequence[int]
        IDs to validate.
    step_number : int
        Current step index.
    """
    if not structure_ids:
        raise ValueError(f"Step {step_number}: empty structure ID list.")
    if any((i is None) or (i < 0) for i in structure_ids):
        bad = [i for i in structure_ids if (i is None) or (i < 0)]
        raise ValueError(f"Step {step_number}: unresolved structure IDs present: {bad}")


def registry_path(output_root: str) -> str:
    """
    Return the path to the global ID registry JSON in the pipeline output root.

    Parameters
    ----------
    output_root : str
        Pipeline root output directory.

    Returns
    -------
    str
        Registry file path.
    """
    return os.path.join(output_root, "id_registry.json")


def get_next_id(output_root: str) -> int:
    """
    Return the next global integer structure ID and advance the registry.

    Parameters
    ----------
    output_root : str
        Pipeline root output directory.

    Returns
    -------
    int
        Next unique integer ID.
    """
    p = registry_path(output_root)
    if os.path.exists(p):
        with open(p) as f:
            data = json.load(f)
    else:
        data = {"next_id": 0}
    nid = data["next_id"]
    data["next_id"] = nid + 1
    with open(p, "w") as f:
        json.dump(data, f, indent=2)
    return nid

def write_synthetic_manifest_for_ensemble(step_number, step_dir, n_structures, operation, engine, output_basename):
    """
    Create a synthetic manifest for a GOAT step where only one ensemble file exists.
    Assigns sequential IDs [0..n_structures-1] and writes to step{N}_manifest.json.

    Parameters
    ----------
    step_number : int
        The step index (e.g., 1).
    step_dir : str
        Path to the step directory.
    n_structures : int
        Number of structures in the ensemble.
    operation : str
        Operation string (e.g., "GOAT").
    engine : str
        Engine string (e.g., "dft").
    output_basename : str
        Filename of the ensemble file (without directory).
    """
    manifest_path = os.path.join(step_dir, f"step{step_number}_manifest.json")
    manifest_data = {
        "step": step_number,
        "operation": operation,
        "engine": engine,
        "structures": [],
        "outputs": [output_basename],
    }
    for idx in range(n_structures):
        manifest_data["structures"].append({
            "id": idx,
            "input": f"step{step_number}_structure_{idx}.inp",
            "output": output_basename
        })
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    logging.info(f"Wrote synthetic manifest for step {step_number} with {n_structures} IDs at {manifest_path}.")


def get_ensemble_ids(step_dir, step_number, n_structures, operation, engine, output_basename):
    """
    Retrieve sequential IDs for a GOAT ensemble file. If the manifest does not exist,
    generate it using write_synthetic_manifest_for_ensemble.

    Parameters
    ----------
    step_dir : str
        Path to the step directory.
    step_number : int
        Step index.
    n_structures : int
        Number of structures in the ensemble.
    operation : str
        Operation string.
    engine : str
        Engine string.
    output_basename : str
        Ensemble file name.

    Returns
    -------
    list[int]
        Sequential IDs [0..n_structures-1].
    """
    manifest_path = os.path.join(step_dir, f"step{step_number}_manifest.json")
    if not os.path.exists(manifest_path):
        logging.info(f"Manifest for step {step_number} not found; creating synthetic manifest.")
        write_synthetic_manifest_for_ensemble(
            step_number, step_dir, n_structures, operation, engine, output_basename
        )
    return list(range(n_structures))

import os
from typing import Optional

_ID_ANYWHERE_RE = re.compile(
    r"step(?P<step>\d+)_structure_(?P<id>\d+)(?:_|\.|$)", re.IGNORECASE
)

def extract_structure_id_from_any_name(name: str) -> Optional[int]:
    """
    Extract a structure ID from any filename or stem containing the pattern
    'step{N}_structure_{ID}[...optional suffixes...]'.

    Examples
    --------
    'step3_structure_0_atom46.out' → 0
    'step4_structure_12_trj.xyz'   → 12
    'step2_structure_5.out'        → 5

    Parameters
    ----------
    name : str
        A basename, full path, or stem.

    Returns
    -------
    int | None
        Parsed ID if found, else None.
    """
    base = os.path.basename(name)
    stem = os.path.splitext(base)[0]
    m = _ID_ANYWHERE_RE.search(stem)
    return int(m.group("id")) if m else None

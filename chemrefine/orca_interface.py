import os
import re
from .utils import Utility
import subprocess
import logging
from pathlib import Path
import sys
import time
import getpass

# chemrefine/orca_interface.py

class OrcaJobSubmitter:
    """
    A lightweight ORCA job submission class for ChemRefine.
    Handles job submission, PAL adjustment, and job monitoring.
    """

    def __init__(self, orca_executable: str = "orca", scratch_dir: str = None, save_scratch: bool = False):
        """
        Initialize the ORCA job submitter.

        Args:
            orca_executable (str): Path to the ORCA executable.
            scratch_dir (str): Path to the scratch directory.
            save_scratch (bool): If True, scratch directories are not deleted after job completion.
        """
        self.orca_executable = orca_executable
        self.scratch_dir = scratch_dir or os.getenv("SCRATCH", "/tmp/orca_scratch")
        self.save_scratch = save_scratch

    def submit_files(self, input_files, max_cores=32, partition="sterling", template_dir=".", output_dir="."):
        """
        Submits multiple ORCA input files to SLURM, managing PAL values, active job tracking,
        and ensuring that the total PAL usage does not exceed max_cores.

        Args:
            input_files (list): List of ORCA input file paths.
            max_cores (int): Maximum total PAL usage allowed.
            partition (str): SLURM partition to monitor jobs.
            template_dir (str): Directory containing SLURM header template.
            output_dir (str): Output directory for SLURM scripts and results.
        """
        total_cores_used = 0
        active_jobs = {}

        for input_file in input_files:
            input_path = Path(input_file).resolve()
            pal_value = self.parse_pal_from_input(input_path)
            pal_value = min(pal_value, max_cores)
            logging.info(f"Setting PAL value to {pal_value} for {input_path.name}")

            # Wait if not enough free cores
            while total_cores_used + pal_value > max_cores:
                logging.info("Waiting for jobs to finish to free up cores...")
                completed_jobs = []
                for job_id, cores in list(active_jobs.items()):
                    if self.is_job_finished(job_id, partition):
                        completed_jobs.append(job_id)
                        total_cores_used -= cores
                        logging.info(f"Job {job_id} completed. Freed {cores} cores.")

                for job_id in completed_jobs:
                    del active_jobs[job_id]

                time.sleep(30)

            slurm_script = self.generate_slurm_script(
                input_file=input_path,
                pal_value=pal_value,
                template_dir=template_dir,
                output_dir=output_dir
            )

            job_id = self.submit_job(slurm_script)
            logging.info(f"Submitted ORCA job with ID: {job_id} for input: {input_path.name}")

            if job_id.isdigit():
                active_jobs[job_id] = pal_value
                total_cores_used += pal_value
            else:
                logging.warning(f"Skipping job tracking for invalid job ID '{job_id}'")

        logging.info("All jobs submitted. Waiting for remaining jobs to complete...")
        while active_jobs:
            completed_jobs = []
            for job_id, cores in list(active_jobs.items()):
                if self.is_job_finished(job_id, partition):
                    completed_jobs.append(job_id)
                    total_cores_used -= cores
                    logging.info(f"Job {job_id} completed. Freed {cores} cores.")

            for job_id in completed_jobs:
                del active_jobs[job_id]

            time.sleep(30)

        logging.info("All calculations finished.")

    def parse_pal_from_input(self, input_file: Path):
        """
        Parse the PAL value from the ORCA input file.

        Args:
            input_file (Path): ORCA input file path.

        Returns:
            int: PAL value (default 1 if not found).
        """
        content = input_file.read_text()
        match = re.search(r"nprocs\s+(\d+)", content, re.IGNORECASE)
        if match:
            pal_value = int(match.group(1))
            logging.info(f"Found PAL value {pal_value} in {input_file}")
            return pal_value
        return 1

    def generate_slurm_script(self, input_file: Path, pal_value: int, template_dir: str, output_dir: str = ".", job_name: str = None):
        """
        Generate a SLURM script by merging user-provided header and ChemRefine additions.
        """
        if job_name is None:
            job_name = input_file.stem

        header_template_path = Path(template_dir) / "orca.slurm.header"
        if not header_template_path.is_file():
            logging.error(f"SLURM header template {header_template_path} not found.")
            raise FileNotFoundError(f"SLURM header template {header_template_path} not found.")

        with open(header_template_path, 'r') as f:
            header_lines = f.readlines()

        sbatch_lines = [line for line in header_lines if line.strip().startswith("#SBATCH")]
        non_sbatch_lines = [line for line in header_lines if not line.strip().startswith("#SBATCH")]

        slurm_file = Path(output_dir) / f"{job_name}.slurm"
        with open(slurm_file, 'w') as f:
            f.write("#!/bin/bash\n")
            f.writelines(sbatch_lines)
            f.write(f"#SBATCH --job-name={job_name}\n")
            f.write(f"#SBATCH --output={job_name}.out\n")
            f.write(f"#SBATCH --error={job_name}.err\n")
            f.write(f"#SBATCH --ntasks={pal_value}\n")
            f.write("#SBATCH --cpus-per-task=1\n\n")
            f.writelines(non_sbatch_lines)
            f.write("\n# Scratch directory management and ORCA execution\n")
            f.write(f"if [ -z \"$ORCA_EXEC\" ]; then\n    ORCA_EXEC={self.orca_executable}\nfi\n\n")
            f.write("export ORIG=$PWD\n")
            f.write("timestamp=$(date +%Y%m%d%H%M%S)\n")
            f.write("random_str=$(tr -dc a-z0-9 </dev/urandom | head -c 8)\n")
            f.write(f"export BASE_SCRATCH_DIR={self.scratch_dir}\n")
            f.write("export SCRATCH_DIR=${BASE_SCRATCH_DIR}/ChemRefine_scratch_${SLURM_JOB_ID}_${timestamp}_${random_str}\n")
            f.write("mkdir -p $SCRATCH_DIR || { echo 'Error: Failed to create scratch directory'; exit 1; }\n")
            f.write(f"cp {input_file} $SCRATCH_DIR/\n")
            f.write("cd $SCRATCH_DIR || { echo 'Error: Failed to change directory'; exit 1; }\n")
            f.write("export OMP_NUM_THREADS=1\n")
            f.write(f"$ORCA_EXEC {input_file.name} > $ORIG/{job_name}.out || {{ echo 'Error: ORCA execution failed.'; exit 1; }}\n")
            f.write("cp *.out $ORIG/\n")
            f.write("cp *.xyz $ORIG/\n")
            if not self.save_scratch:
                f.write("rm -rf $SCRATCH_DIR\n")
            else:
                f.write("echo 'Scratch directory not deleted (save_scratch is True)'\n")

        logging.info(f"Generated SLURM script: {slurm_file}")
        return slurm_file

    def submit_job(self, slurm_script: Path):
        """
        Submit the SLURM script to the scheduler.

        Args:
            slurm_script (Path): Path to the SLURM script.

        Returns:
            str: Job ID or error message.
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
                logging.info(f"Submitted job ID: {job_id}")
                return job_id
            else:
                logging.warning("Failed to extract job ID from sbatch output.")
                return "UNKNOWN"
        except subprocess.CalledProcessError as e:
            logging.error(f"Job submission failed: {e.stderr.strip()}")
            return "ERROR"

    def is_job_finished(self, job_id, partition="sterling"):
        """
        Check if a SLURM job with a given job ID has finished.

        Args:
            job_id (str): SLURM job ID to check.
            partition (str): SLURM partition.

        Returns:
            bool: True if the job is no longer in the queue (finished), False otherwise.
        """
        try:
            username = getpass.getuser()
            command = f"squeue -u {username} -p {partition} -o %i"
            output = subprocess.check_output(command, shell=True, text=True)
            job_ids = output.strip().splitlines()
            return job_id not in job_ids[1:]  # skip header
        except subprocess.CalledProcessError as e:
            logging.info(f"Error running squeue: {e}")
            return False

    def _extract_job_id(self, sbatch_output: str):
        """
        Extract the job ID from sbatch output.

        Args:
            sbatch_output (str): Output from sbatch command.

        Returns:
            str or None: Job ID if found, else None.
        """
        match = re.search(r"Submitted batch job (\d+)", sbatch_output)
        return match.group(1) if match else None
       
class OrcaInterface:
    def __init__(self):
        self.utility = Utility()

    def create_input(self, xyz_files, template, charge, multiplicity, output_dir='./'):
        input_files, output_files = [], []
        logging.info(f"output_dir IN create_input: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        for xyz in xyz_files:
            base = os.path.splitext(os.path.basename(xyz))[0]
            inp = os.path.join(output_dir, f"{base}.inp")
            out = os.path.join(output_dir, f"{base}.out")
            input_files.append(inp)
            output_files.append(out)

            with open(template, "r") as tmpl:
                content = tmpl.read()

            # Strip existing xyzfile lines and clean formatting
            content = re.sub(r'^\s*\*\s+xyzfile.*$', '', content, flags=re.MULTILINE)
            content = content.rstrip() + '\n\n'
            content += f"* xyzfile {charge} {multiplicity} {xyz}\n\n"

            with open(inp, "w") as f:
                f.write(content)

        return input_files, output_files

    def parse_output(self, file_paths, calculation_type, dir='./'):
        """
        Parses ORCA output files for the specified calculation type.
        
        Args:
            file_paths (list): List of .out file paths.
            calculation_type (str): Type of calculation ('goat', 'dft', etc.).
            dir (str): Directory to look for outputs.
        
        Returns:
            tuple: (coordinates, energies)
        """
        coordinates, energies = [], []

        logging.info(f"Parsing calculation type: {calculation_type.upper()}")
        logging.info(f"Looking for output files in directory: {dir}")

        for out_file in file_paths:
            path = os.path.join(dir, os.path.basename(out_file))
            logging.info(f"Checking output file: {path}")

            if not os.path.exists(path):
                logging.warning(f"Output file not found: {path}")
                continue

            with open(path) as f:
                content = f.read()

            if calculation_type.lower() == 'goat':
                final_xyz = path.replace('.out', '.finalensemble.xyz')
                logging.info(f"Looking for GOAT ensemble file: {final_xyz}")

                if os.path.exists(final_xyz):
                    logging.info(f"GOAT ensemble file found: {final_xyz}")
                    with open(final_xyz) as fxyz:
                        lines = fxyz.readlines()

                    current_structure = []
                    for line in lines:
                        if len(line.strip().split()) == 4:
                            current_structure.append(tuple(line.strip().split()))
                    coordinates.append(current_structure)

                    energy_match = re.search(r"^\s*[-]?\d+\.\d+", lines[1])
                    energies.append(float(energy_match.group()) if energy_match else None)
                else:
                    logging.error(f"GOAT ensemble file not found for: {path}")
                    continue
            else:
                logging.info(f"Parsing standard DFT output for: {path}")
                coord_block = re.findall(
                    r"CARTESIAN COORDINATES \\(ANGSTROEM\\)\n-+\n((?:.*?\n)+?)-+\n",
                    content,
                    re.DOTALL
                )
                if coord_block:
                    coords = [line.split() for line in coord_block[-1].strip().splitlines()]
                    coordinates.append(coords)
                else:
                    logging.warning(f"No coordinate block found in: {path}")
                    coordinates.append([])

                energy_match = re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", content)
                if energy_match:
                    energy = float(energy_match[-1])
                    energies.append(energy)
                else:
                    logging.warning(f"No energy found in: {path}")
                    energies.append(None)

        if not coordinates or not energies:
            logging.error(f"Failed to parse {calculation_type.upper()} outputs in directory: {dir}")

        return coordinates, energies


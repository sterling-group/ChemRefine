import os
import re
from .utils import Utility
import subprocess
import logging
from pathlib import Path

# chemrefine/orca_interface.py

class OrcaJobSubmitter:
    """
    A lightweight ORCA job submission class for ChemRefine.

    Features:
    - Adjust PAL value in the input file.
    - Generate a basic SLURM script.
    - Submit ORCA jobs to the SLURM scheduler.
    """

    def __init__(self, orca_executable: str = "orca", scratch_dir: str = None, save_scratch: bool = False):
        """
            Initialize the ORCA job submitter.

            Args:
                orca_executable (str): Path to the ORCA executable.
                scratch_dir (str): Path to the scratch directory. If not provided, the scratch directory is
                                determined as follows:
                                    1. If `scratch_dir` is provided, it is used.
                                    2. If the environment variable 'SCRATCH' is set, it is used.
                                    3. Otherwise, defaults to '/tmp/orca_scratch'.
                save_scratch (bool): If True, scratch directories are not deleted after job completion.

            The scratch directory hierarchy allows the user to override via command-line argument or environment variable.
            """
        self.orca_executable = orca_executable
        self.scratch_dir = scratch_dir or os.getenv("SCRATCH", "/tmp/orca_scratch")
        self.save_scratch = save_scratch

    def submit_files(self, input_files, max_cores, qorca_flags=None):
        """
        Submits multiple ORCA input files to SLURM.

        Args:
            input_files (list): List of input file paths.
            max_cores (int): Maximum allowed cores per job.
            qorca_flags (dict, optional): Additional flags (currently unused).
        """
        for input_file in input_files:
            input_path = Path(input_file)

            # 1️ Parse PAL value
            pal_value = self.parse_pal_from_input(input_path)
            pal_value = min(pal_value, max_cores)

            # 2️ Adjust PAL in the input
            self.adjust_pal_in_input(input_path, pal_value)

            # 3️ Generate SLURM script
            slurm_script = self.generate_slurm_script(input_path, pal_value)

            # 4️ Submit the job
            job_id = self.submit_job(slurm_script)
            logging.info(f"Job submitted with ID: {job_id}")      

    def adjust_pal_in_input(self, input_file: Path, pal_value: int):
        """
        Adjust PAL value in the ORCA input file.
        Adds or updates a '%pal nprocs' block at the end.

        Args:
            input_file (Path): ORCA input file path.
            pal_value (int): PAL value to set.

        Returns:
            bool: True if file was modified, False otherwise.
        """
        content = input_file.read_text()
        original_content = content

        # Try to find and replace existing %pal block
        def replace_pal_block(match):
            block = match.group(0)
            new_block = re.sub(
                r"(nprocs\s+)\d+",
                r"\g<1>{}".format(pal_value),
                block,
                flags=re.IGNORECASE
            )
            return new_block

        new_content = re.sub(
            r"%pal\s+.*?end",
            replace_pal_block,
            content,
            flags=re.IGNORECASE | re.DOTALL
        )

        if new_content == content:
            # Append PAL block at the end
            new_content += f"\n%pal\n   nprocs {pal_value}\nend\n"

        # Overwrite input file only if content changed
        if new_content != original_content:
            input_file.write_text(new_content)
            logging.info(f"Adjusted PAL value to {pal_value} in {input_file}")
            return True

        logging.info(f"No changes made to PAL value in {input_file}")
        return False

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

    def generate_slurm_script(self, input_file: Path, pal_value: int, job_name: str = None):
        """
    Generate a SLURM script to run an ORCA job, including scratch directory handling.

    Args:
        input_file (Path): Path to the ORCA input file.
        pal_value (int): Number of CPUs to allocate for the job.
        job_name (str, optional): Name of the SLURM job. Defaults to the input file stem.

    This script:
        - Exports the scratch directory variable. Priority:
            1. User-provided scratch directory.
            2. Environment variable 'SCRATCH'.
            3. Default '/tmp/orca_scratch'.
        - Copies input and supporting files to the scratch directory.
        - Runs the ORCA job from the scratch directory.
        - Copies output files back to the original directory.
        - Deletes the scratch directory after job completion unless `save_scratch` is True.
    """
        if job_name is None:
            job_name = input_file.stem

        slurm_file = input_file.with_suffix(".slurm")
        with slurm_file.open("w") as f:
            f.write("#!/bin/bash\n")
            f.write("#SBATCH --export=ALL\n")
            f.write(f"#SBATCH --job-name={job_name}\n")
            f.write(f"#SBATCH --output={job_name}.out\n")
            f.write(f"#SBATCH --error={job_name}.err\n")
            f.write(f"#SBATCH --ntasks={pal_value}\n")
            f.write("#SBATCH --cpus-per-task=1\n")
            f.write("#SBATCH --time=24:00:00\n")
            f.write("\nmodule purge\n")
            f.write("# Load ORCA modules here if needed\n")

            f.write("\n# Set scratch directory\n")
            f.write("export ORIG=$PWD\n")
            f.write("timestamp=$(date +%Y%m%d%H%M%S)\n")
            f.write("random_str=$(tr -dc a-z0-9 </dev/urandom | head -c 8)\n")
            if self.scratch_dir:
                f.write(f"export SCRATCH_DIR={self.scratch_dir}\n")
            else:
                f.write(
                    'export SCRATCH_DIR=/home/$USER/scratch/orca_${SLURM_JOB_ID}_${timestamp}_${random_str}\n'
                )
            f.write("mkdir -p $SCRATCH_DIR || { echo 'Error: Failed to create scratch directory'; exit 1; }\n")
            f.write("echo 'SCRATCH_DIR is set to $SCRATCH_DIR'\n")

            f.write("\n# Copy input file and necessary files to scratch\n")
            f.write(f"cp {input_file} $SCRATCH_DIR/ || {{ echo 'Error: Failed to copy input file'; exit 1; }}\n")
            f.write("for file in *.xyz *.pot *.gbw *.hess; do\n")
            f.write("  if [ -e \"$file\" ]; then\n")
            f.write("    cp \"$file\" $SCRATCH_DIR/ || { echo 'Error: Failed to copy $file'; exit 1; }\n")
            f.write("  fi\n")
            f.write("done\n")

            f.write("\n# Change to scratch directory\n")
            f.write("cd $SCRATCH_DIR || { echo 'Error: Failed to change directory'; exit 1; }\n")

            f.write("\n# Start ORCA job\n")
            f.write("export OMP_NUM_THREADS=1\n")
            f.write(f"{self.orca_executable} {input_file.name} > $ORIG/{job_name}.out || {{ echo 'Error: ORCA execution failed.'; exit 1; }}\n")

            f.write("\n# Copy output files back\n")
            f.write("for file in *.xyz *.pot *.gbw *.out *.hess; do\n")
            f.write("  if [ -f \"$file\" ]; then\n")
            f.write("    cp \"$file\" $ORIG/ || { echo 'Error: Failed to copy $file back'; exit 1; }\n")
            f.write("  fi\n")
            f.write("done\n")

            if not self.save_scratch:
                f.write("\n# Clean up scratch directory\n")
                f.write("rm -rf $SCRATCH_DIR || { echo 'Warning: Failed to remove scratch directory'; }\n")
            else:
                f.write("echo 'Not deleting scratch directory (save_scratch is True)'\n")

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
        coordinates, energies = [], []
        for out_file in file_paths:
            path = os.path.join(dir, os.path.basename(out_file))
            if not os.path.exists(path):
                continue
            with open(path) as f:
                content = f.read()

            if calculation_type.lower() == 'goat':
                final_xyz = path.replace('.out', '.finalensemble.xyz')
                if os.path.exists(final_xyz):
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
                coord_block = re.findall(
                    r"CARTESIAN COORDINATES \\(ANGSTROEM\\)\n-+\n((?:.*?\n)+?)-+\n",
                    content,
                    re.DOTALL
                )
                coords = [line.split() for line in coord_block[-1].strip().splitlines()] if coord_block else []
                energy_match = re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", content)
                energy = float(energy_match[-1]) if energy_match else None
                coordinates.append(coords)
                energies.append(energy)

        return coordinates, energies

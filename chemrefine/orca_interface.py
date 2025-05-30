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

    def __init__(self, orca_executable: str = "orca"):
        self.orca_executable = orca_executable

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
        Generate a simple SLURM script to run the ORCA job.

        Args:
            input_file (Path): ORCA input file path.
            pal_value (int): Number of CPUs to request.
            job_name (str, optional): Job name in SLURM.

        Returns:
            Path: Path to the generated SLURM script.
        """
        if job_name is None:
            job_name = input_file.stem

        slurm_file = input_file.with_suffix(".slurm")
        with slurm_file.open("w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"#SBATCH --job-name={job_name}\n")
            f.write(f"#SBATCH --output={job_name}.out\n")
            f.write(f"#SBATCH --error={job_name}.err\n")
            f.write(f"#SBATCH --ntasks={pal_value}\n")
            f.write("#SBATCH --time=24:00:00\n")  # Default 24 hours

            f.write("\nmodule purge\n")
            f.write("# Load ORCA modules here if needed\n")
            f.write("\n")
            f.write(f"srun {self.orca_executable} {input_file.name}\n")

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

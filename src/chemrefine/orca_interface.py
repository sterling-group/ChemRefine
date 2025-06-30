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

    def __init__(self, device='cpu',orca_executable: str = "orca",bind: str = "127.0.0.1:8888",scratch_dir: str = None, save_scratch: bool = False):
        """
        Initialize the ORCA job submitter.

        Args:
            orca_executable (str): Path to the ORCA executable.
            scratch_dir (str): Path to the scratch directory.
            save_scratch (bool): If True, scratch directories are not deleted after job completion.
        """
        self.orca_executable = orca_executable
        self.scratch_dir = scratch_dir
        self.save_scratch = save_scratch
        self.utility = Utility()
        self.device = device  
        self.bind = bind
    def submit_files(self, input_files, max_cores=32, template_dir=".", output_dir=".",device=None,calculation_type='DFT',model_name=None,task_name=None):
        """
        Submits multiple ORCA input files to SLURM, managing PAL values, active job tracking,
        and ensuring that the total PAL usage does not exceed max_cores.

        Args:
            input_files (list): List of ORCA input file paths.
            max_cores (int): Maximum total PAL usage allowed.
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
                    if self.utility.is_job_finished(job_id):
                        completed_jobs.append(job_id)
                        total_cores_used -= cores
                        logging.info(f"Job {job_id} completed. Freed {cores} cores.")

                for job_id in completed_jobs:
                    del active_jobs[job_id]

                time.sleep(10)

            slurm_script = self.generate_slurm_script(
                input_file=input_path,
                pal_value=pal_value,
                template_dir=template_dir,
                output_dir=output_dir,
                device=self.device,
                model_name=model_name,
                task_name=task_name,
                calculation_type=calculation_type,
                bind=self.bind
            )

            job_id = self.utility.submit_job(slurm_script)
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
                if self.utility.is_job_finished(job_id):
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

    def generate_slurm_script(
    self,
    input_file: Path,
    pal_value: int,
    template_dir: str,
    output_dir: str = ".",
    job_name: str = None,
    device: str = "cpu",
    calculation_type: str = "dft",
    model_name: str = "uma-s-1",
    task_name: str = "omol",
    bind: str = "127.0.0.1:8888"
):
        """
        Generate a SLURM script by combining a user-provided header with a consistent footer.

        The header (orca.slurm.header) is defined by the user and contains cluster-specific SLURM settings and module loads.
        This function groups all #SBATCH lines together at the top to comply with SLURM parsing, ensuring no non-SBATCH lines interrupt them.
        The footer includes scratch directory management, file copying, and job execution.

        Args:
            input_file (Path): ORCA input file path.
            pal_value (int): Number of processors to allocate (ntasks).
            template_dir (str): Path to the directory containing the SLURM header.
            output_dir (str): Path to the directory where the SLURM script will be saved.
            job_name (str, optional): Name of the SLURM job. Defaults to the input file stem.

        Returns:
            Path: Path to the generated SLURM script.
        """

        import logging
        from pathlib import Path

        logging.info(f"Using ORCA executable: {self.orca_executable}")
        if job_name is None:
            job_name = input_file.stem
        if not self.scratch_dir:
            logging.warning("scratch_dir not set; defaulting to /tmp/orca_scratch")
            self.scratch_dir = "./tmp/orca_scratch"

        header_name = "cuda.slurm.header" if device == "cuda" else "cpu.slurm.header"
        logging.info(f"Using SLURM header template: {header_name}")
        header_template_path = Path(os.path.abspath(template_dir)) / header_name

        if not header_template_path.is_file():
            logging.error(f"SLURM header template {header_template_path} not found.")
            raise FileNotFoundError(f"SLURM header template {header_template_path} not found.")

        with open(header_template_path, 'r') as f:
            header_lines = f.readlines()

        # Separate SBATCH and non-SBATCH lines to ensure all SBATCH lines are grouped
        sbatch_lines = []
        non_sbatch_lines = []

        for line in header_lines:
            stripped = line.strip()
            if stripped.startswith("#SBATCH"):
                if "--ntasks" in stripped or "--cpus-per-task" in stripped:
                    continue  # skip existing directives that we'll override
                sbatch_lines.append(line.rstrip())
            else:
                non_sbatch_lines.append(line.rstrip())

        # Append job-specific SBATCH directives
        sbatch_lines.append(f"#SBATCH --job-name={job_name}")
        sbatch_lines.append(f"#SBATCH --output=slurm_{job_name}.out")
        sbatch_lines.append(f"#SBATCH --error=slurm_{job_name}.err")
        sbatch_lines.append(f"#SBATCH --ntasks={pal_value}")
        sbatch_lines.append("#SBATCH --cpus-per-task=1")

        # Compose SLURM script
        slurm_file = Path(output_dir) / f"{job_name}.slurm"
        with open(slurm_file, 'w') as f:
            f.write("#!/bin/bash\n\n")
            f.write("\n".join(sbatch_lines))
            f.write("\n\n")
            # Write the rest of the header
            f.write("\n".join(non_sbatch_lines))
            f.write("\n\n")
            # Write scratch management and ORCA execution block
            f.write(f"ORCA_EXEC={self.orca_executable}\n")
            f.write("# Scratch directory management and ORCA execution (generated by ChemRefine)\n")
            f.write("if [ -z \"$ORCA_EXEC\" ]; then\n")
            f.write(f"    ORCA_EXEC={self.orca_executable}\n")
            f.write("fi\n\n")

            f.write("timestamp=$(date +%Y%m%d%H%M%S)\n")
            f.write("random_str=$(tr -dc a-z0-9 </dev/urandom | head -c 8)\n")
            f.write(f"export BASE_SCRATCH_DIR={self.scratch_dir}\n")
            f.write("export SCRATCH_DIR=${BASE_SCRATCH_DIR}/ChemRefine_scratch_${SLURM_JOB_ID}_${timestamp}_${random_str}\n")
            f.write(f"export OUTPUT_DIR={os.path.abspath(output_dir)}\n")

            f.write("mkdir -p $SCRATCH_DIR || { echo 'Error: Failed to create scratch directory'; exit 1; }\n")
            f.write("echo 'SCRATCH_DIR is set to $SCRATCH_DIR'\n\n")

            f.write(f"cp {input_file} $SCRATCH_DIR/ || {{ echo 'Error: Failed to copy input file'; exit 1; }}\n")
            f.write("cd $SCRATCH_DIR || { echo 'Error: Failed to change directory'; exit 1; }\n\n")

            if calculation_type.lower() == "mlff":
                f.write("# Start MLFF socket server before ORCA\n")
                f.write(f"python -m chemrefine.server --model {model_name} --task-name {task_name} --device {device} --bind {bind} & > $OUTPUT_DIR/server.log 2>&1 & \n")
                f.write("SERVER_PID=$!\n")
                f.write("sleep 10\n")
                f.write(f"$ORCA_EXEC {input_file.name} > $OUTPUT_DIR/{job_name}.out || {{ echo 'Error: ORCA execution failed.'; kill $SERVER_PID; exit 1; }}\n")
                f.write("kill $SERVER_PID\n\n")
            else:
                f.write("export OMP_NUM_THREADS=1\n")
                f.write(f"$ORCA_EXEC {input_file.name} > $OUTPUT_DIR/{job_name}.out || {{ echo 'Error: ORCA execution failed.'; exit 1; }}\n\n")


            # File copy commands
            f.write("cp *.out *.xyz *.finalensemble.*.xyz *.finalensemble.xyz $OUTPUT_DIR \n")
            
            if not self.save_scratch:
                f.write("rm -rf $SCRATCH_DIR || { echo 'Warning: Failed to remove scratch directory'; }\n")
            else:
                f.write("echo 'Scratch directory not deleted (save_scratch is True)'\n")

        logging.info(f"Generated SLURM script: {slurm_file}")
        return slurm_file

class OrcaInterface:
    def __init__(self):
        self.utility = Utility()

    def create_input(self, 
                     xyz_files, 
                     template, 
                     charge, 
                     multiplicity, 
                     output_dir='./', 
                     calculation_type=None,
                     model_name=None,
                     task_name=None,
                     device='cuda',
                     bind='127.0.0.1:8888'
):

        input_files, output_files = [], []
        logging.debug(f"output_dir IN create_input: {output_dir}")
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
            if calculation_type and calculation_type.lower() == 'mlff':
                # Add MLFF method block if specified
                run_mlff_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "uma.sh"))
                ext_params = f"--model_name {model_name} --task_name {task_name} --device {device} --bind {bind}"
                content = content.rstrip() + '\n'
                content += f'%method\n  ProgExt "{run_mlff_path}"\n  Ext_Params "{ext_params}"\nend\n\n'

            content += f"* xyzfile {charge} {multiplicity} {xyz}\n\n"

            with open(inp, "w") as f:
                f.write(content)

        return input_files, output_files

    def parse_output(self, file_paths, calculation_type, dir='./'):
        coordinates, energies = [], []

        logging.info(f"Parsing calculation type: {calculation_type.upper()}")
        logging.info(f"Looking for output files in directory: {dir}")

        for out_file in file_paths:
            path = os.path.join(dir, os.path.basename(out_file))
            logging.info(f"Checking output file: {path}")

            if not os.path.exists(path):
                logging.warning(f"Output file not found: {path}")
                continue

            if calculation_type.lower() == 'goat':
                finalensemble_file = path.replace('.out', '.finalensemble.xyz')
                logging.info(f"Looking for GOAT ensemble file: {finalensemble_file}")
                if os.path.exists(finalensemble_file):
                    coords, ens = self.parse_goat_finalensemble(finalensemble_file)
                    coordinates.extend(coords)
                    energies.extend(ens)
                else:
                    logging.error(f"GOAT ensemble file not found for: {path}")
                continue

            if calculation_type.lower() == 'pes':
                coords, ens = self.parse_pes_output(path)
                coordinates.extend(coords)
                energies.extend(ens)
                continue
            
            if calculation_type.lower() == 'docker':
                docker_xyz_file = path.replace('.out', '.struc1.allopt.xyz')
                logging.info(f"Looking for Docker structure file: {docker_xyz_file}")
                if os.path.exists(docker_xyz_file):
                    coords, ens = self.parse_docker_xyz(docker_xyz_file)
                    coordinates.extend(coords)
                    energies.extend(ens)
                else:
                    logging.error(f"Docker structure file not found for: {path}")
                continue

            # Standard DFT parsing
            coords, ens = self.parse_dft_output(path)
            coordinates.extend(coords)
            energies.extend(ens)

        if not coordinates or not energies:
            logging.error(f"Failed to parse {calculation_type.upper()} outputs in directory: {dir}")

        return coordinates, energies

    def parse_goat_finalensemble(self, file_path):
        """
        Parses a .finalensemble.xyz file from GOAT to extract multiple structures.

        Args:
            file_path (str): Path to the .finalensemble.xyz file.

        Returns:
            tuple: (coordinates, energies)
        """
        coordinates_list = []
        energies_list = []

        with open(file_path, 'r') as file:
            lines = file.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.isdigit():
                atom_count = int(line)
                if i + 1 >= len(lines):
                    break  # Avoid index error
                energy_line = lines[i + 1].strip()
                energy_match = re.match(r"(-?\d+\.\d+)", energy_line)
                energy = float(energy_match.group(1)) if energy_match else None

                current_structure = []
                for j in range(i + 2, i + 2 + atom_count):
                    if j >= len(lines):
                        break
                    tokens = lines[j].strip().split()
                    element, x, y, z = tokens[0], float(tokens[1]), float(tokens[2]), float(tokens[3])
                    current_structure.append((element, x, y, z))
                coordinates_list.append(current_structure)
                energies_list.append(energy)
                i += 2 + atom_count
            else:
                i += 1

        logging.info(f"Parsed {len(coordinates_list)} structures and {len(energies_list)} energies from {file_path}.")
        return coordinates_list, energies_list

    def parse_pes_output(self, file_path):
        """
        Parses a PES scan ORCA .out file, extracting only final optimized geometries and energies.

        Args:
            file_path (str): Path to the ORCA output file.

        Returns:
            tuple: (coordinates_list, energies_list)
        """
        coordinates_list = []
        energies_list = []

        with open(file_path, 'r') as f:
            content = f.read()

        logging.info(f"Parsing PES output for: {file_path}")
        blocks = content.split('*** OPTIMIZATION RUN DONE ***')

        for i, block in enumerate(blocks):
            energy_match = re.findall(
                r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", block
            )
            coord_match = re.findall(
                r"CARTESIAN COORDINATES\s+\(ANGSTROEM\)\s*\n-+\n((?:.*?\n)+?)-+\n",
                block, re.DOTALL
            )

            if energy_match and coord_match:
                energy = float(energy_match[-1])
                coords = [line.split() for line in coord_match[-1].strip().splitlines()]
                energies_list.append(energy)
                coordinates_list.append(coords)
                logging.debug(f"Appended PES geometry #{i+1}: energy={energy}")
            else:
                logging.warning(f"Skipping PES block #{i+1}: missing energy or coordinates.")

        return coordinates_list, energies_list

    def parse_docker_xyz(self, file_path):
        """
        Parses a .docker.struc1.allopt.xyz file to extract coordinates and Eopt energies.

        Args:
            file_path (str): Path to the Docker-style XYZ file.

        Returns:
            tuple: (coordinates_list, energies_list)
        """
        coordinates_list = []
        energies_list = []

        with open(file_path, 'r') as file:
            lines = file.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.isdigit():
                atom_count = int(line)
                if i + 1 >= len(lines):
                    break

                comment_line = lines[i + 1].strip()
                energy_match = re.search(r"Eopt=([-+]?\d*\.\d+|\d+)", comment_line)
                energy = float(energy_match.group(1)) if energy_match else None

                current_structure = []
                for j in range(i + 2, i + 2 + atom_count):
                    if j >= len(lines):
                        break
                    tokens = lines[j].strip().split()
                    if len(tokens) >= 4:
                        element = tokens[0]
                        x, y, z = map(float, tokens[1:4])
                        current_structure.append((element, x, y, z))

                coordinates_list.append(current_structure)
                energies_list.append(energy)
                i += 2 + atom_count
            else:
                i += 1

        logging.info(f"Parsed {len(coordinates_list)} Docker structures from {file_path}.")
        return coordinates_list, energies_list

    def parse_dft_output(self,path):
        coordinates = []
        energies = []
         # Standard DFT parsing
        with open(path) as f:
            content = f.read()

        logging.info(f"Parsing standard DFT output for: {path}")
        coord_block = re.findall(
            r"CARTESIAN COORDINATES\s+\(ANGSTROEM\)\s*\n-+\n((?:.*?\n)+?)-+\n",
            content, re.DOTALL
        )
        if coord_block:
            coords = [line.split() for line in coord_block[-1].strip().splitlines()]
            coordinates.append(coords)
            logging.debug(f"Extracted coordinates block: {coords}")
        else:
            logging.warning(f"No coordinate block found in: {path}")
            coordinates.append([])

        energy_match = re.findall(
            r"FINAL SINGLE POINT ENERGY(?: \(From external program\))?\s+(-?\d+\.\d+)",
            content
        )
        if energy_match:
            energy = float(energy_match[-1])
            energies.append(energy)
        else:
            logging.warning(f"No energy found in: {path}")
            energies.append(None)

        return coordinates, energies
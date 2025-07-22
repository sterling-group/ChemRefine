import os
import re
from .utils import Utility
import subprocess
import logging
from pathlib import Path
import sys
import time
import getpass
import numpy as np 
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

    def submit_files(self, 
                     input_files, 
                     max_cores=32, 
                     template_dir=".", 
                     output_dir=".",
                     device=None,
                     operation='OPT+SP',
                     engine='DFT',
                     model_name=None,
                     task_name=None):
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
                operation=operation,
                engine=engine,
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
    operation: str = "OPT+SP",
    engine: str = "DFT",
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

            if engine.lower() == "mlff":
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
                     operation='OPT+SP',
                     engine='DFT',
                     model_name=None,
                     task_name=None,
                     device='cuda',
                     bind='127.0.0.1:8888'):
        """
        Generate ORCA .inp files from xyz inputs, adding MLFF external method if specified.

        Args:
            xyz_files (list): List of xyz file paths.
            template (str): ORCA input template path.
            charge (int): Molecular charge.
            multiplicity (int): Spin multiplicity.
            output_dir (str): Destination directory.
            operation (str): Operation type (e.g., 'GOAT', 'OPT+SP').
            engine (str): Computational engine ('dft' or 'mlff').
            model_name (str): MLFF model name (if using MLFF).
            task_name (str): MLFF task name (if using MLFF).
            device (str): Device for MLFF ('cuda' or 'cpu').
            bind (str): Server bind address for MLFF.

        Returns:
            tuple: Lists of input and output file paths.
        """
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

            # Remove any old xyzfile lines and clean formatting
            content = re.sub(r'^\s*\*\s+xyzfile.*$', '', content, flags=re.MULTILINE)
            content = content.rstrip() + '\n\n'
            if engine and engine.lower() == 'mlff':
                # Add MLFF method block if specified
                run_mlff_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "uma.sh"))
                ext_params = f"--model_name {model_name} --task_name {task_name} --device {device} --bind {bind}"
                content += '%method\n'
                content += f'  ProgExt "{run_mlff_path}"\n'
                content += f'  Ext_Params "{ext_params}"\n'
                content += 'end\n\n'

            content += f"* xyzfile {charge} {multiplicity} {xyz}\n\n"

            with open(inp, "w") as f:
                f.write(content)

        return input_files, output_files

    def parse_output(self, file_paths, operation, dir='./'):
        coordinates, energies = [], []

        logging.info(f"Parsing calculation type: {operation.upper()}")
        logging.info(f"Looking for output files in directory: {dir}")

        for out_file in file_paths:
            path = os.path.join(dir, os.path.basename(out_file))
            logging.info(f"Checking output file: {path}")

            if not os.path.exists(path):
                logging.warning(f"Output file not found: {path}")
                continue

            if operation.lower() == 'goat':
                finalensemble_file = path.replace('.out', '.finalensemble.xyz')
                logging.info(f"Looking for GOAT ensemble file: {finalensemble_file}")
                if os.path.exists(finalensemble_file):
                    coords, ens = self.parse_goat_finalensemble(finalensemble_file)
                    coordinates.extend(coords)
                    energies.extend(ens)
                else:
                    logging.error(f"GOAT ensemble file not found for: {path}")
                continue

            if operation.lower() == 'pes':
                coords, ens = self.parse_pes_output(path)
                coordinates.extend(coords)
                energies.extend(ens)
                continue
            
            if operation.lower() == 'docker':
                docker_xyz_file = path.replace('.out', '.struc1.allopt.xyz')
                logging.info(f"Looking for Docker structure file: {docker_xyz_file}")
                if os.path.exists(docker_xyz_file):
                    coords, ens = self.parse_docker_xyz(docker_xyz_file)
                    coordinates.extend(coords)
                    energies.extend(ens)
                else:
                    logging.error(f"Docker structure file not found for: {path}")
                continue

            if operation.lower() == 'solvator':
                solvator_xyz_file = path.replace('.out', '.solvator.xyz')
                logging.info(f"Looking for Solvator structure file: {solvator_xyz_file}")
                if os.path.exists(solvator_xyz_file):
                    coords, ens = self.parse_solvator_xyz(solvator_xyz_file)
                    coordinates.extend(coords)
                    energies.extend(ens)
                else:
                    logging.error(f"Solvator structure file not found for: {path}")
                continue
            # Standard DFT parsing
            coords, ens = self.parse_dft_output(path)
            coordinates.extend(coords)
            energies.extend(ens)

        if not coordinates or not energies:
            logging.error(f"Failed to parse {operation.upper()} outputs in directory: {dir}")

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
    
    def parse_solvator(self, file_path):
        """
        Parses a solvator.xyz file containing a single XYZ structure without energy.
        Assigns a placeholder energy value of 0.0.

        Args:
            file_path (str): Path to the .solvator.xyz file.

        Returns:
            tuple: (coordinates_list, energies_list) — energy is set to 0.0.
        """
        coordinates_list = []
        energies_list = []

        with open(file_path, 'r') as file:
            lines = file.readlines()

        if len(lines) < 3:
            raise ValueError(f"File {file_path} does not contain a valid XYZ structure.")

        atom_count = int(lines[0].strip())
        current_structure = []

        for line in lines[2:2 + atom_count]:
            tokens = line.strip().split()
            if len(tokens) >= 4:
                element = tokens[0]
                x, y, z = map(float, tokens[1:4])
                current_structure.append((element, x, y, z))

        coordinates_list.append(current_structure)
        energies_list.append(0.0)

        logging.info(f"Parsed 1 solvator structure from {file_path} with placeholder energy 0.0.")
        return coordinates_list, energies_list

    def normal_mode_sampling(self,
                         file_paths,
                         calc_type,
                         template, 
                         charge, 
                         multiplicity, 
                         output_dir, 
                         operation,
                         engine,
                         model_name,
                         step_number,
                         structure_ids,
                         max_cores=32,
                         task_name=None,
                         mlff_model=None,
                         displacement_value=1.0,
                         device='cuda',
                         bind='127.0.0.1:8888'):
        """
        Samples normal modes and optionally removes imaginary frequencies for one or more ORCA output files.

        Parameters
        ----------
        file_paths : str or list of str
            Path(s) to ORCA output file(s).
        calc_type : str
            Type of operation:
            - 'rm_imag': displace along least imaginary frequency.
            - 'normal_modes': displace along a random mode.
        template : str
            ORCA input template path.
        charge : int
            Molecular charge.
        multiplicity : int
            Spin multiplicity.
        output_dir : str
            Base output directory.
        operation : str
            ORCA operation type (e.g., OPT, SP).
        engine : str
            ORCA engine (e.g., DFT or MLFF).
        model_name : str
            Model name for ORCA MLFF input.
        step_number : int
            Step identifier for directory naming.
        structure_ids : list
            List of structure identifiers.
        max_cores : int, optional
            Number of cores per job.
        task_name : str, optional
            ML task name.
        mlff_model : str, optional
            MLFF model name.
        displacement_value : float, optional
            Displacement factor for sampling.
        device : str, optional
            Compute device.
        bind : str, optional
            MLFF server bind address.
        """
        logging.info("Starting normal mode sampling.")
        logging.info(f"Sampling type: {'remove imaginary modes' if calc_type == 'rm_imag' else 'displace random mode'}")

        if isinstance(file_paths, str):
            file_paths = [file_paths]

        imag = calc_type == "rm_imag"
        random_mode = calc_type == "normal_modes"
        normal_output_dir = os.path.join(output_dir, f"{step_number}/normal_modes")

        for file_path in file_paths:
            imag_freq_dict = self.parse_imaginary_frequency(file_path, imag=imag)
            logging.info(f"{len(imag_freq_dict)} imaginary frequencies detected in {file_path}")

            num_atoms = self.get_num_atoms_from_input(file_path)
            normal_mode_tensor = self.parse_normal_modes_tensor_final(file_path, num_atoms)
            coordinates, _ = self.parse_dft_output(file_path)

            pos_coords, neg_coords = self.displace_least_imaginary_mode(
                filepath=file_path,
                imag_freq_dict=imag_freq_dict,
                normal_mode_tensor=normal_mode_tensor,
                coordinates=coordinates,
                displacement_value=displacement_value,
                random_mode=random_mode
            )
            logging.info(f"Successfully displaced coordinates for {file_path}")

            xyz_files = [pos_coords, neg_coords]

            xyz_filenames = self.utility.write_xyz(
                xyz_files,
                step_number=step_number,
                structure_ids=structure_ids,
                output_dir=normal_output_dir
            )

            input_files, output_files = self.orca.create_input(
                xyz_filenames,
                template,
                charge,
                multiplicity,
                output_dir=normal_output_dir,
                operation=operation,
                engine=engine,
                model_name=model_name,
                task_name=task_name,
                device=device,
                bind=bind
            )

            self.submit_orca_jobs(
                input_files,
                max_cores,
                step_dir=normal_output_dir,
                operation=operation,
                engine=engine,
                model_name=mlff_model,
                task_name=task_name,
                device=device
            )

        logging.info("Successfully finished normal mode sampling.")

    def parse_imaginary_frequency(self,file_paths, imag=True):
        import numpy as np
        """
        Parses vibrational frequencies from an ORCA output file.

        Parameters
        ----------
        orca_output_path : str or Path
            Path to the ORCA output file.
        imag : bool, optional
            If True, return only imaginary frequencies (default). If False, return all frequencies.

        Returns
        -------
        dict
            Dictionary mapping frequency index (int) to frequency value (float).
        """
        freqs = {}
        in_freq_block = False
        scaling_found = False

        with open(file_paths, 'r') as f:
            for line in f:
                if 'VIBRATIONAL FREQUENCIES' in line:
                    in_freq_block = True
                    continue
                if in_freq_block and 'Scaling factor for frequencies' in line:
                    scaling_found = True
                    continue
                if in_freq_block and scaling_found and ':' in line and 'cm**-1' in line:
                    try:
                        index_part, value_part = line.strip().split(':', 1)
                        index = int(index_part.strip())
                        value = float(value_part.strip().split()[0])
                        if imag:
                            if '***imaginary mode***' in line:
                                freqs[index] = value
                        else:
                            freqs[index] = value
                    except Exception:
                        continue

        return freqs
    
    def parse_normal_modes_tensor(filepath, num_atoms):
        """
        Parses all normal mode displacement vectors from an ORCA output file into a full tensor.

        Parameters
        ----------
        filepath : str
            Path to the ORCA output file.
        num_atoms : int
            Number of atoms in the system.

        Returns
        -------
        np.ndarray
            Array of shape (num_atoms, 3, n_modes) with displacements.
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()

        collecting = False
        block_rows = []
        all_blocks = []
        current_mode_width = None

        for line in lines:
            if re.match(r'^\s+(\d+\s+)+\d+\s*$', line):  # Header like "0 1 2 3 4 5"
                collecting = True
                if block_rows:
                    all_blocks.append(np.array(block_rows, dtype=float))
                    block_rows = []
                continue
            if collecting:
                if re.match(r'^\s*\d+\s+[-\d.Ee\s]+$', line):
                    parts = line.strip().split()
                    floats = list(map(float, parts[1:]))
                    block_rows.append(floats)
                elif 'IR SPECTRUM' in line or '--------' in line:
                    if block_rows:
                        all_blocks.append(np.array(block_rows, dtype=float))
                    break

        # Check number of rows per block and concatenate horizontally
        for block in all_blocks:
            if block.shape[0] != 3 * num_atoms:
                raise ValueError(f"A block has incorrect number of rows: {block.shape[0]}")
        full_matrix = np.hstack(all_blocks)
        return full_matrix.reshape(num_atoms, 3, -1)
    
    def displace_normal_modes(filepath: str,
                                   imag_freq_dict: dict,
                                   normal_mode_tensor: np.ndarray,
                                   coordinates,
                                   displacement_value: float = 1.0,
                                   random_mode: bool = False):
        """
        Parses coordinates from ORCA output and displaces them along a selected imaginary mode.

        Parameters
        ----------
        filepath : str
            ORCA output file path.
        imag_freq_dict : dict
            Dictionary of imaginary frequencies {mode_index: frequency}.
        normal_mode_tensor : np.ndarray
            Normal mode displacement tensor of shape (n_atoms, 3, n_modes).
        coordinates : list
            List of atomic coordinates (usually parsed from ORCA).
        displacement_value : float
            Scale factor for displacement.
        random_mode : bool
            If True, randomly selects one imaginary mode. Otherwise, selects the one with smallest magnitude.

        Returns
        -------
        tuple
            Positive and negative displaced coordinates (each as np.ndarray of shape (n_atoms, 3)).
        """
        if not imag_freq_dict:
            raise ValueError("No imaginary frequencies found.")

        coords = np.array(coordinates[0], dtype=float)

        if random_mode:
            min_mode_idx = random.choice(list(imag_freq_dict.keys()))
        else:
            min_mode_idx = min(imag_freq_dict.items(), key=lambda kv: abs(kv[1]))[0]

        disp_vector = normal_mode_tensor[:, :, min_mode_idx]

        pos_coords = coords + displacement_value * disp_vector
        neg_coords = coords - displacement_value * disp_vector

        return pos_coords, neg_coords
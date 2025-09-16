import os
import re
from .utils import Utility
import logging
from pathlib import Path
import time
import numpy as np

# chemrefine/orca_interface.py

# -----------GLOBALS-AND-REGEX----------------
EV_PER_HARTREE = 27.211386245988
HARTREE_TO_EV = 27.211386245988  # CODATA 2018
BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_PER_BOHR_TO_EV_PER_A = HARTREE_TO_EV / BOHR_TO_ANGSTROM
ANG_PER_BOHR = 0.529177210903
_ORCA_COORD_BLOCK_RE = re.compile(
    r"CARTESIAN COORDINATES\s+\(ANGSTROEM\)\s*\n-+\n((?:.*?\n)+?)-+\n", re.DOTALL
)
_ORCA_GRAD_BLOCK_RE = re.compile(
    r"CARTESIAN GRADIENT\s*\n-+\n((?:.*?\n)+?)-+\n", re.DOTALL
)
_ORCA_GRAD_LINE_RE = re.compile(
    r"^\s*(\d+)\s+[A-Za-z]{1,3}\s*:\s*"
    r"([+-]?\d*\.?\d+(?:[EeDd][+-]?\d+)?)\s+"
    r"([+-]?\d*\.?\d+(?:[EeDd][+-]?\d+)?)\s+"
    r"([+-]?\d*\.?\d+(?:[EeDd][+-]?\d+)?)\s*$"
)


class OrcaJobSubmitter:
    """
    A lightweight ORCA job submission class for ChemRefine.
    Handles job submission, PAL adjustment, and job monitoring.
    """

    def __init__(
        self,
        device="cpu",
        orca_executable: str = "orca",
        bind: str = "127.0.0.1:8888",
        scratch_dir: str = None,
        save_scratch: bool = False,
    ):
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

    def submit_files(
        self,
        input_files,
        max_cores=32,
        template_dir=".",
        output_dir=".",
        device=None,
        operation="OPT+SP",
        engine="DFT",
        model_name=None,
        task_name=None,
        model_path=None,
    ):
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
                bind=self.bind,
                model_path=model_path,
            )

            job_id = self.utility.submit_job(slurm_script)
            logging.info(
                f"Submitted ORCA job with ID: {job_id} for input: {input_path.name}"
            )

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
        bind: str = "127.0.0.1:8888",
        model_path: str = None,
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
            raise FileNotFoundError(
                f"SLURM header template {header_template_path} not found."
            )

        with open(header_template_path, "r") as f:
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
        with open(slurm_file, "w") as f:
            f.write("#!/bin/bash\n\n")
            f.write("\n".join(sbatch_lines))
            f.write("\n\n")
            # Write the rest of the header
            f.write("\n".join(non_sbatch_lines))
            f.write("\n\n")
            # Write scratch management and ORCA execution block
            f.write(f"ORCA_EXEC={self.orca_executable}\n")
            f.write(
                "# Scratch directory management and ORCA execution (generated by ChemRefine)\n"
            )
            f.write('if [ -z "$ORCA_EXEC" ]; then\n')
            f.write(f"    ORCA_EXEC={self.orca_executable}\n")
            f.write("fi\n\n")

            f.write("timestamp=$(date +%Y%m%d%H%M%S)\n")
            f.write("random_str=$(tr -dc a-z0-9 </dev/urandom | head -c 8)\n")
            f.write(f"export BASE_SCRATCH_DIR={self.scratch_dir}\n")
            f.write(
                "export SCRATCH_DIR=${BASE_SCRATCH_DIR}/ChemRefine_scratch_${SLURM_JOB_ID}_${timestamp}_${random_str}\n"
            )
            f.write(f"export OUTPUT_DIR={os.path.abspath(output_dir)}\n")

            f.write(
                "mkdir -p $SCRATCH_DIR || { echo 'Error: Failed to create scratch directory'; exit 1; }\n"
            )
            f.write("echo 'SCRATCH_DIR is set to $SCRATCH_DIR'\n\n")

            f.write(
                f"cp {input_file} $SCRATCH_DIR/ || {{ echo 'Error: Failed to copy input file'; exit 1; }}\n"
            )
            f.write(
                "cd $SCRATCH_DIR || { echo 'Error: Failed to change directory'; exit 1; }\n\n"
            )

            if engine.lower() == "mlff":
                f.write("# Start MLFF socket server before ORCA\n")
                if model_path:
                    f.write(
                        f"python -m chemrefine.server --model-path {model_path} --device {device} --bind {bind} & > $OUTPUT_DIR/server.log 2>&1 & \n"
                    )
                else:
                    f.write(
                        f"python -m chemrefine.server --model {model_name} --task-name {task_name} --device {device} --bind {bind} & > $OUTPUT_DIR/server.log 2>&1 & \n"
                    )

                f.write("SERVER_PID=$!\n")
                f.write("sleep 10\n")
                f.write(
                    f"$ORCA_EXEC {input_file.name} > $OUTPUT_DIR/{job_name}.out || {{ echo 'Error: ORCA execution failed.'; kill $SERVER_PID; exit 1; }}\n"
                )
                f.write("kill $SERVER_PID\n\n")
            else:
                f.write("export OMP_NUM_THREADS=1\n")
                f.write(
                    f"$ORCA_EXEC {input_file.name} > $OUTPUT_DIR/{job_name}.out || {{ echo 'Error: ORCA execution failed.'; exit 1; }}\n\n"
                )

            # File copy commands
            f.write(
                "cp *.out *.xyz *.finalensemble.*.xyz *.finalensemble.xyz $OUTPUT_DIR \n"
            )

            if not self.save_scratch:
                f.write(
                    "rm -rf $SCRATCH_DIR || { echo 'Warning: Failed to remove scratch directory'; }\n"
                )
            else:
                f.write("echo 'Scratch directory not deleted (save_scratch is True)'\n")

        logging.info(f"Generated SLURM script: {slurm_file}")
        return slurm_file


class OrcaInterface:
    def __init__(self):
        self.utility = Utility()
        self.job_submitter = OrcaJobSubmitter()

    def create_input(
        self,
        xyz_files,
        template,
        charge,
        multiplicity,
        output_dir="./",
        operation="OPT+SP",
        engine="DFT",
        model_name=None,
        task_name=None,
        device="cuda",
        bind="127.0.0.1:8888",
    ):
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
            content = re.sub(r"^\s*\*\s+xyzfile.*$", "", content, flags=re.MULTILINE)
            content = content.rstrip() + "\n\n"
            if engine and engine.lower() == "mlff":
                # Add MLFF method block if specified
                run_mlff_path = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "uma.sh")
                )
                ext_params = f"--bind {bind}"
                content += "%method\n"
                content += f'  ProgExt "{run_mlff_path}"\n'
                content += f'  Ext_Params "{ext_params}"\n'
                content += "end\n\n"

            content += f'%base "{base}_opt" \n'
            content += f"* xyzfile {charge} {multiplicity} {xyz}\n\n"

            with open(inp, "w") as f:
                f.write(content)

        return input_files, output_files

    def parse_output(self, file_paths, operation, dir: str = "./"):
        """
        Dispatch parser depending on the operation type.

        Parameters
        ----------
        file_paths : list
            List of ORCA output files to parse.
        operation : str
            Operation type (goat, pes, docker, solvator, dft).
        dir : str, optional
            Directory containing the output files. Default is current directory.

        Returns
        -------
        tuple
            (coordinates, energies, forces), each a list
        """
        coordinates, energies, forces = [], [], []

        logging.info(f"Parsing calculation type: {operation.upper()}")
        logging.info(f"Looking for output files in directory: {dir}")

        for out_file in file_paths:
            path = os.path.join(dir, os.path.basename(out_file))
            logging.info(f"Checking output file: {path}")

            if not os.path.exists(path):
                logging.warning(f"Output file not found: {path}")
                continue

            op = operation.lower()

            if op == "goat":
                if path.endswith(".finalensemble.xyz"):
                    # Already the ensemble file, no transformation needed
                    finalensemble_file = path
                else:
                    # Turn step1.out -> step1_opt.finalensemble.xyz
                    finalensemble_file = path.replace(".out", "_opt.finalensemble.xyz")

                logging.info(f"Looking for GOAT ensemble file: {finalensemble_file}")
                if os.path.exists(finalensemble_file):
                    coords, ens = self.parse_goat_finalensemble(finalensemble_file)
                    coordinates.extend(coords)
                    energies.extend(ens)
                    forces.extend([None] * len(coords))
                else:
                    logging.error(
                        f"GOAT ensemble file not found for: {finalensemble_file}"
                    )
                continue

            if op == "pes":
                coords, ens = self.parse_pes_output(path)
                coordinates.extend(coords)
                energies.extend(ens)
                forces.extend([None] * len(coords))  # dummy
                continue

            if op == "docker":
                docker_xyz_file = path.replace(".out", ".struc1.allopt.xyz")
                logging.info(f"Looking for Docker structure file: {docker_xyz_file}")
                if os.path.exists(docker_xyz_file):
                    coords, ens = self.parse_docker_xyz(docker_xyz_file)
                    coordinates.extend(coords)
                    energies.extend(ens)
                    forces.extend([None] * len(coords))  # dummy
                else:
                    logging.error(f"Docker structure file not found for: {path}")
                continue

            if op == "solvator":
                solvator_xyz_file = path.replace(".out", ".solventbuild.xyz")
                logging.info(
                    f"Looking for Solvator structure file: {solvator_xyz_file}"
                )
                if os.path.exists(solvator_xyz_file):
                    coords, ens = self.parse_solvator_ensemble(solvator_xyz_file)
                    coordinates.extend(coords)
                    energies.extend(ens)
                    forces.extend([None] * len(coords))  # dummy
                else:
                    logging.error(f"Solvator structure file not found for: {path}")
                continue

            # Default: standard DFT parsing (energies + forces)
            coords, ens, frc = self.parse_dft_output(path)
            coordinates.extend(coords)
            energies.extend(ens)
            forces.extend(frc)

        if not coordinates or not energies:
            logging.error(
                f"Failed to parse {operation.upper()} outputs in directory: {dir}"
            )

        return coordinates, energies, forces

    def parse_dft_output(self, path):
        """
        Parse ORCA DFT output file to extract coordinates, energies, and forces.

        Parameters
        ----------
        path : str
            Path to the ORCA output file.

        Returns
        -------
        tuple[list, list, list]
            coordinates : list
                Each element is a list of atomic coordinates (Å).
            energies : list
                Total energies (Ha) parsed from the file.
            forces : list
                Per-atom forces arrays (N_atoms, 3) in eV/Å.
                Empty list if no gradient block found.
        """
        coordinates, energies, forces = [], [], []

        with open(path, "r") as f:
            content = f.read()

        logging.info(f"Parsing standard DFT output for: {path}")

        # --- Coordinates ---
        coord_block = _ORCA_COORD_BLOCK_RE.findall(content)
        if coord_block:
            coords = [line.split() for line in coord_block[-1].strip().splitlines()]
            coordinates.append(coords)
            logging.debug(f"Extracted coordinates block with {len(coords)} atoms.")
        else:
            logging.warning(f"No coordinate block found in: {path}")
            coordinates.append([])

        # --- Energies ---
        energy_match = re.findall(
            r"FINAL SINGLE POINT ENERGY(?: \(From external program\))?\s+(-?\d+\.\d+)",
            content,
        )
        if energy_match:
            energies.append(float(energy_match[-1]))
        else:
            logging.warning(f"No energy found in: {path}")
            energies.append(None)

        # --- Forces ---
        grad_forces = _orca_parse_all_gradients(content, to_ev_per_A=True)
        if grad_forces:
            # take the last gradient block (final forces after SCF/optimization)
            forces.append(grad_forces[-1])
            logging.debug(
                f"Extracted forces for {len(grad_forces[-1])} atoms " f"(units: eV/Å)."
            )
        else:
            logging.info(f"No gradient block found in: {path}")
            forces.append([])

        return coordinates, energies, forces

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

        with open(file_path, "r") as file:
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
                    element, x, y, z = (
                        tokens[0],
                        float(tokens[1]),
                        float(tokens[2]),
                        float(tokens[3]),
                    )
                    current_structure.append((element, x, y, z))
                coordinates_list.append(current_structure)
                energies_list.append(energy)
                i += 2 + atom_count
            else:
                i += 1

        logging.info(
            f"Parsed {len(coordinates_list)} structures and {len(energies_list)} energies from {file_path}."
        )
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

        with open(file_path, "r") as f:
            content = f.read()

        logging.info(f"Parsing PES output for: {file_path}")
        blocks = content.split("*** OPTIMIZATION RUN DONE ***")

        for i, block in enumerate(blocks):
            energy_match = re.findall(
                r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", block
            )
            coord_match = re.findall(
                r"CARTESIAN COORDINATES\s+\(ANGSTROEM\)\s*\n-+\n((?:.*?\n)+?)-+\n",
                block,
                re.DOTALL,
            )

            if energy_match and coord_match:
                energy = float(energy_match[-1])
                coords = [line.split() for line in coord_match[-1].strip().splitlines()]
                energies_list.append(energy)
                coordinates_list.append(coords)
                logging.debug(f"Appended PES geometry #{i+1}: energy={energy}")
            else:
                logging.warning(
                    f"Skipping PES block #{i+1}: missing energy or coordinates."
                )

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

        with open(file_path, "r") as file:
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

        logging.info(
            f"Parsed {len(coordinates_list)} Docker structures from {file_path}."
        )
        return coordinates_list, energies_list

    def parse_solvator_ensemble(self, file_path):
        """
        Parse SOLVATOR multi-structure XYZ written by solventbuild, with repeating blocks:

            <natoms>\n
            Energy -123.456789   (case-insensitive; variations allowed)\n
            <symbol x y z> * natoms

        Accepts scientific notation and minor variations like "total energy: -123.45".
        Returns
        -------
        (coordinates_list, energies_list)
            coordinates_list: list[list[tuple[str, float, float, float]]]
            energies_list:    list[float | None]
        """
        import re
        import logging

        coords, energies = [], []

        # Match the first float on the line, allowing "Energy", "Total Energy", optional ":" / "=" and sci notation
        energy_re = re.compile(
            r"(?:^\s*(?:energy|total\s+energy)\s*[:=]?\s*|\b)"
            r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
            re.IGNORECASE,
        )

        with open(file_path, "r") as fh:
            lines = fh.readlines()

        i, n = 0, len(lines)
        while i < n:
            line = lines[i].strip()
            # find a block header that is a pure integer
            if not line or not line.isdigit():
                i += 1
                continue

            natoms = int(line)
            if i + 1 >= n:
                break

            # energy line (immediately after natoms)
            m = energy_re.search(lines[i + 1])
            energy = float(m.group(1)) if m else None

            # atom block
            block = []
            start = i + 2
            end = start + natoms
            if end > n:
                logging.warning(
                    f"Unexpected EOF while reading atom block starting at line {i+1} in {file_path}."
                )
                break

            for j in range(start, end):
                parts = lines[j].split()
                if len(parts) < 4:
                    logging.warning(
                        f"Malformed coordinate line at {j+1} in {file_path}: {lines[j].rstrip()}"
                    )
                    block = []
                    break
                sym = parts[0]
                try:
                    x, y, z = map(float, parts[1:4])
                except ValueError:
                    logging.warning(
                        f"Non-numeric coordinates at {j+1} in {file_path}: {lines[j].rstrip()}"
                    )
                    block = []
                    break
                block.append((sym, x, y, z))

            if block and len(block) == natoms:
                coords.append(block)
                energies.append(energy)
            else:
                logging.warning(
                    f"Skipping one SOLVATOR structure (bad/missing atom block near lines {start}-{end})."
                )

            # jump to the next block
            i = end

        missing = sum(e is None for e in energies)
        if missing:
            logging.warning(
                f"{missing} SOLVATOR energy value(s) were missing in {file_path}."
            )
        logging.info(
            f"Parsed {len(coords)} structures and {len(energies)} energies from {file_path}."
        )
        return coords, energies

    def normal_mode_sampling(
        self,
        file_paths,
        calc_type,
        input_template,
        slurm_template,
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
        num_random_modes=1,
        device="cuda",
        bind="127.0.0.1:8888",
        orca_executable="orca",
        scratch_dir=None,
    ):
        """
        Perform normal-mode sampling. For 'random', only generate coords/IDs and defer
        all calculations to the next step. For 'rm_imag', prepare inputs and submit.

        Parameters
        ----------
        file_paths : list[str]
            ORCA output files (one per structure) to read modes/coordinates from.
        calc_type : str
            'rm_imag' for least-imaginary-mode displacement; 'random' for random mode sampling.
        input_template : str
            ORCA input template path.
        slurm_template : str
            SLURM header/template directory used by OrcaJobSubmitter.submit_files.
        charge : int
            Molecular charge.
        multiplicity : int
            Spin multiplicity.
        output_dir : str
            BASE outputs directory; XYZs go under step{n}/normal_mode_sampling.
        operation : str
            Operation label ('OPT', 'SP', 'OPT+SP', ...).
        engine : str
            'dft' or 'mlff'.
        model_name : str
            MLFF model name when `engine='mlff'`.
        step_number : int
            Current step index (1-based).
        structure_ids : list[str]
            IDs aligned with `file_paths`.
        max_cores : int, optional
            Max total cores across submissions (rm_imag only).
        task_name : str, optional
            MLFF task variant (if applicable).
        mlff_model : str, optional
            Deprecated alias for `model_name`; if provided, overrides at submission.
        displacement_value : float, optional
            Displacement magnitude (Å).
        num_random_modes : int, optional
            Random modes to sample per structure when `calc_type='random'`.
        device : str, optional
            'cpu' or 'cuda' (for MLFF engine).
        bind : str, optional
            MLFF server bind address (passed through to input generation).
        orca_executable : str, optional
            ORCA executable path/name used by OrcaJobSubmitter.
        scratch_dir : str, optional
            Scratch directory used by OrcaJobSubmitter.

        Returns
        -------
        tuple[list, list]
            For 'rm_imag': (filtered_coords, original_ids).
            For 'random' : (all_displaced_coords, displaced_ids), with no submissions.
        """
        import os
        import re
        import logging
        import contextlib

        def _has_normal_modes(path: str) -> bool:
            """Quick check that the ORCA output contains a normal-mode table."""
            hdr = False
            try:
                with open(path, "r") as fh:
                    for line in fh:
                        if "VIBRATIONAL FREQUENCIES" in line:
                            hdr = True
                        if hdr and re.match(r"^\s*(\d+\s+)+\d+\s*$", line):
                            return True
            except Exception:
                return False
            return False

        logging.info("***Starting normal mode sampling.***")
        logging.info(
            f"Sampling type: {calc_type} x{num_random_modes if calc_type == 'random' else 1}"
        )

        # Accumulators
        all_pos_coords, all_neg_coords = [], []
        all_pos_ids, all_neg_ids = [], []
        rm_imag_inp_files = []

        # Directories
        per_step_dir = os.path.join(output_dir, f"step{step_number}")
        normal_output_dir = os.path.join(per_step_dir, "normal_mode_sampling")
        os.makedirs(normal_output_dir, exist_ok=True)

        skipped = 0
        for file_path, sid in zip(file_paths, structure_ids):
            if not _has_normal_modes(file_path):
                logging.warning(
                    f"Skipping ID {sid}: no vibrational frequencies in {file_path}."
                )
                skipped += 1
                continue

            try:
                imag_flag = calc_type != "random"
                imag_freq_dict = self.parse_imaginary_frequency(
                    file_path, imag=imag_flag
                )

                coordinates, energies, forces = self.parse_dft_output(file_path)
                if not coordinates:
                    logging.warning(
                        f"Skipping ID {sid}: no coordinates parsed from {file_path}."
                    )
                    skipped += 1
                    continue

                num_atoms = len(coordinates[0])
                normal_mode_tensor = self.parse_normal_modes_tensor(
                    file_path, num_atoms=num_atoms
                )
            except Exception as e:
                logging.warning(
                    f"Skipping ID {sid}: failed to parse modes from {file_path} ({e})."
                )
                skipped += 1
                continue

            if calc_type == "rm_imag":
                pos_coords, neg_coords = self.displace_normal_modes(
                    filepath=file_path,
                    imag_freq_dict=imag_freq_dict,
                    normal_mode_tensor=normal_mode_tensor,
                    coordinates=coordinates,
                    displacement_value=displacement_value,
                    random_mode=False,
                )

                pos_xyz = self.write_displaced_xyz(
                    [pos_coords[0]], step_number, [f"{sid}_pos"], output_dir
                )
                neg_xyz = self.write_displaced_xyz(
                    [neg_coords[0]], step_number, [f"{sid}_neg"], output_dir
                )

                inp_files, _ = self.create_input(
                    xyz_files=pos_xyz + neg_xyz,
                    input_template=input_template,
                    charge=charge,
                    multiplicity=multiplicity,
                    output_dir=normal_output_dir,
                    operation=operation,
                    engine=engine,
                    model_name=model_name,
                    task_name=task_name,
                    device=device,
                    bind=bind,
                )

                all_pos_coords.append(pos_coords[0])
                all_neg_coords.append(neg_coords[0])
                all_pos_ids.append(f"{sid}_pos")
                all_neg_ids.append(f"{sid}_neg")
                rm_imag_inp_files.extend(inp_files)

            elif calc_type == "random":
                # Pass-through behavior: only generate coords/IDs; do NOT create inputs or submit.
                pos_coords, neg_coords, pos_ids, neg_ids, _ = (
                    self.generate_random_displacements(
                        sid=sid,
                        file_path=file_path,
                        normal_mode_tensor=normal_mode_tensor,
                        coordinates=coordinates,
                        num_random_modes=num_random_modes,
                        displacement_value=displacement_value,
                        step_number=step_number,
                        input_template=input_template,
                        charge=charge,
                        multiplicity=multiplicity,
                        output_dir=output_dir,  # BASE for XYZ writer
                        engine=engine,
                        model_name=model_name,
                        task_name=task_name,
                        device=device,
                        bind=bind,
                        normal_output_dir=normal_output_dir,  # ignored when create_inp=False
                        operation=operation,
                        create_inp=False,
                    )
                )

                all_pos_coords.extend(pos_coords)
                all_neg_coords.extend(neg_coords)
                all_pos_ids.extend(pos_ids)
                all_neg_ids.extend(neg_ids)

            else:
                raise ValueError("calc_type must be 'rm_imag' or 'random'.")

        if skipped:
            logging.info(
                f"Normal-mode sampling: skipped {skipped} file(s) without usable frequency data."
            )

        # ---- SUBMISSION: only for rm_imag branch ----
        if calc_type == "rm_imag":
            logging.info(
                f"Total rm_imag input files prepared: {len(rm_imag_inp_files)}"
            )
            if rm_imag_inp_files:
                abs_inp_files = [
                    (
                        p
                        if os.path.isabs(p)
                        else os.path.abspath(os.path.join(normal_output_dir, p))
                    )
                    for p in rm_imag_inp_files
                ]
                abs_inp_files = [p for p in abs_inp_files if os.path.isfile(p)]
                if not abs_inp_files:
                    logging.error(
                        "Prepared 0 valid INP files after existence check; submission skipped."
                    )
                else:
                    orig = os.getcwd()
                    try:
                        logging.info(
                            f"Switching to working directory for submission: {normal_output_dir}"
                        )
                        os.chdir(normal_output_dir)

                        submitter = OrcaJobSubmitter(
                            scratch_dir=scratch_dir,
                            orca_executable=orca_executable,
                            device=device,
                        )
                        submitter.submit_files(
                            input_files=abs_inp_files,
                            max_cores=max_cores,
                            template_dir=slurm_template,
                            output_dir=normal_output_dir,
                            engine=engine,
                            operation=operation,
                            model_name=(
                                mlff_model if mlff_model is not None else model_name
                            ),
                            task_name=task_name,
                        )
                        logging.info(
                            "ORCA submissions dispatched for rm_imag displacements."
                        )
                    except Exception as e:
                        logging.error(
                            f"Error while submitting ORCA jobs in {normal_output_dir}: {e}"
                        )
                        raise
                    finally:
                        with contextlib.suppress(Exception):
                            os.chdir(orig)
                        logging.info(f"Returned to original directory: {orig}")
            else:
                logging.warning("No rm_imag input files generated; submission skipped.")

        # ---- RETURNS ----
        if calc_type == "rm_imag":
            filtered_coords, filtered_tmp_ids = self.select_lowest_imaginary_structures(
                step_number=step_number,
                pos_ids=all_pos_ids,
                neg_ids=all_neg_ids,
                directory=output_dir,
            )
            original_ids = [tid.rsplit("_", 1)[0] for tid in filtered_tmp_ids]
            return filtered_coords, original_ids
        else:
            return all_pos_coords + all_neg_coords, all_pos_ids + all_neg_ids

    def parse_imaginary_frequency(self, file_paths, imag=True):
        """
        Parses vibrational frequencies from an ORCA output file.

        Parameters
        ----------
        file_paths : str
            Path to the ORCA output file.
        imag : bool, optional
            If True, return only imaginary frequencies. If False, return all frequencies except first 5.

        Returns
        -------
        dict
            Dictionary mapping frequency index (int) to frequency value (float).
        """
        freqs = {}
        in_freq_block = False
        scaling_found = False

        with open(file_paths, "r") as f:
            for line in f:
                if "VIBRATIONAL FREQUENCIES" in line:
                    in_freq_block = True
                    continue
                if in_freq_block and "Scaling factor for frequencies" in line:
                    scaling_found = True
                    continue
                if in_freq_block and scaling_found and ":" in line and "cm**-1" in line:
                    try:
                        index_part, value_part = line.strip().split(":", 1)
                        index = int(index_part.strip())
                        value = float(value_part.strip().split()[0])

                        if imag:
                            if "***imaginary mode***" in line:
                                freqs[index] = value
                        else:
                            if index <= 5:
                                continue  # Skip the first 5 real modes
                            freqs[index] = value
                    except Exception:
                        continue

        return freqs

    def parse_normal_modes_tensor(self, filepath, num_atoms):
        """
        Parse normal-mode displacement vectors from an ORCA output into
        an array of shape (num_atoms, 3, n_modes).

        Raises
        ------
        ValueError
            If no normal-mode blocks are found in the file.
        """
        with open(filepath, "r") as f:
            lines = f.readlines()

        collecting = False
        block_rows, all_blocks = [], []

        for line in lines:
            # Mode-column header: "   1   2   3 ..."
            if re.match(r"^\s*(\d+\s+)+\d+\s*$", line):
                collecting = True
                if block_rows:
                    all_blocks.append(np.array(block_rows, dtype=float))
                    block_rows = []
                continue

            if collecting:
                # Displacement rows: "<atom_index>  dX dY dZ [ per mode columns...]"
                if re.match(r"^\s*\d+\s+[-\d.Ee\s]+$", line):
                    parts = line.strip().split()
                    floats = list(map(float, parts[1:]))
                    block_rows.append(floats)
                    continue

                # End of a block/section
                if "IR SPECTRUM" in line or "------------" in line:
                    if block_rows:
                        all_blocks.append(np.array(block_rows, dtype=float))
                    break

        if not all_blocks:
            raise ValueError(
                f"No normal-mode blocks found in '{filepath}'. "
                "Ensure this is a frequency job output (FREQ) with printed normal modes."
            )

        for b in all_blocks:
            if b.shape[0] != 3 * num_atoms:
                raise ValueError(
                    f"Malformed normal-mode block rows: got {b.shape[0]}, expected {3*num_atoms}."
                )

        full_matrix = np.hstack(all_blocks)
        return full_matrix.reshape(num_atoms, 3, -1)

    def displace_normal_modes(
        self,
        filepath: str,
        imag_freq_dict: dict,
        normal_mode_tensor: np.ndarray,
        coordinates,
        displacement_value: float = 1.0,
        random_mode: bool = False,
    ):
        """
        Displaces atomic coordinates along a selected imaginary mode for each molecule.

        Parameters
        ----------
        filepath : str
            Path to the frequency file (unused, retained for compatibility).
        imag_freq_dict : dict
            Dictionary mapping normal mode index to imaginary frequency.
        normal_mode_tensor : np.ndarray
            Array of shape (n_atoms, 3, n_modes) containing displacement vectors.
        coordinates : list of list of list
            Input coordinates with shape: [[['C', x, y, z], ...], ...] (one or more molecules).
        displacement_value : float
            Magnitude of the displacement.
        random_mode : bool
            If True, displace along a random imaginary mode. Else, use the least negative.

        Returns
        -------
        tuple
            (pos_displaced, neg_displaced): two lists of displaced structures in the same shape as input.
        """
        if not random_mode:
            if not imag_freq_dict or len(imag_freq_dict) == 0:
                raise ValueError("No imaginary frequencies found.")

        if random_mode:
            import random

            mode_idx = random.choice(list(imag_freq_dict.keys()))
        else:
            mode_idx = min(imag_freq_dict.items(), key=lambda kv: abs(kv[1]))[0]

        disp_vector = normal_mode_tensor[:, :, mode_idx]

        pos_displaced = []
        neg_displaced = []

        for mol in coordinates:
            if not all(len(atom) == 4 for atom in mol):
                raise ValueError("Each atom must have [symbol, x, y, z].")

            symbols = [atom[0] for atom in mol]
            positions = np.array([[float(x), float(y), float(z)] for _, x, y, z in mol])

            pos = positions + displacement_value * disp_vector
            neg = positions - displacement_value * disp_vector

            pos_atoms = [
                [symbols[i], f"{pos[i][0]:.4f}", f"{pos[i][1]:.4f}", f"{pos[i][2]:.4f}"]
                for i in range(len(symbols))
            ]
            neg_atoms = [
                [symbols[i], f"{neg[i][0]:.4f}", f"{neg[i][1]:.4f}", f"{neg[i][2]:.4f}"]
                for i in range(len(symbols))
            ]

            pos_displaced.append(pos_atoms)
            neg_displaced.append(neg_atoms)

        return pos_displaced, neg_displaced

    def write_displaced_xyz(
        self, structures, step_number, structure_ids, output_dir="."
    ):
        """
        Writes XYZ files for displaced structures (positive/negative modes).
        Accepts nested structure format: [[['C', x, y, z], ...]] per structure.

        Parameters
        ----------
        structures : list of list of list
            List of structures (each structure is a list of atoms).
        step_number : int
            Step number for filename convention.
        structure_ids : list
            Identifiers (e.g. ["0_pos", "0_neg"]).
        output_dir : str
            BASE outputs directory (this function appends step{n}/normal_mode_sampling).
        """
        import os
        import logging

        logging.info(
            f"Writing Ensemble XYZ files to {output_dir} for step {step_number}"
        )
        base_name = f"step{step_number}"
        written = []

        # This function appends the subfolders; ensure the final parent exists:
        for structure, sid in zip(structures, structure_ids):
            # Flatten structure if nested
            if isinstance(structure[0], list) and isinstance(structure[0][0], str):
                atom_list = structure
            else:
                atom_list = [atom for mol in structure for atom in mol]

            # Comment line
            if str(sid).endswith("_pos"):
                comment = "Displaced along +imaginary mode"
            elif str(sid).endswith("_neg"):
                comment = "Displaced along -imaginary mode"
            else:
                comment = ""

            # Build final path under BASE/step{n}/normal_mode_sampling
            nested_dir = os.path.join(
                output_dir, f"step{step_number}", "normal_mode_sampling"
            )
            os.makedirs(nested_dir, exist_ok=True)  # <-- ensure parent exists
            output_file = os.path.join(nested_dir, f"{base_name}_structure_{sid}.xyz")

            with open(output_file, "w") as f:
                f.write(f"{len(atom_list)}\n{comment}\n")
                for element, x, y, z in atom_list:
                    f.write(f"{element} {x} {y} {z}\n")

            written.append(os.path.abspath(output_file))

        return written

    def select_lowest_imaginary_structures(
        self, directory, pos_ids, neg_ids, step_number
    ):
        """
        Selects structures with exactly one imaginary frequency and lowest energy.

        Parameters
        ----------
        directory : str
            Path to the base output directory (typically 'outputs').
        pos_ids : list
            List of positive structure IDs (e.g. ['0_pos', '1_pos']).
        neg_ids : list
            List of negative structure IDs (e.g. ['0_neg', '1_neg']).
        step_number : int
            Step number in the workflow (e.g. 1, 2, 3).

        Returns
        -------
        tuple
            (filtered_coordinates, filtered_ids)
        """
        selected_coords = []
        selected_ids = []
        base_dir = os.path.join(directory, f"step{step_number}", "normal_mode_sampling")

        for pos_id, neg_id in zip(pos_ids, neg_ids):
            pos_path = os.path.join(
                base_dir, f"step{step_number}_structure_{pos_id}.out"
            )
            neg_path = os.path.join(
                base_dir, f"step{step_number}_structure_{neg_id}.out"
            )

            pos_coords_list, pos_energies, _ = self.parse_dft_output(pos_path)
            neg_coords_list, neg_energies, _ = self.parse_dft_output(neg_path)

            pos_imag_freqs = self.parse_imaginary_frequency(pos_path, imag=True)
            neg_imag_freqs = self.parse_imaginary_frequency(neg_path, imag=True)

            pos_valid = len(pos_imag_freqs) == 1
            neg_valid = len(neg_imag_freqs) == 1

            if pos_valid and neg_valid:
                if pos_energies[0] < neg_energies[0]:
                    selected_coords.append(pos_coords_list[0])
                    selected_ids.append(pos_id)
                    logging.info(
                        f"Both '{pos_id}' and '{neg_id}' have one imaginary frequency. "
                        f"Selected '{pos_id}' due to lower energy ({pos_energies[0]:.6f} eV)."
                    )
                else:
                    selected_coords.append(neg_coords_list[0])
                    selected_ids.append(neg_id)
                    logging.info(
                        f"Both '{pos_id}' and '{neg_id}' have one imaginary frequency. "
                        f"Selected '{neg_id}' due to lower energy ({neg_energies[0]:.6f} eV)."
                    )
            elif pos_valid:
                selected_coords.append(pos_coords_list[0])
                selected_ids.append(pos_id)
                logging.info(f"Only '{pos_id}' has one imaginary frequency. Selected.")
            elif neg_valid:
                selected_coords.append(neg_coords_list[0])
                selected_ids.append(neg_id)
                logging.info(f"Only '{neg_id}' has one imaginary frequency. Selected.")
            else:
                logging.error(
                    f"Neither '{pos_id}' nor '{neg_id}' has exactly one imaginary frequency.\n"
                    f"Unable to remove imaginary frequencies. Try a higher displacement value."
                )
                raise RuntimeError("Imaginary frequency removal failed. Aborting.")

        return selected_coords, selected_ids

    def generate_random_displacements(
        self,
        sid,
        file_path,
        normal_mode_tensor,
        coordinates,
        num_random_modes,
        displacement_value,
        step_number,
        input_template,
        charge,
        multiplicity,
        output_dir,
        engine,
        model_name,
        task_name,
        device,
        bind,
        normal_output_dir,
        operation,
        create_inp=False,
    ):
        """
        Generate multiple random mode displacements for a given structure.
        [docstring unchanged]
        """
        all_pos_coords, all_neg_coords = [], []
        all_pos_ids, all_neg_ids = [], []

        # Parse all frequencies (imag=False) for use in random_mode=True
        imag_freq_dict = self.parse_imaginary_frequency(file_path, imag=False)

        for i in range(num_random_modes):
            pos_coords, neg_coords = self.displace_normal_modes(
                filepath=file_path,
                imag_freq_dict=imag_freq_dict,
                normal_mode_tensor=normal_mode_tensor,
                coordinates=coordinates,
                displacement_value=displacement_value,
                random_mode=True,
            )

            all_pos_coords.append(pos_coords[0])
            all_neg_coords.append(neg_coords[0])
            all_pos_ids.append(f"{sid}_rand{i}_pos")
            all_neg_ids.append(f"{sid}_rand{i}_neg")

            logging.info(f"Generated random mode displacement {i} for {sid}")

        # Write all XYZ files at once
        logging.info(
            f"Writing {len(all_pos_coords)} positive and {len(all_neg_coords)} negative XYZ files for random modes."
        )
        pos_xyz = self.write_displaced_xyz(
            all_pos_coords, step_number, all_pos_ids, output_dir
        )
        neg_xyz = self.write_displaced_xyz(
            all_neg_coords, step_number, all_neg_ids, output_dir
        )
        xyz_files = pos_xyz + neg_xyz

        # Generate ORCA input files
        if create_inp:
            logging.info(
                f"Creating ORCA input files for {len(xyz_files)} random mode displacements."
            )
            input_files, _ = self.create_input(
                xyz_files,
                input_template,
                charge,
                multiplicity,
                output_dir=normal_output_dir,
                operation=operation,
                engine=engine,
                model_name=model_name,
                task_name=task_name,
                device=device,
                bind=bind,
            )

            return all_pos_coords, all_neg_coords, all_pos_ids, all_neg_ids, input_files
        return all_pos_coords, all_neg_coords, all_pos_ids, all_neg_ids, []

    def has_normal_modes(self, filepath: str) -> bool:
        """
        Fast check: does this ORCA output include a normal-mode table?
        Returns True if a 'VIBRATIONAL FREQUENCIES' section and any normal-mode
        column header are present.
        """
        hdr_seen = False
        with open(filepath, "r") as f:
            for line in f:
                if "VIBRATIONAL FREQUENCIES" in line:
                    hdr_seen = True
                # column header like: "   1    2    3"
                if hdr_seen and re.match(r"^\s*(\d+\s+)+\d+\s*$", line):
                    return True
        return False


def _orca_parse_all_gradients(content: str, to_ev_per_A: bool = False):
    """
    Parse all CARTESIAN GRADIENT blocks in an ORCA output and return raw gradients.

    Parameters
    ----------
    content : str
        Full ORCA output file contents.
    to_ev_per_A : bool, optional
        Convert Hartree/Bohr gradients to eV/Å forces. Default False (keep Hartree/Bohr).

    Returns
    -------
    list[np.ndarray]
        Each entry is an (N_atoms, 3) array of forces.
        By default in Hartree/Bohr (raw ORCA units).
    """
    blocks = _ORCA_GRAD_BLOCK_RE.findall(content)
    out = []
    for b in blocks:
        rows = []
        for line in b.strip().splitlines():
            m = _ORCA_GRAD_LINE_RE.match(line)
            if not m:
                continue
            dEdx = float(m.group(2).replace("D", "E"))
            dEdy = float(m.group(3).replace("D", "E"))
            dEdz = float(m.group(4).replace("D", "E"))
            fx, fy, fz = -dEdx, -dEdy, -dEdz  # F = -∇E
            if to_ev_per_A:
                fx *= HARTREE_PER_BOHR_TO_EV_PER_A
                fy *= HARTREE_PER_BOHR_TO_EV_PER_A
                fz *= HARTREE_PER_BOHR_TO_EV_PER_A
            rows.append([fx, fy, fz])
        if rows:
            out.append(np.array(rows, dtype=float))
    return out

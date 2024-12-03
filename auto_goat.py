import re
from pathlib import Path
import subprocess
import argparse
import os 
import sys 
import time

def parse_arguments():
    parser = argparse.ArgumentParser(description='Code to automate the process of conformer searching, submits initial XTB calculation and improves precision')
    parser.add_argument('input_file',help='Intial ORCA file for GOAT optimization for CHEAP level of theory')

    #Optional arguments
    parser.add_argument(
    '-cores','-c',type=int,default=16,
    help='Max number of cores used throughout conformer search. For multiple files it will be cores/PAL')
    
    args = parser.parse_args()
    return args

def detect_goat_output(input_dir):
    """
    Detecs if there are GOAT outputs in the directory, if so continues with other steps without running GOAT. 

    Args: 
    Input Dir

    Returns: 
    Bool for running calculation
    """
    for files in os.listdir(input_dir):
        if files.endswith('finalensemble.xyz'):
            print('Final Ensemble File Already Exists Continuing with other Steps')
            run_calc = False
            break
        else:
            run_calc = True
    if run_calc == True:
        print("No GOAT output found in directory, running GOAT.")
    return run_calc

def submit_qorca(input_file):
    """
    Uses our in-house code for submitting ORCA calculations

    Args:
        Input File

    Returns: 
            JobID of the submitted file
    """
    command = ["qorca", input_file] 
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    jobid = result.stdout.split()[-1]
    return jobid

def read_input_file(input_path):
    """Read the ORCA input file."""
    if not input_path.is_file():
        print(
            f"Error: Input file '{input_path}' does not exist. "
            "Please check the file path and try again.",
            file=sys.stderr
        )
        sys.exit(1)
    with input_path.open('r') as f:
        lines = f.readlines()
    return lines

def parse_pal_from_input(lines):
    """
    Parse PAL value from ORCA input file lines.

    Args:
        lines (list): List of strings representing lines in the ORCA input file.

    Returns:
        int: The PAL value found in the input file, or None if not found.
    """
    input_text = ''.join(lines)
    # Use regex to find %pal ... end blocks, regardless of formatting
    pal_blocks = re.findall(r'%pal\s*(.*?)\s*end', input_text, re.IGNORECASE | re.DOTALL)
    for pal_block in pal_blocks:
        # Search for nprocs value within pal_block
        match = re.search(r'nprocs\s+(\d+)', pal_block, re.IGNORECASE)
        if match:
            pal_value = int(match.group(1))
            return pal_value
    # Also check for pal in '!' line
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith('!'):
            tokens = stripped_line[1:].strip().split()
            for token in tokens:
                match = re.match(r'pal(\d+)', token.lower())
                if match:
                    pal_value = int(match.group(1))
                    return pal_value
    return None

def parse_coordinates(data):
    # If data is already a list of lines, no need to split
    if isinstance(data, str):
        lines = data.strip().split('\n')
    else:
        lines = data  # Assume it's already a list
    
    result = []
    recording = False

    for line in lines:
        line = line.strip()
        if line.startswith('* xyz'):
            # Parse charge and multiplicity
            _, _, charge, multiplicity = line.split()
            charge, multiplicity = int(charge), int(multiplicity)
            recording = True
            current_structure = {
                "header": f"* xyz {charge} {multiplicity}",
                "atoms": []
            }
        elif line.startswith('*'):
            if recording:
                result.append(current_structure)
                recording = False
        elif recording:
            # Parse atom data
            parts = line.split()
            atom = f"{parts[0]:<2} {float(parts[1]):>12.6f} {float(parts[2]):>12.6f} {float(parts[3]):>12.6f}"
            current_structure["atoms"].append(atom)
    
    return result

def submit_goat(input_file):
    """
    Submit a single file for calculation and wait until the job finishes.

    Args:
        input_file (str): Path to the input file to be submitted.
    """
    # Submit the job and get the job ID
    jobid = submit_qorca(input_file)
    print(f"Job {jobid} submitted for file {input_file}. Waiting for it to finish...")

    # Check if the job is still running
    while not is_job_finished(jobid):
        print(f"Job {jobid} is still running. Checking again in 30 seconds...")
        time.sleep(30)  # Wait for 30 seconds before checking again

    print(f"Job {jobid} for file {input_file} has finished.")

def write_preopt_coordinates(structures, output_file,cores):
    with open(output_file, "w") as f:
        f.write('%pal nprocs' + str(cores) +'\n')
        f.write('\n!opt pbe def2svp \n')
        f.write('%geom\nMaxIter 50 \nConvergence loose \nend \n')
        for structure in structures:
            f.write("\n" + structure["header"] + "\n")
            f.write("\n".join(structure["atoms"]) + "\n")
            f.write("*\n")

def get_input_dir(input_file):
    # Get directory from input file
    input_dir = os.path.dirname(input_file)
    
    # If no directory is specified, default to the current working directory
    if not input_dir:  # Empty string evaluates to False
        input_dir = os.getcwd()
    
    return input_dir

def parse_ensemble_coordinates(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    structures = []
    current_structure = []
    for line in lines:
        # Split the line into columns
        columns = line.strip().split()
        # Check if the line is a coordinate line (4 columns)
        if len(columns) == 4:
            current_structure.append(line.strip())
        # Check if the line starts a new structure (number of atoms)
        elif len(columns) == 1 and columns[0].isdigit():
            if current_structure:
                structures.append(current_structure)
                current_structure = []

    # Add the last structure if it exists
    if current_structure:
        structures.append(current_structure)

    return structures

def write_ensemble_coordinates(structures):
    print("Writing Ensemble XYZ files")
    base_name = 'step2'
    xyz_filenames = []
    for i, structure in enumerate(structures, start=1):
        output_file = os.path.join('./', f"{base_name}_structure_{i}.xyz")
        xyz_filenames.append(output_file)
        with open(output_file, 'w') as file:
            # Write the number of atoms (line 1) and a blank line (line 2)
            file.write(f"{len(structure)}\n\n")
            # Write the coordinates
            for atom in structure:
                file.write(f"{atom}\n")
    return xyz_filenames

def create_orca_input(xyz_files, template='step2.inp', output_dir='./'):
    input_files = []
    output_files = []
    for file in xyz_files:
        base_name = os.path.splitext(os.path.basename(file))[0]
        input_file = os.path.join(output_dir, f"{base_name}.inp")
        output_file = os.path.join(output_dir, f"{base_name}.out")
        input_files.append(input_file)
        output_files.append(output_file)
        with open(template, "r") as tmpl:
            content = tmpl.read().replace("molecule.xyz", file)
        with open(input_file, "w") as inp:
            inp.write(content)
        print(f" Writing {input_file}:")
    return input_files,output_files    

def submit_multiple_files(max_cores, input_files, partition="sterling"):
    """
    Submit multiple calculations based on available cores, handling the case
    where max_cores is less than the PAL value detected in the input files.

    Parameters:
    - max_cores (int): Maximum number of cores to use at one time.
    - input_files (list): List of input files for calculations.
    - partition (str): SLURM partition to submit jobs.
    """
    username = subprocess.check_output("whoami", text=True).strip()
    jobs = []  # List to track submitted job IDs

    while input_files or jobs:
        # Debugging: print current job status
        print(f"Checking jobs. Jobs running: {len(jobs)}")
        
        # Remove completed jobs from the list by checking if they're finished
        jobs = [job for job in jobs if not is_job_finished(job, partition)]
        
        # Debugging: print jobs that are still running
        print(f"Jobs still running: {jobs}")
        
def submit_multiple_files(input_files,max_cores=16,partition="sterling"):
    """
    Submit multiple calculations based on available cores, ensuring the number of running jobs 
    does not exceed the specified max_cores. Submissions are sequential, and the function waits 
    for cores to free up before submitting more.
    
    Args:
        max_cores (int): Maximum number of cores to use simultaneously.
        input_files (list): List of input file paths for calculations.
        partition (str): Partition name for SLURM queue.
    """
    total_cores_used = 0
    active_jobs = {}  # Map of job_id to cores used for tracking active jobs
    
    for input_file in input_files:
        input_path = Path(input_file)
        try:
            # Read the input file and extract the PAL value
            lines = read_input_file(input_path)
            pal_value = parse_pal_from_input(lines)
            if pal_value is None:
                print(f"Error: PAL value not found in input file {input_file}. Skipping...")
                continue
            
            cores_needed = pal_value
            
            # Wait if there aren't enough free cores to submit the next job
            while total_cores_used + cores_needed > max_cores:
                print("Waiting for jobs to finish to free up cores...")
                
                # Check active jobs and remove completed ones
                completed_jobs = []
                for job_id, cores in active_jobs.items():
                    if is_job_finished(job_id, partition):
                        completed_jobs.append(job_id)
                        total_cores_used -= cores
                
                # Remove completed jobs from active_jobs
                for job_id in completed_jobs:
                    del active_jobs[job_id]
                
                time.sleep(30)  # Check every 30 seconds

            # Submit the job
            print(f"Submitting job for {input_file} requiring {cores_needed} cores...")
            job_id = submit_qorca(input_file)
            active_jobs[job_id] = cores_needed
            total_cores_used += cores_needed

        except Exception as e:
            print(f"Error processing input file {input_file}: {e}")
            continue

    # Wait for all remaining jobs to finish
    print("All jobs submitted. Waiting for remaining jobs to complete...")
    while active_jobs:
        completed_jobs = []
        for job_id, cores in active_jobs.items():
            if is_job_finished(job_id, partition):
                completed_jobs.append(job_id)
                total_cores_used -= cores

        for job_id in completed_jobs:
            del active_jobs[job_id]

        time.sleep(30)

    print("All calculations finished.")

def is_job_finished(job_id, partition="sterling"):
    """
    Check if a SLURM job with a given job ID has finished.
    
    Parameters:
    - job_id (str): The SLURM job ID to check.
    - partition (str): The SLURM partition to check. Default is "sterling".
    
    Returns:
    - bool: True if the job is not in the queue (finished), False otherwise.
    """
    try:
        # Get the current username using `whoami`
        username = subprocess.check_output("whoami", text=True).strip()
        
        # Construct the squeue command
        command = f"squeue -u {username} -p {partition} -o %i"
        
        # Execute the command and capture the output
        output = subprocess.check_output(command, shell=True, text=True)
        
        # Split the output into lines and check if job_id is present
        job_ids = output.splitlines()
        
        # The first line is usually the header, skip it
        if job_id in job_ids[1:]:
            return False  # Job is still in the queue
        else:
            return True   # Job has finished or is not in the queue
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return False
    
def parse_last_orca_total_energy(file_path):
    """Parse the last total energy from an ORCA output file, supporting both XTB and DFT calculations."""
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Regular expressions for both XTB and DFT energy patterns
    xtb_energy_pattern = r":: total energy\s+(-?\d+\.\d+) Eh"
    dft_energy_pattern = r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)"
    
    # Check for the presence of DFT or XTB specific lines and apply the corresponding pattern
    if re.search(dft_energy_pattern, content):
        match = re.search(dft_energy_pattern, content)
    elif re.search(xtb_energy_pattern, content):
        match = re.search(xtb_energy_pattern, content)
    else:
        sys.exit(f"Error: Total energy not found for step 2 in file: {file_path}")
    
    if match:
        # Extract and return the energy
        energy = float(match.group(1))
        return energy

def find_lowest_energy_file(file_list):
    """Find the file with the lowest total energy."""
    energy_file_mapping = {}
    
    for file in file_list:
        try:
            energy = parse_last_orca_total_energy(file)
            energy_file_mapping[file] = energy
        except ValueError as e:
            print(e)
    
    # Determine the file with the lowest energy
    if energy_file_mapping:
        lowest_energy_file = min(energy_file_mapping, key=energy_file_mapping.get)
        lowest_energy = energy_file_mapping[lowest_energy_file]
        return lowest_energy_file, lowest_energy
    else:
        raise ValueError("No valid energies found in the file list.")

def parse_final_coordinates(file_path):
    """
    Extract the Cartesian coordinates after the final energy evaluation in an ORCA output file.
    
    Args:
        file_path (str): Path to the ORCA output file.
    
    Returns:
        list: A list of tuples where each tuple represents an atom's type and its (x, y, z) coordinates.
    """
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Pattern to locate the final energy evaluation section
    final_energy_pattern = r"\*{3} FINAL ENERGY EVALUATION AT THE STATIONARY POINT \*{3}.*?CARTESIAN COORDINATES \(ANGSTROEM\)\n-+\n((?:\s*[A-Za-z]+\s+-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+\n)+)"
    
    # Match the section with Cartesian coordinates
    match = re.search(final_energy_pattern, content, re.DOTALL)
    if not match:
        raise ValueError("Final coordinates section not found in the file.")
    
    # Extract the coordinate block
    coordinates_block = match.group(1)
    
    # Parse individual lines of coordinates
    coordinates = []
    for line in coordinates_block.strip().split('\n'):
        parts = line.split()
        atom_type = parts[0]
        x, y, z = map(float, parts[1:4])
        coordinates.append((atom_type, x, y, z))
    
    return coordinates

def write_xyz(coordinates):
    """
    Write atomic coordinates to an XYZ file.
    
    Args:
        coordinates (list): A list of tuples containing atom type and (x, y, z) coordinates.
        output_file (str): Path to the output XYZ file.
    """
    # Number of atoms
    num_atoms = len(coordinates)
    output_file = './step3.xyz'
    with open(output_file, 'w') as file:
        # Write the number of atoms
        file.write(f"{num_atoms}\n \n")
        # Write the atomic data
        for atom in coordinates:
            atom_type, x, y, z = atom
            file.write(f"{atom_type} {x:.6f} {y:.6f} {z:.6f}\n")
    return output_file

def main():
    #Get parse arguments
    args = parse_arguments()
    cores = args.cores
    input_file = args.input_file
    input_dir = get_input_dir(args.input_file)
    #Detect if GOAT has been run in this directory if not run it
    goat_calc = detect_goat_output(input_dir)
    if goat_calc: 
        jobid = submit_goat(args.input_file)
    #Parse final ensemble geometries 
    base_name = Path(input_file).stem
    final_ensemble_file = Path(base_name + '.finalensemble.xyz')
    coordinates = parse_ensemble_coordinates(final_ensemble_file)
    #Create XYZ files with ensemble coordinates
    xyz_filenames = write_ensemble_coordinates(coordinates)
    #Using template input file creates a ORCA input that reads ensemble files
    input_files,output_files = create_orca_input(xyz_filenames)
    #Submits and checks for completion of Step 2 ensemble
    submit_multiple_files(input_files,cores)
    try:
        lowest_file, lowest_energy = find_lowest_energy_file(output_files)
        print(f"The file with the lowest energy is {lowest_file} with an energy of {lowest_energy} Eh.")
    except ValueError as e:
        print(e)
    coordinates = parse_final_coordinates(lowest_file)
    step3_xyz = write_xyz(coordinates)
    step3_xyz=[step3_xyz]
    step3_input_file,step3_output_file = create_orca_input(step3_xyz,template='step3.inp')
    submit_multiple_files(step3_input_file,cores)

if __name__ == "__main__":
    main()



import numpy as np
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
    '-c','-cores',type=int,default=8,
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
    for file in xyz_files:
        base_name = os.path.splitext(os.path.basename(file))[0]
        input_file = os.path.join(output_dir, f"{base_name}.inp")
        input_files.append(input_file)
        with open(template, "r") as tmpl:
            content = tmpl.read().replace("molecule.xyz", file)
        with open(input_file, "w") as inp:
            inp.write(content)
        print(f" Writing {input_file}:")
    return input_files    

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
    jobs = []  # List to track submitted jobs

    while input_files or jobs:
        # Remove completed jobs from the list
        jobs = [job for job in jobs if is_running(username, partition)]

        if input_files:
            # Parse PAL value from the first input file
            try:
                pal_value = parse_pal_from_input(input_files[0])
            except ValueError:
                print("Error: PAL value not found in input file.")
                return
            
            # Check if max_cores is less than PAL value
            if max_cores < pal_value:
                print(f"Warning: Max cores ({max_cores}) is less than the required PAL ({pal_value}). Defaulting to running 1 calculation at a time.")
                cores_per_job = 1  # Only run 1 calculation at a time
            else:
                cores_per_job = max_cores // pal_value  # Use available cores for multiple calculations
            
            # Limit jobs to available cores
            num_jobs = min(len(input_files), cores_per_job)

            # Submit jobs
            for _ in range(num_jobs):
                input_file = input_files.pop(0)
                job_id = submit_job(input_file, partition)
                print(f"Submitted job {job_id} for input file {input_file}")
                jobs.append(job_id)

        if jobs:
            print("Waiting for jobs to finish...")
            time.sleep(30)  # Wait before checking again




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


def main():
    args = parse_arguments()
    input_file = args.input_file
    input_dir = get_input_dir(args.input_file)
    goat_calc = detect_goat_output(input_dir)
    if goat_calc: 
        jobid = submit_qorca(args.input_file)
    final_ensemble_file = Path(input_file.split('/')[-1].split('.')[0] + '.finalensemble.xyz')
    coordinates = parse_ensemble_coordinates(final_ensemble_file)
    xyz_filenames = write_ensemble_coordinates(coordinates)
    input_files = create_orca_input(xyz_filenames)
    submit_multiple_files(args.cores,input_files)

if __name__ == "__main__":
    main()



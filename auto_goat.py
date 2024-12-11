import re
from pathlib import Path
import subprocess
import argparse
import os 
import sys 
import time
import yaml  # type: ignore
import numpy as np  # type: ignore
import glob
import shutil
import pandas as pd # type: ignore

def parse_arguments():
    parser = argparse.ArgumentParser(description='Code to automate the process of conformer searching, submits initial XTB calculation and improves precision')
    parser.add_argument('input_file',help='YAML input file containing instructions for automated workflow')
    
    #Optional arguments
    parser.add_argument(
    '-cores','-c',type=int,default=16,
    help='Max number of cores used throughout conformer search. For multiple files it will be cores/PAL')
    
    parser.add_argument(
    '-skip',action='store_true',default=False,
    help='Skips the first step because it was already run, and starts with the next steps.')
    args = parser.parse_args()
    return args

def submit_qorca(input_file):
    """
    Uses our in-house code for submitting ORCA calculations

    Args:
        Input File

    Returns: 
            JobID of the submitted file
    """
    command = ["qorca", "-x", "compute-2-07-01,compute-2-07-02", input_file]
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

def get_input_dir(input_file):
    # Get directory from input file
    input_dir = os.path.dirname(input_file)
    
    # If no directory is specified, default to the current working directory
    if not input_dir:  # Empty string evaluates to False
        input_dir = os.getcwd()
    
    return input_dir

def create_orca_input(xyz_files, template, output_dir='./'):
    input_files = []
    output_files = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for file in xyz_files:
        base_name = os.path.splitext(os.path.basename(file))[0]
        input_file = os.path.join(output_dir, f"{base_name}.inp")
        output_file = os.path.join(output_dir, f"{base_name}.out")
        input_files.append(input_file)
        output_files.append(output_file)
        with open(template, "r") as tmpl:
            content = tmpl.read().replace("molecule.xyz", file)
        with open(input_file, "w") as inp:
            inp.write(content + ' ')
        print(f" Writing {input_file}:")
    return input_files,output_files    
        
def submit_files(input_files,max_cores=16,partition="sterling"):
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
                    print(f"Job ID {job_id} is running with {cores}")
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

def parse_orca_output(file_paths, calculation_type, dir='./'):
    """
    Parse ORCA output files for specified calculation types.

    Parameters:
        file_paths (list): List of file names (without directory) to process.
        calculation_type (str): Type of calculation ('goat', 'dft', 'mlff').
        dir (str): Directory to read files from. Defaults to './'.

    Returns:
        tuple: A list of coordinates and a list of energies.
    """
    calculation_type = calculation_type.lower()
    all_coordinates_list = []
    all_energies_list = []
    
    for file_name in file_paths:
        file_path = os.path.join(dir, file_name)  # Combine directory and file name
        
        with open(file_path, 'r') as f:
            content = f.read()

        basename = os.path.splitext(file_path)[0]  # Get the base name of the current file
        
        if calculation_type == 'goat':
            # Parse the corresponding .finalensemble.xyz file
            finalensemble_file = f"{basename}.finalensemble.xyz"
            try:
                with open(finalensemble_file, 'r') as file:
                    lines = file.readlines()

                current_structure = []
                energy = None
                atom_count = None
                for i, line in enumerate(lines):
                    columns = line.strip().split()
                    # Check if the line contains the number of atoms
                    if len(columns) == 1 and columns[0].isdigit():
                        if current_structure:  # If there's a current structure, save it
                            # Append the current structure and its corresponding energy
                            all_coordinates_list.append([
                                (atom.split()[0], float(atom.split()[1]), float(atom.split()[2]), float(atom.split()[3]))
                                for atom in current_structure
                            ])
                            all_energies_list.append(energy)
                            current_structure = []
                        atom_count = int(columns[0])
                        # The next line should contain the energy value
                        if i + 1 < len(lines):
                            energy_line = lines[i + 1].strip()
                            energy_match = re.match(r"(-?\d+\.\d+)", energy_line)
                            if energy_match:
                                energy = energy_match.group(1)
                            else:
                                energy = None
                    # Add coordinate lines to the current structure
                    elif len(columns) == 4:  # Coordinates
                        current_structure.append(line.strip())

                # Add the last structure if it exists
                if current_structure:
                    all_coordinates_list.append([
                        (atom.split()[0], float(atom.split()[1]), float(atom.split()[2]), float(atom.split()[3]))
                        for atom in current_structure
                    ])
                    all_energies_list.append(energy)

            except FileNotFoundError:
                raise ValueError(f"Corresponding .finalensemble.xyz file not found: {finalensemble_file}")
        
        elif calculation_type in ['dft', 'mlff']:
            # Extract only the final Cartesian coordinates (Angstrom)
            final_coordinates_block = re.findall(
                r"CARTESIAN COORDINATES \(ANGSTROEM\)\n-+\n((?:.*?\n)+?)-+\n",
                content,
                re.DOTALL  # Ensure multiline matching
            )

            if final_coordinates_block:
                final_coordinates = [line.split() for line in final_coordinates_block[-1].strip().splitlines()]
            else:
                final_coordinates = []

            # Extract only the final energy (if available)
            final_energy = re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", content)
            energy = final_energy[-1] if final_energy else None

            # Append the coordinates and energy
            all_coordinates_list.append(final_coordinates)  # Append final coordinates
            all_energies_list.append(energy)  # Append corresponding energy

        else:
            raise ValueError("Invalid calculation_type. Choose from 'goat', 'dft', or 'mlff'.")

    return all_coordinates_list, all_energies_list

def filter_structures(coordinates_list, energies, id_list, method, **kwargs):
    """
    Filters structures based on the provided method while preserving original ID order.

    Parameters:
        coordinates_list (list): List of coordinate blocks (parsed from ORCA output).
        energies (list): List of energies corresponding to each structure.
        id_list (list): List of IDs corresponding to each structure.
        method (str): Filtering method. Options are 'energy_window', 'boltzmann', 'integer'.
        **kwargs: Additional arguments for filtering methods.
            - 'energy_window': `window` (float) - Energy window from the lowest energy.
            - 'boltzmann': `percentage` (float) - Percentage of Boltzmann probability.
            - 'integer': `num_structures` (int) - Number of structures to select.

    Returns:
        filtered_coordinates (list): List of selected coordinate blocks.
        filtered_ids (list): List of IDs of selected structures, preserving original order.
    """
    if len(coordinates_list) != len(energies) or len(energies) != len(id_list):
        raise ValueError(
            f"Mismatch in list lengths: coordinates_list ({len(coordinates_list)}), "
            f"energies ({len(energies)}), and id_list ({len(id_list)}) must have the same length."
        )

    energies = np.array([float(e) for e in energies])
    sorted_indices = np.argsort(energies)  # Sort energies to determine favored structures

    if method == 'energy_window':
        window = kwargs.get('window', 0.5)
        min_energy = np.min(energies)
        favored_indices = [i for i in sorted_indices if energies[i] <= min_energy + window]

    elif method == 'boltzmann':
        # Constants
        R_kcalmol_K = 0.0019872041  # kcal/(mol·K)
        temperature = 298.15
        hartree_to_kcalmol = 627.5  # Conversion factor from Hartrees to kcal/mol
        percentage = kwargs.get('percentage',99)
        # Convert energies from Hartrees to kcal/mol
        energies_kcalmol = energies * hartree_to_kcalmol

        # Calculate minimum energy
        min_energy = np.min(energies_kcalmol)

        # Calculate Boltzmann weights (without normalization)
        delta_E = energies_kcalmol - min_energy  # Energy differences (ΔE)
        boltzmann_weights = np.exp(-delta_E / (R_kcalmol_K * temperature))

        # Normalize Boltzmann weights to sum to 1
        boltzmann_probs = boltzmann_weights / np.sum(boltzmann_weights)

        # Calculate cumulative probability distribution
        cumulative_probs = np.cumsum(boltzmann_probs)

        # Determine the cutoff probability (percentage of total probability)
        cutoff_prob = percentage / 100.0

        # Find the indices where cumulative probability exceeds the cutoff
        favored_indices = [i for i, prob in enumerate(cumulative_probs) if prob <= cutoff_prob]
        

    elif method == 'integer':
        num_structures = kwargs.get('num_structures', 5)
        num_structures = min(num_structures, len(coordinates_list))  # Ensure we don't exceed the list length
        if num_structures <= 0 or num_structures >= len(coordinates_list):  # If num_structures is larger than or equal to the list size
            favored_indices = sorted_indices  # Return all indices (sorted by energy)
        else:
                favored_indices = sorted_indices[:num_structures]  # Return top 'num_structures' based on energy sorting
    else:
        raise ValueError("Invalid method. Choose from 'energy_window', 'boltzmann', or 'integer'.")

    # Create a mask to preserve only the favored structures while maintaining original order
    mask = [i in favored_indices for i in range(len(coordinates_list))]
    filtered_coordinates = [coord for coord, keep in zip(coordinates_list, mask) if keep]
    filtered_ids = [id_ for id_, keep in zip(id_list, mask) if keep]

    return filtered_coordinates, filtered_ids

def write_xyz(structures, step_number, structure_ids):
    print("Writing Ensemble XYZ files")
    base_name = f"step{step_number}"
    xyz_filenames = []
    
    for structure, structure_id in zip(structures, structure_ids):
        output_file = os.path.join('./', f"{base_name}_structure_{structure_id}.xyz")
        xyz_filenames.append(output_file)
        with open(output_file, 'w') as file:
            # Write the number of atoms (line 1) and a blank line (line 2)
            file.write(f"{len(structure)}\n\n")
            # Write the coordinates
            for atom in structure:
                element, x, y, z = atom  # Unpack the atom's data
                file.write(f"{element} {x} {y} {z}\n")
    
    return xyz_filenames

def move_step_files(step_number):
    """
    Move all files starting with 'step{step_number}' to a directory named 'step{step_number}'.

    Parameters:
        step_number (int): The step number whose files need to be moved.
    """
    step_dir = f"step{step_number}"

    # Create the directory if it doesn't exist
    if not os.path.exists(step_dir):
        os.makedirs(step_dir)

    # Find all files starting with 'step{step_number}'
    files_to_move = glob.glob(f"step{step_number}*")

    # Move each file to the directory
    for file in files_to_move:
        shutil.move(file, step_dir)

    print(f"Moved files for step {step_number} to directory '{step_dir}'")

def save_step_csv(energies, ids, step_number, temperature=298.15, filename="steps.csv", precision=8):
    """
    Appends energy data for a given step to a CSV file, with step number included.
    
    Parameters:
        energies (list or array): List of energy values in Hartrees.
        ids (list or array): List of corresponding IDs.
        step_number (int): Step number to label the CSV page.
        temperature (float): Temperature in Kelvin (default is 298.15 K).
        filename (str): Name of the output CSV file (default is "steps.csv").
        precision (int): Decimal precision for output values (default is 8).
    """
    # Conversion factor from Hartrees to kcal/mol
    hartree_to_kcalmol = 627.5
    # Gas constant in kcal/(mol*K)
    R_kcalmol_K = 0.0019872041  # kcal/(mol*K)
    
    # Convert inputs to DataFrame
    df = pd.DataFrame({'Conformer': ids, 'Energy (Hartrees)': energies})
    
    # Ensure Energy column is numeric
    df['Energy (Hartrees)'] = pd.to_numeric(df['Energy (Hartrees)'], errors='coerce')
    if df['Energy (Hartrees)'].isnull().any():
        raise ValueError("Non-numeric or missing energy values found. Please clean the input data.")
    
    # Convert energy to kcal/mol
    df['Energy (kcal/mol)'] = df['Energy (Hartrees)'] * hartree_to_kcalmol
    
    # Sort by Energy in kcal/mol
    df = df.sort_values(by='Energy (kcal/mol)', ascending=True).reset_index(drop=True)
    
    # Calculate energy differences (dE) in kcal/mol
    df['dE (kcal/mol)'] = df['Energy (kcal/mol)'] - df['Energy (kcal/mol)'].min()
    
    # Calculate Boltzmann weights using the corrected formula
    delta_E_over_RT = df['dE (kcal/mol)'] / (R_kcalmol_K * temperature)
    df['Boltzmann Weight'] = np.exp(-delta_E_over_RT)
    
    # Normalize Boltzmann weights to sum to 1
    total_weight = df['Boltzmann Weight'].sum()
    df['Boltzmann Weight'] /= total_weight  # Normalize to ensure sum equals 1
    
    # Calculate % Total (percentage contribution of each Boltzmann weight)
    df['% Total'] = df['Boltzmann Weight'] * 100
    
    # Calculate cumulative percentages
    df['% Cumulative'] = df['% Total'].cumsum()
    
    # Round values to desired precision
    df = df.round({'Energy (kcal/mol)': precision, 'dE (kcal/mol)': precision, 
                   'Boltzmann Weight': precision, '% Total': precision, 
                   '% Cumulative': precision})
    
    # Add step number as a new column
    df.insert(0, 'Step', step_number)
    
    # Save to CSV
    mode = 'w' if step_number == 1 else 'a'  # Write if first step, append otherwise
    header = step_number == 1  # Write header only for the first step
    df.to_csv(filename, mode=mode, index=False, header=header)
    print(f"Step {step_number} data saved to {filename}.")


def main():
    #Get parse arguments
    args = parse_arguments()
    cores = args.cores
    yaml_input = args.input_file
    skip = args.skip
        # Load the YAML configuration
    with open(yaml_input, 'r') as file:
        config = yaml.safe_load(file)

    steps = config.get('steps', [])
    calculation_functions = ["GOAT","DFT","XTB","MLFF"]

    for step in steps:
        step_number = step['step']
        calculation_type = step['calculation_type']
        sample_method = step['sample_type']['method']
        parameters = step['sample_type']['parameters']

        # Ensure calculation type is valid
        if calculation_type not in calculation_functions:
            raise ValueError(f"Invalid calculation type '{calculation_type}' in step {step_number}. Exiting...")

        # Validate the presence of the input file
        input_file = f"step{step_number}.inp"
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file '{input_file}' not found for step {step_number}. Exiting...")
        
        # Call the respective function for the calculation type
        print(f"Running step {step_number}: {calculation_type} with sampling method '{sample_method}'")

        #Initialize 1st step
        if step_number == 1:
            xyz_file = "step1.xyz"
            inp_file = "step1.inp"
            xyz_filenames = [xyz_file]
            if skip:
                print("Skipping Step 1...")
                coordinates,energies = parse_orca_output(output_files,calculation_type,dir='./step1')
            else:
                input_files,output_files = create_orca_input(xyz_filenames,template=inp_file)
                submit_files(input_files,cores)
                coordinates,energies = parse_orca_output(output_files,calculation_type)
            ids = [i for i in range(1, len(energies) + 1)]
            save_step_csv(energies,ids,step_number)
            filtered_coordinates,filtered_ids = filter_structures(coordinates,energies,ids,sample_method,parameters=parameters) 
            if not skip:
                move_step_files(1)
            continue

        #For loop body
        if not calculation_type == 'MLFF':
            xyz_filenames = write_xyz(filtered_coordinates,step_number,filtered_ids)

            #Create template for ORCA Input
            input_template = f"step{step_number}.inp"
            input_files,output_files = create_orca_input(xyz_filenames,template=input_template)

            #Submit Files
            submit_files(input_files,cores)

            #Parse and filter
            coordinates,energies = parse_orca_output(output_files,calculation_type)
            save_step_csv(energies,filtered_ids,step_number)
            filtered_coordinates, filtered_ids = filter_structures(coordinates,energies,filtered_ids,sample_method,parameters=parameters)

        else:
            #TODO Add MLFF functionality 
            raise ValueError("We are still working on this feature")
        move_step_files(step_number)
        
if __name__ == "__main__":
    main()



import re
from pathlib import Path
import subprocess
import argparse
import os 
import sys 
import time
import getpass
import yaml  # type: ignore
import numpy as np  # type: ignore
import glob
import shutil
import pandas as pd # type: ignore
import logging

# Constants
HARTREE_TO_KCAL_MOL = 2625.49964/4.184  # Conversion factor from Hartrees to kcal/mol
R_KCAL_MOL_K = 8.314462618e-3/4.184  # Gas constant in kcal/(mol*K)
DEFAULT_TEMPERATURE = 298.15  # Temperature in Kelvin
DEFAULT_CORES = 16
DEFAULT_MAX_CORES = 32
DEFAULT_ENERGY_WINDOW = 0.5  # Hartrees
DEFAULT_BOLTZMANN_PERCENTAGE = 99  # Percentage
JOB_CHECK_INTERVAL = 5  # Seconds between job status checks
CSV_PRECISION = 8  # Decimal precision for CSV output

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Code to automate the process of conformer searching, submits initial XTB calculation and improves precision"
    )
    parser.add_argument(
        "input_file",
        help="YAML input file containing instructions for automated workflow",
    )
    parser.add_argument(
        "--maxcores", type=int, default=DEFAULT_CORES,
        help="Max number of cores used throughout conformer search. For multiple files it will be cores/PAL"
    )
    parser.add_argument(
        "--skip", action="store_true", default=False,
        help="Skips the first step because it was already run, and starts with the next steps."
    )
    
    # Parse known and unknown arguments
    args, unknown = parser.parse_known_args()

    return args, unknown


def submit_qorca(input_file, qorca_flags=None):
    """
    Submits the ORCA calculation using qorca, showing output and extracting job info.

    Args:
        input_file (str): Path to the ORCA input file.
        qorca_flags (list, optional): A list of additional flags for qorca.

    Returns:
        tuple: (JobID, PAL_value) of the submitted file.
    """
    # Base command
    qorca_path = os.path.join(os.path.dirname(__file__), "vendor", "qorca", "qorca")
    command = ["python3", qorca_path, input_file]

    if qorca_flags:
        command.extend(qorca_flags)

    logging.info(f"Running qorca command: {' '.join(command)}")

    try:
        # Run the command and show output in real-time
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        
        # Print qorca output so user can see PAL messages
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        # Extract job ID from output
        jobid_match = re.search(r'Submitted job (\d+)', result.stdout)
        if jobid_match:
            jobid = jobid_match.group(1)
        else:
            # Fallback parsing - try to find job ID in last line
            lines = result.stdout.strip().split('\n')
            for line in reversed(lines):
                if line.strip():
                    jobid = line.strip().split()[-1]
                    break
            else:
                raise ValueError("Could not extract job ID from qorca output")
          # Extract PAL value from qorca output (check both stdout and stderr)
        combined_output = result.stdout + "\n" + (result.stderr or "")
        pal_value = extract_pal_from_qorca_output(combined_output)
        
        if pal_value is not None:
            logging.info(f"Extracted PAL value: {pal_value} from qorca output")
        else:
            logging.warning("Could not extract PAL value from qorca output - using default")
            # Debug: show what we're trying to parse
            logging.debug(f"Full qorca output for debugging:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
        
        return jobid, pal_value
        
    except subprocess.CalledProcessError as e:
        # Handle command execution failure
        logging.error(f"qorca command failed with return code {e.returncode}")
        if e.stdout:
            logging.error(f"qorca stdout: {e.stdout}")
        if e.stderr:
            logging.error(f"qorca stderr: {e.stderr}")
            print(e.stderr, file=sys.stderr)
        raise ValueError(f"qorca command failed: {e}")
    except Exception as e:
        logging.error(f"Unexpected error running qorca: {e}")
        raise



def read_input_file(input_path):
    """Read the ORCA input file."""
    if not input_path.is_file():
        logging.info(
            f"Error: Input file '{input_path}' does not exist. "
            "Please check the file path and try again.",
            file=sys.stderr
        )
        sys.exit(1)
    with input_path.open('r') as f:
        lines = f.readlines()
    return lines

# Removed parse_pal_from_input - now handled by qorca

def extract_pal_from_qorca_output(qorca_output):
    """
    Extract final PAL value from qorca output.
    
    Args:
        qorca_output (str): Output from qorca command
        
    Returns:
        int or None: The final PAL value used, or None if not found
    """
    # Look for qorca's PAL adjustment messages (order matters - more specific first)
    patterns = [
        # Message when qorca overwrites PAL value in input file (most common)
        r"Input file '[^']*' overwritten with specified PAL value \((\d+)\)",
        r"Input file '[^']*' overwritten with adjusted PAL value \((\d+)\)",
        
        # Message when PAL is already correct
        r"PAL value in '[^']*' is already set to (\d+)",
        
        # Command line specification patterns
        r"-p\s+(\d+)",
        r"--pal\s+(\d+)",
        
        # SLURM resource allocation messages
        r"#SBATCH --ntasks=(\d+)",
        r"Number of CPUs \((\d+)\)",
        
        # General PAL patterns
        r"nprocs\s+(\d+)",
        r"PAL.*?(\d+)",
        r"pal(\d+)",
        
        # Fallback patterns
        r"Using (\d+) cores",
        r"cores?\s*[=:]\s*(\d+)",
    ]
    
    for pattern in patterns:
        pal_match = re.search(pattern, qorca_output, re.IGNORECASE | re.MULTILINE)
        if pal_match:
            return int(pal_match.group(1))
    
    return None

def submit_goat(input_file, qorca_flags=None):
    """
    Submit a single file for calculation and wait until the job finishes.

    Args:
        input_file (str): Path to the input file to be submitted.
        qorca_flags (list, optional): Additional flags to pass to qorca.
    """
    # Submit the job and get the job ID
    jobid, pal_value = submit_qorca(input_file, qorca_flags=qorca_flags)
    logging.info(f"Job {jobid} submitted for file {input_file}. Waiting for it to finish...")

    # Check if the job is still running
    while not is_job_finished(jobid):
        logging.info(f"Job {jobid} is still running. Checking again in {JOB_CHECK_INTERVAL} seconds...")
        time.sleep(JOB_CHECK_INTERVAL)  # Wait before checking again

    logging.info(f"Job {jobid} for file {input_file} has finished.")


def get_input_dir(input_file):
    # Get directory from input file
    input_dir = os.path.dirname(input_file)
    
    # If no directory is specified, default to the current working directory
    if not input_dir:  # Empty string evaluates to False
        input_dir = os.getcwd()
    
    return input_dir

def create_orca_input(xyz_files, template, charge, multiplicity, output_dir='./'):
    input_files = []
    output_files = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logging.info(f"Writing ORCA input files")    
    
    for file in xyz_files:
        base_name = os.path.splitext(os.path.basename(file))[0]
        input_file = os.path.join(output_dir, f"{base_name}.inp")
        output_file = os.path.join(output_dir, f"{base_name}.out")
        input_files.append(input_file)
        output_files.append(output_file)
        
        with open(template, "r") as tmpl:
            content = tmpl.read()
        
        # Remove any existing * xyzfile lines (case insensitive)
        # This pattern matches lines that start with * and contain xyzfile
        content = re.sub(r'^\s*\*\s+xyzfile.*$', '', content, flags=re.MULTILINE | re.IGNORECASE)
        
        # Clean up any multiple consecutive newlines left by removal
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        
        # Ensure content ends with proper spacing before adding xyzfile line
        content = content.rstrip() + '\n\n'
        
        # Add the new * xyzfile line for this specific XYZ file
        formatted_content = content + f"* xyzfile {charge} {multiplicity} {file}\n\n"

        # Write the formatted content to the input file
        with open(input_file, "w") as inp:
            inp.write(formatted_content)
        
        logging.info(f"Created {input_file} with charge {charge}, multiplicity {multiplicity}, and XYZ file {file}")
    
    return input_files, output_files
        
def submit_files(input_files, max_cores=DEFAULT_MAX_CORES, qorca_flags=None):
    """
    Submit multiple calculations based on available cores, ensuring the number of running jobs 
    does not exceed the specified max_cores. Submissions are sequential, and the function waits 
    for cores to free up before submitting more.
    
    Args:
        input_files (list): List of input file paths for calculations.
        max_cores (int): Maximum number of cores to use simultaneously.
        qorca_flags (list, optional): Additional flags to pass to qorca.
    """
    total_cores_used = 0
    active_jobs = {}  # Map of job_id to cores used for tracking active jobs
    
    for input_file in input_files:
        try:
            # We don't pre-parse PAL since qorca will handle it and potentially modify it
            # For now, assume a default core allocation that will be corrected by qorca output
            cores_needed = DEFAULT_CORES  # Default assumption
            
            # Wait if there aren't enough free cores to submit the next job
            while total_cores_used + cores_needed > max_cores:
                logging.info("Waiting for jobs to finish to free up cores...")
                  # Check active jobs and remove completed ones
                completed_jobs = []
                for job_id, cores in active_jobs.items():
                    logging.info(f"Job ID {job_id} is running with {cores} cores")
                    if is_job_finished(job_id):
                        completed_jobs.append(job_id)
                        total_cores_used -= cores
                
                # Remove completed jobs from active_jobs
                for job_id in completed_jobs:
                    del active_jobs[job_id]
                
                time.sleep(JOB_CHECK_INTERVAL)            # Submit the job and get actual PAL value used
            logging.info(f"Submitting job for {input_file}...")
            job_id, actual_pal_value = submit_qorca(input_file, qorca_flags=qorca_flags)
            
            # Use actual PAL value if available, otherwise use default
            actual_cores = actual_pal_value if actual_pal_value is not None else cores_needed
            active_jobs[job_id] = actual_cores
            total_cores_used += actual_cores
            
            logging.info(f"Job {job_id} submitted using {actual_cores} cores (PAL value: {actual_pal_value})")
            logging.info(f"Total cores in use: {total_cores_used}/{max_cores}")

        except Exception as e:
            logging.error(f"Error processing input file {input_file}: {e}")
            continue    # Wait for all remaining jobs to finish
    logging.info("All jobs submitted. Waiting for remaining jobs to complete...")
    while active_jobs:
        completed_jobs = []
        for job_id, cores in active_jobs.items():
            if is_job_finished(job_id):
                completed_jobs.append(job_id)
                total_cores_used -= cores

        for job_id in completed_jobs:
            del active_jobs[job_id]

        time.sleep(JOB_CHECK_INTERVAL)

    logging.info("All calculations finished.")


def is_job_finished(job_id):
    """
    Check if a SLURM job with a given job ID has finished.
    
    Parameters:
    - job_id (str): The SLURM job ID to check.
    
    Returns:
    - bool: True if the job is not in the queue (finished), False otherwise.
    """
    try:
        # Get the current username using getpass (more efficient than subprocess)
        username = getpass.getuser()
        
        # Use modern subprocess approach with list of arguments (no shell=True)
        command = ["squeue", "-u", username, "-o", "%i", "-h"]
        
        # Execute the command and capture the output
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True,  # Python 3.7+ (returns string instead of bytes)
            timeout=30,  # Prevent hanging
            check=True   # Raise CalledProcessError on non-zero exit
        )
        
        # Split the output into lines and check if job_id is present
        job_ids = result.stdout.strip().splitlines()
        
        # No header to skip since we used -h flag
        return job_id not in job_ids  # True if job finished, False if still running
        
    except subprocess.CalledProcessError as e:
        logging.warning(f"Error running squeue command: {e}")
        return True  # Assume finished on error to avoid infinite loops
    except subprocess.TimeoutExpired:
        logging.warning(f"squeue command timed out for job {job_id}")
        return True  # Assume finished on timeout
    except Exception as e:
        logging.warning(f"Unexpected error checking job {job_id}: {e}")
        return True

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
        # Ensure only the base name is appended to the directory
        base_file_name = os.path.basename(file_name)
        file_path = os.path.join(dir, base_file_name)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

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
            - 'energy_window': `energy` (float) - Energy window from the lowest energy.
            - 'boltzmann': `weight` (float) - Percentage of Boltzmann probability.
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
    parameters = kwargs.get('parameters', {})
    energies = np.array([float(e) for e in energies])
    sorted_indices = np.argsort(energies)  # Sort energies to determine favored structures

    if method == 'energy_window':
        logging.info("Filtering structures based on energy window.")
        energy = parameters.get('energy', DEFAULT_ENERGY_WINDOW)  # Default is 0.5 Hartrees
        unit = parameters.get('unit', 'hartree')  # Assume 'hartree' if no unit is specified
        logging.info(f"Filtering Energy window: {energy} {unit}")
        if unit.lower() == 'kcal/mol':
            energy /= HARTREE_TO_KCAL_MOL  # Convert kcal/mol to Hartrees
            logging.info(f"Converted energy window to Hartrees: {energy:.6f}")

        min_energy = np.min(energies)
        favored_indices = [i for i in sorted_indices if energies[i] <= min_energy + energy]

    elif method == 'boltzmann':
        logging.info("Filtering structures based on Boltzmann probability.")

        if len(energies) == 0:
            logging.warning("No structures available for Boltzmann filtering. Returning empty lists.")
            return [], []

        if len(energies) == 1:
            logging.info("Only one structure available; returning it.")
            return coordinates_list, id_list
        
        # Constants
        temperature = DEFAULT_TEMPERATURE
        percentage = parameters.get('weight', DEFAULT_BOLTZMANN_PERCENTAGE)  # User-specified probability threshold
        logging.info(f"Filtering Boltzmann probability: {percentage}%")

        # Convert energies from Hartrees to kcal/mol
        energies_kcalmol = energies * HARTREE_TO_KCAL_MOL

        if energies_kcalmol.size == 0:
            logging.warning("No energies available after conversion. Returning empty results.")
            return [], []

        # Sort energies and retain original indices
        sorted_indices = np.argsort(energies_kcalmol)
        sorted_energies = energies_kcalmol[sorted_indices]

        # Compute energy differences (relative to minimum energy)
        min_energy = np.min(sorted_energies)
        delta_E = sorted_energies - min_energy

        # Compute Boltzmann weights
        boltzmann_weights = np.exp(-delta_E / (R_KCAL_MOL_K * temperature))
        boltzmann_probs = boltzmann_weights / np.sum(boltzmann_weights)

        # Compute cumulative probabilities
        cumulative_probs = np.cumsum(boltzmann_probs)

        # Define cutoff probability based on user input
        cutoff_prob = percentage / 100.0

        # Find indices where cumulative probability is within the threshold
        favored_indices_sorted = [i for i, prob in enumerate(cumulative_probs) if prob <= cutoff_prob]

        # Ensure at least the first structure exceeding the cutoff is included
        if favored_indices_sorted and favored_indices_sorted[-1] < len(cumulative_probs) - 1:
            favored_indices_sorted.append(favored_indices_sorted[-1] + 1)

        # Map back to original indices
        favored_indices = sorted_indices[favored_indices_sorted]

        # Apply selection mask
        mask = [i in favored_indices for i in range(len(coordinates_list))]
        filtered_coordinates = [coord for coord, keep in zip(coordinates_list, mask) if keep]
        filtered_ids = [id_ for id_, keep in zip(id_list, mask) if keep]

        logging.info(f"Selected {len(filtered_coordinates)} structures based on '{method}' method.")

    elif method == 'integer':
        logging.info("Filtering structures based on integer count.")
        num_structures = parameters.get('num_structures')
        logging.info("Number of structures to select: %d", num_structures)
        num_structures = min(num_structures, len(coordinates_list))  # Ensure we don't exceed the list length
        if num_structures <= 0 or num_structures >= len(coordinates_list):
            logging.info("Your input is either 0 or bigger than the total structures, taking all of the structures.")  # If num_structures is larger than or equal to the list size
            favored_indices = sorted_indices  # Return all indices (sorted by energy)
        else:
                favored_indices = sorted_indices[:num_structures]  # Return top 'num_structures' based on energy sorting
    else:
        raise ValueError("Invalid method. Choose from 'energy_window', 'boltzmann', or 'integer'.")

    # Create a mask to preserve only the favored structures while maintaining original order
    mask = [i in favored_indices for i in range(len(coordinates_list))]
    filtered_coordinates = [coord for coord, keep in zip(coordinates_list, mask) if keep]
    filtered_ids = [id_ for id_, keep in zip(id_list, mask) if keep]
    logging.info(f"Selected {len(filtered_coordinates)} structures based on '{method}' method.")
    
    return filtered_coordinates, filtered_ids

def write_xyz(structures, step_number, structure_ids):
    logging.info("Writing Ensemble XYZ files")
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
    Excludes .inp template files which should remain in the working directory.
    If a file already exists in the destination, rename the existing file by adding a prefix 'old_'.

    Parameters:
        step_number (int): The step number whose files need to be moved.
    """
    step_dir = f"step{step_number}"

    # Create the directory if it doesn't exist
    os.makedirs(step_dir, exist_ok=True)

    if step_number == 1: 
        # Get all files starting with step1, but exclude .inp template files
        files_to_move = [f for f in glob.glob(f"step{step_number}*") 
                        if os.path.isfile(f) and not f.endswith('.inp')]
    else: 
        # For other steps, only move structure files
        files_to_move = glob.glob(f"step{step_number}_structure*")
    
    # Move each file to the directory
    for file in files_to_move:
        try:
            dest_path = os.path.join(step_dir, os.path.basename(file))
            
            # Check if the file exists in the destination
            if os.path.exists(dest_path):
                old_path = os.path.join(step_dir, f"old_{os.path.basename(file)}")
                os.rename(dest_path, old_path)  # Rename the existing file with 'old_' prefix
            
            shutil.move(file, dest_path)
        except Exception as e:
            logging.error(f"There was an error moving the file {file}: {e}")
    
    logging.info(f"Step {step_number} files organized.")

def save_step_csv(energies, ids, step_number, temperature=DEFAULT_TEMPERATURE, filename="steps.csv", precision=CSV_PRECISION):
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
    # Convert inputs to DataFrame
    df = pd.DataFrame({'Conformer': ids, 'Energy (Hartrees)': energies})
    
    # Ensure Energy column is numeric
    df['Energy (Hartrees)'] = pd.to_numeric(df['Energy (Hartrees)'], errors='coerce')
    if df['Energy (Hartrees)'].isnull().any():
        raise ValueError("Non-numeric or missing energy values found. Please clean the input data.")
    
    # Convert energy to kcal/mol
    df['Energy (kcal/mol)'] = df['Energy (Hartrees)'] * HARTREE_TO_KCAL_MOL
    
    # Sort by Energy in kcal/mol
    df = df.sort_values(by='Energy (kcal/mol)', ascending=True).reset_index(drop=True)
    
    # Calculate energy differences (dE) in kcal/mol
    df['dE (kcal/mol)'] = df['Energy (kcal/mol)'] - df['Energy (kcal/mol)'].min()
    
    # Calculate Boltzmann weights using the corrected formula
    delta_E_over_RT = df['dE (kcal/mol)'] / (R_KCAL_MOL_K * temperature)
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
    logging.info(f"Step {step_number} data saved to {filename}.")

def check_if_dir_and_skip_step(step_number):
    """
    If the directory of the step exists skip this step and continue to the next step.
    
    Parameters:
        step_number (int): The step number to skip.
    """
    # Move all files starting with 'step{step_number}' to a directory named 'step{step_number}'
    
    # Log the skipped step
    logging.info(f"Step {step_number} skipped. Files moved to 'step{step_number}' directory.")
    if os.path.exists(f"step{step_number}"):
        output_files = [os.path.join(f"step{step_number}", file) for file in os.listdir(f"step{step_number}") if file.endswith('.out')]
        return output_files
    else:
        return False
    
def main():
    #Get parse arguments
    args, qorca_flags = parse_arguments()
    cores = args.maxcores
    yaml_input = args.input_file
    skip = args.skip
    
    # Log CLI integration information
    if qorca_flags:
        logging.info(f"Passing qorca flags: {' '.join(qorca_flags)}")
    else:
        logging.info("No additional qorca flags specified")
        
    # Load the YAML configuration
    with open(yaml_input, 'r') as file:
        config = yaml.safe_load(file)

    steps = config.get('steps', [])
    charge = config.get('charge')
    multiplicity = config.get('multiplicity')
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
        if not skip:
            input_file = f"step{step_number}.inp"
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Input file '{input_file}' not found for step {step_number}. Exiting...")
        
        # Call the respective function for the calculation type
        logging.info(f"Running step {step_number}: {calculation_type} with sampling method '{sample_method}'")

        #Initialize 1st step
        if step_number == 1:
            xyz_file = "step1.xyz"
            inp_file = "step1.inp"
            xyz_filenames = [xyz_file]
            if skip:
                output_files=check_if_dir_and_skip_step(step_number)
                logging.info("Skipping Step 1...")
                coordinates,energies = parse_orca_output(output_files,calculation_type,dir='./step1')
            else:
                input_files,output_files = create_orca_input(xyz_filenames,template=inp_file,charge=charge,multiplicity=multiplicity)
                submit_files(input_files,cores,qorca_flags=qorca_flags)
                coordinates,energies = parse_orca_output(output_files,calculation_type)
            ids = [i for i in range(0, len(energies))]
            save_step_csv(energies,ids,step_number)
            filtered_coordinates,filtered_ids = filter_structures(coordinates,energies,ids,sample_method,parameters=parameters) 
            if not skip:
                move_step_files(1)
            continue

        #For loop body
        if not calculation_type == 'MLFF':
            if skip and check_if_dir_and_skip_step(step_number):
                output_files=check_if_dir_and_skip_step(step_number)
                coordinates,energies = parse_orca_output(output_files,calculation_type,dir=f'./step{step_number}')
                continue
            else:
                xyz_filenames = write_xyz(filtered_coordinates,step_number,filtered_ids)

                #Create template for ORCA Input
                input_template = f"step{step_number}.inp"
                input_files,output_files = create_orca_input(xyz_filenames,template=input_template,charge=charge,multiplicity=multiplicity)

                #Submit Files
                submit_files(input_files,cores,qorca_flags=qorca_flags)

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



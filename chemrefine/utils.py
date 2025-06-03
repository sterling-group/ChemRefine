import re

import glob
import shutil
import logging
from .constants import HARTREE_TO_KCAL_MOL, R_KCAL_MOL_K, CSV_PRECISION

class Utility:
    def extract_pal_from_qorca_output(self, output):
        patterns = [
            r"PAL.*?(\d+)", r"nprocs\s+(\d+)", r"--pal\s+(\d+)", r"pal(\d+)", r"Using (\d+) cores"
        ]
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    def save_step_csv(self, energies, ids, step, filename="steps.csv", T=298.15, output_dir="."):
        """
        Save step-wise energies and Boltzmann weights to a CSV file.

        Parameters:
        - energies (list): List of energy values in Hartrees.
        - ids (list): List of structure IDs.
        - step (int): Current step number.
        - filename (str): Output CSV file name (default: steps.csv).
        - T (float): Temperature in Kelvin for Boltzmann weighting.
        - output_dir (str): Directory to save the CSV file.
        """
        import os
        import pandas as pd
        import numpy as np

        # Calculate energies
        df = pd.DataFrame({'Conformer': ids, 'Energy (Hartrees)': energies})
        df['Energy (kcal/mol)'] = df['Energy (Hartrees)'] * HARTREE_TO_KCAL_MOL
        df['dE (kcal/mol)'] = df['Energy (kcal/mol)'] - df['Energy (kcal/mol)'].min()
        dE_RT = df['dE (kcal/mol)'] / (R_KCAL_MOL_K * T)
        df['Boltzmann Weight'] = np.exp(-dE_RT)
        df['Boltzmann Weight'] /= df['Boltzmann Weight'].sum()
        df['% Total'] = df['Boltzmann Weight'] * 100
        df['% Cumulative'] = df['% Total'].cumsum()
        df.insert(0, 'Step', step)
        df = df.round(CSV_PRECISION)

        # Construct output path
        output_path = os.path.join(output_dir, filename)
        os.makedirs(output_dir, exist_ok=True)

        # Write CSV
        mode = 'w' if step == 1 else 'a'
        header = step == 1
        df.to_csv(output_path, mode=mode, index=False, header=header)
        logging.info(f"Saved CSV for step {step} to {output_path}")


    def move_step_files(self, step_number):
        """
        Moves all files starting with 'step{step_number}' into a dedicated directory.

        Parameters:
        - step_number (int): The step number to organize files for.
        """
        step_dir = f"step{step_number}"
        os.makedirs(step_dir, exist_ok=True)

        files = glob.glob(f"step{step_number}*")
        for file in files:
            dest = os.path.join(step_dir, os.path.basename(file))
            if os.path.exists(dest):
                os.rename(dest, os.path.join(step_dir, f"old_{os.path.basename(file)}"))
            shutil.move(file, dest)


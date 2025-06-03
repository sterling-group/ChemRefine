#!/bin/bash
#SBATCH --partition=your_partition
#SBATCH --time=24:00:00
#SBATCH --mem=4G

# Load modules, set environment variables
module purge
module load orca

# Set ORCA executable path (user can override)
ORCA_EXEC=/path/to/orca

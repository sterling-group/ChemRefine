#!/bin/bash
#SBATCH --partition=<your_partition>
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G              # Limit memory to allow multiple jobs on the same node
#SBATCH --time=72:00:00
#SBATCH --exclude=compute-2-09-05
#SBATCH --job-name=auto_goat
#SBATCH --output=%x_%j.out   # Saves output to auto_goat_JOBID.out
#SBATCH --error=%x_%j.err    # Saves error log

# Ensure the script allows for shared node usage
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the calculation
python -u <path>/auto_goat.py -c 256 input.yaml


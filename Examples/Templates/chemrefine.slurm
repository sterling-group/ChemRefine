#!/bin/bash
#SBATCH --partition=<your_partition>
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G              # Limit memory to allow multiple jobs on the same node
#SBATCH --time=72:00:00
#SBATCH --exclude=g-07-02
#SBATCH --job-name=conformer_search
#SBATCH --output=%x.out   # Saves output to auto_goat_JOBID.out
#SBATCH --error=%x.err    # Saves error log

# Ensure the script allows for shared node usage
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the calculation
chemrefine input.yaml --maxcores 480 

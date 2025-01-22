#!/bin/bash
#SBATCH     --partition=sterling
#SBATCH     --nodes=1
#SBATCH     --ntasks= 1
#SBATCH     --time=72:00:00
#SBATCH     --cpus-per-task: 1

python -u <path_of_code>/auto_goat.py -c 256 input.yaml


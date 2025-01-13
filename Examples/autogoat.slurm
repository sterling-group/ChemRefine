#!/bin/bash
#SBATCH     --partition=sterling
#SBATCH     --nodes=1
#SBATCH     --ntasks=32
#SBATCH     --time=72:00:00

module activate autogoat
python <path_of_code>/auto_goat.py -c 256 input.yaml


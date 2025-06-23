#!/bin/bash


#source ~/miniconda3/etc/profile.d/conda.sh
#conda activate chemrefine  # or your correct env name

python -m chemrefine.client "$@"


# Conformer Sampling Tutorial

--8<-- "docs/_includes/schema-note.md"


This tutorial demonstrates how to use **ChemRefine** for conformational sampling with an initial **global geometry optimization (GOAT)** and ensemble generation.

## Overview

Conformer sampling is the first step in exploring molecular flexibility and generating diverse geometries.  
ChemRefine automates this process by running a **global optimization** followed by ensemble generation, producing a set of candidate structures for further refinement.

The workflow:

1. **Global Optimization (GOAT):**  
   Performs a stochastic search of the potential energy surface (PES) to identify low-energy conformers.  
2. **Ensemble Generation:**  
   Collects the lowest-energy structures into an ensemble for downstream calculations (e.g., DFT, MLIP).
3. **Level of theory benchmarking:**   
   We're going to refine the level of theory starting from simple GFN2-xTB, UMA-S-1, PBE-D4, ωB97X-D4, B2PLYP 

---

## Prerequisites

- Installed **ChemRefine** (see [Installation Guide](../get-started/install.md))  
- Access to an **ORCA executable**  
- Example molecule and YAML input from the repository  

---

## Input Files

For this tutorial, we will use **Pd(PPh₃)₄**.

- 📄 [View Input YAML](https://github.com/sterling-group/ChemRefine/blob/main/examples/tutorials/conformational_sampling/input.yaml)  
- 📄 [View step1.xyz](https://github.com/sterling-group/ChemRefine/blob/main/examples/tutorials/conformational_sampling/step1.xyz)  

## Orca Input Files

You can find the ORCA input files [here](https://github.com/sterling-group/ChemRefine/tree/main/examples/tutorials/conformational_sampling/templates)

### Interactive 3D Viewer

<div id="viewer" data-xyz="examples/tutorials/conformational_sampling/step1.xyz"
     style="width: 100%; height: 400px; position: relative;"></div>

--8<-- "docs/_includes/viewer.md"



---


## YAML Configuration

The YAML input for conformer sampling is also included in the tutorial folder:

➡️ [examples/tutorials/conformational_sampling/input.yaml](https://raw.githubusercontent.com/sterling-group/ChemRefine/main/examples/tutorials/conformational_sampling/input.yaml)



This is the shipped config, included verbatim — the same file `tests/test_examples.py` validates on every CI run:

```yaml
--8<-- "examples/tutorials/conformational_sampling/input.yaml"
```
## How to Run

Before running ChemRefine, ensure that:

- The **ChemRefine Enviroment** is activated
- The **ORCA executable** is installed and available in your `PATH`  
- The **template directory** (`./templates/`) is correctly set up  
- The **input structure file** (e.g., `input.xyz`) is prepared  

### Option 1: Run from the Command Line

You can launch ChemRefine directly from the command line:

```bash
chemrefine run input.yaml --maxcores <N>
```

Here N is the max number of simultaneous cores you want to use.

### Option 2: Run ChemRefine with SLURM script

On HPC systems with SLURM, you can submit ChemRefine as a batch job.
A ready-to-use SLURM script template is available at:


```bash
#!/bin/bash
#SBATCH --partition=<your_partition>
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G              # Limit memory to allow multiple jobs on the same node
#SBATCH --time=72:00:00
#SBATCH --exclude=g-07-02
#SBATCH --job-name=conformer_search
#SBATCH --output=%x.out   
#SBATCH --error=%x.err    # Saves error log

# Ensure the script allows for shared node usage
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the calculation
chemrefine run input.yaml --maxcores 480 
```

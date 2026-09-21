# MLIP Training Tutorial

--8<-- "docs/_includes/schema-note.md"


This tutorial demonstrates how to use **ChemRefine** to train a **Machine Learning Interatomic Potential (MLIP)** using DFT data generated during the workflow.

## Overview

Training an MLIP involves generating reference data, running the training process, and validating the trained model on new configurations.  
ChemRefine automates this multi-step process:

1. **Global Optimization (GOAT)**  
   Performs a global search of the PES to identify low-energy conformers.  

2. **Normal Mode Sampling (NMS)**  
   Generates additional diverse geometries by displacing atoms along vibrational modes (`nms: true`, `target: random`).  

3. **Reference DFT Optimizations (OPT+SP)**  
   Provides high-quality energies and forces for MLIP training.  

4. **MLIP Training (`mlip-train`)**  
   Fine-tunes a potential on the generated DFT dataset. `task_name` picks which library
   trains — the same word that picks the one that runs — and `model_name` is the foundation
   model it starts from (unset means training from scratch); `task_name` and `device` are
   required. Which backends train is the
   [backends table](../engines/installing.md#available-backends)'s business — it is
   generated from the registry, so it cannot go stale the way a list here would. This
   tutorial trains MACE; see
   [`examples/tutorials/fairchem_finetune`](https://github.com/sterling-group/ChemRefine/tree/main/examples/tutorials/fairchem_finetune)
   for the FAIRChem counterpart. Adding another library is one dropped-in module under
   `engines/mlip/backends/`, beside that library's calculator and sharing its one
   environment declaration.
   The `step4.yaml` template is MACE's own training config with `$PLACEHOLDERS` where the
   dataset paths go ([MACE docs](https://github.com/ACEsuit/mace)). ChemRefine writes the
   dataset with MACE's own label keys and carries the system's charge and multiplicity into
   it, so no `energy_key` / `forces_key` line is needed.

   The structures pass through unchanged — the model is the artifact — so step 5 receives the
   whole ensemble. The model lands at `outputs/step4/train/train_stagetwo.model`, a path you
   can write into step 5 before the training has run.

5. **MLIP Validation (`opt_sp` with `mlip-extopt`)**  
   Applies the trained model to evaluate new structures, testing its accuracy and efficiency.  

---

## Prerequisites

- Installed **ChemRefine** (see [Installation Guide](../get-started/install.md))  
- Access to an **ORCA executable** (for DFT reference calculations)  
- Example molecule and YAML input from the repository  

---

## Input Files

We start with an initial structure located in the templates folder:

- 📄 [View input.yaml](https://github.com/sterling-group/ChemRefine/blob/main/examples/tutorials/mlip_training/input.yaml)  
- 📄 [View step1.xyz](https://github.com/sterling-group/ChemRefine/blob/main/examples/tutorials/mlip_training/step1.xyz)  

## Orca Input Files

You can find the ORCA input files [here](https://github.com/sterling-group/ChemRefine/tree/main/examples/tutorials/mlip_training/templates)

---

### Interactive 3D Viewer

<!-- chemrefine:structure examples/tutorials/mlip_training/step1.xyz -->




## YAML Configuration

The full YAML input for this MLIP training workflow is included:

➡️ [examples/tutorials/mlip_training/input.yaml](https://raw.githubusercontent.com/sterling-group/ChemRefine/main/examples/tutorials/mlip_training/input.yaml)

Download the template files [here](https://github.com/sterling-group/ChemRefine/tree/main/examples/tutorials/mlip_training/templates)

This is the shipped config, included verbatim — the same file `tests/test_examples.py` validates on every CI run:

```yaml
--8<-- "examples/tutorials/mlip_training/input.yaml"
```

---

## How to Run

Before running ChemRefine, ensure that:

- The **ChemRefine environment** is activated  
- The **ORCA executable** path is correct  
- The **template directory** (`./templates/`) contains the initial structure  
- The YAML config matches your dataset and workflow  

### Option 1: Run from the Command Line

```bash
chemrefine run input.yaml --maxcores <N>
```

Here N is the number of simultaneous cores you want to use.

### Option 2: Run with SLURM Script

On HPC systems with SLURM, submit the training workflow as a batch script:


```bash
#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --job-name=mlip_training
#SBATCH --output=%x.out
#SBATCH --error=%x.err

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

chemrefine run input.yaml --maxcores 8
```

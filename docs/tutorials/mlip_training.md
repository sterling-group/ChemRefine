# MLIP Training Tutorial

!!! note "Schema note"
    The YAML excerpts on this page are abbreviated for illustration. For the authoritative schema (`sample:`, `input:`, `options:` blocks, …) see the [main schema page](../index.md) and the example in [Examples/input.yaml](https://github.com/sterling-group/ChemRefine/blob/main/Examples/input.yaml).


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
   Trains a potential (e.g., MACE) on the generated DFT dataset. As of writing, ChemRefine can only train/finetune with MACE. We need a MACE input yaml; an explanation can be found [here](https://github.com/ACEsuit/mace).

5. **MLIP Validation (`opt_sp` with `mlip-extopt`)**  
   Applies the trained model to evaluate new structures, testing its accuracy and efficiency.  

---

## Prerequisites

- Installed **ChemRefine** (see [Installation Guide](../user-guide/installation.md))  
- Access to an **ORCA executable** (for DFT reference calculations)  
- Example molecule and YAML input from the repository  

---

## Input Files

We start with an initial structure located in the templates folder:

- 📄 [View input.yaml](https://github.com/sterling-group/ChemRefine/blob/main/Examples/Tutorials/MLIPTraining/input.yaml)  
- 📄 [View step1.xyz](https://github.com/sterling-group/ChemRefine/blob/main/Examples/Tutorials/MLIPTraining/step1.xyz)  

## Orca Input Files

You can find the ORCA input files [here](https://github.com/sterling-group/ChemRefine/tree/main/Examples/Tutorials/MLIPTraining/templates)

---

### Interactive 3D Viewer

<div id="viewer" style="width: 100%; height: 400px; position: relative;"></div>

<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<script>
  let viewer = $3Dmol.createViewer("viewer", { backgroundColor: "white" });

  fetch("https://raw.githubusercontent.com/sterling-group/ChemRefine/main/Examples/Tutorials/MLIPTraining/step1.xyz")
    .then(r => r.text())
    .then(data => {
      viewer.addModel(data, "xyz");   // force XYZ format
      viewer.setStyle({}, {stick:{radius:0.15}, sphere:{scale:0.25}});
      viewer.zoomTo();
      viewer.render();
    })
    .catch(err => console.error("Could not load XYZ:", err));
</script>




## YAML Configuration

The full YAML input for this MLIP training workflow is included:

➡️ [Examples/Tutorials/MLIPTraining/input.yaml](https://raw.githubusercontent.com/sterling-group/ChemRefine/main/Examples/Tutorials/MLIPTraining/input.yaml)

Download the template files [here](https://github.com/sterling-group/ChemRefine/tree/main/Examples/Tutorials/MLIPTraining/templates)

Example content:

```yaml
template_dir: ./templates
scratch_dir: /scratch/
output_dir: ./outputs
executables: { orca: /orca/orca_6_1_0_avx2/orca }

charge: 0
multiplicity: 1

input: ./templates/step1.xyz

steps:
  - step: 1
    operation: goat
    engine: orca
    sample: { method: min, count: 15 }

  # Augment the dataset with normal-mode-sampled geometries.
  - step: 2
    operation: opt_sp
    engine: orca
    nms: true
    options: { target: random, displacement_value: 1.0, num_random_displacements: 1 }
    sample: { method: min, count: 0 }

  # DFT labels (energies + forces) for training.
  - step: 3
    operation: opt_sp
    engine: orca
    sample: { method: min, count: 0 }

  # Train a MACE model on the labelled structures.
  - step: 4
    engine: mlip-train
    operation: mlip_train
    sample: { method: min, count: 0 }

  # Validate the trained model via the MLIP gradient server.
  - step: 5
    operation: opt_sp
    engine: mlip-extopt
    options:
      model_name: ../step4/checkpoints_dir/goat_model_run-123_stagetwo.model
      task_name: mace_off
      device: cuda
    sample: { method: min, count: 0 }
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

➡️ [Example ChemRefine SLURM script](https://raw.githubusercontent.com/sterling-group/ChemRefine/main/Examples/Templates/chemrefine.slurm)

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

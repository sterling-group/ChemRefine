# MLIP Training Tutorial

This tutorial demonstrates how to use **ChemRefine** to train a **Machine Learning Interatomic Potential (MLIP)** using DFT data generated during the workflow.

## Overview

Training an MLIP involves generating reference data, running the training process, and validating the trained model on new configurations.  
ChemRefine automates this multi-step process:

1. **Global Optimization (GOAT)**  
   Performs a stochastic search of the PES to identify low-energy conformers.  

2. **Normal Mode Sampling (NMS)**  
   Generates additional diverse geometries by displacing atoms along vibrational modes.  

3. **Reference DFT Optimizations (OPT+SP)**  
   Provides high-quality energies and forces for MLIP training.  

4. **MLIP Training (MLFF_TRAIN)**  
   Trains a potential (e.g., MACE) on the generated DFT dataset.  

5. **MLIP Validation (OPT+SP with MLFF)**  
   Applies the trained model to evaluate new structures, testing its accuracy and efficiency.  

---

## Prerequisites

- Installed **ChemRefine** (see [Installation Guide](./installation.md))  
- Access to an **ORCA executable** (for DFT reference calculations)  
- Example molecule and YAML input from the repository  

---

## Input Files

We start with an initial structure located in the templates folder:

- 📄 [View step1.xyz](https://github.com/sterling-group/ChemRefine/blob/mkdocs/Examples/Tutorials/MLIP-Training/step1.xyz)  
- 📥 [Download step1.xyz](https://raw.githubusercontent.com/sterling-group/ChemRefine/mkdocs/Examples/Tutorials/MLIP-Training/step1.xyz)

---

### Interactive 3D Viewer

<div id="viewer" style="width: 100%; height: 400px; position: relative;"></div>

<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<script>
  let viewer = $3Dmol.createViewer("viewer", { backgroundColor: "white" });

  fetch("https://raw.githubusercontent.com/sterling-group/ChemRefine/mkdocs/Examples/Tutorials/MLIP-Training/step1.xyz")
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

➡️ [Examples/Tutorials/MLIP-Training/input.yaml](https://raw.githubusercontent.com/sterling-group/ChemRefine/mkdocs/Examples/Tutorials/MLIP-Training/input.yaml)

Example content:

```yaml
orca_executable: /mfs/io/groups/sterling/software-tools/orca/orca_6_1_0_avx2/orca
charge: 0
multiplicity: 1

initial_xyz: ./templates/step1.xyz

steps:
  - step: 1
    operation: "GOAT"
    engine: "DFT"
    sample_type:
      method: "integer"
      parameters:
       num_structures: 15

  - step: 2
    operation: "OPT+SP"
    engine: "DFT"
    normal_mode_sampling: True
    normal_mode_sampling_parameters:
      calc_type: "random"
      displacement_vector: 1.0
      num_random_displacements: 1
    sample_type:
      method: "integer"
      parameters:
       num_structures: 0

  - step: 3
    operation: "OPT+SP"
    engine: "DFT"
    sample_type:
      method: "integer"
      parameters:
        num_structures: 0

  - step: 4
    operation: "MLFF_TRAIN"
    sample_type:
      method: "integer"
      parameters:
        num_structures: 0

  - step: 5
    operation: "OPT+SP"
    engine: "MLFF"
    mlff:
      model_name: "../step3/checkpoints_dir/goat_model_run-123_stagetwo.model"
      task_name: "mace_off"
      device: "cuda"
    sample_type:
      method: "integer"
      parameters:
       num_structures: 0
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
chemrefine input.yaml --maxcores <N>
```

Here N is the number of simultaneous cores you want to use.

### Option 2: Run with SLURM Script

On HPC systems with SLURM, submit the training workflow as a batch script:

➡️ [Example ChemRefine SLURM script](https://raw.githubusercontent.com/sterling-group/ChemRefine/mkdocs/Examples/Templates/chemrefine.slurm)

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --job-name=mlip_training
#SBATCH --output=%x.out
#SBATCH --error=%x.err

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

chemrefine input.yaml --maxcores 8
```

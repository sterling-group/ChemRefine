# Redox Reaction Tutorial

This tutorial demonstrates how to use **ChemRefine** to study **redox processes**, including electron transfer reactions, charge-state changes, and energy evaluation with both MLFF and DFT levels of theory.

---

## Overview

Redox chemistry is central to catalysis, batteries, and energy materials.  
ChemRefine automates redox workflows by allowing you to:

1. **Prepare input geometries** for different charge states.  
2. **Run optimizations** at MLFF or DFT levels of theory.  
3. **Evaluate redox potentials** by comparing total energies of oxidized and reduced species.  
4. **Apply solvation corrections** if required.  

---

## Prerequisites

- Installed **ChemRefine** (see [Installation Guide](../INSTALL.md))  
- Access to an **ORCA executable**  
- Example input (`redox_input.yaml`) from this tutorial folder  
- Initial structure (`step1.xyz`)  

---

## Input Files

We start with an initial structure located in the templates folder:

- 📄 [View input.yaml](https://github.com/sterling-group/ChemRefine/blob/mkdocs/Examples/Tutorials/Redox/input.yaml)  
- 📄 [View Input XYZ](https://github.com/sterling-group/ChemRefine/blob/mkdocs/Examples/Tutorials/Redox/step1.xyz)  
## Orca Input Files

You can find the ORCA input files [here](https://github.com/sterling-group/ChemRefine/tree/mkdocs/Examples/Tutorials/Redox/templates)

### Interactive 3D Viewer

<div id="viewer" style="width: 100%; height: 400px; position: relative;"></div>

<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<script>
  let viewer = $3Dmol.createViewer("viewer", { backgroundColor: "white" });

  fetch("https://raw.githubusercontent.com/sterling-group/ChemRefine/mkdocs/Examples/Tutorials/Redox/step1.xyz")
    .then(r => r.text())
    .then(data => {
      viewer.addModel(data, "xyz");   // force XYZ format
      viewer.setStyle({}, {stick:{radius:0.15}, sphere:{scale:0.25}});
      viewer.zoomTo();
      viewer.render();
    })
    .catch(err => console.error("Could not load XYZ:", err));
</script>

---

## YAML Configuration

➡️ [Examples/Tutorials/Redox/redox_input.yaml](https://raw.githubusercontent.com/sterling-group/ChemRefine/mkdocs/Examples/Tutorials/Redox/input.yaml)

Example content (excerpt):

```yaml
template_dir: ./templates
scratch_dir: /scratch/redox_jobs
output_dir: ./outputs/redox
orca_executable: /mfs/io/groups/sterling/software-tools/orca/orca_6_1_0_avx2/orca

charge: 0
multiplicity: 1

initial_xyz: ./templates/step1.xyz

steps:
  - step: 1
    operation: "GOAT"
    engine: "DFT"
    sample_type:
      method: "boltzmann"  
      parameters:
       weight: 95 

  - step: 2
    operation: "OPT+SP"
    engine: "DFT"
    sample_type:
      method: "energy_window"
      parameters:
        energy: 10      
        unit: kcal/mol

  - step: 3
    operation: "OPT+SP"
    engine: "MLFF"
    charge: -1 
    multiplicity: 2
    mlff:
      model_name: "uma-s-1"
      task_name: "omol"
      device: "cuda"
    sample_type:
      method: "integer"  
      parameters:
       num_structures: 0  
              
  - step: 4
    operation: "OPT+SP"
    engine: "MLFF"
    charge: 0
    multiplicity: 1
    mlff:
      model_name: "uma-s-1"
      task_name: "omol"
      device: "cuda"
    sample_type:
      method: "integer"  
      parameters:
       num_structures: 0  

  - step: 5
    operation: "OPT+SP"
    engine: "MLFF"
    charge: 1
    multiplicity: 2
    mlff:
      model_name: "uma-s-1"
      task_name: "omol"
      device: "cuda"
    sample_type:
      method: "integer"  
      parameters:
       num_structures: 0  

  - step: 6
    operation: "OPT+SP"
    engine: "DFT"
    charge: -1
    multiplicity: 2
    sample_type:
      method: "integer"
      parameters:
       num_structures: 0  

  -  step: 7
     operation: "OPT+SP"
     engine: "DFT"
     charge: 0
     multiplicity: 1
     sample_type:
      method: "integer"
      parameters:
       num_structures: 0  
 
  -  step: 8
     operation: "OPT+SP"
     engine: "DFT"
     charge: 1
     multiplicity: 2
     sample_type:
      method: "integer"
      parameters:
       num_structures: 0  
```

This workflow optimizes the neutral, reduced (–1), and oxidized (+1) charge states.

---

## How to Run

Before running ChemRefine, ensure that:

- The **ChemRefine environment** is activated  
- The **ORCA executable** is in your `PATH`  
- The **template directory** (`./templates/`) is set up  
- The **input structure file** (e.g., `step1.xyz`) is prepared  

### Option 1: Run from the Command Line

```bash
chemrefine redox_input.yaml --maxcores <N>
```

Here `<N>` is the maximum number of simultaneous cores.  

### Option 2: Run with SLURM

On HPC systems with SLURM:

```bash
sbatch ./Examples/templates/chemrefine.slurm
```

➡️ [Example ChemRefine SLURM script](https://raw.githubusercontent.com/sterling-group/ChemRefine/mkdocs/Examples/Templates/chemrefine.slurm)

---

## Expected Outputs

- **Neutral optimization** in `outputs/redox/step1/`  
- **Reduced state** in `outputs/redox/step2/`  
- **Oxidized state** in `outputs/redox/step3/`  

Each directory contains `.out` logs, `.xyz` geometries, and total energy values.  
These energies can be compared to compute **redox potentials**.  

---

## Notes & Tips

- Modify `charge` and `multiplicity` values to match your redox states.  
- Use MLFF first for speed, then recheck with DFT.  
- Solvation can be included with an additional **SOLVATOR** step.  
- Always verify convergence in `.out` files.  

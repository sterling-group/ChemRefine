# Spin State Tutorial

This tutorial demonstrates how to use **ChemRefine** to investigate different **spin states** of a molecule and compare predictions between **DFT** and **machine-learned force fields (MLFFs)**.

---

## Overview

Electronic spin states play a critical role in catalysis, magnetism, and redox chemistry.  
ChemRefine automates spin exploration with the following workflow:

1. **Initialize geometry** from an input structure.  
2. **Optimize structures** at multiple spin multiplicities.  
3. **Compare MLFF vs DFT predictions** for spin energetics and geometries.  
4. **Extract spin energy gaps** for further analysis.  
---

## Prerequisites

- Installed **ChemRefine** (see [Installation Guide](../install.md))  
- Access to an **ORCA executable**  
- Example input (`spin_input.yaml`) from this tutorial folder  
- Initial structure (`step1.xyz`)  

---

## Input Files

For this tutorial, we will use the provided **spin input file**.

- 📄 [View spin_input.yaml](https://github.com/sterling-group/ChemRefine/blob/mkdocs/Examples/Tutorials/Spin/spin_input.yaml)  
- 📥 [Download spin_input.yaml](https://raw.githubusercontent.com/sterling-group/ChemRefine/mkdocs/Examples/Tutorials/Spin/input.yaml)  

- 📄 [View step1.xyz](https://github.com/sterling-group/ChemRefine/blob/mkdocs/Examples/Tutorials/Spin/heme.xyz)  
- 📥 [Download step1.xyz](https://raw.githubusercontent.com/sterling-group/ChemRefine/mkdocs/Examples/Tutorials/Spin/heme.xyz)  

### Interactive 3D Viewer

<div id="viewer" style="width: 100%; height: 400px; position: relative;"></div>

<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<script>
  let viewer = $3Dmol.createViewer("viewer", { backgroundColor: "white" });

  fetch("https://raw.githubusercontent.com/sterling-group/ChemRefine/mkdocs/Examples/Tutorials/Spin/heme.xyz")
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

➡️ [Examples/Tutorials/Spin/spin_input.yaml](https://raw.githubusercontent.com/sterling-group/ChemRefine/mkdocs/Examples/Tutorials/Spin/input.yaml)

Example content (excerpt):

```yaml
template_dir: ./templates
scratch_dir: /scratch/
output_dir: ./outputs
orca_executable: /orca

charge: 0
multiplicity: 5 

# Optional: Override default initial structure (default is /template_dir/step1.xyz)
initial_xyz: ./templates/step1.xyz

steps:
  - step: 1
    operation: "OPT+SP"
    engine: "DFT"
    sample_type:
      method: "integer"  
      parameters: 
        num_structures: 0

  - step: 2
    operation: "OPT+SP"
    engine: "DFT"
    charge: 0 
    multiplicity: 5
    sample_type:
      method: "integer"
      parameters:
        num_structures: 0

  - step: 3
    operation: "OPT+SP"
    engine: "DFT"
    charge: 0 
    multiplicity: 3
    sample_type:
      method: "integer"
      parameters:
        num_structures: 0   

  - step: 4
    operation: "OPT+SP"
    engine: "DFT"
    charge: 0 
    multiplicity: 1
    sample_type:
      method: "integer"
      parameters:
        num_structures: 0   

  - step: 5
    operation: "OPT+SP"
    engine: "MLFF"
    charge: 0
    multiplicity: 5
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
    engine: "MLFF"
    charge: 0
    multiplicity: 3
    mlff:
      model_name: "uma-s-1"
      task_name: "omol"
      device: "cuda"
    sample_type:
      method: "integer"  
      parameters:
       num_structures: 0   

  - step: 7
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
```

This workflow optimizes the same molecule in **singlet and triplet spin states** using both MLFF and DFT.

---

## How to Run

Before running ChemRefine, ensure that:

- The **ChemRefine environment** is activated  
- The **ORCA executable** is in your `PATH`  
- The **template directory** (`./templates/`) is set up  
- The **input structure file** (e.g., `step1.xyz`) is prepared  

### Option 1: Run from the Command Line

```bash
chemrefine spin_input.yaml --maxcores <N>
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

- **Singlet optimization (MLFF + DFT)** in `outputs/spin/step1/` and `outputs/spin/step2/`  
- **Triplet optimization (MLFF + DFT)** in `outputs/spin/step3/` and `outputs/spin/step4/`  

Each directory contains `.out` logs, `.xyz` geometries, and total energy values.  
You can compare these to evaluate **spin gaps** and test **MLFF accuracy vs DFT**.  

---

## Notes & Tips

- Extend to higher spin states by adding more steps.  
- Use MLFF first for speed, then benchmark with DFT.  
- Monitor **ΔE(S=0 → S=2)** to quantify spin crossover energetics.  
- Spin states may converge to different geometries — always visualize final `.xyz` files.  

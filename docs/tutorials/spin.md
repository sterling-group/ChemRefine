# Spin State Tutorial

!!! note "Schema note"
    The YAML excerpts on this page are abbreviated for illustration. For the authoritative schema (`sample:`, `input:`, `options:` blocks, …) see the [main schema page](../index.md) and the example in [Examples/input.yaml](https://github.com/sterling-group/ChemRefine/blob/main/Examples/input.yaml).


This tutorial demonstrates how to use **ChemRefine** to investigate different **spin states** of a molecule and compare predictions between **DFT** and **machine-learned interatomic potentials (MLIPs)**.

---

## Overview

Electronic spin states play a critical role in catalysis, magnetism, and redox chemistry.  
ChemRefine automates spin exploration with the following workflow:

1. **Initialize geometry** from an input structure.  
2. **Optimize structures** at multiple spin multiplicities.  
3. **Compare MLIP vs DFT predictions** for spin energetics and geometries.  
4. **Extract spin energy gaps** for further analysis.  
---

## Prerequisites

- Installed **ChemRefine** (see [Installation Guide](../user-guide/installation.md))  
- Access to an **ORCA executable**  
- Example input (`input.yaml`) from this tutorial folder  
- Initial structure (`step1.xyz`)  

---

## Input Files

We start with an initial structure located in the templates folder:

- 📄 [View input.yaml](https://github.com/sterling-group/ChemRefine/blob/main/Examples/Tutorials/Spin/heme_catalyst/input.yaml)  
- 📄 [View step1.xyz](https://github.com/sterling-group/ChemRefine/blob/main/Examples/Tutorials/Spin/heme_catalyst/templates/step1.xyz)  


  ## Orca Input Files

You can find the ORCA input files [here](https://github.com/sterling-group/ChemRefine/tree/main/Examples/Tutorials/Spin/heme_catalyst/templates)

### Interactive 3D Viewer

<div id="viewer" style="width: 100%; height: 400px; position: relative;"></div>

<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<script>
  let viewer = $3Dmol.createViewer("viewer", { backgroundColor: "white" });

  fetch("https://raw.githubusercontent.com/sterling-group/ChemRefine/main/Examples/Tutorials/Spin/heme.xyz")
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

➡️ [Examples/Tutorials/Spin/heme_catalyst/input.yaml](https://raw.githubusercontent.com/sterling-group/ChemRefine/main/Examples/Tutorials/Spin/heme_catalyst/input.yaml)

Example content:

```yaml
template_dir: ./templates
scratch_dir: /scratch/
output_dir: ./outputs
executables: { orca: /orca }

charge: 0
multiplicity: 5

input: ./templates/step1.xyz

steps:
  - step: 1
    operation: opt_sp
    engine: orca
    sample: { method: min, count: 0 }

  - step: 2
    operation: opt_sp
    engine: orca
    charge: 0
    multiplicity: 5
    sample: { method: min, count: 0 }

  - step: 3
    operation: opt_sp
    engine: orca
    charge: 0
    multiplicity: 3
    sample: { method: min, count: 0 }

  - step: 4
    operation: opt_sp
    engine: orca
    charge: 0
    multiplicity: 1
    sample: { method: min, count: 0 }

  - step: 5
    operation: opt_sp
    engine: mlip-extopt
    charge: 0
    multiplicity: 5
    options: { model_name: uma-s-1, task_name: omol, device: cuda }
    sample: { method: min, count: 0 }

  - step: 6
    operation: opt_sp
    engine: mlip-extopt
    charge: 0
    multiplicity: 3
    options: { model_name: uma-s-1, task_name: omol, device: cuda }
    sample: { method: min, count: 0 }

  - step: 7
    operation: opt_sp
    engine: mlip-extopt
    charge: 0
    multiplicity: 1
    options: { model_name: uma-s-1, task_name: omol, device: cuda }
    sample: { method: min, count: 0 }
```

This workflow optimizes the same molecule at **multiplicities 5, 3, and 1** using both DFT and MLIP.

---

## How to Run

Before running ChemRefine, ensure that:

- The **ChemRefine environment** is activated  
- The **ORCA executable** is in your `PATH`  
- The **template directory** (`./templates/`) is set up  
- The **input structure file** (e.g., `step1.xyz`) is prepared  

### Option 1: Run from the Command Line

```bash
chemrefine run input.yaml --maxcores <N>
```

Here `<N>` is the maximum number of simultaneous cores.  

### Option 2: Run with SLURM

On HPC systems with SLURM:

```bash
sbatch ./Examples/Templates/chemrefine.slurm
```

➡️ [Example ChemRefine SLURM script](https://raw.githubusercontent.com/sterling-group/ChemRefine/main/Examples/Templates/chemrefine.slurm)

---

## Expected Outputs

- **DFT-optimized spin states** (multiplicities 5 / 3 / 1) in `outputs/step2/`–`outputs/step4/`  
- **MLIP-optimized spin states** (multiplicities 5 / 3 / 1) in `outputs/step5/`–`outputs/step7/`  

Each directory contains `.out` logs, `.xyz` geometries, and total energy values.  
You can compare these to evaluate **spin gaps** and test **MLIP accuracy vs DFT**.  

---

## Notes & Tips

- Extend to higher spin states by adding more steps.  
- Use MLIP first for speed, then benchmark with DFT.  
- Monitor **ΔE(S=2 → S=0)** to quantify spin crossover energetics.  
- Spin states may converge to different geometries — always visualize final `.xyz` files.  

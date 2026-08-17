# Redox Reaction Tutorial

!!! note "Schema note"
    The YAML excerpts on this page are abbreviated for illustration. For the authoritative schema (`sample:`, `input:`, `options:` blocks, …) see the [configuration reference](../user-guide/configuration.md) and the example in [examples/quickstart/input.yaml](https://github.com/sterling-group/ChemRefine/blob/main/examples/quickstart/input.yaml).


This tutorial demonstrates how to use **ChemRefine** to study **redox processes**, including electron transfer reactions, charge-state changes, and energy evaluation with both MLIP and DFT levels of theory.

---

## Overview

Redox chemistry is central to catalysis, batteries, and energy materials.  
ChemRefine automates redox workflows by allowing you to:

1. **Prepare input geometries** for different charge states.  
2. **Run optimizations** at MLIP or DFT levels of theory.  
3. **Evaluate redox potentials** by comparing total energies of oxidized and reduced species.  
4. **Apply solvation corrections** if required.  

---

## Prerequisites

- Installed **ChemRefine** (see [Installation Guide](../user-guide/installation.md))  
- Access to an **ORCA executable**  
- Example input (`input.yaml`) from this tutorial folder  
- Initial structure (`step1.xyz`)  

---

## Input Files

We start with an initial structure located in the templates folder:

- 📄 [View input.yaml](https://github.com/sterling-group/ChemRefine/blob/main/examples/tutorials/redox/dimethylaniline/input.yaml)  
- 📄 [View Input XYZ](https://github.com/sterling-group/ChemRefine/blob/main/examples/tutorials/redox/dimethylaniline/step1.xyz)  
## Orca Input Files

You can find the ORCA input files [here](https://github.com/sterling-group/ChemRefine/tree/main/examples/tutorials/redox/dimethylaniline/templates)

### Interactive 3D Viewer

<div id="viewer" style="width: 100%; height: 400px; position: relative;"></div>

<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<script>
  let viewer = $3Dmol.createViewer("viewer", { backgroundColor: "white" });

  fetch("https://raw.githubusercontent.com/sterling-group/ChemRefine/main/examples/tutorials/redox/dimethylaniline/step1.xyz")
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

➡️ [examples/tutorials/redox/dimethylaniline/input.yaml](https://raw.githubusercontent.com/sterling-group/ChemRefine/main/examples/tutorials/redox/dimethylaniline/input.yaml)

Example content:

```yaml
template_dir: ./templates
scratch_dir: /scratch/
output_dir: ./outputs
executables: { orca: /orca/orca_6_1_0_avx2/orca }

charge: 0
multiplicity: 1

input: ./step1.xyz

steps:
  - step: 1
    operation: goat
    engine: orca
    sample: { method: boltzmann, percent_cumulative: 95 }

  - step: 2
    operation: opt_sp
    engine: orca
    sample: { method: min, window_kcalmol: 10 }

  - step: 3
    operation: opt_sp
    engine: mlip-extopt
    charge: -1
    multiplicity: 2
    options: { model_name: uma-s-1, task_name: omol, device: cuda }
    sample: { method: min, count: 0 }

  - step: 4
    operation: opt_sp
    engine: mlip-extopt
    charge: 0
    multiplicity: 1
    options: { model_name: uma-s-1, task_name: omol, device: cuda }
    sample: { method: min, count: 0 }

  - step: 5
    operation: opt_sp
    engine: mlip-extopt
    charge: 1
    multiplicity: 2
    options: { model_name: uma-s-1, task_name: omol, device: cuda }
    sample: { method: min, count: 0 }

  - step: 6
    operation: opt_sp
    engine: orca
    charge: -1
    multiplicity: 2
    sample: { method: min, count: 0 }

  - step: 7
    operation: opt_sp
    engine: orca
    charge: 0
    multiplicity: 1
    sample: { method: min, count: 0 }

  - step: 8
    operation: opt_sp
    engine: orca
    charge: 1
    multiplicity: 2
    sample: { method: min, count: 0 }
```

This workflow optimizes the neutral, reduced (–1), and oxidized (+1) charge states with both MLIP and DFT.

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

On HPC systems with SLURM, the same command submits each calculation as its own job
(`dispatch: auto` detects `sbatch`; no wrapper script is needed):

```bash
chemrefine run input.yaml
```


---

## Expected Outputs

- **MLIP-optimized charge states** (reduced / neutral / oxidized) in `outputs/step3/`–`outputs/step5/`  
- **DFT-optimized charge states** in `outputs/step6/`–`outputs/step8/`  

Each directory contains `.out` logs, `.xyz` geometries, and total energy values.  
These energies can be compared to compute **redox potentials**.  

---

## Notes & Tips

- Modify `charge` and `multiplicity` values to match your redox states.  
- Use MLIP first for speed, then recheck with DFT.  
- Solvation can be included with an additional **solvator** step.  
- Always verify convergence in `.out` files.  

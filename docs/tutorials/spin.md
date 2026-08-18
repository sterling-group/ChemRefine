# Spin State Tutorial

--8<-- "docs/_includes/schema-note.md"


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

- 📄 [View input.yaml](https://github.com/sterling-group/ChemRefine/blob/main/examples/tutorials/spin/heme_catalyst/input.yaml)  
- 📄 [View step1.xyz](https://github.com/sterling-group/ChemRefine/blob/main/examples/tutorials/spin/heme_catalyst/step1.xyz)  


  ## Orca Input Files

You can find the ORCA input files [here](https://github.com/sterling-group/ChemRefine/tree/main/examples/tutorials/spin/heme_catalyst/templates)

### Interactive 3D Viewer

<div id="viewer" data-xyz="examples/tutorials/spin/heme_catalyst/step1.xyz"
     style="width: 100%; height: 400px; position: relative;"></div>

--8<-- "docs/_includes/viewer.md"

---

## YAML Configuration

➡️ [examples/tutorials/spin/heme_catalyst/input.yaml](https://raw.githubusercontent.com/sterling-group/ChemRefine/main/examples/tutorials/spin/heme_catalyst/input.yaml)

This is the shipped config, included verbatim — the same file `tests/test_examples.py` validates on every CI run:

```yaml
--8<-- "examples/tutorials/spin/heme_catalyst/input.yaml"
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

On HPC systems with SLURM, the same command submits each calculation as its own job
(`dispatch: auto` detects `sbatch`; no wrapper script is needed):

```bash
chemrefine run input.yaml
```


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

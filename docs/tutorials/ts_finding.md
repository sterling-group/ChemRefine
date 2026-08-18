# Transition State (TS) Finding Tutorial

--8<-- "docs/_includes/schema-note.md"


This tutorial demonstrates how to use **ChemRefine** to locate and validate **transition states (TS)** using a stepwise pipeline that combines a **PES scan, optimizations, and normal-mode sampling**.

---

## Overview

Transition states are critical for understanding chemical reactivity and kinetics.  
ChemRefine automates TS exploration with the following workflow:

1. **PES Scan:** Explore bond distances/angles to identify high-energy regions.  
2. **TS Optimization (Top 5):** Optimize the five highest-energy structures from the PES scan.  
3. **Normal-Mode Sampling:** Run the frequency analysis and displace along the imaginary mode, keeping geometries with exactly one imaginary mode (`nms: true`, `target: ts`).  
4. **Final SP Calculation:** Compute the single-point energy on the corrected TS structure.  

---

## Prerequisites

- Installed **ChemRefine** (see [Installation Guide](../get-started/install.md))  
- Access to an **ORCA executable**  
- Example input (`input.yaml`) from this tutorial folder  
- Initial structure (`step1.xyz`)  

---

## Input Files

We start with an initial structure located in the templates folder:

- 📄 [View input.yaml](https://github.com/sterling-group/ChemRefine/blob/main/examples/tutorials/transition_state/input.yaml)  
- 📄 [View step1.xyz](https://github.com/sterling-group/ChemRefine/blob/main/examples/tutorials/transition_state/step1.xyz)  

 ## Orca Input Files

You can find the ORCA input files [here](https://github.com/sterling-group/ChemRefine/tree/main/examples/tutorials/transition_state/templates)

### Interactive 3D Viewer

<div id="viewer" data-xyz="examples/tutorials/transition_state/step1.xyz"
     style="width: 100%; height: 400px; position: relative;"></div>

--8<-- "docs/_includes/viewer.md"

---

## YAML Configuration

➡️ [examples/tutorials/transition_state/input.yaml](https://github.com/sterling-group/ChemRefine/blob/main/examples/tutorials/transition_state/input.yaml)

The scan coordinates and any geometry constraints live in the step's ORCA `.inp`
template (e.g. a `%geom Scan ... end` block), not in the YAML.

This is the shipped config, included verbatim — the same file `tests/test_examples.py` validates on every CI run:

```yaml
--8<-- "examples/tutorials/transition_state/input.yaml"
```

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

Here `<N>` is the total core budget, not a job count: the scheduler runs as many jobs at
once as fit in it, each charged the cores its step declares.  

### Option 2: Run with SLURM

On HPC systems with SLURM, the same command submits each calculation as its own job
(`dispatch: auto` detects `sbatch`; no wrapper script is needed):

```bash
chemrefine run input.yaml
```


---

## Expected Outputs

- **PES scan geometries** in `outputs/step1/`  
- **Top 5 optimized candidates** in `outputs/step2/`  
- **Normal-mode-sampled (corrected) TS geometries** in `outputs/step3/`  
- **Final single-point energy** in `outputs/step4/`  

Each directory contains `.out` logs, `.xyz` geometries, and summary files.  

---

## Notes & Tips

- Increase the PES scan resolution for difficult reactions.  
- Ensure only **one imaginary frequency** is present for a valid TS.  
- Use `nms: true` with `target: ts` on the optimisation step to remove spurious modes.  
- Always double-check `.xyz` files to confirm correct TS geometry.  

### Identifying Good vs Bad Imaginary Modes

ChemRefine helps distinguish **spurious imaginary modes** (bad TS guesses) from **true transition states**.  

- ❌ Bad imaginary frequency:  
![Bad Imaginary](./bad_imag.gif)  

- ✅ Corrected good imaginary frequency:  
![Good Imaginary](./good_imag.gif)  

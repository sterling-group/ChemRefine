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

---

## Identifying Good vs Bad Imaginary Modes

A transition state is a *first-order* saddle point: it has exactly one imaginary mode, **and
that mode is the reaction coordinate**. A frequency table only ever answers the first half.
Both structures below are real states of this workflow — a step-2 candidate, and what step
3's normal-mode sampling made of it. Drag either viewer to turn the molecule while it moves.

### ❌ Before: a candidate carrying a spurious mode

<!-- chemrefine:mode ts-bad -->

The wavenumber is not what is wrong here. This structure *does* have the reaction
coordinate — mode 6, at −295.24 cm⁻¹, draws both forming C–N bonds in together. What
disqualifies it is the mode above: a wag of the phenyl ring that changes the N19–C5
distance by 0.000 Å and the N17–C6 distance by −0.002 Å per unit of displacement. It goes
nowhere near the reaction it is supposed to describe. Two imaginary modes make this a
second-order saddle, not a transition state — which is exactly what step 3 is for.

### ✅ After: normal-mode sampling has removed it

<!-- chemrefine:mode ts-good -->

Step 3 displaced the candidate along its imaginary modes and re-optimised each child. One
came back still carrying two imaginary modes and was discarded; the other came back with
one, drawn above. The terminal azide nitrogen and the two alkene carbons carry most of the
motion, the phenyl ring is almost still, and both forming bonds move in phase — the
concerted, asynchronous cycloaddition, and nothing else. That is a transition state, and
it is the geometry this tutorial ships as `step1.xyz`.

### Making this view for your own run

Both figures above are three-kilobyte extracts of the frequency outputs that produced them,
written by the call below and animated in the browser. The same file is what the
[workflow builder](../playground.md)'s structure pane draws when you give it a mode index,
so anything you can see here you can see for a run of your own.

<!-- chemrefine:mode recipe -->

`mode_index` is the index the frequency table prints, and the three extra columns the file
carries are the per-atom displacements a viewer reads as `dx/dy/dz`.

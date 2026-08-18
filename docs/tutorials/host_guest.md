# Host–Guest Docking Tutorial

--8<-- "docs/_includes/schema-note.md"


This tutorial demonstrates how to use **ChemRefine** for a host–guest docking workflow, followed by machine-learning refinement, DFT validation, and explicit solvation.  

We will start with an initial structure (`step1.xyz`) and progressively refine docking poses through MLIP and DFT optimization.

## Overview



- **Step 1 – Docking (DFT)**  
  Generate 5 initial docking poses of the guest molecule into the host cavity using XTB-level scoring.

- **Step 2 – MLIP Optimization**  
  Refine docked structures using the **UMA-S-1 model** (`omol` task) through the MLIP gradient server.  
  - GPU acceleration is enabled (`device: cuda`).  
  - Retains structures within **10 kcal/mol** of the lowest energy.

- **Step 3 – DFT Re-optimization**  
  The lowest-energy structure is re-optimized at the DFT level for accuracy.  

- **Step 4 – Solvation**  
  Add explicit solvent molecules around the final optimized host–guest complex for solvation analysis.  

- **Step 5 - DFT calculations**                                                                  

   DFT calculations for each solvent molecule to get solvation free energies. 

## Input Files

We start with an initial structure located in the templates folder:

- 📄 [View input.yaml](https://github.com/sterling-group/ChemRefine/blob/main/examples/tutorials/host_guest/input.yaml)  
- 📄 [View step1.xyz](https://github.com/sterling-group/ChemRefine/blob/main/examples/tutorials/host_guest/step1.xyz)  

## Orca Input Files

You can find the ORCA input files [here](https://github.com/sterling-group/ChemRefine/tree/main/examples/tutorials/host_guest/templates)

---

### Interactive 3D Viewer

<div id="viewer" data-xyz="examples/tutorials/host_guest/step1.xyz"
     style="width: 100%; height: 400px; position: relative;"></div>

--8<-- "docs/_includes/viewer.md"

## 1. Input File

This is the shipped config, included verbatim — the same file `tests/test_examples.py` validates on every CI run:

```yaml
--8<-- "examples/tutorials/host_guest/input.yaml"
```

---


## 2. Running the Workflow

From the command line:

```bash
chemrefine run input.yaml --maxcores 16
```

This runs the workflow locally within a 16-core budget — as many jobs at once as fit in
it, each charged the cores its step declares, not 16 jobs.  

On an HPC cluster with SLURM, the same command submits each calculation as its own job
(`dispatch: auto` detects `sbatch`; no wrapper script is needed):

```bash
chemrefine run input.yaml
```

---

## 4. Expected Outputs

- **Docked poses** from Step 1 in `outputs/step1/`  
- **Refined MLIP structures** with energies in `outputs/step2/`  
- **Validated DFT structures** in `outputs/step3/`  
- **Final solvated complex** in `outputs/step4/`  
- **Free Energy Solvation Energies** in `outputs/step5/`

Each step directory contains `.out` logs, `.xyz` geometries, and summary files.  

---

## 5. Notes & Tips

- Adjust `count` in Step 1 to explore more docking poses.  
- Use MLIP refinement for speed, then confirm results with DFT.  
- Solvation step can be skipped by removing Step 4.  
- Large jobs should always be submitted via SLURM.  

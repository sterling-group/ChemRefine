# Host–Guest Docking Tutorial

!!! note "Schema note"
    The YAML excerpts on this page are abbreviated for illustration. For the authoritative schema (`sample:`, `input:`, `options:` blocks, …) see the [main schema page](../index.md) and the example in [examples/input.yaml](https://github.com/sterling-group/ChemRefine/blob/main/examples/input.yaml).


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

<div id="viewer" style="width: 100%; height: 400px; position: relative;"></div>

<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<script>
  let viewer = $3Dmol.createViewer("viewer", { backgroundColor: "white" });

  fetch("https://raw.githubusercontent.com/sterling-group/ChemRefine/main/examples/tutorials/host_guest/step1.xyz")
    .then(r => r.text())
    .then(data => {
      viewer.addModel(data, "xyz");   // force XYZ format
      viewer.setStyle({}, {stick:{radius:0.15}, sphere:{scale:0.25}});
      viewer.zoomTo();
      viewer.render();
    })
    .catch(err => console.error("Could not load XYZ:", err));
</script>

## 1. Input File

Below is a complete example of an input file (`input.yaml`) for a docking study:

```yaml
template_dir: ./templates
scratch_dir: /scratch/
output_dir: ./outputs
executables: { orca: /orca/orca_6_1_0_avx2/orca }

# Global system settings
charge: 0
multiplicity: 1

input: ./templates/step1.xyz

steps:
  # Step 1 — docking poses of the guest in the host cavity.
  - step: 1
    operation: docker
    engine: orca
    sample: { method: min, count: 5 }

  # Step 2 — refine with an MLIP gradient server (ORCA-driven).
  - step: 2
    operation: opt_sp
    engine: mlip-extopt
    options: { model_name: uma-s-1, task_name: omol, device: cuda }
    sample: { method: min, window_kcalmol: 10 }

  # Step 3 — validate the best candidate with DFT.
  - step: 3
    operation: opt_sp
    engine: orca
    charge: -1
    multiplicity: 1
    sample: { method: min, count: 1 }

  # Step 4 — explicit solvation.
  - step: 4
    operation: solvator
    engine: orca
    sample: { method: min, count: 0 }

  # Step 5 — DFT on the solvated complex.
  - step: 5
    operation: opt_sp
    engine: orca
    charge: -1
    multiplicity: 1
    sample: { method: min, window_kcalmol: 10 }
```

---


## 2. Running the Workflow

From the command line:

```bash
chemrefine run input.yaml --maxcores 16
```

This runs the workflow locally with up to 16 parallel jobs.  

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

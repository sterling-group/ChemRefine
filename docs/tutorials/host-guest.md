# Host–Guest Docking Tutorial

This tutorial demonstrates how to use **ChemRefine** for a host–guest docking workflow, followed by machine-learning refinement, DFT validation, and explicit solvation.  

We will start with an initial structure (`step1.xyz`) and progressively refine docking poses through MLFF and DFT optimization.

## Overview

Orca 6.0+ has added a featured called [Docker](https://www.faccts.de/docs/orca/6.0/manual/contents/typical/docker.html), which finds the best way of putting two systems and putting them together in the their best possible interaction. We will run the following workflow:

1. **Host-Guest Docking (DOCKER):**     
    Performs a stochastic search for the best interaction between the host molecule and the guest Cl<sup>-</sup> ion. We will refine with integer taking the 5 best structures. 

2. **MLIP Optimization:**     
    We will be using UMA-S-1, for geometry optimization

## Input Files

For this tutorial, we will use the following files.

- 📄 [View step1.xyz](https://github.com/sterling-group/ChemRefine/blob/mkdocs/Examples/Tutorials/Host-Guest/step1.xyz)  
- 📥 [Download step1.xyz](https://raw.githubusercontent.com/sterling-group/ChemRefine/mkdocs/Examples/Tutorials/Host-Guest/step1.xyz)

---

### Interactive 3D Viewer

<div id="viewer" style="width: 100%; height: 400px; position: relative;"></div>

<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<script>
  let viewer = $3Dmol.createViewer("viewer", { backgroundColor: "white" });

  fetch("https://raw.githubusercontent.com/sterling-group/ChemRefine/mkdocs/Examples/Tutorials/Host-Guest/step1.xyz")
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
scratch_dir: /scratch/ganymede2/dal950773/orca_files/
output_dir: ./fixed_charge
orca_executable: /mfs/io/groups/sterling/software-tools/orca/orca_6_1_0_avx2/orca

# Global system settings
charge: 0
multiplicity: 1

# Optional: Override default initial structure
initial_xyz: ./templates/step1.xyz

# === Step-by-step workflow ===
steps:
  # Step 1: Perform docking with DFT
  - step: 1
    operation: "DOCKER"
    engine: "DFT"
    sample_type:
      method: "integer"
      parameters:
        num_structures: 5   # Generate 5 docked structures

  # Step 2: Refine docking poses with MLFF
  - step: 2
    operation: "OPT+SP"
    engine: "MLFF"
    charge: -1
    multiplicity: 1
    mlff:
      model_name: "uma-s-1"
      task_name: "omol"
      device: "cuda"
    sample_type:
      method: "energy_window"
      parameters:
        energy: 10
        unit: kcal/mol

  # Step 3: Validate best candidates with DFT
  - step: 3
    operation: "OPT+SP"
    engine: "DFT"
    charge: -1
    multiplicity: 1
    sample_type:
      method: "integer"
      parameters:
        num_structures: 1

  # Step 4: Solvation refinement
  - step: 4
    operation: "SOLVATOR"
    engine: "DFT"
    sample_type:
      method: "integer"
      parameters:
        num_structures: 0

  - step: 5
        engine: "DFT"
        operation: "OPT+SP"
        charge: -1
        multiplicity: 1
        sample_type:
        method: "energy_window"
        parameters:
            energy: 10
            unit: kcal/mol

```

---

## 2. Workflow Explanation

- **Step 1 – Docking (DFT)**  
  Generate 5 initial docking poses of the guest molecule into the host cavity using DFT-level scoring.

- **Step 2 – MLFF Optimization**  
  Refine docked structures using the **UMA-S-1 MLFF model** (`omol` task).  
  - GPU acceleration is enabled (`device: cuda`).  
  - Retains structures within **10 kcal/mol** of the lowest energy.

- **Step 3 – DFT Re-optimization**  
  The lowest-energy MLFF structure is re-optimized at the DFT level for accuracy.  

- **Step 4 – Solvation**  
  Add explicit solvent molecules around the final optimized host–guest complex for solvation analysis.  

- **Step 5 - DFT calculations**                                                                  

   DFT calculations for each solvent molecule to get solvation free energies. 

---

## 3. Running the Workflow

From the command line:

```bash
chemrefine input.yaml --maxcores 16
```

This runs the workflow locally with up to 16 parallel jobs.  

On an HPC cluster with SLURM:

```bash
sbatch ./Examples/templates/chemrefine.slurm
```

---

## 4. Expected Outputs

- **Docked poses** from Step 1 in `outputs/step1/`  
- **Refined MLFF structures** with energies in `outputs/step2/`  
- **Validated DFT structures** in `outputs/step3/`  
- **Final solvated complex** in `outputs/step4/`  
- **Free Energy Solvation Energies** in `outputs/step5/`

Each step directory contains `.out` logs, `.xyz` geometries, and summary files.  

---

## 5. Notes & Tips

- Adjust `num_structures` in Step 1 to explore more docking poses.  
- Use MLFF refinement for speed, then confirm results with DFT.  
- Solvation step can be skipped by removing Step 4.  
- Large jobs should always be submitted via SLURM.  

# Transition State (TS) Finding Tutorial

This tutorial demonstrates how to use **ChemRefine** to locate and validate **transition states (TS)** using a stepwise pipeline that combines **PES scans, optimizations, frequency analysis, and final single-point energy evaluation**.

---

## Overview

Transition states are critical for understanding chemical reactivity and kinetics.  
ChemRefine automates TS exploration with the following workflow:

1. **PES Scan:** Explore bond distances/angles to identify high-energy regions.  
2. **TS Optimization (Top 5):** Optimize the five highest-energy structures from the PES scan.  
3. **Frequency Analysis:** Confirm the presence of one imaginary mode.  
4. **Imaginary Mode Displacement:** Remove the mode and generate corrected geometries.  
5. **Final SP Calculation:** Compute high-level single-point energies on the corrected TS structure.  

---

## Prerequisites

- Installed **ChemRefine** (see [Installation Guide](../install.md))  
- Access to an **ORCA executable**  
- Example input (`ts_input.yaml`) from this tutorial folder  
- Initial structure (`step1.xyz`)  

---

## Input Files

For this tutorial, we will use the provided **transition state input file**.

- 📄 [View ts_input.yaml](https://github.com/sterling-group/ChemRefine/blob/mkdocs/Examples/Tutorials/Transition-State/ts_input.yaml)  
- 📥 [Download ts_input.yaml](https://raw.githubusercontent.com/sterling-group/ChemRefine/mkdocs/Examples/Tutorials/Transition-State/ts_input.yaml)  

- 📄 [View step1.xyz](https://github.com/sterling-group/ChemRefine/blob/mkdocs/Examples/Tutorials/Transition-State/step1.xyz)  
- 📥 [Download step1.xyz](https://raw.githubusercontent.com/sterling-group/ChemRefine/mkdocs/Examples/Tutorials/Transition-State/step1.xyz)  

### Interactive 3D Viewer

<div id="viewer" style="width: 100%; height: 400px; position: relative;"></div>

<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<script>
  let viewer = $3Dmol.createViewer("viewer", { backgroundColor: "white" });

  fetch("https://raw.githubusercontent.com/sterling-group/ChemRefine/mkdocs/Examples/Tutorials/Transition-State/step1.xyz")
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

➡️ [Examples/Tutorials/Transition-State/ts_input.yaml](https://raw.githubusercontent.com/sterling-group/ChemRefine/mkdocs/Examples/Tutorials/Transition-State/ts_input.yaml)

Example content (excerpt):

```yaml
template_dir: ./templates
scratch_dir: /scratch/ts_jobs
output_dir: ./outputs/ts
orca_executable: /mfs/io/groups/sterling/software-tools/orca/orca_6_1_0_avx2/orca

initial_xyz: ./templates/step1.xyz

steps:
  # Step 1: PES scan
  - step: 1
    operation: "PES"
    engine: "DFT"
    sample_type:
      method: "integer"
      parameters:
        num_structures: 20
    constraints:
      bonds: [(82-91), (0-79), (0-80), (0-82)]

  # Step 2: Optimize top 5 high-energy structures
  - step: 2
    operation: "OPT+SP"
    engine: "DFT"
    sample_type:
      method: "energy_window"
      parameters:
        energy: top5

  # Step 3: Frequency calculation
  - step: 3
    operation: "FREQ"
    engine: "DFT"

  # Step 4: Displace along imaginary mode
  - step: 4
    operation: "NORMAL_MODE_SAMPLING"
    engine: "DFT"
    parameters:
      mode: imaginary

  # Step 5: Final SP calculation on corrected TS
  - step: 5
    operation: "SP"
    engine: "DFT"
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
chemrefine ts_input.yaml --maxcores <N>
```

Here `<N>` is the maximum number of simultaneous jobs.  

### Option 2: Run with SLURM

On HPC systems with SLURM:

```bash
sbatch ./Examples/templates/chemrefine.slurm
```

➡️ [Example ChemRefine SLURM script](https://raw.githubusercontent.com/sterling-group/ChemRefine/mkdocs/Examples/Templates/chemrefine.slurm)

---

## Expected Outputs

- **PES scan geometries** in `outputs/ts/step1/`  
- **Top 5 optimized candidates** in `outputs/ts/step2/`  
- **Frequency analysis files** in `outputs/ts/step3/`  
- **Imaginary mode displacement results** in `outputs/ts/step4/`  
- **Final single-point energy** in `outputs/ts/step5/`  

Each directory contains `.out` logs, `.xyz` geometries, and summary files.  

---

## Notes & Tips

- Increase the PES scan resolution for difficult reactions.  
- Ensure only **one imaginary frequency** is present for a valid TS.  
- Use `NORMAL_MODE_SAMPLING` to visualize imaginary modes.  
- Always double-check `.xyz` files to confirm correct TS geometry.  

### Identifying Good vs Bad Imaginary Modes

ChemRefine helps distinguish **spurious imaginary modes** (bad TS guesses) from **true transition states**.  

- ❌ Bad imaginary frequency:  
![Bad Imaginary](./bad_imag.gif)  

- ✅ Corrected good imaginary frequency:  
![Good Imaginary](./good_imag.gif)  

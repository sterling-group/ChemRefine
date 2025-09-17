# Conformer Sampling Tutorial

This tutorial demonstrates how to use **ChemRefine** for conformational sampling with an initial **global geometry optimization (GOAT)** and ensemble generation.

## Overview

Conformer sampling is the first step in exploring molecular flexibility and generating diverse geometries.  
ChemRefine automates this process by running a **global optimization** followed by ensemble generation, producing a set of candidate structures for further refinement.

The workflow:

1. **Global Optimization (GOAT):**  
   Performs a stochastic search of the potential energy surface (PES) to identify low-energy conformers.  
2. **Ensemble Generation:**  
   Collects the lowest-energy structures into an ensemble for downstream calculations (e.g., DFT, MLFF).
3. **Level of theory benchmarking:**   
   We're going to refine the level of theory starting from simple GFN2-xTB, UMA-S-1, PBE-D4, ωB97X-D4, B2PLYP 

---

## Prerequisites

- Installed **ChemRefine** (see [Installation Guide](./installation.md))  
- Access to an **ORCA executable**  
- Example molecule and YAML input from the repository  

---

## Input Files

For this tutorial, we will use **Pd(PPh₃)₄**.  
The XYZ file is provided in the repository:

➡️ [Examples/Tutorials/Conformational Sampling/PdPPh3_4.xyz](https://raw.githubusercontent.com/sterling-research-group/ChemRefine/main/Examples/Tutorials/Conformational-Sampling/step1.xyz)
- 📂 File location in repo: `Examples/Tutorials/Conformational Sampling/PdPPh3_4.xyz`

---

### Interactive 3D Viewer

<div id="viewer" style="width: 100%; height: 400px; position: relative;"></div>

<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<script>
  let element = document.getElementById("viewer");
  let config = { backgroundColor: "white" };
  let viewer = $3Dmol.createViewer(element, config);

  // Load XYZ directly from raw GitHub link
  $3Dmol.download("https://raw.githubusercontent.com/sterling-research-group/ChemRefine/main/Examples/Tutorials/Conformational-Sampling/step1.xyz", viewer, {}, function() {
    viewer.setStyle({}, {stick:{radius:0.15}, sphere:{scale:0.25}});
    viewer.zoomTo();
    viewer.render();
  });
</script>

---


## YAML Configuration

The YAML input for conformer sampling is also included in the tutorial folder:

➡️ [Examples/Tutorials/Conformational Sampling/input.yaml](./Examples/Tutorials/Conformational-Sampling/input.yaml)



Example content:

```yaml
charge: 0
multiplicity: 1

initial_xyz: ./Examples/Tutorials/Conformational Sampling/PdPPh3_4.xyz

template_dir: ./templates
scratch_dir: /scratch/
output_dir: ./outputs
orca_executable: /orca
# Sequential ORCA Input Configuration File
# Define each step with its specific parameters.
#This workflow reflects using GOAT and refining methods to improve the accuracy
charge: 0
multiplicity: 1 

# Optional: Override default initial structure (default is /template_dir/step1.xyz)
initial_xyz: ./templates/step1.xyz

steps:
  - step: 1
    calculation_type: "GOAT"
    sample_type:
      method: "integer"  
      parameters:
       num_structures: 15  #This energy is in Hartrees.

  # Step 1: Using MLFF to refine the calculation
  - step: 2
    calculation_type: "MLFF"
    mlff:
      model_name: "uma-s-1"
      task_name: "omol"
      device: "cuda"
    sample_type:
      method: "integer"
      parameters:
       num_structures: 15 
              
  - step: 3
    calculation_type: "DFT"
    sample_type:
      method: "integer"
      parameters:
        num_strucures: 15      

  - step: 4
    calculation_type: "DFT"
    sample_type:
      method: "integer"
      parameters:
        num_strucures: 15
  
  - step: 5
    calculation_type: "DFT"
    sample_type:
      method: "integer"
      parameters:
        num_structures: 15
```


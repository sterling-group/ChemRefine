# Redox Reaction Tutorial

--8<-- "docs/_includes/schema-note.md"


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

- Installed **ChemRefine** (see [Installation Guide](../get-started/install.md))  
- Access to an **ORCA executable**  
- Example input (`input.yaml`) from this tutorial folder  
- Initial structure (`step1.xyz`)  

---

## Input Files

We start with an initial structure located beside the config, at the example's root
(`input: ./step1.xyz`):

- 📄 [View input.yaml](https://github.com/sterling-group/ChemRefine/blob/main/examples/tutorials/redox/dimethylaniline/input.yaml)  
- 📄 [View Input XYZ](https://github.com/sterling-group/ChemRefine/blob/main/examples/tutorials/redox/dimethylaniline/step1.xyz)  
## Orca Input Files

You can find the ORCA input files [here](https://github.com/sterling-group/ChemRefine/tree/main/examples/tutorials/redox/dimethylaniline/templates)

### Interactive 3D Viewer

<!-- chemrefine:structure examples/tutorials/redox/dimethylaniline/step1.xyz -->

---

## YAML Configuration

➡️ [examples/tutorials/redox/dimethylaniline/input.yaml](https://raw.githubusercontent.com/sterling-group/ChemRefine/main/examples/tutorials/redox/dimethylaniline/input.yaml)

This is the shipped config, included verbatim — the same file `tests/test_examples.py` validates on every CI run:

```yaml
--8<-- "examples/tutorials/redox/dimethylaniline/input.yaml"
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

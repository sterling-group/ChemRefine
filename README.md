[![GitHub release](https://img.shields.io/github/v/release/sterling-group/ChemRefine.svg)](https://github.com/sterling-group/ChemRefine/releases/)
[![Paper](https://img.shields.io/badge/Paper-ChemRefine-blue)](https://doi.org/10.26434/chemrxiv-2025-cvg1x) 
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![GitHub issues](https://img.shields.io/github/issues/sterling-group/ChemRefine.svg)](https://github.com/sterling-group/ChemRefine/issues/)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://sterling-group.github.io/ChemRefine/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17187169.svg)](https://doi.org/10.5281/zenodo.17187169)

![chemrefinelogo](https://github.com/user-attachments/assets/ae7b1ad5-0d90-445c-be83-ddcb76fa85c3)

# **Automated Workflow for Conformer Sampling and Refinement.**

This repository contains a streamlined Python code for automated ORCA workflow for conformer sampling, TS finding,  and refinement for DFT and MLIPs. The code automates the process of progressively refining the level of theory, eliminating the need for manual intervention. This code seamlessly integrates state-of-the-art MLIP's that can be accessed through ORCA inputs. This code is meant for HPC slurm submission system. Using an input yaml file we are able to automate the process of submitting calculations and then choosing a sampling method to choose the favored conformations, to then refine the calculation with more precise methods.

---

## **Features**
- **Automated workflow** for conformer sampling and refinement
- **Progressive refinement** of computational level across multiple steps
- **Intelligent sampling** with multiple selection algorithms (energy window, Boltzmann, integer-based)
- **HPC integration** with automatic SLURM job management and resource optimization
- **Built-in analysis** with CSV output and structure filtering
- **Flexible configuration** via YAML input files
- **Error reduction** and efficient resource utilization
- **Machine Learning Interatomic potentials** integration using pretrained `mace` and `FairChem models` models for fast geometry optimisation, molecular dynamics, and more.


---

## **Installation**

### **Development Installation**
```bash

#Pip install[Recommended]

pip install "chemrefine @ git+https://github.com/sterling-group/ChemRefine.git@main"

# Installing from Source
git clone  https://github.com/sterling-group/ChemRefine.git
cd ChemRefine

# Install in development mode
pip install -e .
```

### **Requirements**
- **Python 3.11+ and < 3.14** with the following dependencies:
  - `numpy`, `pyyaml`, `pandas`, `ase`, `rdkit`, `scikit-learn`
  - `pydantic >= 2` — typed config validation
  - `typer >= 0.12` — CLI framework
- Optional `[mlff]` extras (for MLFF engines): `torch >= 2.8`, `mace-torch`, `fairchem-core`, `flask`, `waitress`, `requests`
- **ORCA 6.0+** — quantum chemistry calculations
- **SLURM** — HPC job scheduling
---
## **Tutorial** 

You can find examples for running multiple calculations that were in our publication in our [Tutorial](https://sterling-group.github.io/ChemRefine/tutorials/)

## **Quick Start**
 
### **1. Prepare Input Files**

Create the required input files in your working directory:

- **YAML Configuration** (`input.yaml`): Defines the workflow steps
- **Initial XYZ** (`step1.xyz`): Starting molecular geometry  
As of 1.3.1, we have added the ability to start with multiple xyz files. You can either point to a singular xyz file or a directory full of XYZ files. We have also added the ability to read a CSV file with smiles strings as input structure. 
- **ORCA Templates** (`step1.inp`, `step2.inp`, `step3.inp`... `orca.slurm.header`, `mlff.slurm.header`): Calculation templates for each step

You must provide **one ORCA input file** (e.g., `step1.inp`, `step2.inp`, etc.) for **each step** defined in your `input.yaml` configuration file, these must be found where you defined your `template` directory . For example, if your `input.yaml` specifies three ORCA steps, then you need three corresponding ORCA input files in your templates directory.

ChemRefine provides seamless MLIP integration through the use of the tool ExtOpt in Orca, which uses the ORCA optimization codes paired with ASE, you can use any optimization function of ORCA with MLIPS. For more [information](https://github.com/faccts/orca-external-tools).

In addition to these input files, you must include one of each:
- **`cpu.slurm.header`**: A SLURM submission script header with your cluster-specific job settings (e.g., partition, time limit, memory).
- **`cuda.slurm.header`**: Required for MLFF jobs. Include your GPU node configuration here so MLFF calculations run under SLURM.


### **2. Run the Workflow**

```bash
# Full pipeline from step 1
chemrefine run input.yaml

# Resume from cached steps (skip work already completed)
chemrefine resume input.yaml

# Override the YAML's max_cores
chemrefine run input.yaml --maxcores 128

# Validate config without launching anything
chemrefine run input.yaml --dry-run

# Background execution (recommended for long HPC runs)
nohup chemrefine run input.yaml --maxcores 128 &
```

### **Error Correction**

ChemRefine writes a fingerprint-keyed cache to `<step_dir>/_cache/step.pkl`
plus a JSON sidecar. Steps whose YAML config + parent IDs haven't changed
are skipped on the next `resume`. Targeted re-execution:

```bash
# Re-execute the latest step (or a named one)
chemrefine rerun input.yaml             # latest step
chemrefine rerun input.yaml refine      # step named "refine"
chemrefine rerun input.yaml 3           # step number 3

# Rebuild a step's parsed cache without re-submitting jobs
chemrefine rebuild-cache input.yaml 2

# Re-run normal-mode sampling for a step
chemrefine rebuild-nms input.yaml 4
```

### **3. Monitor Progress**

The tool provides detailed logging and creates organized output directories for each step:
```
step1/          # Conformer generation outputs
step2/          # First refinement level outputs  
step3/          # Final high-level calculations
steps.csv       # Summary of energies and structures
```

---

# ChemRefine Operations and Engines

## Operations
| Operation   | Description                                                                 |
|-------------|-----------------------------------------------------------------------------|
| OPT+SP      | General optimization followed by a single-point calculation                 |
| DOCKER      | Host–guest docking workflow                                                 |
| SOLVATOR    | Explicit solvation for a molecule                                           |
| PES         | Parse potential energy surface (PES) scan energies                          |
| MLFF_TRAIN  | Train or fine-tune a machine-learned force field (MLFF)                     |

---

## Engines

### 1. DFT
- **Description:** Quantum mechanical electronic structure calculations (e.g., ORCA).
- **Usable operations:** `OPT+SP`, `DOCKER`, `SOLVATOR`, `PES`

### 2. MLFF
- **Description:** Machine-learned force fields (fast surrogates for DFT).
- **Usable operations:** `OPT+SP`, `DOCKER`, `SOLVATOR`, `PES`, `MLFF_TRAIN`

#### (a) UMA Models
| Model Variant      | Task Types (Domain)                           |
|--------------------|-----------------------------------------------|
| uma-s-1            | omol, oc20, omat, odac, omc                   |
| uma-s1.1           | omol, oc20, omat, odac, omc                   |
| eSEN-sm-direct     | omol, oc20, omat, odac, omc                   |
| eSEN-sm-conserving | omol, oc20, omat, odac, omc                   |

**Task type domains:**
- **omol** → molecules  
- **oc20** → catalysis  
- **omat** → inorganic materials  
- **odac** → MOFs  
- **omc** → molecular crystals  

#### (b) MACE Models
| Task Type   | Domain / Intended Use             |
|-------------|-----------------------------------|
| mace_off    | Mace potential trained on SPICE dataset (small,medium,large)  |
| mace_omol   | MACE potential trained on OMol25 (extralarge model)                       
| mace_mp     | MACE potential trained on Inorganic materials (Materials Project) |



## **Input Files Description**

### **YAML Configuration File**

ChemRefine v4 uses a flat top-level schema with a `steps:` list. Each step's
numeric `step:` is the canonical identifier; the optional `name:` is a
human-readable label that appears in logs, CSV headers, and output
directory names (`step1_screen/`, `step2_refine/`, ...). Engine-specific
knobs live under `options:`; the sampling block is a discriminated union
keyed on `method:`.

```yaml
template_dir: ./templates
scratch_dir:  ./scratch
output_dir:   ./outputs
input:        ./step1.xyz       # xyz file, dir of xyz files, or csv of SMILES
charge: 0
multiplicity: 1
max_cores: 64
slurm_template: cpu.slurm.header
orca_executable: orca

steps:
  - step: 1
    name: screen                 # optional, drives the directory name (step1_screen/)
    engine: mlff
    operation: opt_sp
    options:
      model_name: medium
      task_name: mace_off
      device: cuda
      bind: 127.0.0.1:8888
    sample:
      method: boltzmann
      percent_cumulative: 99

  - step: 2
    name: refine
    engine: orca
    operation: opt_sp
    template: dft_opt.inp        # optional: defaults to stepN.inp
    sample:
      method: energy_window
      window_kcal: 3.0

  - step: 3
    name: high_level
    engine: orca
    operation: opt_sp
    charge: -1                   # per-step override
    multiplicity: 2
    sample:
      method: integer
      count: 5
    nms: true                    # opt in to normal-mode sampling
```

**Sample methods**: `boltzmann` (cumulative %), `energy_window`
(kcal/mol window from the lowest), `integer` (N lowest by energy),
`high_energy` (N highest, e.g. for PES sampling). Any of them accepts
`by_parent: true` to apply the filter within each parent-ID group.

**Engines**: `orca`, `mlff`, `mlff-direct`, `pyscf`, `pyscf-direct`, plus
the `fake` engine used by the test suite.


### **ORCA Template Files**

1. **First Input File** (`step1.inp`):
   - Generally includes **GOAT specifications** for conformer optimization or another conformer sampler. 
   - Uses cheap level of theory (e.g., XTB) for initial sampling
   - Example: `! GOAT XTB`

2. **Subsequent Input Files** (`step2.inp`, `step3.inp`, etc.):
   - Progressive refinement with higher-level methods
   - **Recommended**: Include frequency calculations in final step
   - Example: `! B3LYP def2-TZVP FREQ`

3. **Initial XYZ File** (`step1.xyz`):
   - Starting molecular geometry
   - Standard XYZ format with atom count, comment line, and coordinates

---

## **Sampling Methods**

### **Energy Window** 
```yaml
method: "energy_window"
parameters:
  window: 0.5  # Hartrees
```
Selects conformers within specified energy range of the global minimum.

### **Boltzmann Population**
```yaml
method: "boltzmann"
parameters:
  percentage: 95  # Cumulative population %
```
Selects conformers based on Boltzmann population at given temperature.

### **Integer Count**
```yaml
method: "integer" 
parameters:
  count: 10  # Number of conformers
```
Selects the N lowest-energy conformers.

---

### **Example Multi-Step Workflows**
The tool supports complex multi-step refinement protocols:
1. **Step 1**: GOAT or other conformer generation (XTB level)
2. **Step 2**: Machine Learning interatomic potential optimization (uma-s-1/omol)
2. **Step 3**: DFT geometry optimization (B3LYP/def2-SVP)
3. **Step 4**: High-level single points (B3LYP/def2-TZVP + frequencies)

### **Resource Management**
- Automatic core allocation based on ORCA PAL settings
- Intelligent job queuing to maximize cluster utilization
- Real-time monitoring of SLURM job status

---

## **Project Structure**
```
chemrefine/
├── src/chemrefine          # Main package code
├── Examples/               # Example input files and SLURM scripts
├── README.md               # This file
├── LICENSE                 # License
└── pyproject.toml          # Package configuration
```

---

## **Contributing**

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality  
4. Submit a pull request

---

## **Citation**

If you use ChemRefine in your research, please cite:

```bibtex
@software{ChemRefine,
  title={ChemRefine},
  author={Ignacio Migliaro,Markus G.S. Weiss,Alistair J. Sterling},
  url={https://doi.org/10.26434/chemrxiv-2025-cvg1x},
  year={2025}
}
```

---

## **License**

This project is licensed under the GNU AFFERO GENERAL PUBLIC LICENSE- see the [LICENSE](LICENSE) file for details.


---

## **Support**

For questions, issues, or feature requests:
- 📧 Email: ignacio.migliaro@utdallas.edu
- 🐛 Issues: [GitHub Issues](https://github.com/sterling-group/ChemRefine/issues)
- 📖 Documentation: [README.md](https://sterling-group.github.io/ChemRefine/)




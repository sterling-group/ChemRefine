![chemrefinelogo](https://github.com/user-attachments/assets/ae7b1ad5-0d90-445c-be83-ddcb76fa85c3)

# **Automated Workflow for Conformer Sampling and Refinement.**

This repository contains a streamlined Python code for conformer sampling and refinement for DFT and MLIPs. The code automates the process of progressively refining the level of theory, eliminating the need for manual intervention. This code is meant for HPC slurm submission system, minor modifications must be made if you use HPC outside of our group (UTD). Using an input yaml file we are able to automate the process of submitting calculations and then choosing a sampling method to choose the favored conformations, to then refine the calculation with more precise methods.

---

## **Features**
- **Automated workflow** for conformer sampling and refinement
- **Progressive refinement** of computational level across multiple steps
- **Intelligent sampling** with multiple selection algorithms (energy window, Boltzmann, integer-based)
- **HPC integration** with automatic SLURM job management and resource optimization
- **Built-in analysis** with CSV output and structure filtering
- **Flexible configuration** via YAML input files
- **Error reduction** and efficient resource utilization
- **Machine Learning Interatomic potentials** integration using pretrained `mace` models for fast geometry optimisation.

---

## **Installation**

### **Development Installation**
```bash
# Clone the repository
git clone --recursive https://github.com/sterling-research-group/ChemRefine.git
cd ChemRefine

# Install in development mode
pip install -e .
```

### **From PyPI** (when available)
```bash
pip install ChemRefine
```

### **Requirements**
- **Python 3.6+** with the following dependencies:
  - `numpy` - Numerical computations
  - `pyyaml` - YAML configuration parsing  
  - `pandas` - Data analysis and CSV handling
  - `ase` - Geometry handling and optimisation
  - `mace-torch` - Machine learning force fields
- **ORCA 6.0+** - Quantum chemistry calculations
- **SLURM** - Job scheduling system
- **QORCA** - Included as submodule for ORCA job submission
---

## **Quick Start**

### **1. Prepare Input Files**

Create the required input files in your working directory:

- **YAML Configuration** (`input.yaml`): Defines the workflow steps
- **Initial XYZ** (`step1.xyz`): Starting molecular geometry  
- **ORCA Templates** (`step1.inp`, `step2.inp`, `step3.inp`... `orca.slurm.header`, `mlff.slurm.header`): Calculation templates for each step

You must provide **one ORCA input file** (e.g., `step1.inp`, `step2.inp`, etc.) for **each step** defined in your `input.yaml` configuration file. For example, if your `input.yaml` specifies three steps, then you need three corresponding ORCA input files in your templates directory.

In addition to these input files, you must include one of each:
- **`orca.slurm.header`**: A SLURM submission script header with your cluster-specific job settings (e.g., partition, time limit, memory).
- **`mlff.slurm.header`**: Required for MLFF jobs. Include your GPU node configuration here so MLFF calculations run under SLURM.

Make sure to specify the path to your **ORCA 6.0+** executable in the `ORCA_EXEC` line of your header file(s). Adjust any other parameters (such as modules or memory) to fit your cluster environment.


### **2. Run the Workflow**

```bash
# Basic usage
chemrefine input.yaml

# With custom core count
chemrefine input.yaml --maxcores 128

# Background execution (recommended for HPC)
nohup chemrefine input.yaml --maxcores 128 &

# Skip any step (if already completed)
chemrefine input.yaml --skip
```

### **3. Monitor Progress**

The tool provides detailed logging and creates organized output directories for each step:
```
step1/          # GOAT conformer generation outputs
step2/          # First refinement level outputs  
step3/          # Final high-level calculations
steps.csv       # Summary of energies and structures
```

---

## **Input Files Description**

### **YAML Configuration File**
```yaml
template_dir: <location of template_files>
scratch_dir:  <location of your scratch directory>
output_dir: <location of your output directory>
charge: 0
multiplicity: 1
steps:
  - step: 1
    template: "step1.inp"
    calculation_type: "GOAT"
    sampling:
      method: "integer"
      parameters:
        count: 10
  - step: 2
    template: "step2.inp"
    calculation_type: "DFT"
    sampling:
      method: "energy_window"
      parameters:
        window: 0.5
  - step: 3
    calculation_type: "MLFF"
    sampling:
      method: "integer"
      parameters:
        num_structures: 1
```

The optional MLFF step uses a pretrained model from `mace` (default `"mol"`) to optimise the provided geometries before proceeding with higher-level methods.
If a CUDA-capable GPU is detected, the MLFF optimisation runs on the GPU; otherwise it falls back to the CPU automatically.
To avoid downloading the model each time, set the environment variable `CHEMREFINE_MLFF_CHECKPOINT` to the path of a locally downloaded checkpoint **or** place the file as `chemrefine/models/<model>.model` within this repository.

### **ORCA Template Files**

1. **First Input File** (`step1.inp`):
   - Must include **GOAT specifications** for conformer optimization
   - Uses cheap level of theory (e.g., XTB) for initial sampling
   - Example: `! XTB2 GOAT`

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
  temperature: 298.15  # Kelvin
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

### **Multi-Step Workflows**
The tool supports complex multi-step refinement protocols:
1. **Step 1**: GOAT conformer generation (XTB level)
2. **Step 2**: DFT geometry optimization (B3LYP/def2-SVP)
3. **Step 3**: High-level single points (B3LYP/def2-TZVP + frequencies)

### **Resource Management**
- Automatic core allocation based on ORCA PAL settings
- Intelligent job queuing to maximize cluster utilization
- Real-time monitoring of SLURM job status

---

## **Project Structure**
```
auto-conformer-goat/
├── chemrefine              # Main package code
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

If you use Auto-Conformer-GOAT in your research, please cite:

```bibtex
@software{auto_conformer_goat,
  title={Auto-Conformer-GOAT: Automated Conformer Sampling and Refinement},
  author={Sterling Research Group},
  url={https://github.com/sterling-research-group/auto-conformer-goat},
  year={2025}
}
```

---

## **License**

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

---

## **Support**

For questions, issues, or feature requests:
- 📧 Email: ignacio.migliaro@utdallas.edu
- 🐛 Issues: [GitHub Issues](https://github.com/sterling-research-group/ChemRefine/issues)
- 📖 Documentation: [README.md](README.md)



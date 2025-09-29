# Installation and Setup Guide

## Package Installation

### Option 1: Installation from Source (Recommended)
```bash
git clone https://github.com/sterling-group/ChemRefine.git
cd ChemRefine

#Normal install 
pip install . 
# Install in development mode
pip install -e .

# Verify installation
chemrefine --help
```

### Option 2: Installation from source through pip
```bash
# Install package
pip install "chemrefine @ git+https://github.com/sterling-group/ChemRefine.git@main"

#Editable/development Install
pip install -e "git+https://github.com/sterling-group/ChemRefine.git@main#egg=chemrefine"

```

## Dependencies
- **Python 3.6+** with the following dependencies:
  - `numpy` - Numerical computations
  - `pyyaml` - YAML configuration parsing  
  - `pandas` - Data analysis and CSV handling
  - `ase` - Geometry handling and optimisation
  - `mace-torch` - Machine learning force fields
  - `torch == 2.5.1` - Machine Learning (if you use later version of Pytorch it might not work with UMA models)
  - `e3nn == 0.4.4` MACE models work with this version of e3nn, FAIRCHEM models use a newer version, but they are backwards compatible from our testing. 
### External Requirements

- **ORCA 6.0+** - Quantum chemistry calculations
- **SLURM** - Job scheduling system for HPC
- **MACE-torch** - MLIP platform for MACE architecture
- **FAIRChem** - MLIP platform for UMA and esen models
- **Sevenn** - MLIP platfrom for Sevenn models

MACE models use an ealier version of e3nn and of Pytorch, we have determined that FairChem models allow for backwards compatiblity with these libraries, but due to their strict package controls, they do not allow them to be installed through pip. We have included a patched version of FairChem that relaxes these requirements. 

### Using FAIRChem models

FAIRChem models require a special type of permission through HuggingFace to be able to use them. Please follow the Tutorial from 
## Verification

After installation, verify everything works:

```bash
# Test command-line interface
chemrefine --help

# Test with example files
cd Examples/
chemrefine input.yaml --maxcores 32
```

## License Information

The GNU Affero General Public License is a free, copyleft license for
software and other kinds of works, specifically designed to ensure
cooperation with the community in the case of network server software.

## Troubleshooting

### Common Issues

1. **ORCA not found**: Ensure that Orca is installed correctly, and its directly pointed to.
2. **ORCA not accessible**: Check the path for ORCA and point at the binary file not the directory.
3. **SLURM errors**: Verify SLURM configuration for your cluster
4. **Permission errors**: Check file permissions in working directory

### Getting Help

- Check the main [README.md](../README.md) for usage examples
- Check out our tutorial at [Tutorial](https://sterling-group.github.io/ChemRefine/)
- Review [Examples/](Examples/) directory for sample inputs
- Open an issue on GitHub for bugs or feature requests

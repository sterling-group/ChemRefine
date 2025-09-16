# Installation and Setup Guide

## Package Installation

### Option :Installation (Recommended)
```bash
git clone --recursive https://github.com/sterling-research-group/ChemRefine.git
cd ChemRefine

# Install in development mode
pip install -e .

# Verify installation
chemrefine --help

# Clone repository
git clone https://github.com/sterling-research-group/ChemRefine.git
cd ChemRefine

# Install package6
pip install .
```

# Install package
pip install .


## Dependencies
- **Python 3.6+** with the following dependencies:
  - `numpy` - Numerical computations
  - `pyyaml` - YAML configuration parsing  
  - `pandas` - Data analysis and CSV handling
  - `ase` - Geometry handling and optimisation
  - `mace-torch` - Machine learning force fields
  - `torch == 2.5.1` - Machine Learning (if you use later version of Pytorch it might not work with UMA models)
### External Requirements

- **ORCA 6.0+** - Quantum chemistry calculations
- **SLURM** - Job scheduling system for HPC
- **MACE-torch** - MLIP platform for MACE architecture
- **FAIRChem** - MLIP platform for UMA and esen models

## Verification

After installation, verify everything works:

```bash
# Test command-line interface
chemrefine --help

# Test with example files
cd Examples/
chemrefine input.yaml --maxcores 32
```

## Development Setup

For developers who want to contribute:

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run tests
pytest

## License Information

This software is licensed under the GNU General Public License v3.0. By installing and using this software, you agree to the terms of the GPL-3.0 license. See the [LICENSE](LICENSE) file for complete terms.

## Troubleshooting

### Common Issues

1. **QORCA not found**: Ensure submodules are initialized
2. **ORCA not accessible**: Check ORCA installation and PATH
3. **SLURM errors**: Verify SLURM configuration for your cluster
4. **Permission errors**: Check file permissions in working directory

### Getting Help

- Check the main [README.md](../README.md) for usage examples
- Review [Examples/](Examples/) directory for sample inputs
- Open an issue on GitHub for bugs or feature requests

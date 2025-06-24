# Installation and Setup Guide

## Package Installation

### Option 1: Development Installation (Recommended)
```bash
# Clone with submodules
git clone --recursive https://github.com/sterling-research-group/auto-conformer-goat.git
cd auto-conformer-goat

# Install in development mode
pip install -e .

# Verify installation
auto-goat --help
```

### Option 2: Direct Installation from Source
```bash
# Clone repository
git clone https://github.com/sterling-research-group/auto-conformer-goat.git
cd auto-conformer-goat

# Initialize submodules
git submodule update --init --recursive

# Install package
pip install .
```

## Dependencies

The package automatically installs Python dependencies:
- `numpy` - Numerical computations
- `pyyaml` - YAML parsing
- `pandas` - Data handling

### External Requirements
- **ORCA 6.0+**: Must be installed and accessible in your environment
- **SLURM**: For HPC job submission
- **Python 3.6+**: The package supports Python 3.6 and newer

## Verification

After installation, verify everything works:

```bash
# Check package installation
python -c "import autogoat; print(autogoat.__version__)"

# Test command-line interface
auto-goat --help

# Test with example files
cd Examples/
auto-goat input.yaml --maxcores 32
```

## Development Setup

For developers who want to contribute:

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/

# Type checking
mypy src/
```

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

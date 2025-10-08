# Installation and Setup Guide


```bash

#Pip install[Recommended]

pip install "chemrefine @ git+https://github.com/sterling-group/ChemRefine.git"

# Installing from Source
git clone  https://github.com/sterling-group/ChemRefine.git
cd ChemRefine

# Install in development mode
pip install -e .
```

### **Requirements**
Everything is managed through the pip installation. 
- **Python 3.6+ or < 3.13** with the following dependencies:
  - `numpy` - Numerical computations
  - `pyyaml` - YAML configuration parsing  
  - `pandas` - Data analysis and CSV handling
  - `ase` - Geometry handling and optimisation
  - `mace-torch` - Machine learning force fields
  - `torch == 2.5.1` - Machine Learning (if you use later version of Pytorch it might not work with UMA models)

- **ORCA 6.0+** - Quantum chemistry calculations
- **SLURM** - Job scheduling system
- **MLIP Engines** - MACE, FAIRChem, Sevenn, Orb
---


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

#Test version

chemrefine --version

# Test with example files
cd Examples/
chemrefine input.yaml --maxcores 32
```
### Instantiate FAIRChem models. 
Make sure you have a Hugging Face account, have already applied for model access to the
[UMA model repository](https://huggingface.co/facebook/UMA), and for [OMol25 model repository](https://huggingface.co/facebook/OMol25) and have logged in to Hugging Face using an access token. Make sure to do these before runnining any of the models as the permission process may take ~10+ minutes to be processed. You can create a token by going into your profile avatar in the the top right corner and clicking on access token or by clicking [here](https://huggingface.co/settings/tokens).
You can use the following to save an auth token,
```bash
huggingface-cli login
```

## License Information

This software is licensed under the GNU AFFERO GENERAL PUBLIC LICENSE By installing and using this software, you agree to the terms of the AGPL v3 license. See the [LICENSE](LICENSE) file for complete terms.



### Common Issues

1. **QORCA not found**: Ensure submodules are initialized
2. **ORCA not accessible**: Check ORCA installation and PATH
3. **SLURM errors**: Verify SLURM configuration for your cluster
4. **Permission errors**: Check file permissions in working directory
5. **Server Connection Refused**: For MLIPS a local server is created to not have a constant overhead of transferring model to GPU. Sometimes when the MLIP fails with this error the true error is not the server not connecting but the one found in the slurm_step*.out generated. Make sure if using FairChem models you have activated HuggingFace. 

### Getting Help

- Check the main [README.md](../README.md) for usage examples
- Check [Issues](https://github.com/sterling-group/ChemRefine/issues) if a similar issue has been encountered. 
- Review [Examples/](Examples/) directory for sample inputs
- Open an issue on GitHub for bugs or feature requests

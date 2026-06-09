[![GitHub release](https://img.shields.io/github/v/release/sterling-group/ChemRefine.svg)](https://github.com/sterling-group/ChemRefine/releases/)
[![Paper](https://img.shields.io/badge/Paper-ChemRefine-blue)](https://doi.org/10.26434/chemrxiv-2025-cvg1x)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://sterling-group.github.io/ChemRefine/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17187169.svg)](https://doi.org/10.5281/zenodo.17187169)

![chemrefinelogo](https://github.com/user-attachments/assets/ae7b1ad5-0d90-445c-be83-ddcb76fa85c3)

# ChemRefine

Automated, interoperable manager for computational-chemistry workflows.
ChemRefine drives multi-step conformer sampling and refinement through
ORCA, MLIPs (MACE / FAIRChem / SevenNet / ORB / CHGNet), and PySCF, with
SLURM submission, caching, and resumable runs built in.

📖 **Full documentation:** <https://sterling-group.github.io/ChemRefine/>

## Install

```bash
pip install "chemrefine @ git+https://github.com/sterling-group/ChemRefine.git@main"

# With the default MLIP backends (MACE + FAIRChem):
pip install "chemrefine[mlip] @ git+https://github.com/sterling-group/ChemRefine.git@main"
```

Requires Python 3.11–3.13 and ORCA 6+. SLURM is optional — the generated
`.slurm` script runs unchanged under `bash` for local execution. The
[install guide](https://sterling-group.github.io/ChemRefine/INSTALL/) covers the
per-backend MLIP extras (MACE / FAIRChem / SevenNet / ORB / CHGNet), PySCF, and GPU setup.

## Run

ChemRefine is driven by a single YAML config. A minimal two-step
workflow (MLFF screen → DFT refine):

```yaml
template_dir: ./templates
scratch_dir:  ./scratch
output_dir:   ./outputs
input:        ./step1.xyz
charge: 0
multiplicity: 1
max_cores: 64
slurm_template: cpu.slurm.header
executables: { orca: orca }

steps:
  - step: 1
    name: screen
    engine: mlip
    operation: opt_sp
    options: { model_name: medium, task_name: mace_off, device: cuda }
    sample: { method: boltzmann, percent_cumulative: 99 }

  - step: 2
    name: refine
    engine: orca
    operation: opt_sp
    template: dft_opt.inp
    sample: { method: energy_window, window_kcal: 3.0 }
```

```bash
chemrefine run input.yaml                  # full pipeline from step 1
chemrefine resume input.yaml               # honor cache where valid
chemrefine rebuild-cache input.yaml refine # invalidate one step, resume
chemrefine --help                          # full subcommand list
```

See the [tutorials](https://sterling-group.github.io/ChemRefine/tutorials/)
for conformer sampling, TS finding, host–guest docking, MLIP training,
and redox/spin workflows.

## Citation

If you use ChemRefine in published work, please cite the
[ChemRxiv preprint](https://doi.org/10.26434/chemrxiv-2025-cvg1x) and
the [Zenodo DOI](https://doi.org/10.5281/zenodo.17187169) of the version
you used.

## License

[AGPL v3](LICENSE).

# Installation

## Pip (recommended)

```bash
pip install "chemrefine @ git+https://github.com/sterling-group/ChemRefine.git"

# With the MLFF backends (torch, mace-torch, fairchem, flask):
pip install "chemrefine[mlff] @ git+https://github.com/sterling-group/ChemRefine.git"
```

## From source

```bash
git clone https://github.com/sterling-group/ChemRefine.git
cd ChemRefine
pip install -e .[dev,test,docs,mlff]
pre-commit install   # optional: run ruff + interrogate on every commit
```

## Requirements

- **Python 3.11–3.13**
- **ORCA 6.0+** — quantum-chemistry calculations
- **SLURM** — HPC job scheduler (optional for local runs; the same
  `.slurm` script can be executed with `bash` directly)

The base install pulls `numpy`, `pyyaml`, `pandas`, `ase`, `rdkit`,
`scikit-learn`, `pydantic >= 2`, and `typer >= 0.12`. The `[mlff]`
extra adds `torch >= 2.8, < 2.9`, `mace-torch >= 0.3.16`,
`e3nn == 0.4.4`, `fairchem-core` (Sterling Group's patched fork),
plus `flask`, `waitress`, and `requests` for the gradient server.

## Verification

```bash
chemrefine --version
chemrefine --help
chemrefine run Examples/input.yaml --dry-run    # validates the YAML, no jobs run
```

## FAIRChem model access

The UMA / OMol models on Hugging Face require manual access approval
(allow ~10 minutes). Apply at the
[UMA repository](https://huggingface.co/facebook/UMA) and the
[OMol25 repository](https://huggingface.co/facebook/OMol25), then
authenticate locally:

```bash
huggingface-cli login
```

## Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| `chemrefine: command not found` | Activate the env where you installed ChemRefine (`pip show chemrefine` to confirm). |
| `ORCA not accessible` | Set `orca_executable` in the YAML to an absolute path, or put ORCA on `$PATH`. |
| `sbatch: command not found` | Either SLURM isn't installed locally — run the generated `.slurm` script with `bash` instead — or activate the cluster's SLURM module. |
| `Server crashed during startup` (MLFF) | Check the per-job `server_${SLURM_JOB_ID}.log`; common causes are out-of-memory at model load or a missing HuggingFace token for FAIRChem. |
| `PackageNotFoundError: ChemRefine` at runtime | `pip install -e .` again — the editable install was removed. |

## License

ChemRefine is released under [AGPL v3](https://github.com/sterling-group/ChemRefine/blob/main/LICENSE).

## Getting help

- [Project issues](https://github.com/sterling-group/ChemRefine/issues) — search before opening a new one
- [Example tutorials](tutorials/index.md)
- [Project README](https://github.com/sterling-group/ChemRefine#readme) for the elevator pitch

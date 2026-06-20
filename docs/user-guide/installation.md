# Installation

## Pip (recommended)

```bash
pip install "chemrefine @ git+https://github.com/sterling-group/ChemRefine.git"

# With the default MLIP backends (MACE + FAIRChem):
pip install "chemrefine[mlip] @ git+https://github.com/sterling-group/ChemRefine.git"
pip install "fairchem-core @ git+https://github.com/sterling-group/fairchem-patched.git@main#subdirectory=packages/fairchem-core"
```

The second command installs the patched `fairchem-core` fork. It cannot be a
regular dependency of the `[mlip]` extra — PyPI rejects packages whose metadata
carries direct git dependencies — so it is a documented second step until
upstream `fairchem-core` co-installs with `mace-torch` (see
[MLIP backends](#mlip-backends)). Skip it if you only use MACE.

## From source

```bash
git clone https://github.com/sterling-group/ChemRefine.git
cd ChemRefine
pip install -e .[dev,test,docs,mlip]
pre-commit install   # optional: run ruff + interrogate on every commit
```

## Requirements

- **Python 3.11–3.13**
- **ORCA 6.0+** — quantum-chemistry calculations
- **SLURM** — HPC job scheduler (optional for local runs; the same
  `.slurm` script can be executed with `bash` directly)

The base install pulls `numpy`, `pyyaml`, `pandas`, `ase`, `rdkit`,
`pydantic >= 2`, and `typer >= 0.12`. Optional extras layer backends on top.

### MLIP backends

Each MLIP backend ships in its own extra, so you install only what you use. The
default `[mlip]` install is **MACE + FAIRChem** — they co-exist because
`mace-torch` pins `e3nn == 0.4.4` and Sterling Group's patched `fairchem-core`
fork relaxes the upstream `e3nn >= 0.5` floor down to `>= 0.4.4`, so both share
e3nn 0.4.4. The fork is installed as a **second pip step** (it is a direct git
reference, which PyPI does not allow inside package metadata):

```bash
pip install "fairchem-core @ git+https://github.com/sterling-group/fairchem-patched.git@main#subdirectory=packages/fairchem-core"
```

| Extra | `task_name`(s) it enables | Pulls | Notes |
|-------|---------------------------|-------|-------|
| `[mlip]` | MACE + FAIRChem (the two rows below) | — | default; `[mlff]` is an alias |
| `[mlip-mace]` | `mace_off`, `mace_mp`, `mace_omol`, `custom_mace` | `torch`, `e3nn==0.4.4`, `mace-torch` | |
| `[mlip-fairchem]` | `omol`, `omat`, `odac`, `oc20`, `oc22`, `oc25`, `omc` | `torch`, `e3nn==0.4.4` | UMA / eSEN checkpoints; + the patched `fairchem-core` step above |
| `[mlip-sevenn]` | `sevenn` | `sevenn` (→ `e3nn>=0.5`, `torch-geometric`) | **dedicated env** — `e3nn>=0.5` clashes with MACE's `==0.4.4` |
| `[mlip-orb]` | `orb` | `orb-models` (→ `torch>=2.8`) | **dedicated env**; needs **Python ≥ 3.12** |
| `[mlip-chgnet]` | `chgnet` | `chgnet` (→ `torch`, `pymatgen`) | **dedicated env**; pulls `pymatgen` |

`sevenn` / `orb` / `chgnet` each drag their own (conflicting) torch/e3nn tree, so
install each in a **separate environment** — never combine them with `[mlip]`.
They don't pin a CUDA build of `torch`, so for GPU install the matching `torch`
first. Selecting a backend whose package is missing raises a clear error naming
the extra to install.

### Other extras

- `[pyscf]` — `pyscf` for the PySCF engine / PySCF-ExtOpt gradients
  (plus `[server]`); `[pyscf-gpu]` adds `gpu4pyscf-cuda12x` + `cutensor-cu12`
  (CUDA 12; CUDA-11 hosts swap in the `-cuda11x` wheels).
- `[server]` — just `flask` + `waitress` (the ExtOpt HTTP server);
  pulled in automatically by every MLIP extra and `[pyscf]`.

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
| `ORCA not accessible` | Set `executables: { orca: ... }` in the YAML to an absolute path, or put ORCA on `$PATH`. |
| `sbatch: command not found` | Either SLURM isn't installed locally — run the generated `.slurm` script with `bash` instead — or activate the cluster's SLURM module. |
| `Server crashed during startup` (MLIP) | Check the per-job `server_${SLURM_JOB_ID}.log`; common causes are out-of-memory at model load or a missing HuggingFace token for FAIRChem. |
| `PackageNotFoundError: ChemRefine` at runtime | `pip install -e .` again — the editable install was removed. |

## License

ChemRefine is released under [AGPL v3](https://github.com/sterling-group/ChemRefine/blob/main/LICENSE).

## Getting help

- [Project issues](https://github.com/sterling-group/ChemRefine/issues) — search before opening a new one
- [Example tutorials](../tutorials/index.md)
- [Project README](https://github.com/sterling-group/ChemRefine#readme) for the elevator pitch

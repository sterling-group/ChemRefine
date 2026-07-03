# Installation

ChemRefine is a **light orchestrator core** plus optional **backend environments**. The
core (pipeline, ORCA driving, SLURM) installs anywhere; each MLIP backend lives in its own
environment because their torch/e3nn stacks conflict (MACE pins `e3nn==0.4.4`,
FAIRChem/SevenNet need `e3nn>=0.5`, …) — one Python process can only hold one family.
Two ways to get a backend:

- **single backend** — install its extra straight into the ChemRefine env
  (`pip install "chemrefine[mlip]"` = FAIRChem/UMA) and run as usual;
- **several (conflicting) backends** — keep the core light and provision one *managed env*
  per backend with `chemrefine backends install <extra>`. Envs are resolved **by name** at
  run time — no interpreter paths in your YAML — and reused by every later run.

## Python already installed (pip / venv)

```bash
pip install "chemrefine @ git+https://github.com/sterling-group/ChemRefine.git"   # core
chemrefine backends install mlip-fairchem      # provision the backends you use (once)
```

## From scratch (no Python on the system)

[`uv`](https://docs.astral.sh/uv/) is a single binary that needs no pre-installed Python
and can bootstrap one:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.13
uv tool install "chemrefine @ git+https://github.com/sterling-group/ChemRefine.git"
chemrefine backends install mlip-fairchem
```

## Conda

```bash
conda create -n chemrefine python=3.13 -y && conda activate chemrefine
pip install "chemrefine @ git+https://github.com/sterling-group/ChemRefine.git"
chemrefine backends install mlip-fairchem      # backend envs are built WITH conda too
```

`chemrefine backends install` always builds the managed envs with the **same tool that
created the current env** (conda / uv / venv, detected automatically), so everything stays
in the ecosystem you already use.

## HPC

Provision on a **login node** (needs internet), run anywhere: a managed env is a plain
directory of files — offline compute nodes just execute its `python`. Envs live under
`$CHEMREFINE_HOME` (default: alongside the install when writable, else `~/.chemrefine`);
point it at a project/shared filesystem to share provisioned backends across machines:

```bash
export CHEMREFINE_HOME=/projects/mygroup/chemrefine   # optional; put it on shared storage
chemrefine backends install mlip-mace mlip-fairchem   # once, on the login node
```

## From source

```bash
git clone https://github.com/sterling-group/ChemRefine.git
cd ChemRefine
pip install -e .[dev,test,docs]
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

Each MLIP backend ships in its own extra and its own environment (their torch/e3nn trees
conflict — none co-install). `chemrefine backends install <extra>` provisions the managed
env; alternatively install exactly one extra into the main env for the single-backend case.

| Extra | `task_name`(s) it enables | Pulls |
|-------|---------------------------|-------|
| `[mlip]` = `[mlip-fairchem]` | `omol`, `omat`, `odac`, `oc20`, `oc22`, `oc25`, `omc` | `fairchem-core` (→ `torch`, `e3nn>=0.5`); UMA / eSEN checkpoints, default `uma-s-1p2` |
| `[mlip-mace]` | `mace_off`, `mace_mp`, `mace_omol`, `custom_mace` | `torch`, `e3nn==0.4.4`, `mace-torch` |
| `[mlip-sevenn]` | `sevenn` | `sevenn` (→ `e3nn>=0.5`, `torch-geometric`) |
| `[mlip-orb]` | `orb` | `orb-models` (→ `torch>=2.8`); needs **Python ≥ 3.12** |
| `[mlip-chgnet]` | `chgnet` | `chgnet` (→ `torch`, `pymatgen`) |

`[mlff]` remains an alias of `[mlip]`. The backends don't pin a CUDA build of `torch`, so
for GPU install the matching `torch` first (or let the extra resolve the default build).
Every run validates its steps' backends **up front**: a step whose backend is neither
importable nor provisioned fails before any job submits, naming the
`chemrefine backends install <extra>` fix.

### Multiple MLIP backends in one run

Because each step resolves its own backend env by name, one pipeline can mix models whose
dependency stacks conflict — e.g. a broad MACE screen refined by UMA:

```bash
chemrefine backends install mlip-mace mlip-fairchem   # once
```

```yaml
steps:
  - step: 1                       # broad screen — MACE (e3nn 0.4.4)
    engine: mlip
    operation: opt_sp
    options: { task_name: mace_off, model_name: medium }
    sample: { method: boltzmann, percent_cumulative: 99 }

  - step: 2                       # refine — UMA (e3nn >= 0.5)
    engine: mlip
    operation: opt_sp
    options: { task_name: omol, model_name: uma-s-1p2 }
    sample: { method: min, count: 5 }
```

Step 1 runs each structure with the `mlip-mace` env's Python, step 2 with
`mlip-fairchem`'s — the conflicting stacks never share a process. The same works for
`mlip-extopt` steps (the gradient server launches from the step's env) and for `pyscf`.
The `options.backend_python` knob overrides the resolution with an explicit interpreter
(escape hatch — normally envs are resolved by name only).

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
chemrefine backends list                        # known backends + provisioned envs
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
| `backend '…' is not available` at run start | The step's backend is neither importable nor provisioned — run `chemrefine backends install <extra>` (on HPC: on a login node), or install `chemrefine[<extra>]` into the main env. |
| `PackageNotFoundError: ChemRefine` at runtime | `pip install -e .` again — the editable install was removed. |

## License

ChemRefine is released under [AGPL v3](https://github.com/sterling-group/ChemRefine/blob/main/LICENSE).

## Getting help

- [Project issues](https://github.com/sterling-group/ChemRefine/issues) — search before opening a new one
- [Example tutorials](../tutorials/index.md)
- [Project README](https://github.com/sterling-group/ChemRefine#readme) for the elevator pitch

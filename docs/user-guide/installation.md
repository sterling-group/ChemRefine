# Installation

ChemRefine is a **light orchestrator core** plus optional **compute backends**. The core
(pipeline, ORCA and Q-Chem driving, SLURM) installs anywhere in seconds; neither
quantum-chemistry program needs a backend environment of its own. Backends (MLIPs,
PySCF) are added afterwards — either into the same environment or as isolated
[managed environments](#compute-backends) when you use several.

## Requirements

- **Python 3.11–3.14**
- **ORCA 6.0+** — quantum-chemistry calculations
- **Q-Chem** — optional alternative QM program ([its own options and install
  keys](configuration.md#engine-options))
- **SLURM** — HPC job scheduler (optional for local runs; the same `.slurm` script can be
  executed with `bash` directly)

The base install pulls `numpy`, `pyyaml`, `pandas`, `ase`, `rdkit`, `pydantic >= 2`, and
`typer >= 0.12`. Optional extras layer backends on top.

## Install ChemRefine

!!! note "PyPI release pending"
    The `chemrefine` package name below refers to the upcoming **v2.0.0 PyPI release**.
    Until it is published, substitute the [Git form](#from-git-until-the-pypi-release)
    wherever `chemrefine` appears as an install target — everything else is identical.

=== "pip"

    ```bash
    python -m venv ~/chemrefine-env && source ~/chemrefine-env/bin/activate
    pip install chemrefine
    ```

=== "uv"

    [`uv`](https://docs.astral.sh/uv/) is a single binary that needs no pre-installed
    Python and can bootstrap one — the fastest path on a bare system:

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv python install 3.13        # only if the system has no Python
    uv tool install chemrefine    # isolated install, `chemrefine` on PATH
    ```

=== "conda"

    ```bash
    conda create -n chemrefine python=3.13 -y
    conda activate chemrefine
    pip install chemrefine
    ```

### From Git (until the PyPI release)

The current development version installs straight from GitHub — a drop-in replacement for
`chemrefine` in any command above:

```bash
pip install "chemrefine @ git+https://github.com/sterling-group/ChemRefine.git"
```

### From source (contributors)

```bash
git clone https://github.com/sterling-group/ChemRefine.git
cd ChemRefine
pip install -e ".[dev]"
pre-commit install   # REQUIRED — CI runs these same hooks
```

## Compute backends

The MLIP libraries have mutually incompatible dependency stacks, so one Python process can
only host one backend family. Two ways to add one:

- **Single backend** — install its extra straight into the ChemRefine env and run as usual:

    ```bash
    pip install "chemrefine[mlip]"        # FAIRChem / UMA (the default backend)
    ```

- **Several (conflicting) backends** — keep the core light and provision one *managed
  environment* per backend:

    ```bash
    chemrefine backends install mlip-mace mlip-fairchem
    ```

    Each env is built once with the **same tool that created the current env** (conda / uv
    / venv, detected automatically), reused by every later run, and resolved **by name** at
    run time — no interpreter paths in your YAML. Managed envs are built **from the same
    source as the orchestrator**: a PyPI install pins the version, a Git or
    `pip install -e .` install reinstalls from that same repo/checkout. `chemrefine backends
    list` shows what is provisioned. Every run validates its steps' backends **up front**: a
    step whose backend is neither importable nor provisioned fails before any job submits,
    naming the fix.

### Available backends

These are the names `chemrefine backends install` accepts — and the only ones it accepts.
`[mlip]` and `[mlff]` are pip extras that alias `[mlip-fairchem]`; neither is a backend
name.

<!-- chemrefine:backends -->

The backends don't pin a CUDA build of `torch`, so for GPU install the matching `torch`
first (or let the extra resolve the default build). FAIRChem's default checkpoint is
`uma-s-1p2`.

`task_name` is the only key that selects a backend. To run a model you fine-tuned yourself,
name the library that trained it and point `model_path` at the checkpoint — there is no
`custom_<library>` task for any of them. (`custom_mace` still resolves, as a MACE alias kept
for v1 configs.)

### Multiple MLIP backends in one run

Because each step resolves its own backend env by name, one pipeline can mix models whose
dependency stacks conflict — e.g. a broad MACE screen refined by UMA:

```bash
chemrefine backends install mlip-mace mlip-fairchem   # once
```

```yaml
steps:
  - step: 1                       # broad screen — MACE
    engine: mlip
    operation: opt_sp
    options: { task_name: mace_off, model_name: medium }
    sample: { method: boltzmann, percent_cumulative: 99 }

  - step: 2                       # refine — UMA
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

### Every pip extra

The full `pip install "chemrefine[…]"` vocabulary, including the ones that are not
compute backends:

<!-- chemrefine:extras -->

`[server]` is flask + waitress — the ExtOpt HTTP server — and every backend extra pulls
it in. `[gui]` is the workflow builder, `[mcp]` the AI-agent tool server, `[agent]` the
terminal chat; `[dev]`, `[docs]` and `[test]` are for working on ChemRefine itself.
`[pyscf-gpu]` targets CUDA 12 — CUDA-11 hosts swap in the `-cuda11x` wheels.

## HPC

Provision on a **login node** (needs internet), run anywhere: a managed env is a plain
directory of files — offline compute nodes just execute its `python`. Envs live under
`$CHEMREFINE_HOME` (default: alongside the install when writable, else `~/.chemrefine`);
point it at a project/shared filesystem to share provisioned backends across machines:

```bash
export CHEMREFINE_HOME=/projects/mygroup/chemrefine   # optional; put it on shared storage
chemrefine backends install mlip-mace mlip-fairchem   # once, on the login node
```

## Verification

```bash
chemrefine --version
chemrefine --help
chemrefine backends list                        # known backends + provisioned envs
chemrefine run examples/quickstart/input.yaml --dry-run    # validates the YAML, no jobs run
```

## FAIRChem model access

The UMA / OMol checkpoints are gated on Hugging Face: request access to the
[UMA repository](https://huggingface.co/facebook/UMA) and authenticate your machine with a
Hugging Face token. Follow the
[FAIRChem documentation](https://fair-chem.github.io/) for the current access +
authentication steps — they are defined upstream and may change.

## Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| `chemrefine: command not found` | Activate the env where you installed ChemRefine (`pip show chemrefine` to confirm). |
| `ORCA not accessible` | Set `executables: { orca: ... }` in the YAML to an absolute path, or put ORCA on `$PATH`. |
| `sbatch: command not found` | Either SLURM isn't installed locally — run the generated `.slurm` script with `bash` instead — or activate the cluster's SLURM module. |
| `Server crashed during startup` (MLIP) | Check the per-job `server_${SLURM_JOB_ID}.log`; common causes are out-of-memory at model load or a missing HuggingFace token for FAIRChem. |
| `backend '…' is not available` at run start | The step's backend is neither importable nor provisioned — run `chemrefine backends install <extra>` (on HPC: on a login node), or install `chemrefine[<extra>]` into the main env. |
| `No matching distribution found for chemrefine==…` during `backends install` | Upgrade ChemRefine — older versions could only provision backends from a published PyPI release; current ones reinstall from the same Git/source install as the orchestrator. |
| `sbatch failed …` on a machine that isn't a cluster | An `sbatch` binary on PATH made auto-detection pick SLURM; set `dispatch: local` in the YAML to force the local runner. |
| `PackageNotFoundError: ChemRefine` at runtime | `pip install -e .` again — the editable install was removed. |

## License

ChemRefine is released under
[AGPL v3](https://github.com/sterling-group/ChemRefine/blob/main/LICENSE).

## Getting help

- [Project issues](https://github.com/sterling-group/ChemRefine/issues) — search before opening a new one
- [Example tutorials](../tutorials/index.md)
- [Project README](https://github.com/sterling-group/ChemRefine#readme) for the elevator pitch

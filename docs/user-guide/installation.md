# Installation

ChemRefine is a **light orchestrator core** plus optional **compute backends**. The core
(pipeline, ORCA driving, SLURM) installs anywhere in seconds; backends (MLIPs, PySCF, Qiskit) are
added afterwards — either into the same environment or as isolated
[managed environments](#compute-backends) when you use several.

## Requirements

- **Python 3.11–3.13**
- **ORCA 6.0+** — quantum-chemistry calculations
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
pre-commit install   # optional: run ruff + interrogate on every commit
```

## Compute backends

The MLIP libraries have mutually incompatible dependency stacks, so one Python process can
only host one backend family. Two ways to add one:

- **Single backend** — install its extra straight into the ChemRefine env and run as usual:

    ```bash
    pip install "chemrefine[mlip]"        # FAIRChem / UMA (the default backend)
    # or, for the Qiskit engine:
    pip install "chemrefine[qiskit]"      # Qiskit Nature + Algorithms + PySCF
    # or "chemrefine[qiskit-aer]" to add Aer estimators
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

### MLIP backends

| Extra | `task_name`(s) it enables | Pulls |
|-------|---------------------------|-------|
| `[mlip]` = `[mlip-fairchem]` | `omol`, `omat`, `odac`, `oc20`, `oc22`, `oc25`, `omc` | `fairchem-core`; UMA / eSEN checkpoints, default `uma-s-1p2` |
| `[mlip-mace]` | `mace_off`, `mace_mp`, `mace_omol` | `mace-torch` |
| `[mlip-sevenn]` | `sevenn` | `sevenn` |
| `[mlip-orb]` | `orb` | `orb-models`; needs **Python ≥ 3.12** |
| `[mlip-chgnet]` | `chgnet` | `chgnet` |

`[mlff]` remains an alias of `[mlip]`. The backends don't pin a CUDA build of `torch`, so
for GPU install the matching `torch` first (or let the extra resolve the default build).

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
The same managed-environment resolution applies to `qiskit`. The
`options.backend_python` knob overrides the resolution with an explicit interpreter
(escape hatch — normally envs are resolved by name only).

### Other extras

- `[pyscf]` — `pyscf` for the PySCF engine / PySCF-ExtOpt gradients (plus `[server]`);
  `[pyscf-gpu]` adds `gpu4pyscf-cuda12x` + `cutensor-cu12` (CUDA 12; CUDA-11 hosts swap in
  the `-cuda11x` wheels).
- `[qiskit]` — Qiskit `>=1.4,<2.0`, Qiskit Nature `>=0.8,<0.9`, Qiskit Algorithms
  `>=0.4,<0.5`, and PySCF for modular exact, VQE, and ADAPT-VQE
  electronic-structure single points.
- `[qiskit-aer]` — standard CPU Qiskit Aer `>=0.17,<0.18` plus everything in
  `[qiskit]`, enabling the `aer_statevector` and `aer_shots` estimators. See
  [Qiskit ground-state calculations](qiskit.md).
- `[server]` — just `flask` + `waitress` (the ExtOpt HTTP server); pulled in automatically
  by every MLIP extra and `[pyscf]`.

### Qiskit Nature

The Qiskit engine can share the main environment:

```bash
pip install "chemrefine[qiskit]"
# Or include Aer:
pip install "chemrefine[qiskit-aer]"
```

For a source checkout, use:

```bash
pip install -e ".[qiskit]"
# Or include Aer:
pip install -e ".[qiskit-aer]"
```

Or provision it as a managed backend so the scientific stack stays isolated:

```bash
chemrefine backends install qiskit
# Use this backend name for Aer estimators:
chemrefine backends install qiskit-aer
chemrefine backends list
```

The four built-in estimators all run locally and need no cloud credentials:
`statevector`, the lightweight shot-based `basic_backend`, exact-expectation
`aer_statevector`, and finite-shot `aer_shots`. The first two use `[qiskit]`;
the Aer estimators require `[qiskit-aer]`. ChemRefine resolves their managed
backend names accordingly, so a mixed pipeline may provision both `qiskit` and
`qiskit-aer`. IBM Runtime and quantum-hardware submission are not included.

Aer GPU wheels are a separate Linux distribution and require a compatible CUDA
stack. Build a managed or custom backend environment in which
`qiskit-aer-gpu>=0.17,<0.18` replaces standard `qiskit-aer`; do not install both
distributions in one environment. A custom environment needs the same pinned
Qiskit, Nature, Algorithms, PySCF, and ChemRefine source/version as the main
installation. Point a step at its interpreter only when it cannot be
provisioned under the normal `qiskit-aer` backend name:

```yaml
steps:
  - step: 1
    engine: qiskit
    operation: sp
    options:
      estimator: aer_statevector
      device: cuda
      backend_python: /absolute/path/to/qiskit-gpu-env/bin/python
```

`device: cuda` makes ChemRefine request GPU capacity and tells Aer to use its
GPU device. It is valid only for an Aer estimator. GPU simulation changes where
the classical simulator runs; it does not connect the job to quantum hardware.

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

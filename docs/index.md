![chemrefinelogo](https://github.com/user-attachments/assets/ae7b1ad5-0d90-445c-be83-ddcb76fa85c3)

# ChemRefine

Automated, interoperable manager for computational-chemistry workflows.
A YAML file describes a chain of stages (e.g. MLIP screen → DFT refine
→ frequency analysis); ChemRefine writes the engine inputs, submits
SLURM jobs under a global core budget, parses the outputs, filters
survivors by energy, and feeds the next stage. Re-runs are fingerprint-
cached so unchanged steps skip automatically.

[Get started](user-guide/index.md){ .md-button .md-button--primary }
[How it works](concepts/index.md){ .md-button }

---

## Features

- **Step-based pipeline** driven by a single Pydantic-validated YAML
  config — every knob lives in one place.
- **Engine plugins** behind a narrow Protocol: ORCA, MLIPs
  (MACE / FAIRChem / SevenNet / ORB / CHGNet), and PySCF — each as a
  direct engine or an ORCA-driven `-extopt` gradient server. New engines
  drop in via a registry decorator (see [Adding an Engine](developer/adding-an-engine.md)).
- **Filtering** by Boltzmann cumulative weight, or the lowest / highest
  structures by count or energy window (`min` / `max`, the latter PES-style),
  with optional per-parent grouping.
- **SLURM-aware**: the PAL/`nprocs` value in each input participates in a
  total-core throttle so concurrent jobs never exceed `max_cores`; whole
  steps can also go out as job arrays.
- **Resumable**: each step's parsed results are cached with a SHA-1
  fingerprint of the step config + the parent structures (IDs and content).
  A change anywhere in that surface invalidates the cache.
- **Hierarchical IDs**: every conformer carries its lineage
  (`0` → `0-1` → `0-1-2`) so survivors trace back to their root structure.

## Quickstart

Install (see the [install guide](user-guide/installation.md) for backends + GPU):

```bash
pip install "chemrefine @ git+https://github.com/sterling-group/ChemRefine.git"
```

Describe the pipeline in one YAML file (full
[configuration reference](user-guide/configuration.md)):

```yaml
template_dir: ./templates
output_dir:   ./outputs
input:        ./step1.xyz
max_cores: 64
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
    sample: { method: min, window_kcalmol: 3.0 }
```

Run it (full [CLI reference](user-guide/cli.md)):

```bash
chemrefine run input.yaml                   # full pipeline from step 1
chemrefine resume input.yaml                # skip cached steps; retry failures
chemrefine run input.yaml --maxcores 128    # override max_cores
chemrefine run input.yaml --dry-run         # validate + describe; no jobs
```

## Where to go next

- **[User Guide](user-guide/index.md)** — install, the YAML configuration
  reference, and the CLI.
- **[Tutorials](tutorials/index.md)** — worked examples: conformer sampling,
  TS finding, docking, MLIP training, redox, spin.
- **[Concepts](concepts/index.md)** — the architecture & code flow, the
  fingerprint cache, filtering, and normal-mode sampling.
- **[API Reference](api/index.md)** — the orchestration core and the engine contract.
- **[Migrating from v1.3.1](migrating-from-main.md)** — old keys/flags map to v2.

Requires **Python 3.11–3.13** and **ORCA 6+**; SLURM is optional (the generated
`.slurm` script runs unchanged under `bash` locally).

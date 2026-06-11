![chemrefinelogo](https://github.com/user-attachments/assets/ae7b1ad5-0d90-445c-be83-ddcb76fa85c3)

# ChemRefine

Automated, interoperable manager for computational-chemistry workflows.
A YAML file describes a chain of stages (e.g. MLFF screen → DFT refine
→ frequency analysis); ChemRefine writes the engine inputs, submits
SLURM jobs under a global core budget, parses the outputs, filters
survivors by energy, and feeds the next stage. Re-runs are fingerprint-
cached so unchanged steps skip automatically.

---

## Features

- **Step-based pipeline** driven by a single Pydantic-validated YAML
  config — every knob lives in one place.
- **Engine plugins** with a narrow Protocol: ORCA, MLIPs
  (MACE / FAIRChem / SevenNet / ORB / CHGNet), and PySCF — each as a
  direct engine or an ORCA-driven ``-extopt`` gradient server. New
  engines drop in via a registry decorator.
- **Filtering** by Boltzmann cumulative weight, energy window,
  fixed integer count, or "highest N" (PES-style), with optional
  per-parent grouping.
- **SLURM-aware**: the PAL/`nprocs` value in each input file
  participates in a total-core throttle so concurrent jobs never exceed
  ``max_cores``.
- **Resumable**: each step's parsed results are pickled with a SHA-1
  fingerprint of the step config + the parent structures (IDs and
  content). A change anywhere in that surface — the YAML, the seed
  file, an upstream result — invalidates the cache.
- **Hierarchical IDs**: every conformer carries its lineage
  (``0`` → ``0-1`` → ``0-1-2``) so survivors can be traced back to
  their root structure across all steps.

## Install

See the [install guide](INSTALL.md) for the full setup, including GPU
backends, MACE checkpoints, and FairChem authentication.

Quick path:

```bash
pip install "chemrefine @ git+https://github.com/sterling-group/ChemRefine.git@main"
pip install "chemrefine[mlip] @ git+https://github.com/sterling-group/ChemRefine.git@main"
pip install "fairchem-core @ git+https://github.com/sterling-group/fairchem-patched.git@main#subdirectory=packages/fairchem-core"
```

Requires **Python 3.11–3.13** and **ORCA 6+**; SLURM is optional. The
``[mlip]`` extra pulls torch and mace-torch plus the gradient-server deps
(flask, waitress); the patched ``fairchem-core`` fork is the documented
second pip step above (PyPI forbids direct git deps in package metadata).
``[mlff]`` is an alias for ``[mlip]``.

## Quickstart

### 1 — Prepare inputs

Three things on disk:

- A YAML config (see the schema below).
- A starting geometry: an ``.xyz`` file (multi-frame files seed one
  structure per frame), a directory of ``.xyz`` files, or a CSV of
  SMILES (one column named ``smiles``).
- An ORCA input template per step in ``template_dir`` (default name:
  ``stepN.inp``; override with ``template:`` on the step).

### 2 — Write the YAML

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
    name: screen              # optional label → directory step1_screen/
    engine: mlip
    operation: opt_sp
    options:
      model_name: medium
      task_name: mace_off
      device: cuda
    sample:
      method: boltzmann
      percent_cumulative: 99

  - step: 2
    name: refine
    engine: orca
    operation: opt_sp
    template: dft_opt.inp
    charge: -1                 # per-step overrides allowed
    multiplicity: 2
    sample:
      method: energy_window
      window_kcal: 3.0

  - step: 3
    engine: orca
    operation: opt_sp
    sample: { method: integer, count: 5 }
    nms: true                  # opt-in normal-mode sampling
```

### 3 — Run

```bash
chemrefine run input.yaml                   # full pipeline from step 1
chemrefine resume input.yaml                # skip cached steps
chemrefine rebuild-cache input.yaml refine  # invalidate one step
chemrefine rebuild-nms input.yaml 3
chemrefine rerun input.yaml                 # rerun the latest step
chemrefine run input.yaml --maxcores 128    # override YAML's max_cores
chemrefine run input.yaml --dry-run         # validate + describe; no jobs
```

Step targets accept either the numeric ``step:`` or the optional
``name:``. The output layout is:

```
outputs/
├── step1_screen/      *.inp *.out *.xyz  _cache/
├── step2_refine/      *.inp *.out *.xyz  _cache/
├── step3/             *.inp *.out *.xyz  _cache/
└── steps.csv          Boltzmann summary per surviving structure
```

## Sampling methods

| Method          | Required field          | Description                                            |
|-----------------|-------------------------|--------------------------------------------------------|
| `boltzmann`     | `percent_cumulative`    | Keep until cumulative Boltzmann weight reaches this %  |
| `energy_window` | `window_kcal`           | Keep all within window (kcal/mol) of the lowest energy |
| `integer`       | `count`                 | Keep the `count` lowest-energy structures              |
| `high_energy`   | `count`                 | Keep the `count` *highest*-energy (e.g. PES sampling)  |

Add `by_parent: true` to apply the method per parent-ID group instead
of globally — useful for keeping diversity across branches after a
fan-out step.

## Engines

| Engine          | Driver                        | Backend                                       |
|-----------------|-------------------------------|-----------------------------------------------|
| `orca`          | SLURM + ORCA                  | DFT via ORCA 6+                                |
| `mlip`          | template script (SLURM/local) | MACE / FAIRChem / SevenNet / ORB / CHGNet, direct (no ORCA) |
| `mlip-extopt`   | SLURM + ORCA + server         | MLIP gradient server driving ORCA `%method`   |
| `mlip-train`    | template script (SLURM/local) | MLIP model training                           |
| `pyscf`         | template script (SLURM/local) | PySCF / gpu4pyscf, direct (no ORCA)           |
| `pyscf-extopt`  | SLURM + ORCA + server         | PySCF gradient server driving ORCA `%method`  |
| `fake`          | in-process                    | Test stub used by the pipeline tests          |

## Operations

Engines decide which operations they support. The current ORCA mapping:

| Operation     | Description                                                  |
|---------------|--------------------------------------------------------------|
| `opt_sp`      | Geometry optimisation + single point                         |
| `goat`        | GOAT conformer ensemble                                      |
| `pes`         | PES scan (one frame per converged scan point)                |
| `docker`      | Host–guest docking ensemble                                  |
| `solvator`    | Explicit solvation ensemble                                  |
| `mlip_train`  | MLIP model training (writes inputs + submits training job)   |

Engines that don't know how to handle a given operation raise
`OutputParseError` (or, for engines that can't perform an operation,
`NotImplementedError`) so the pipeline fails loudly instead of
returning empty data.

## Caching & resumability

Each step writes `{step_dir}/_cache/step.pkl` plus a JSON sidecar with
a SHA-1 fingerprint of the step's full config + parent structure IDs.
The next invocation skips the step if the fingerprint matches. Change
the YAML — even one field — and that step (and every downstream step
that depended on its survivors) re-runs.

The matching JSON sidecar is human-readable so you can `cat` it to
audit what was cached without unpickling.

## Tutorials

Worked examples covering the patterns from our publication live under
[Tutorials](tutorials/index.md):

- [Conformer Sampling](tutorials/conformer_sampling.md)
- [Transition State Finding](tutorials/ts_finding.md)
- [Host–Guest Docking and Microsolvation](tutorials/host-guest.md)
- [MLIP Training and Normal Mode Sampling](tutorials/mlip_training.md)
- [Redox properties](tutorials/redox.md)
- [Spin Properties](tutorials/spin.md)

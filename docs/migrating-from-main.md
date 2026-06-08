# Migrating from v1.3.1 (main) YAML

ChemRefine v2 loads your existing v1.3.1 workflow YAML **unchanged** — the config
loader rewrites the old keys to the current schema at parse time (you'll see a
one-line deprecation warning per legacy feature). This page documents that
mapping so you can modernise your files when convenient.

Everything below is rewritten automatically **except `calculation_type`**, which
must be replaced by hand with `engine:` + `operation:`.

## Top level

| v1.3.1                          | v2                                |
|---------------------------------|-----------------------------------|
| `orca_executable: /path/orca`   | `executables: { orca: /path/orca }` |
| `initial_xyz: ./seed.xyz`       | `input: ./seed.xyz`               |

## Per step

| v1.3.1                                         | v2                                            |
|------------------------------------------------|-----------------------------------------------|
| `engine: "DFT"`                                | `engine: orca`                                |
| `engine: "MLFF"` + an `mlff:` block            | `engine: mlip-extopt` (ORCA-driven, as before)|
| `engine: "MLFF"` / `mlff` (no block)           | `engine: mlip` (the **new** direct, in-process path) |
| `mlff:` / `pyscf:` / `trainer:` block          | `options: { … }` (the engine block → options) |
| `operation: "OPT+SP"` / `"GOAT"`               | `operation: opt_sp` / `goat`                  |
| `operation: "MLFF_TRAIN"`                      | `engine: mlip-train`                          |
| `normal_mode_sampling: true`                   | `nms: true`                                   |
| `normal_mode_sampling_parameters: { … }`       | folded into `options:` (see NMS table below)  |
| `calculation_type: "DFT"` **(manual)**         | `engine: orca` + `operation: opt_sp`          |

### Normal-mode sampling knobs

`normal_mode_sampling_parameters` keys move into `options:` and are renamed:

| v1.3.1                          | v2 `options` key       | notes |
|---------------------------------|------------------------|-------|
| `calc_type: rm_imag` (default)  | `target: ts`           | keep exactly one imaginary mode (first-order saddle / TS) |
| `calc_type: random`             | `target: random`       | sampling along random modes |
| `displacement_vector`           | `displacement_value`   | ± displacement magnitude (Å) |
| `num_random_displacements`      | `num_random_displacements` | unchanged |

> v2 adds a third target, `target: minimum` (remove **all** imaginary modes → a true minimum), which
> v1.3.1 did not have. A bare `normal_mode_sampling: true` maps to `target: ts` to match main's default.

> **Scheduling.** NMS runs as **two throttled phases with a barrier**. Round 1 (every structure) completes
> under the `max_cores`/`max_gpus` budget, *then* round 2 — the imaginary-frequency removal of the flagged
> structures — runs as a **separate** throttled batch that inherits the same budget, `device`, and SLURM
> header. So removal starts only **after** round 1 fully drains, not the moment an individual structure
> finishes; the two rounds never share the budget at the same time.

> **MLFF was ORCA-driven in v1.3.1.** `engine: MLFF` ran ORCA using a machine-learned
> gradient server — that is `mlip-extopt` in v2. The bare `mlip`/`mlff` engine in v2
> is the *new* direct (template-driven, no ORCA) path. A v1.3.1 step is recognised by
> its `mlff:` block and mapped to `mlip-extopt` so it keeps doing what it did.

## Sampling

`sample_type: { method, parameters: { … } }` becomes a flat `sample: { method, … }`,
with these per-method parameter renames:

| method         | v1.3.1 parameter        | v2 key               |
|----------------|-------------------------|----------------------|
| `boltzmann`    | `weight`                | `percent_cumulative` |
| `integer`      | `num_structures`        | `count`              |
| `high_energy`  | `num_structures`        | `count`              |
| `energy_window`| `energy` (+ `unit`)     | `window_kcal`        |

## Example

```yaml
# v1.3.1
orca_executable: /orca
initial_xyz: ./seed.xyz
steps:
  - step: 1
    operation: "OPT+SP"
    engine: "MLFF"
    mlff: { model_name: uma-s-1, task_name: omol, device: cuda }
    sample_type: { method: boltzmann, parameters: { weight: 95 } }

# v2 (what the loader produces; also what you'd write by hand)
executables: { orca: /orca }
input: ./seed.xyz
steps:
  - step: 1
    operation: opt_sp
    engine: mlip-extopt
    options: { model_name: uma-s-1, task_name: omol, device: cuda }
    sample: { method: boltzmann, percent_cumulative: 95 }
```

## Command line

v1.3.1 was flag-style; v2 uses subcommands. Old invocations are translated automatically (you'll see a
one-line warning), so existing scripts keep working:

| v1.3.1                              | v2                                       |
|-------------------------------------|------------------------------------------|
| `chemrefine CONFIG`                 | `chemrefine run CONFIG`                  |
| `chemrefine CONFIG --skip`          | `chemrefine resume CONFIG`               |
| `chemrefine CONFIG --rebuild_cache [N]` | `chemrefine rebuild-cache CONFIG [N]` |
| `chemrefine CONFIG --rebuild_nms [N]`   | `chemrefine rebuild-nms CONFIG [N]`   |
| `chemrefine CONFIG --rerun_errors [N]`  | `chemrefine rerun-errors CONFIG [N]`  |
| `--maxcores N`                      | `--maxcores N` (unchanged)               |

## Failure handling & recovery

Each step takes `on_failure: stop | skip | best` (in the YAML). For a step where, say, 2 of 5 structures
fail:

- **`skip`** (default) — drop the 2 failures and continue with the 3 survivors.
- **`best`** — keep all 5, backfilling the 2 failures with the best geometry obtained.
- **`stop`** — run every structure to completion, cache the 3 successes, then **halt** before the next
  step.

The `_cache/failed_jobs.json` ledger records which structures failed under **every** policy (so you can
always see them), but only a `stop` step leaves failures *pending*. To recover:

- **`chemrefine resume CONFIG`** — re-attempt the pending failures (only a `stop` step has any) and
  continue.
- **`chemrefine rerun-errors CONFIG [N]`** — re-attempt only step N's pending failures (latest if no N).
- **`chemrefine rerun CONFIG [N]`** — redo the whole step N from scratch.

## Cores, GPUs & devices

`max_cores` is a **CPU** budget; jobs declare their core count (ORCA from `%pal nprocs`, the direct
`mlip`/`pyscf` engines from `options.cores`) and the throttler keeps the in-flight total under it.

- **On SLURM**, the scheduler is the real arbiter of both CPU (`cpu` TRES) and GPU (`gres/gpu`, where a unit
  may be a whole card *or* a MIG slice, so several jobs can share one physical GPU). `max_cores` is just
  chemrefine's dispatch ceiling; GPU placement is left to SLURM.
- **Locally** (no `sbatch`), chemrefine is the only arbiter: jobs now run **in the background, in parallel**
  under `max_cores`, with each thread-based engine (pyscf/mlip) pinned to its `options.cores` via
  `OMP_NUM_THREADS` (ORCA stays MPI with `OMP=1`). A new **`max_gpus`** budget caps concurrent GPU jobs —
  default auto: **unlimited under SLURM**, the **detected device count** (`nvidia-smi -L`, counting MIG
  instances) off-cluster — so a single-GPU desktop serialises CUDA jobs while CPU jobs keep parallelising.
  Concurrent local GPU jobs are pinned to distinct devices via `CUDA_VISIBLE_DEVICES`.

`device: cuda` (mlip) / `gpu: true` (pyscf) sets the in-process compute device **and** makes a run step
request a GPU node — the engine auto-picks `cuda.slurm.header` over the global `slurm_template`. A per-step
`slurm_template:` override wins over that pick (for multiple GPU partitions). ORCA is CPU-only, so a "GPU run"
is ORCA-on-CPUs + the gradient server-on-GPU, **one job** on a GPU (or MIG) node.

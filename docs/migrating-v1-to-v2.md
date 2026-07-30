# Migrating from v1 to v2

ChemRefine v2 loads your existing v1.3.1 workflow YAML **unchanged** — the config
loader rewrites the old keys to the current schema at parse time (you'll see a
one-line deprecation warning per legacy feature). This page documents that
mapping so you can modernise your files when convenient.

Everything below is rewritten automatically **except `calculation_type`**, which
must be replaced by hand with `engine:` + `operation:`.

!!! warning "The compatibility layer is scheduled for removal in 3.0"

    Reading a v1 file is a **2.x** guarantee, not a permanent one. Every key on this
    page was renamed before 2.0, so the translation has nothing left to learn — it can
    only accumulate. Two modules hold all of it, and 3.0 deletes both:

    | module | translates |
    |--------|-----------|
    | `chemrefine/config_legacy.py` | the YAML keys on this page |
    | `chemrefine/cli_legacy.py` | the v1 flag-style command line (`--input x.yaml --skip`) |

    Nothing else in the package knows the old names, so a 3.0 config file is a 2.x
    config file that emitted no deprecation warnings. **Run once on 2.x, fix what it
    warns about, and you are done** — there is no separate migration step later.

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

`sample_type: { method, parameters: { … } }` becomes a flat `sample: { method, … }`.
The methods were also renamed — `integer`/`high_energy` are now `min`/`max`, and
`energy_window` folded into `min` with a `window_kcalmol` knob:

| v1.3.1 method (+ parameter) | v2 |
|-----------------------------|----|
| `boltzmann` (`weight`)              | `boltzmann` (`percent_cumulative`) |
| `integer` (`num_structures`)        | `min` (`count`) |
| `high_energy` (`num_structures`)    | `max` (`count`) |
| `energy_window` (`energy` + `unit`) | `min` (`window_kcalmol`) |

`min` / `max` take **exactly one** of `count` / `window_kcalmol`. All the old
spellings auto-translate with one deprecation warning each.

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

## v2 default changes

A few defaults differ from earlier expectations (all overridable):

- **`on_failure` defaults to `stop`** (was `skip`) — failures halt the run after caching successes rather
  than being silently dropped. Set `on_failure: skip` per step to restore the drop-and-continue behaviour.
- **`max_cores` defaults to `4`** (a safe local default) — raise it in the YAML or with `--maxcores`.
- **`--maxgpus`** is a new flag mirroring `--maxcores` (overrides `max_gpus`).
- **PySCF**: `basis` is now required (and `xc` is required for `method: dft`) — no silent level of theory;
  `df` defaults **on**; `gpu` is derived from a new `device` knob (`cuda` ⇒ attempt GPU, with CPU fallback).
- **Direct `pyscf` templates** can now read `$METHOD` / `$XC` / `$BASIS` (parity with direct `mlip`).

## Failure handling & recovery

Each step takes `on_failure: stop | skip | best` (in the YAML). For a step where, say, 2 of 5 structures
fail:

- **`stop`** (default) — run every structure to completion, cache the 3 successes, then **halt** before
  the next step, so failures are never silently dropped.
- **`skip`** — drop the 2 failures and continue with the 3 survivors.
- **`best`** — keep all 5, backfilling the 2 failures with the best geometry obtained.

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

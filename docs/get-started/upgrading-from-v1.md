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
    | `chemrefine/cli_legacy.py` | the v1 flag-style command line (`x.yaml --skip`) |

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

> **Scheduling.** The two rounds share **one** queue and one `max_cores`/`max_gpus` budget. A structure's
> displaced children are submitted the moment *its own* round-1 job finishes, alongside whatever is still
> running — not after the whole of round 1 drains, and not one parent at a time. So the slots freed by
> the early finishers go to round-2 work instead of idling, and a step with 50 unresolved structures
> uses the whole budget rather than one parent's worth of it. See
> [Normal-Mode Sampling](../workflow/nms.md).

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

The `energy` value also changes **units** on the way: v1 read it as hartree unless
`unit: kcal/mol` was explicit, and `window_kcalmol` is kcal/mol by definition — so the
translation converts the number (`energy: 0.5` becomes `window_kcalmol: 313.755`) and
the deprecation warning names both values. An explicit `unit: kcal/mol` crosses
unchanged.

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

v1 had no per-step failure policy and no way to re-attempt part of a run. v2 has
`on_failure: stop | skip | best` per step and six recovery commands, all described in
[When a run fails](../running/when-a-run-fails.md) — the same page a v2 user reaches
for, rather than a second account of it here.

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

## Adopting a v1 output tree

The compatibility layer reads v1 **configs**; it does not read v1 **output trees**. If you
have a finished v1 run whose calculations you want to keep — days of DFT that v2's
`rebuild-cache` could re-parse instead of re-running — the tree has to be brought to the
v2 layout first. The differences are mechanical:

| | v1.3.1 | v2 |
|---|--------|----|
| File layout | flat: `step1/step1_structure_0.out` | per-structure dirs: `step1/0/step1_0.out` |
| Basenames | `step{N}_structure_{id}.*` | `step{N}_{id}.*` |
| Ensemble sidecars | on an `_opt` base: `step1_structure_0_opt.finalensemble.xyz` | on the `.out` stem: `step1_0.finalensemble.xyz` |
| Step metadata | `step1_manifest.json`, `_cache/step1.json` + `step1.pkl` | `_cache/manifest.json` (v2 ignores the v1 files) |

What v2's parsers actually read is small: the `.out` (all ORCA operations), plus — for
`goat` / `docker` / `solvator` — the ensemble sidecar **named after the `.out` stem**. The
`_opt` rename is the step most hand-migrations miss: everything else can be in place and
the step still ledgers as `unparseable`, because a successful GOAT run's ensemble sits
under a stem v2 never looks at. For one structure of a GOAT step:

```bash
cd outputs/<run>/step1
mkdir 0
for f in step1_structure_0*; do mv "$f" "0/${f/step1_structure_0/step1_0}"; done
# fold v1's `_opt` base into the v2 stem for the files the parser reads:
mv 0/step1_0_opt.finalensemble.xyz 0/step1_0.finalensemble.xyz
```

Then write the step's `_cache/manifest.json` — the record `rebuild-cache` requires, naming
which input produced which output for which structure id (absolute paths):

```json
{
  "operation": "goat",
  "engine": "orca",
  "fingerprint": "",
  "files": [
    {
      "input": "/abs/path/outputs/<run>/step1/0/step1_0.inp",
      "output": "/abs/path/outputs/<run>/step1/0/step1_0.out",
      "id": "0"
    }
  ]
}
```

The empty `fingerprint` is accepted deliberately: a manifest that cannot *prove* the
outputs match the current config is exactly what a pre-v2 tree looks like, and you — by
writing it — are the one asserting they do. That assertion is real: the YAML you rebuild
under must be the configuration that produced these outputs (template, options, charge,
multiplicity), because a re-parse checks none of that.

Then rebuild step by step, in order — each `rebuild-cache N` serves steps `1..N-1` from
the caches the previous rounds wrote and ends the run at `N`:

```bash
chemrefine rebuild-cache input.yaml 1
chemrefine rebuild-cache input.yaml 2
# … up to the last step
```

Two things to expect. A trailing
`ChemRefineError: step N halted (on_failure=stop)` after a rebuild is **not** the rebuild
failing — the cache was written; it is the failure ledger reporting structures the
re-parse classified as failed, exactly as a fresh run would (check
`_cache/failed_jobs.json`, and note v2's default `on_failure` is `stop` where v1 skipped).
And the v1 metadata (`step1_manifest.json`, `_cache/step1.json`, `step1.pkl`) can stay put:
v2 ignores it, and sweeps the `.pkl` on the next `run`/`rerun`.

This recipe is validated for single-structure operations (`opt_sp` / `sp` / `freq`) and
the ensemble ones (`goat` / `docker` / `solvator`). A v1 NMS step's displaced-child layout
differs more deeply — re-running it under v2 (`chemrefine rerun N`) is the supported path
there.

# Migrating from v1 to v2

ChemRefine v2 reads a v1.3.1 workflow YAML through a translation layer: the config
loader rewrites the old keys to the current schema at parse time and reports every
rewrite as a one-line deprecation warning naming what to write instead. This page
documents that mapping so you can modernise your files when convenient.

Everything below is rewritten automatically except three things, which are edits
by hand:

- `calculation_type`, replaced with `engine:` + `operation:`;
- a step with no `engine:`, which v1.3.1 read as DFT and v2 refuses — write
  `engine: orca`;
- a training step's template, which moves from MACE's own YAML in `stepN.inp` to a
  `stepN.yaml` with placeholders — see [Training steps](#training-steps).

v1.3.1's README also showed a `sampling:` key on a step, which v1.3.1 never read (it read
`sample_type:`) and v2 refuses: write `sample:` — see [Sampling](#sampling).

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

A half-migrated file can carry a v1 key beside its v2 spelling — `initial_xyz` and
`input`, a block's key and the same key under `options:`, `energy` and
`window_kcalmol`. Where both say the same, that is one setting. Where they differ, the
file is refused with both keys named: v1.3.1 never read the v2 key and v2 never reads
the v1 one, so either choice would be a guess — keep one.

## Top level

| v1.3.1                          | v2                                |
|---------------------------------|-----------------------------------|
| `orca_executable: /path/orca`   | `executables: { orca: /path/orca }` |
| `initial_xyz: ./seed.xyz`       | `input: ./seed.xyz`               |

## Per step

| v1.3.1                                         | v2                                            |
|------------------------------------------------|-----------------------------------------------|
| `engine: "DFT"`                                | `engine: orca` (an engine name is read in any case) |
| `engine: "MLFF"`, with or without an `mlff:` block | `engine: mlip-extopt` — ORCA over the model, as in v1.3.1 ([details](#machine-learned-steps)) |
| `engine: "MLIP"` with an `mlip:` block         | `engine: mlip-extopt`, as for MLFF ([details](#machine-learned-steps)) |
| `mlff:` / `mlip:` / `trainer:` block           | `options: { … }` on the step v1.3.1 used it on; dropped elsewhere |
| `operation: "OPT+SP"` / `"GOAT"`               | unchanged — read in any case, `+` for `_`     |
| `operation: "MLFF_TRAIN"`                      | `engine: mlip-train` ([training steps](#training-steps)) |
| `normal_mode_sampling: true`                   | `nms: true`                                   |
| `normal_mode_sampling_parameters: { … }`       | folded into `options:` (see NMS table below)  |
| `calculation_type: "DFT"` **(manual)**         | `engine: orca` + `operation: opt_sp`          |

### Normal-mode sampling knobs

`normal_mode_sampling_parameters` keys move into `options:` and are renamed. v1.3.1 read
them only on a step with `normal_mode_sampling: true`, and only the three below, so on any
other step, and any other key, they are dropped with a warning:

| v1.3.1                          | v2 `options` key       | notes |
|---------------------------------|------------------------|-------|
| `calc_type: rm_imag` (default)  | `target: ts`           | keep exactly one imaginary mode (first-order saddle / TS) |
| `calc_type: random`             | `target: random`       | sampling along random modes |
| `displacement_vector`           | `displacement_value`   | ± displacement magnitude (Å) |
| `num_random_displacements`      | `num_random_displacements` | how many modes `random` displaces along |

> v2 adds a third target, `target: minimum` (remove **all** imaginary modes → a true minimum), which
> v1.3.1 did not have. A bare `normal_mode_sampling: true` maps to `target: ts` to match v1.3.1's default.

> **Scheduling.** The two rounds share **one** queue and one `max_cores`/`max_gpus` budget. A structure's
> displaced children are submitted the moment *its own* round-1 job finishes, alongside whatever is still
> running, rather than after the whole of round 1 drains — so the slots freed by the early finishers go
> to round-2 work instead of idling. See [Normal-Mode Sampling](../workflow/nms.md).

## Machine-learned steps

v1.3.1 ran an `engine: MLFF` step through ORCA, with a machine-learned gradient server
standing in for ORCA's own method — `mlip-extopt` in v2, which is where the step is
translated. Its `mlff:` block becomes the step's `options:`. v1.3.1 read four keys from
the block, each with a default of its own, and v2's defaults differ, so a key the block
leaves unset is written out as v1.3.1 ran it:

| v1.3.1 key | v1.3.1 ran it unset as | translated to |
|------------|------------------------|---------------|
| `task_name` | `mace_off` | `task_name: mace_off` |
| `model_name` | `medium` | `model_name: medium` under `mace_off` and `mace_mp`, the families whose loader read it; under any other task v1.3.1 ignored the default or failed on it, so the library's own default runs. Under `mace_omol`, whose loader ignored `model_name`, a value the block names is dropped |
| `device` | `cuda` | `device: cuda` — on a host with no GPU, write `device: cpu` |
| `bind` | `127.0.0.1:8888` | dropped: the server's address is managed for you |

Under `mace_off` and `mace_mp` — the families whose loader read a file — a `model_name`
that names a checkpoint (a value holding a `/` that is not an `https:` URL) becomes
`model_path`, with a leading `~` expanded as the job's shell expanded it. A relative path
is then read from the config file's directory, where v1.3.1 read it from the job's
scratch directory: the MLIPTraining tutorial's `../step3/checkpoints_dir/…` is written
relative to the config file instead (`./outputs/step3/…`).

v1.3.1 chose SevenNet by a `model_name` starting `sevenn` and read the release from
`task_name` (`{model_name: sevennet, task_name: 7net-0}`), when no earlier library's
task matched; that pair is translated to `task_name: sevenn` with the release as
`model_name`.

v1.3.1's README also put `model_name` and `task_name` on an MLFF step itself, beside or
instead of the block. v1.3.1 read only the block, so such a step ran its defaults (MACE-OFF
`medium`); the translation honours what the step names and moves the values into
`options:`, unless the block names the same key, whose value is the one v1.3.1 ran.

Each written-out default and each dropped key is reported with a deprecation warning of
its own. Any other key in the block configured nothing in v1.3.1 and is dropped, reported;
so is a block on a step v1.3.1 never used it on — an `mlff:` block beside `engine: DFT`
stays a DFT step, as it ran.

The template is what makes the step run the model: `! ExtOpt` hands ORCA's gradients to
the server. A v1.3.1 template without it ran ORCA's own method under the MLFF label while
the server idled, and 2.0 refuses such a template when the run starts — add `ExtOpt` to run
the model the step names, or write `engine: orca` to keep what v1.3.1 computed.

A v1.3.1 `engine: MLIP` step with an `mlip:` block is translated the same way, to
`mlip-extopt`: v1.3.1 read that block with `mlff`'s keys and defaults but never started the
server for it, so the step ran its template as ORCA's own method — give the template
`! ExtOpt` to run the model the block names, or write `engine: orca` to keep what v1.3.1
computed. Without a block, `engine: MLIP` keeps its name: in v2 that is the direct engine,
a different path, where the model runs a Python template with no ORCA.

FAIRChem's model names are the installed `fairchem-core`'s, and v1.3.1's `uma-s-1` is not
among the current library's — name a current one, such as `uma-s-1p2`; the library refuses
a name it does not have when the job loads the model.

### Training steps

`operation: MLFF_TRAIN` (or `MLIP_TRAIN`) becomes `engine: mlip-train` +
`operation: mlip_train`, and the `trainer:` block becomes its `options:`. v1.3.1 trained
MACE alone, read three keys from the block and split off a *test* set and nothing else —
validation was the MACE template's own `valid_fraction` — so the translation keeps that
partition and writes out what v1.3.1 used where the step names none:

| v1.3.1 | meant | translated to |
|--------|-------|---------------|
| (always MACE) | the library | `task_name: mace_off` |
| `trainer.device` (`cuda`) | the job's device | `device` (`cuda` when unset) |
| `trainer.valid_fraction` (`0.1`) | the **test** split | `test_fraction` (`0.1` when unset) |
| `trainer.seed` (`42`) | the split's seed | `seed` |
| — | no validation split of its own | `valid_fraction: 0` |

Any other key in the block configured nothing in v1.3.1 and is dropped, reported — a
`trainer.model_name` included, which would otherwise turn a fit from scratch into a
fine-tune. To have ChemRefine carve the validation set instead, set `valid_fraction` and
hand it to the template as `valid_file: $VALID_SET`.

The template is the edit by hand: v1.3.1 read MACE's own
`run_train` YAML from `stepN.inp` and rewrote its file paths, while v2 renders
`stepN.yaml` and hands it the dataset through placeholders — see [Training a
potential](../engines/index.md#training-a-potential-mlip-train).

## Sampling

`sample_type: { method, parameters: { … } }` becomes a flat `sample: { method, … }`.
The methods were also renamed — `integer`/`high_energy` are now `min`/`max`, and
`energy_window` folded into `min` with a `window_kcalmol` knob:

| v1.3.1 method (+ parameters) | v2 |
|------------------------------|----|
| `boltzmann` (`weight`, `temperature`) | `boltzmann` (`percent_cumulative`, `temperature_k`) |
| `integer` (`num_structures`)        | `min` (`count`) |
| `high_energy` (`num_structures`)    | `max` (`count`) |
| `energy_window` (`energy` + `unit`) | `min` (`window_kcalmol`) |

`min` / `max` take **exactly one** of `count` / `window_kcalmol`. All the old
spellings auto-translate with one deprecation warning each. Each method read only its own
parameters, so any other key in the block is dropped with a warning, and where the block
named none v1.3.1's own default is written out: `energy_window`'s 0.5 hartree,
`integer`'s every structure (`count: 0`), `high_energy`'s one (`count: 1`).

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
    mlff: { model_name: medium, task_name: mace_off, bind: "127.0.0.1:8888" }
    sample_type: { method: boltzmann, parameters: { weight: 95 } }

# v2 (what the loader produces; also what you'd write by hand)
executables: { orca: /orca }
input: ./seed.xyz
steps:
  - step: 1
    operation: opt_sp
    engine: mlip-extopt
    options: { model_name: medium, task_name: mace_off, device: cuda }
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

A few defaults differ from v1.3.1's (all overridable):

- **`on_failure` defaults to `stop`** — a failure halts the run once the step's successes are cached, as
  v1.3.1 stopped a step with any output it could not parse. `on_failure: skip` drops the failures and
  continues; `best` backfills them.
- **`max_cores` defaults to `4`** (a safe local default) — raise it in the YAML or with `--maxcores`.
- **`--maxgpus`** is a new flag mirroring `--maxcores` (overrides `max_gpus`).
- **A machine-learned step written in v2's terms** runs v2's defaults — `task_name: omol`
  (FAIRChem), the library's own model and `device: cpu`. A step translated from v1.3.1 has
  v1.3.1's written out instead; see [Machine-learned steps](#machine-learned-steps).

## Failure handling & recovery

v1.3.1 stopped a step with any output it could not parse, and `--rerun_errors` resubmitted
the jobs it recorded. v2 has
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
which input produced which output for which structure id. Paths are relative to the step
directory, so the record moves with the tree (absolute paths are read too):

```json
{
  "operation": "goat",
  "engine": "orca",
  "fingerprint": "",
  "files": [
    {
      "input": "0/step1_0.inp",
      "output": "0/step1_0.out",
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
v2 never reads it.

This recipe is validated for single-structure operations (`opt_sp` / `sp` / `freq`) and
the ensemble ones (`goat` / `docker` / `solvator`). A v1 NMS step's displaced-child layout
differs more deeply — re-running it under v2 (`chemrefine rerun N`) is the supported path
there.

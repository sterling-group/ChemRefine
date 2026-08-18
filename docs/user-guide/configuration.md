# Configuration

A ChemRefine run is driven by a single YAML file: top-level defaults, then an
ordered list of `steps`. The file is validated by Pydantic — an unknown key or a
malformed value fails fast with a `ConfigError` before any job is submitted.

A minimal two-step workflow (MLIP screen → DFT refine):

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
    name: screen
    engine: mlip
    operation: opt_sp
    options: { model_name: medium, task_name: mace_off, device: cuda }
    sample: { method: boltzmann, percent_cumulative: 99 }

  - step: 2
    name: refine
    engine: orca
    operation: opt_sp
    template: dft_opt.inp
    sample: { method: min, window_kcalmol: 3.0 }
```

## Top-level keys

!!! note "Relative paths are resolved against the config file"
    `template_dir`, `output_dir`, `scratch_dir`, and `input` — when given as
    relative paths — resolve against the **directory containing the YAML file**,
    not the process working directory. So `chemrefine run proj/input.yaml` from
    anywhere finds `proj/templates` and writes `proj/outputs`. Absolute paths are
    used as-is.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `template_dir` | path | `./templates` | Directory holding the per-step engine templates and SLURM headers. |
| `scratch_dir` | path | `None` | Fast node-local working directory base. Unset ⇒ a per-calc `_work_…` dir is derived under `output_dir`; on HPC point it at node scratch (e.g. `/scratch/$USER`). Must differ from `output_dir`. |
| `output_dir` | path | `./outputs` | Where per-step results, caches, and `steps.csv` are written. |
| `input` | path | `None` | Seed structures: an `.xyz` (one structure per frame), a directory of `.xyz`, or a `.csv` of SMILES (column `smiles`). Unset falls back to `templates/step1.xyz`. |
| `charge` | int | `0` | Global molecular charge (per-step `charge` overrides). |
| `multiplicity` | int ≥ 1 | `1` | Global spin multiplicity `2S+1` (per-step `multiplicity` overrides). |
| `max_cores` | int ≥ 1 | `4` | Total CPU budget the throttler enforces across concurrent jobs. The `--maxcores` flag overrides this. |
| `max_gpus` | int ≥ 0 / `None` | `None` (auto) | Concurrent-GPU budget. `None` auto-resolves: unlimited under SLURM (the scheduler places GPUs via `--gres`), the detected device count (`nvidia-smi -L`) locally. The `--maxgpus` flag overrides this. |
| `slurm_template` | str | `cpu.slurm.header` | Default SLURM header basename in `template_dir`. A GPU step auto-picks `cuda.slurm.header`. |
| `slurm_array` | bool | `False` | Submit each step as SLURM job array(s) instead of one job per structure (ignored when running locally). |
| `dispatch` | `auto` / `local` / `slurm` | `auto` | How jobs are executed. `auto` submits via `sbatch` when it is on PATH and runs the generated scripts locally via `bash` otherwise. `local` forces the local runner even when an `sbatch` binary exists (e.g. a workstation with SLURM client tools but no reachable cluster); `slurm` requires `sbatch` and fails fast when it is missing. |
| `job_timeout_seconds` | float > 0 / `None` | `None` (wait forever) | How long a step may go with **nothing finishing** before giving up with exit code `8`. The clock restarts on every completed job, so a batch that keeps draining never trips it however long the whole step takes — size it against the longest *single* job, not the step. `None` is right under SLURM: the partition's own time limit already bounds the job. Set it when nothing else will — a `dispatch: local` run, or a cluster where a job can sit in `PD` indefinitely. It bounds *waiting*, not compute, and means the same thing on the per-job and `slurm_array` paths. |
| `executables` | map | `{}` | Tool → path map for external-binary engines, e.g. `{ orca: /opt/orca/orca }`. Importable backends (mlip, pyscf) need no entry. An engine may document extra keys: the qchem engine reads `qc` and `qcaux` as install roots — see its options tab below. |
| `steps` | list | — | **Required.** The ordered pipeline stages; `step:` numbers must form a contiguous `1..N`. |

## Per-step keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `step` | int ≥ 1 | — | **Required.** 1-based step number; drives directory naming and order. |
| `name` | str | `None` | Optional filesystem-safe label (letters/digits/`_`/`-`, not all-digits). Directory becomes `stepN_name/`; usable as a CLI target. |
| `engine` | str | — | **Required.** One of `orca`, `qchem`, `mlip`, `mlip-extopt`, `mlip-train`, `pyscf`, `pyscf-extopt`. |
| `operation` | str | `None` | Engine-defined: `opt_sp`, `sp`, `freq`, `pes`, `goat`, `docker`, `solvator`. (`mlip_train` is a legacy spelling: it named a step *kind*, which `engine: mlip-train` already says. Old configs are still translated.) **Optional** — when omitted, ORCA infers the run type from the template's `!` keyword lines (`GOAT`/`DOCKER`/`SOLVATOR`/a `%geom Scan` block/`Opt`/`OptTS`/`Freq`; `#` comments are ignored, matching is case-insensitive), defaulting to a single point if it finds no run-type keyword. An explicit value always wins — give it when inspection can't decide. |
| `template` | str |  `stepN.{inp,in,py}` | Engine input template basename (relative to `template_dir` if not absolute). |
| `slurm_template` | str | global | Per-step SLURM header override. |
| `charge` / `multiplicity` | int | global | Per-step overrides of the global values. |
| `options` | map | `{}` | Engine-specific knobs (see below). |
| `sample` | map | `None` | Survivor filter (see below). `None` keeps every structure. |
| `nms` | bool | `False` | Opt-in normal-mode sampling (honoured only for an NMS-capable engine: ORCA / ExtOpt / Q-Chem). Requires a frequency calc: an ORCA NMS step whose template has no `Freq` keyword is rejected at prepare time (set `operation` explicitly to override). The `target` (`minimum`/`ts`) is inferred from the template — `OptTS` → `ts`, else `minimum` — unless `options.target` is set. |
| `on_failure` | `stop`/`skip`/`best` | `stop` | What to do when some structures fail (after the convergence auto-retry below): `stop` (default) caches the successes then halts so failures are never silently dropped; `skip` drops them and continues; `best` keeps all, backfilling a failed structure with the best geometry obtained for it or, failing that, the input it was submitted with — including on step 1, where that input is a seed with no energy computed yet. A structure that *did not converge* is first retried once from its best geometry — the failed attempt is archived under `stepN/<id>/attemptK/` — before this policy applies. |

## Sample (survivor filter)

`sample` selects which structures advance to the next step. Common knobs:
`by_parent` (default `false` — filter globally; `true` filters within each
parent-ID group), `temperature_k` (default `298.15`, used by Boltzmann), and
`energy_type` (default `electronic`) — the energy the filter sorts/selects on:
`electronic`/`E`, `gibbs`/`G`, `enthalpy`/`H`, or `electronic_zero_point`/`E_ZPE`.
The non-electronic types require a frequency calc (thermochemistry); filtering
raises a clear error if the chosen energy wasn't computed.

`min` and `max` take **exactly one** selector: `count` (keep N) or
`window_kcalmol` (keep all within that energy window).

| `method` | Selector(s) | Keeps |
|----------|-------------|-------|
| `boltzmann` | `percent_cumulative` (default `99`) | Structures until the cumulative Boltzmann weight reaches the percentage. |
| `min` | `count` **or** `window_kcalmol` | The `count` lowest-energy structures (`0` = keep all), or all within `window_kcalmol` of the minimum. |
| `max` | `count` (≥ 1) **or** `window_kcalmol` | The `count` *highest*-energy structures, or all within `window_kcalmol` of the maximum (PES-style sampling). |

## Engines

Every registered engine, what template it reads, and what it can do:

<!-- chemrefine:engines -->

An engine with no declared `operation:` vocabulary treats the field as a free label —
only the engines listed above interpret it. `NMS` is the `nms: true` capability. See
[Installation → available backends](installation.md#available-backends) for the
`Backend env` column.

## Engine options

`options` is a free per-engine dict; each engine validates its own keys.

=== "mlip / mlip-extopt"

    | Key | Default | Description |
    |-----|---------|-------------|
    | `model_name` (aliases `model`, `size`) | `uma-s-1p2` | Model weights (a MACE size, a FAIRChem checkpoint, a SevenNet/ORB id). |
    | `task_name` (alias `task`) | `omol` | Method/head — **the only thing that selects the backend builder**. |
    | `model_path` | `None` | A local checkpoint to load *instead of* `model_name`, with the library `task_name` named. Selects nothing itself: to run a model an `mlip-train` step produced, name the same `task_name` it trained with. Relative paths resolve against the config file's directory. |
    | `device` | `cpu` | `cuda` or `cpu`. CPU is the floor that always runs; asking for a GPU is one line, whereas a wrong `cuda` default schedules a CPU job whose script then asks for a device it wasn't given. |
    | `cores` | `1` | Per-structure core budget. |
    | `backend_python` | `None` | Explicit interpreter for the backend (escape hatch). Normally unset: the step's managed env is resolved by name — see [Installation → available backends](installation.md#available-backends). |

=== "pyscf / pyscf-extopt"

    | Key | Default | Description |
    |-----|---------|-------------|
    | `method` | `dft` | `dft` or `hf`. |
    | `xc` | — (**required** for `dft`) | Exchange-correlation functional. No silent default — name it explicitly. |
    | `basis` | — (**required**) | Orbital basis set. No silent default — name it explicitly. |
    | `df` | `True` | Density fitting / RI (defaults on — large speed-up, negligible cost). |
    | `strict_scf` | `True` | Refuse to serve a gradient from an SCF that did not converge. PySCF returns the last iterate rather than raising, and ORCA's `.out` reports only *its own* geometry convergence — so a loose result would rank against converged siblings unmarked. Set `false` for a knowingly loose SCF. |
    | `device` | `cpu` | Compute device; drives `gpu` when `gpu` is unset (`cuda` ⇒ attempt GPU). |
    | `gpu` | derived from `device` | Attempt `gpu4pyscf` if installed (falls back to CPU). Set explicitly to override the `device`-derived default. |
    | `save_tensors` | `False` | Dump 1e/2e MO tensors after the SCF. |
    | `localized` | `False` | Boys-localize before tensor extraction. |
    | `tensor_folder` | `tensors` | Output dir for `save_tensors` `.npz`. A relative path (the default) is copied back into the structure's own dir (`outputs/stepN/<id>/tensors/`); an absolute path writes there directly. |
    | `cores` | `1` | Per-structure core budget. |
    | `backend_python` | `None` | Explicit interpreter for the backend (escape hatch). Normally unset: the `pyscf` managed env is resolved by name. |

=== "qchem"

    | Key | Default | Description |
    |-----|---------|-------------|
    | `cores` | `1` | OpenMP threads, rendered as `-nt N` (with `OMP_NUM_THREADS` and `QC_THREADS` exported beside it) and allocated as `--ntasks=1 --cpus-per-task=N`. Q-Chem takes its parallelism **on the command line**, never in the input file — so the number lives here, where ORCA's lives in the template's `%pal` (the rule is: the number lives where the program natively reads it). |
    | `nprocs` | `None` | **Opt-in MPI**: `-mpi -np P` (plus `-nt N` when `cores` > 1), allocated as `--ntasks=P --cpus-per-task=N` and charged as `P×N` cores (over `max_cores` is refused, not clamped). Q-Chem's MPI covers only some methods — leave this unset unless you know your method and build support it. The MPI install facts (`QCRSH`/`QCMPI` exports, MPI module loads) belong in the SLURM header. |
    | `save` | `False` | Copy the key scratch files (MO coefficients — the `.gbw`-analogue) back to the structure's dir (`outputs/stepN/<id>/<stem>/`). The job always runs with a savename so Q-Chem keeps them in scratch; this decides whether they come home. |
    | `device` | `cpu` | Inherited, and the Q-Chem engine never reads it: the job is launched the same way either way. The *pipeline* still acts on it — `cuda` charges a GPU against `max_gpus` and auto-picks `cuda.slurm.header` — so leave it at `cpu` for Q-Chem steps, or they queue for a device nothing will use. |

    **Install environment** (`executables`, config-level — machine facts, like the `orca`
    binary): `qchem` names the wrapper explicitly; `qc` names the install root and makes the
    run block export `QC`, extend `PATH` with `$QC/bin:$QC/bin/perl`, and invoke
    `$QC/bin/qchem`; `qcaux` overrides the auxiliary-file root for installs that keep it
    *beside* the QC root. With `qc` set and `qcaux` unset, `QCAUX=$QC/qcaux` is exported —
    Q-Chem's own documented default, just spelled visibly. A cluster whose header carries
    `module load qchem` needs none of the three.

    **Memory**: declare `mem_total` in the template's `$rem` block (Q-Chem's counterpart of
    ORCA's `%maxcore`) and the SLURM request follows it — a header whose own
    `--mem`/`--mem-per-cpu` already covers the declaration stands untouched; a short or
    absent one is extended to fit. No declaration → the header's memory policy stands.

    **Header rule**: a threaded Q-Chem job cannot span nodes — carry `#SBATCH --nodes=1` in
    the header your qchem steps use. `QCSCRATCH` needs no line anywhere: ChemRefine points
    it at the per-job work dir, and `scratch_dir` relocates that to fast local disk.

    Migrating a qcsetup-style environment file: `module load …`, compiler
    `LD_LIBRARY_PATH` exports and hostname conditionals go into the SLURM header body
    verbatim (the header's non-`#SBATCH` lines run in every job — locally too, so guard
    cluster-only commands with `command -v module >/dev/null && module load …`, or keep a
    separate local header and point `slurm_template` at it); `export QC=…`/`QCAUX=…` become
    the `executables` keys above; the `QCSCRATCH` line becomes `scratch_dir`;
    `QCRSH`/`QCMPI` stay in the header and only matter for MPI.

    ```yaml
    template_dir: ./templates
    output_dir: ./outputs
    input: ./input.xyz
    max_cores: 16
    executables:
      qc: /groups/sterling/software-tools/qchem/qchem700
      qcaux: /groups/sterling/software-tools/qchem/qcaux
    steps:
      - step: 1
        engine: qchem            # templates/step1.in with $rem … $end; the $molecule
        options: { cores: 8 }    # block is generated per structure into job 1
    ```

=== "mlip-train"

    | Key | Default | Description |
    |-----|---------|-------------|
    | `task_name` (alias `task`) | — (**required**) | Which library trains — the same word that selects a backend for inference. No default: the inference default names a foundation model to *run*, which is a different choice. Trainable today: `mace_off`, `mace_mp`, `mace_omol`, and every FAIRChem head (`omol`, `omat`, `odac`, `oc20`, `oc22`, `oc25`, `omc`) — but see the FAIRChem note below. |
    | `model_name` (aliases `model`, `size`) | `uma-s-1p2` | The foundation model a run **starts from** — MACE's `foundation_model`. Set it to `""` to train from scratch, which on a few-dozen-structure dataset is rarely what you want. |
    | `model_path` | `None` | A local checkpoint to continue from, loaded by the library `task_name` names. Relative paths resolve against the config file's directory. |
    | `device` | — (**required**) | `cuda` or `cpu`. No default either way: `cpu` would silently hand you a job that grinds for days, `cuda` would silently charge the GPU budget and swap the SLURM header for a step that never asked. |
    | `gpus` | `1` | Data-parallel width (MACE's `--nproc_per_node`). ChemRefine never writes `--gres` — this is what it charges to the GPU budget and passes to the trainer; the *allocation* is the `--gres` line in your own `cuda.slurm.header`, so raise both together. |
    | `cores` | the step's whole budget | Unset means all of `max_cores`: the single training job *is* the step, unlike a per-structure step that shares the budget. |
    | `valid_fraction` | `0.1` | Share held out to validate on (steers training). A non-zero fraction always yields at least one structure. |
    | `test_fraction` | `0.0` | Share held out for a final evaluation the training never sees. Distinct from `valid_fraction`, and off by default — on a small dataset a test set is a luxury the training set cannot afford. |
    | `seed` | `42` | Seed for the split, so a re-run partitions identically. |
    | `backend_python` | `None` | Explicit interpreter for the trainer (escape hatch). Normally unset: the managed env is resolved by name. |

    The step's template is the **trainer's own config** (`stepN.yaml`), rendered rather than
    patched: ChemRefine substitutes `$TRAIN_SET`, `$VALID_SET`, `$TEST_SET`, `$RUN_DIR`,
    `$RUN_NAME`, `$DEVICE`, `$SEED`, `$NGPUS`, `$CORES`, `$FOUNDATION_MODEL`, `$CHARGE` and
    `$MULTIPLICITY`, and leaves everything else — including a config's own `${...}`
    interpolations — untouched. A template that never references the dataset placeholder its
    backend needs is rejected before anything is submitted.

    The trained model lands at `<step dir>/train/train[_stagetwo].model`, beside a
    `trained_model.json` recording which run produced it. That path is predictable *before*
    the training runs, so a later step can name it in `model_path`; retraining changes that
    step's cache key, so it re-runs rather than serving a result computed with the old weights.

    !!! note "FAIRChem templates must follow fairchem's own fine-tuning recipe"

        The template for a FAIRChem head is FAIRChem's hydra config, and its moving parts
        are not optional: the dataset stanza needs
        `transforms: {common_transform: {dataset_name: …}}` (the collater dispatches on the
        name that transform stamps), `tasks_list` defines fresh `energy`/`forces` tasks
        bound to that dataset name, and the model node is
        `initialize_finetuning_model` with replacement heads. This is the shape of
        fairchem's own `configs/uma/finetune/uma_sm_finetune_template.yaml`, and the
        shipped example follows it. A config that instead tags data via
        `a2g_args: {task_name: …}` or reuses the checkpoint's task list fails inside
        FAIRChem's collater with `TypeError: unhashable type: 'list'`.

        Fine-tuning a UMA checkpoint is a **GPU-scale job**: on CPU the optimizer states and
        conservative-force graph need roughly 8 GB for `uma-s`. Size `cores`/`device`
        accordingly.

=== "nms (when nms: true)"

    | Key | Default | Description |
    |-----|---------|-------------|
    | `target` | inferred | `minimum` removes every imaginary mode; `ts` keeps the reaction coordinate and removes the rest; `random` explores. Default: **inferred** from the template (`OptTS` → `ts`, else `minimum`); set this to override. |
    | `displacement_value` | `1.0` | ± displacement (Å) per selected mode. |
    | `num_random_displacements` | `1` | `random` only: how many modes to draw. |
    | `ts_mode_index` | `None` | `ts` only: which imaginary mode to keep (default: the largest-magnitude one). Must name a mode the structure's frequency calculation flagged imaginary — an index that matches none is rejected per structure, since which modes are imaginary is a property of the result rather than of the config. |
    | `seed` | `42` | RNG seed for reproducible `random` selection. |

## Output layout

Each structure gets its **own directory** under the step dir, so a calculation's
files sit together and never collide across structures. The input geometry is
kept as `…_inp.xyz` distinct from the engine's output geometry:

```
outputs/
├── step1_screen/
│   ├── 0/                     step1_0_inp.xyz  step1_0.{inp,out,xyz,...}  step1_0.runlog
│   │   └── 0_m5_pos/          NMS round-2 (a displaced re-run) nests under its parent
│   ├── 1/                     …
│   ├── step1_ensemble.xyz     every final structure of the step, one multi-frame XYZ
│   ├── step1_survivors.xyz    the subset the `sample:` filter kept for the next step
│   └── _cache/                step.json + arrays.npz, manifest.json, failed_jobs.json
├── step2_refine/<id>/…
└── steps.csv                  Boltzmann summary per surviving structure
```

The two ensemble files are the step's results as geometry: `stepN_ensemble.xyz`
holds **every** parsed final structure (a GOAT step's whole conformer ensemble,
say), `stepN_survivors.xyz` only what the `sample:` filter passed on. Frames are
sorted ascending by the step's own ranking energy (the `sample.energy_type`,
electronic by default — the same energy `steps.csv` reports), and each comment
line carries `stepN id=<id> E=<hartree> Eh`, so a frame is traceable to its
structure directory and its `steps.csv` row. Both files are rewritten
deterministically on every run of the step — `resume`, a cache hit, and
`rebuild-cache` regenerate them byte-identically.

NMS round-2 — re-optimising a displaced geometry — is treated like any "redo this
structure" step: the child lives in a sub-directory *inside* its parent's directory
(`stepN/<parent>/<child>/`), the same place a failure re-run would go.

## Legacy (v1.3.1) configs

Old keys are auto-translated with one deprecation warning each
(`initial_xyz`→`input`, `orca_executable`→`executables`, `sample_type`→`sample`,
the `mlff:`/`pyscf:` engine blocks → `options:`, `normal_mode_sampling`→`nms`, …).
Only `calculation_type` is a hard error. See
[Migrating from v1 to v2](../migrating-v1-to-v2.md).

# Engines & backends

An **engine** is what a step computes with. Every step names one in `engine:`, and the
engine decides what its template looks like, which `operation:` words it understands, and
whether it can do [normal-mode sampling](../workflow/nms.md).

<!-- chemrefine:engines -->

That table is generated from the registry when these docs are built, so it cannot fall
behind the code. Reading it:

- **Step template** — the file the engine reads for each step, `templates/stepN.<suffix>`,
  and the format it is written in. This is where a QM engine's real settings live.
- **`operation:`** — the vocabulary that engine interprets. An engine listing *any* treats
  the field as a free label; ORCA additionally infers the run type from its template when
  `operation:` is omitted.
- **NMS** — whether `nms: true` is honoured. Detected by `isinstance`, never a flag.
- **Backend env** — the Python stack the engine needs, if any, and the name
  [`chemrefine backends install`](installing.md) takes. ORCA and Q-Chem need none: they
  are programs you install yourself and name in `executables:`.

`options:` is a free per-engine dict and each engine validates its own keys — an
undeclared key is reported by `chemrefine validate` as a silent no-op rather than
ignored. The sections below are those keys, one engine family at a time.

## ORCA (`orca`)

ORCA is configured **through its template**, not through `options:` — it declares no
option model at all, which is why the table above shows none. The `!` keyword lines,
`%pal` core count, `%maxcore` memory, `%geom` scan and constraint blocks all live in
`templates/stepN.inp`, which is an ORCA input with the geometry block generated per
structure.

That is also why ORCA can infer `operation:`: when the key is omitted, the template's
keyword lines decide (`GOAT` / `DOCKER` / `SOLVATOR` / a `%geom Scan` block / `Opt` /
`OptTS` / `Freq`, case-insensitively, `#` comments ignored), defaulting to a single point.
An explicit `operation:` always wins — set it when inspection cannot decide.

`executables: { orca: /path/to/orca }` names the binary; an absolute path is required for
ORCA's own MPI launcher to work.

## Q-Chem (`qchem`)

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

## Machine-learned potentials (`mlip`, `mlip-extopt`)

| Key | Default | Description |
|-----|---------|-------------|
| `model_name` (aliases `model`, `size`) | `""` (the library's own default) | Model weights, in whatever spelling the library `task_name` selected uses — a size for MACE, a checkpoint name for FAIRChem (its default is `uma-s-1p2`), an id for the others. Unset means the chosen library picks its own default — one spelling could not be right for every library at once. |
| `task_name` (alias `task`) | `omol` | Method/head — **the only thing that selects the backend builder**. |
| `model_path` | `None` | A local checkpoint to load *instead of* `model_name`, with the library `task_name` named. Selects nothing itself: to run a model an `mlip-train` step produced, name the same `task_name` it trained with. Relative paths resolve against the config file's directory. |
| `device` | `cpu` | `cuda` or `cpu`. CPU is the floor that always runs; asking for a GPU is one line, whereas a wrong `cuda` default schedules a CPU job whose script then asks for a device it wasn't given. |
| `cores` | `1` | Per-structure core budget. |
| `backend_python` | `None` | Explicit interpreter for the backend (escape hatch). Normally unset: the step's managed env is resolved by name — see [Installing engines & backends](installing.md#available-backends). |

## Training a potential (`mlip-train`)

| Key | Default | Description |
|-----|---------|-------------|
| `task_name` (alias `task`) | — (**required**) | Which library trains — the same word that selects a backend for inference. No default: the inference default names a foundation model to *run*, which is a different choice. Which libraries can train is the `Engines` column of the [backend table](installing.md#available-backends) — a backend without a trainer is refused by name. See also the FAIRChem note below. |
| `model_name` (aliases `model`, `size`) | `""` (train from scratch) | The foundation model a run **starts from** — MACE's `foundation_model`. Unset trains from scratch (`started_from: scratch` in the runlog), which on a few-dozen-structure dataset is rarely what you want — name the foundation model to fine-tune. |
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

The device, the seed and the foundation weights are **plan facts**: the step's options
are their source of truth — the scheduler charges by them and the cache fingerprint
records them — and how they reach the training program depends only on what kind of
program it is. A library trained through ChemRefine's shared driver (one with no
training CLI of its own) receives them on the driver's **own command line**, so they
hold whether or not the template references their placeholders, and a template value
that *contradicts* the step's options is refused by name. A library run through its own
CLI reads only the config the template becomes, so there each plan fact the library
reads from its config is a **required placeholder** — a template that never references
it is rejected before anything is submitted, instead of the library's own fallback
quietly overriding the step.

The trained model lands under `<step dir>/train/` at a name fixed per library — MACE's
`train[_stagetwo].model`, FAIRChem's `train/checkpoints/final/inference_ckpt.pt`,
SevenNet's `checkpoint_best.pth`, CHGNet's `train.pth.tar`, ORB's `train.ckpt` — beside a
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

### MLIP training templates, per backend

The template is always the *trainer's* config; what that means differs per library, and
each trainer refuses a template missing its own required placeholders before anything
submits:

- **MACE** — MACE's own training YAML with `$TRAIN_SET`, `$RUN_NAME`, `$RUN_DIR`,
  `device: $DEVICE` and `seed: $SEED` where those values go; the dataset is written with
  MACE's own `REF_*` label keys, so no `energy_key`/`forces_key` line is needed. The
  [MLIP training tutorial](../tutorials/mlip_training.md) walks a complete one.
- **FAIRChem** — fairchem's hydra config per the note above; requires `$TRAIN_SET`,
  `$VAL_SET`, `$RUN_DIR`, `$RUN_NAME`, `device_type: $DEVICE` and the `seed: $SEED`
  keys the shipped example threads through `job` and the dataset stanzas.
- **SevenNet** — SevenNet's own `input.yaml` (start from `sevenn preset fine_tune >
  step{N}.yaml`), with `$TRAIN_SET` in `data.load_trainset_path` and `device: $DEVICE`
  (absent, SevenNet takes cuda whenever torch sees one — the step's own device must win);
  `$VALID_SET` / `$FOUNDATION_MODEL` → `train.continue.checkpoint` as wanted. A
  validation split is required: `checkpoint_best.pth`, the model the step adopts, is
  written when the validation metric improves.
- **CHGNet** — chemrefine's own small schema (CHGNet has no config format): `train_set:
  $TRAIN_SET`, `valid_set: $VALID_SET`, `run_name: $RUN_NAME`, plus the `Trainer` knobs
  (`epochs`, `learning_rate`, `batch_size`, `targets`). The step runs chemrefine's shared
  train driver inside the backend env; the device, the seed and a `model_path`/
  `model_name` to start from arrive on the driver's command line from the step's options
  — never through the template.
- **ORB** — the same driver route: `train_set: $TRAIN_SET`, `run_name: $RUN_NAME`, and a
  `base_model:` naming the pretrained loader (the architecture — the step's `model_name`
  serves when the template names none); a local checkpoint arrives as the step's
  `model_path`, on the driver's command line. No validation file — orb's fine-tune loop
  is train-only.

Whatever trained, running the result is the same one line: `model_path:` pointing at the
artifact, with the same `task_name`.

## PySCF (`pyscf`, `pyscf-extopt`)

Both engines read the SCF selection — `pyscf` renders your `stepN.py` with `$METHOD` /
`$XC` / `$BASIS` / `$DF`, `pyscf-extopt` builds a gradient server from the same values. The
**ExtOpt only** rows are the ones a server has to do *for* you: on the ExtOpt path ORCA drives
and there is no `stepN.py`, so anything after the SCF has nowhere else to live. In a direct
step that work is yours to call — `chemrefine.engines.pyscf._runtime` exports
`get_active_space_tensors` and `save_tensors` — so those knobs are rejected there by name
rather than accepted and ignored.

| Key | Default | Description |
|-----|---------|-------------|
| `method` | `dft` | `dft` or `hf`. |
| `xc` | — (**required** for `dft`) | Exchange-correlation functional. No silent default — name it explicitly. |
| `basis` | — (**required**) | Orbital basis set. No silent default — name it explicitly. |
| `df` | `True` | Density fitting / RI (defaults on — large speed-up, negligible cost). |
| `strict_scf` *(ExtOpt only)* | `True` | Refuse to serve a gradient from an SCF that did not converge. PySCF returns the last iterate rather than raising, and ORCA's `.out` reports only *its own* geometry convergence — so a loose result would rank against converged siblings unmarked. Set `false` for a knowingly loose SCF. |
| `device` | `cpu` | Compute device; drives `gpu` when `gpu` is unset (`cuda` ⇒ attempt GPU). |
| `gpu` | derived from `device` | Attempt `gpu4pyscf` if installed (falls back to CPU). Set explicitly to override the `device`-derived default. |
| `save_tensors` *(ExtOpt only)* | `False` | Dump 1e/2e MO tensors after the SCF. |
| `localized` *(ExtOpt only)* | `False` | Boys-localize before tensor extraction. |
| `tensor_folder` *(ExtOpt only)* | `tensors` | Output dir for `save_tensors` `.npz`. A relative path (the default) is copied back into the structure's own dir (`outputs/stepN/<id>/tensors/`); an absolute path writes there directly. |
| `cores` | `1` | Per-structure core budget. |
| `backend_python` | `None` | Explicit interpreter for the backend (escape hatch). Normally unset: the `pyscf` managed env is resolved by name. |

## The ExtOpt engines (`mlip-extopt`, `pyscf-extopt`)

Two of the engines end in `-extopt`, and the table shows why they look like ORCA: their
step template is an ORCA `.inp`, they answer to ORCA's `operation:` vocabulary, and they
can do NMS. They *are* ORCA runs — ORCA drives the optimiser, the geometry steps and the
frequency analysis — with one substitution: where ORCA would call its own SCF for energies
and gradients, it calls the backend instead.

```
ORCA (%method ProgExt)  →  extopt bridge  →  local HTTP server  →  MACE / UMA / PySCF
        ↑                                                                   │
        └──────────────────── .engrad: energy + gradient ←──────────────────┘
```

The trade is worth making in both directions:

- **The backend gets ORCA's machinery.** Its optimiser, its coordinate system, its
  numerical Hessian — so an MLIP can locate a transition state and produce a real
  frequency table, which is what makes `nms: true` meaningful for a potential at all.
- **ORCA gets the backend's speed.** The expensive part of each geometry step is the
  energy and gradient, and that is exactly the part the potential replaces.

The direct engines (`mlip`, `pyscf`) skip ORCA entirely: they render a `stepN.py` that
calls the library, which is faster and simpler when a single point or a plain optimisation
is all that is wanted. Pick `-extopt` when you need ORCA to be in charge of the geometry.

The server is started per job, and it is a real trust boundary — loopback only, a
kernel-assigned port, a per-run bearer token, and a `0600` sidecar for it. The details are
in [Security & trust boundaries](../internals/security.md#the-extopt-compute-server); the
place it sits in the run is in [Architecture](../internals/architecture.md#submit-compute-flow).
